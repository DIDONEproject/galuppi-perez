import pickle
import warnings
from collections import namedtuple
from datetime import datetime
from pathlib import Path

import numpy as np
from autosklearn.classification import AutoSklearnClassifier
from autosklearn.metrics import balanced_accuracy as autosklearn_balanced_accuracy
from autosklearn.metrics import mean_absolute_error as autosklearn_mae
from sklearn import model_selection
from sklearn.exceptions import ConvergenceWarning
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

from .data import load_features
from .model_wrappers import DidoneBagging, DidoneClassifier
from .plotting import plot_time_performance, plotly_save
from .validation import crossvalidation

warnings.filterwarnings(action="ignore", category=ConvergenceWarning)


# These lines are to fix a bug in autosklearn
backedup_balanced_accuracy = autosklearn_balanced_accuracy._score_func


def corrected_balanced_accuracy(x, y, **kwargs):
    return backedup_balanced_accuracy(np.round(x), np.round(y), **kwargs)


autosklearn_balanced_accuracy._score_func = corrected_balanced_accuracy
# bug fixed


def automl(
    label=None,
    data_x_y=None,
    prehook=None,
    post_split_hook=None,
    output_dir="automl",
    interpretable="none",
    splitter=None,
    autosklearn_kwargs={},
    tmp_folder=None,
    automl_time=4 * 3600,
    crossvalidation_kwargs={},
    skipsearches=False,
    skipbagfitting=False,
):
    """
    Performs a full optimization and validation using auto-sklearn.

    Parameters
    ----------

    `label` : str or None
        the label that you want to classify or to regress
        TODO: regression not supported yet!

        If None, `data_x_y` should be provided. Ignored if `data_x_y` is given.

    `data_x_y` : Optional[Tuple[array-like]]
        A tuple containing X and y data. If given, `label` is ignored. If not
        given, `label` sohuldn't be None. If given, `pre_split_hook` is ignored
        too.

    `prehook` : callable or None
        A callable that is called after having loaded the data but before of any processing. It receives the
        output of ``load_features`` (i.e. 3 ``pandas.Data`` `data`, `X`, and
        `y`) and must return two variables: `X` and `y`. Aside from this, it
        can do anything it likes, even returning other data (e.g. even data not
        coming from musiF); Ignored if `data_x_y` is not None.

    `model_fname` : str or pathlib.Path
        The path where the automl model is saved. You can load it with
        something like: ``model = pickle.load(open('./automl.pkl', 'rb'))``
        If `None`, it is not saved.

    `interpretable` : str
        A string that says if using interpretable pipelines or not.
        Accepted values are:
        * ``'linear'`` : in this case, only linear models are considered
        * ``'tree'`` : only decision trees are considered
        * ``'none'`` : none models are considered, so that the built model will
            likely be not interpretable

        When using ``'linear'``, ``'tree'``, feature processing is set to
        include only feature selection methods (this can still be overridden by
        arguments passed via `autosklearn_kwargs`). Finally, explanotary plots
        and visualizations are created for the best model and saved to file in
        folders like ``model_fname.stem + '_' + interpretable + '-' +
        a_number``.

        Specifically, if the best classifier is a tree, its decision tree is
        transformed in graphviz format and a ``graphviz.Source`` object is
        returned. If, instead, the best classifier is a linear model, the
        coefficients that explain the 50% of the total sum are plotted in a bar
        plot. The ``plotly.graph_objects.Figure`` object is returned. In both
        cases, the output is rendered by ipython/jupyter.

        An exception is returned if ``'linear'`` is used in a multi-label
        classification problem.

    `splitter` : Optional[sklearn.model_selection._split.BaseCrossValidator]
        An sklearn-like splitter used for evaluating the whole automl
        optimisation. If None, an hold-out evaluation is used, by using a
        `StratifiedShuffleSplit` with a custom seed and only one split. The
        ratio used is 80-20.

    `automl_time` : int
        Number of seconds used for the AutoML fitting procedure

    `autosklearn_kwargs` : dict
        Optional arguments for
        ``autosklearn.classification.AutoSklearnClassifier``
        The provided dict will overwrite our defaults:
            ```
            dict(
                time_left_for_this_task=automl_time,
                n_jobs=-1,
                seed=2311,
                memory_limit=3000,
                per_run_time_limit=automl_time // 10,
                ensemble_size=5,
                ensemble_nbest=10,
                max_models_on_disc=10,
                tmp_folder=tmp_folder,
                metric=autosklearn_balanced_accuracy,
                resampling_strategy=splitter,
            )
            ```

    `crossvalidation_kwargs` : dict
        Optional arguments for ``crossvalidation``, except ``y``.

    Returns
    -------

    `NamedTuple` :
        a tuple containing the following data under these 3 fields:
        `crossval_scores`, `crossval_figure`

    `list[Union[float, np.ndarray]]` :
        the scores computed after the crossvalidation of the model. See
        ``crossvalidation()`` for more info about the used scores.

    `plotly.graph_objects.Figure` or None :
        the object containing the figure created after crossvalidation, if
        plotting is enabled (default).See ``crossvalidation()`` for more info
        about enabling/disabling plotting.

    """
    assert (
        data_x_y is not None or label is not None
    ), "Please, specify label or data_x_y"

    # create the directory for the plots
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    if data_x_y is None:
        data, X, y = load_features(label)

        if prehook:
            X, y = prehook(data, X, y)
    else:
        X, y = data_x_y

    # reset index to prevent possible mismatching inserted by `prehook`
    X.reset_index(drop=True, inplace=True)
    y.reset_index(drop=True, inplace=True)

    if splitter is None:
        splitter = model_selection.StratifiedShuffleSplit(
            train_size=0.8, n_splits=1, random_state=213
        )

    # encoding categories
    x_encoder = OrdinalEncoder()
    feature_names = X.columns
    X = x_encoder.fit_transform(X)
    X_ = X.copy()
    y_encoder = LabelEncoder()
    encoded_y = y_encoder.fit_transform(y)

    ensemble_nbest = 10
    our_autosklearn_kwargs = dict(
        time_left_for_this_task=automl_time,
        n_jobs=-1,
        seed=2311,
        memory_limit=30000,
        per_run_time_limit=automl_time // ensemble_nbest,
        ensemble_size=5,
        ensemble_nbest=ensemble_nbest,
        max_models_on_disc=ensemble_nbest,
        tmp_folder=tmp_folder,
        metric=autosklearn_balanced_accuracy,
        resampling_strategy=splitter,
    )

    # including only interpretable algorithms
    if interpretable != "none":
        classifiers_include = []
        feature_include = []
        if interpretable == "tree":
            classifiers_include.append("decision_tree")
            feature_include.append("no_preprocessing")

        if interpretable == "linear":
            classifiers_include += ["passive_aggressive", "sgd", "lda", "liblinear_svc"]
            feature_include += ["pca", "truncatedSVD", "fast_ica"]

        our_autosklearn_kwargs["include"] = {
            "classifier": classifiers_include,
            "feature_preprocessor": feature_include,
        }

    for k, v in autosklearn_kwargs.items():
        our_autosklearn_kwargs[k] = v

    # Create auto-sklearn classifier
    ensemble = AutoSklearnClassifier(**our_autosklearn_kwargs)

    # Fitting the model
    if skipsearches:
        # fnames = sorted(Path(output_dir).glob("automl-*.pkl"), key=lambda x: x.stat().st_mtime)
        classifier = pickle.load(open("classifier.pkl", "rb"))
    else:
        print("Starting AutoML")
        ensemble.fit(X, encoded_y, dataset_name="Didone")
        # DidoneClassifier is an object that prevents the call to `fit` and instead
        # uses `refit`
        classifier = DidoneClassifier(
            ensemble.automl_.models_,
            ensemble.automl_.ensemble_,
            autosklearn_mae,
            x_encoder,
            y_encoder,
            standardize=False,
            n_jobs=our_autosklearn_kwargs["n_jobs"],
        )

        if output_dir:
            time_stamp = datetime.now().strftime("%m_%d-%H_%M")
            pickle.dump(
                ensemble, open(Path(output_dir) / f"automl-{time_stamp}.pkl", "wb")
            )
            plot_time_performance(
                ensemble, fname=output_dir / f"automl_optimization-{time_stamp}.svg"
            )
            pickle.dump(classifier, open(Path(output_dir) / "ensemble.pkl", "wb"))

    if not skipbagfitting:
        bag = DidoneBagging.default_init(classifier)
        print("Building best model using the whole dataset")
        bag.fit(X_, y)
        if output_dir:
            pickle.dump(bag, open(Path(output_dir) / "bag.pkl", "wb"))

    print("Evaluating the best automl pipeline")

    scores, validation_fig = crossvalidation(
        classifier,
        X_,
        y,
        label_encoder=y_encoder,
        plotting=True,
        stdout=False,
        cv=splitter,
        n_jobs=-1,
        classes=y_encoder.classes_,
        **crossvalidation_kwargs,
    )

    # Saving model
    if output_dir:
        plotly_save(validation_fig, Path(output_dir) / "crossvalidation.svg")
        pickle.dump(scores, open(Path(output_dir) / "crossval_scores.pkl", "wb"))

    AutoMLResult = namedtuple("AutoMLResult", ["crossval_scores", "crossval_figure"])
    return AutoMLResult(scores, validation_fig)
