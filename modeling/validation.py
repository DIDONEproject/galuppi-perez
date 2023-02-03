"""
This module implements some utilities to train and validate effective models on
the Didone dataset. Example of usage:

Classifying passions: ``automl('Passion')``
Classifying composers: ``automl('Composer')``

TODO add:
 * example to classify complex things (e.g. decade)
 * example to validate custom pipeline
 * example to use restricted pipeline
 * example to regress somethings
"""
from copy import copy

import numpy as np
import pandas as pd
from sklearn import metrics, model_selection

from .conf_matrix_hack import CustomEval
from .plotting import plot_confusion_matrices

import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings(action="ignore", category=ConvergenceWarning)

def crossvalidation(
    estimator,
    X,
    y=None,
    label_encoder=None,
    plotting=True,
    scoring=["balanced_accuracy_score", "matthews_corrcoef", "confusion_matrix"],
    cv=10,
    n_jobs=-1,
    verbose=1,
    return_train_score=True,
    title=None,
    stdout=False,
    classes=None,
    **kwargs,
):
    """
    Crossvalidate a classifier and plots informative evaluation metrics.

    You can pass all the arguments of
    ``sklearn.model_selection.cross_validate``.

    Note, however, that some defaults are overriden so that it (by
    default):
    * performs 10-fold stratified cross-validation
    * collects the following measures: balanced accuracy, matthews
      correlation coefficient and confusion matrix
    * uses all the available virtual processors
    * returns train scores to control overfitting

    Moreover, ``scoring`` must be a list of strings or a single strings
    containing function names from ``sklearn.metrics``.

    If `stdout` is True, some detailed results about the crossvalidation are
    printed into the standard output.

    If `classes` is used, the crossvalidation forgets the trained models,
    saving RAM.

    Plotting
    --------

    A further parameter ``plotting`` is used to decide if the summary plots are
    created or not.

    Note that only the first score is available in plot, the others will still
    be printed to terminal.

    If leave-one-out is used, then the scores are normalized to the whole
    matrix, otherwise to the single rows.

    To disable plotting, just use a ``None`` or ``False`` or empty string.

    The argument `title` is the title used for the plotting. If `None`, a
    default title will be used.

    Returns
    -------

    `list[Union[float, np.ndarray]]` :
        scores computed during cross-validation

    `plotly.graph_objectw.Figure` :
        the final plot, ``None`` if ``plotting`` is disabled

    """
    _scoring = copy(scoring)

    # put scoring in a list if it's a string
    if type(_scoring) is str:
        _scoring = [_scoring]

    # if user asks to plot, add the confusion matrix!
    if plotting and "confusion_matrix" not in _scoring:
        _scoring.append("confusion_matrix")

    # create the callables
    # note: this is needed because some metrics available in sklearn.metrics
    # would not be usable otherwise ( e.g. confusion_matrix, matthews_corrcoef)
    for i, sc in enumerate(_scoring):
        if sc == "confusion_matrix":
            # Note: CustomEval is a dummy class which inherits from
            # np.ndarray and represents one number only; a confusion matrix is actually
            # stored in the field `.eval["matrix"]`. This is needed because sklearn
            # doesn't accept that the return value of the metric is not a
            # number. See `conf_matrix_hack.py` for more info.
            labels = np.unique(y)
            _scoring[i] = (sc, metrics.make_scorer(CustomEval(labels)))
        else:
            _scoring[i] = (sc, metrics.make_scorer(getattr(metrics, sc)))

    # Note: cross_validate performs a `clone` of the estimator, so it always
    # starts from scratch (i.e., no warm_start allowed here)

    # evaluating scores
    scores = model_selection.cross_validate(
        estimator=estimator,
        X=X,
        y=y,
        scoring=dict(_scoring),
        cv=cv,
        n_jobs=n_jobs,
        return_train_score=return_train_score,
        verbose=verbose,
        return_estimator=classes is None,
        **kwargs,
    )

    # trying to get a nice output to the console...
    if stdout:
        for k, v in scores.items():
            if not k.startswith(("train", "test")):
                continue
            print("-----")
            if type(v) is list:
                print(f"{k:<20}")
                for i in v:
                    print(pd.DataFrame(i.matrix))
                    print("--")
            elif type(v) in [float, int]:
                print(f"{k:<20}: {v:.2e}")
            elif type(v) is np.ndarray:
                if np.any(np.isnan(v)):
                    raise RuntimeError(
                        "Found a nan in the output scores of crossvalidation, this is luckily due to an error during cross-validation. Please see the above logs for further information."
                    )
                print(f"{k:<20}:\n {pd.DataFrame(v)}")
                print(f"\t{'avg':<20}: {np.mean(v)}")
                print(f"\t{'std':<20}: {np.std(v)}")
            else:
                print(f"{k:<20}: {v:<20}")

    bal_acc = scores["test_balanced_accuracy_score"]
    print(f"Avg balanced accuracy: {bal_acc.mean():.2f}")
    print(f"Std balanced accuracy: {bal_acc.std():.2f}")

    title = title or f"Results of {cv}-fold cross-validation"

    # plotting, if needed...
    if plotting:
        if classes is not None:
            labels = classes
        elif label_encoder is not None:
            labels = label_encoder.classes_
        else:
            labels = scores["estimator"][0].classes_
        fig = plot_confusion_matrices(
            [i.eval["matrix"] for i in scores["train_confusion_matrix"]],
            [i.eval["matrix"] for i in scores["test_confusion_matrix"]],
            scores["train_" + _scoring[0][0]],
            scores["test_" + _scoring[0][0]],
            classes=labels,
            title=title,
            score_name=_scoring[0][0].replace("_", " "),
        )
    else:
        fig = None

    return scores, fig
