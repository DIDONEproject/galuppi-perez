from pathlib import Path

import graphviz as gv
import numpy as np
import pandas as pd
import plotly.express as px
# from glmnet import LogitNet
import plotly.graph_objects as go
import sklearn
from autosklearn.pipeline.components.classification import (decision_tree, lda,
                                                            liblinear_svc,
                                                            passive_aggressive,
                                                            sgd)
from plotly.subplots import make_subplots
from scipy import stats
from sklearn import calibration, linear_model, tree
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import (LogisticRegressionCV,
                                  PassiveAggressiveClassifier,
                                  RidgeClassifierCV, SGDClassifier)
from sklearn.metrics import balanced_accuracy_score, make_scorer
from sklearn.svm import LinearSVC
from sklearn.tree import export_graphviz

from .easy_tools import get_xy
from .inspection import (FeatureDimredError, partial_dependence,
                         pipeline_backtrack)

balanced_accuracy = make_scorer(balanced_accuracy_score)


class InspectionError(RuntimeError):
    pass


def get_feature_category(feature, scale):
    scale = scale[::-1]
    feature = feature.lower()
    if "js_" in feature:
        return "jSymbolic", scale[-8]
    elif "m21_" in feature:
        return "music21", scale[-9]
    elif "interval" in feature:
        return "Melodic", scale[-1]
    elif "degree" in feature:
        return "Melodic", scale[-1]
    elif "motion" in feature:
        return "Melodic", scale[-1]
    elif "leaps" in feature:
        return "Melodic", scale[-1]
    elif "dsc_avg" in feature:
        return "Melodic", scale[-1]
    elif "dsc_prp" in feature:
        return "Melodic", scale[-1]
    elif "asc_avg" in feature:
        return "Melodic", scale[-1]
    elif "asc_prp" in feature:
        return "Melodic", scale[-1]
    elif "tempo" in feature:
        return "Rhythm", scale[-2]
    elif "rhythm" in feature:
        return "Rhythm", scale[-2]
    elif "duration" in feature:
        return "Rhythm", scale[-2]
    elif "harmony" in feature:
        return "Harmony", scale[-3]
    elif "dyn" in feature:
        return "Dynamics", scale[-4]
    elif "texture" in feature:
        return "Texture", scale[-5]
    elif "density" in feature:
        return "Texture", scale[-5]
    elif "presence" in feature:
        return "Texture", scale[-5]
    elif "ambitus" in feature:
        return "Ambitus", scale[-6]
    elif "highestnoteindex" in feature:
        return "Ambitus", scale[-6]
    elif "lowestnoteindex" in feature:
        return "Ambitus", scale[-6]
    elif "syllab" in feature:
        return "Lyrics", scale[-7]
    else:
        print(feature + ": category `Other`")
        return "Other", scale[-10]


def hide_duplicate_traces_from_legend(fig):
    names = set()
    fig.for_each_trace(
        lambda trace: trace.update(showlegend=False)
        if (trace.name in names)
        else names.add(trace.name)
    )


def graphviz_save(dot_source, fname):
    dot_source.save(fname.with_suffix(".dot"))
    with open(fname.with_suffix(".svg"), "w") as f:
        f.write(dot_source.pipe(encoding="utf-8"))


def plotly_save(fig, fname):
    fname = Path(fname)
    fig.write_html(fname.with_suffix(".html"))
    fig.write_image(fname)


def plot_confusion_matrices(
    cm1s,
    cm2s,
    score1s,
    score2s,
    classes,
    title,
    score_name,
    fname=None,
    confidence_level=0.95,
):
    """
    This creates a plot with two confusion matrices with ratio of matched data
    and a histogram and a score for each confusion matrix.

    ``cm1s``, ``cm2s``, ``score1s``, ``score2s`` should be lists of matrices
    and scores.

    It returns the figure object without plotting nor saving.

    If leave-one-out is used, then the scores are normalized to the whole
    matrix, otherwise to the single rows.
    """

    z_stat = -stats.norm.ppf((1 - confidence_level) / 2)
    # mean and error for scores
    score1 = [np.mean(score1s), z_stat * stats.sem(score1s, ddof=1)]
    score2 = [np.mean(score2s), z_stat * stats.sem(score2s, ddof=1)]
    # total number of data per each class and each fold
    sum_cm1s = np.sum(cm1s, axis=2)
    sum_cm2s = np.sum(cm2s, axis=2)
    # ratio of confusion matrices
    cm1s /= sum_cm1s[..., None]
    cm2s /= sum_cm2s[..., None]
    # if some value in sum_cm1s was 0, then we got a nan, so we replace it with 0
    cm1s[np.isnan(cm1s)] = 0
    cm2s[np.isnan(cm2s)] = 0
    # confusion matrices average and error
    cm1 = [np.mean(cm1s, axis=0), z_stat * stats.sem(cm1s, axis=0, ddof=1)]
    cm2 = [np.mean(cm2s, axis=0), z_stat * stats.sem(cm2s, axis=0, ddof=1)]
    # total number of data per each class on average (histogram)
    sum_cm1s = sum_cm1s.mean(axis=0)
    sum_cm2s = sum_cm2s.mean(axis=0)

    # structure and titles with scores
    fig = make_subplots(
        rows=1,
        cols=8,
        specs=[
            [
                {"colspan": 3},
                None,
                None,
                {"colspan": 1},
                {"colspan": 3},
                None,
                None,
                {"colspan": 1},
            ]
        ],
        subplot_titles=(
            f"Train dataset: {score1[0]:.2f} ± {score1[1]:.2f}",
            "Histogram train set",
            f"Test dataset: {score2[0]:.2f} ± {score2[1]:.2f}",
            "Histogram test set",
        ),
    )

    # confusion matrix left and bar plot left
    trace = px.imshow(cm1[0], x=classes, y=classes, zmin=0, zmax=1).data[0]
    trace.text = cm1[1]
    trace.texttemplate = "%{z:.2f} ± %{text:.2f}"
    fig.add_trace(trace, row=1, col=1)
    fig.add_trace(px.bar(x=sum_cm1s, y=classes, text=sum_cm1s).data[0], row=1, col=4)

    # confusion matrix right and bar plot right
    trace = px.imshow(cm2[0], x=classes, y=classes, zmin=0, zmax=1).data[0]
    trace.text = cm2[1]
    trace.texttemplate = "%{z:.2f} ± %{text:.2f}"
    fig.add_trace(trace, row=1, col=5)
    fig.add_trace(px.bar(x=sum_cm2s, y=classes, text=sum_cm2s).data[0], row=1, col=8)

    # title and subtitle
    fig.update_layout(
        title=title
        + f"<br><sup>Training and test scores in terms of {score_name}. Errors computed with {confidence_level} confidence level</sup>"
    )
    fig.update_yaxes(tickangle=90)

    if fname:
        plotly_save(fig, fname)
    return fig


def toGraphObject(src):
    """
    Thanks Ray Ronnaret https://stackoverflow.com/a/67143816
    """
    lst = str(src).splitlines()
    is_direct_graph = False
    skip_index = 0

    def has_comment(idx):
        return lst[idx].find("//") != -1

    def is_direct_graph(idx):
        return lst[idx].find("digraph ") != -1

    while has_comment(skip_index):
        skip_index += 1

    if is_direct_graph(skip_index):
        g = gv.Digraph()
    else:
        g = gv.Graph()

    g.body.clear()

    # initial and closing statement are added by graphviz...
    lst = lst[skip_index + 1 : -1]

    for idx, row in enumerate(lst):
        if not has_comment(idx):
            g.body.append(row)
    return g


def tree_to_graphviz(*args, fname=None, weight=1.0, **kwargs):
    """
    This function simply exports the classifier into graphviz source and
    plots it using `graphviz` module. It returns the `graphviz.Source` object.
    """

    if type(args[0]) is sklearn.pipeline.Pipeline:
        args[0] = args[0][-1]

    dot_source = export_graphviz(
        *args, filled=True, rounded=True, special_characters=True, **kwargs
    )

    # converting to gv.Graph and adding title
    g = toGraphObject(dot_source)
    g.attr(label=rf"\nEnsemble weight: {weight}")

    # converting to gv.Source object (for saving easily)
    g = gv.Source(g.source, format="svg")

    if fname:
        graphviz_save(g, fname)

    return g


def plot_time_performance(automl_data, fname=None):
    """
    Plot the performance over time of the best mdels in automl_data

    You can use `easy_tools.get_automl` to load the most recent `automl_data`.
    """
    fig = px.line(
        automl_data.performance_over_time_,
        x="Timestamp",
        y=automl_data.performance_over_time_.columns,
    )
    if fname:
        plotly_save(fig, fname)
    return fig


def get_classifier_type(model, feature_names=None):
    """
    Returns the type of a classifier. Supported: 'didone', 'tree' and 'linear'.
    If the classifier is a pipeline, performs backtracking of the features.

    CalibratedClassifierCV is supported only if `ensemble` is False.

    Returns:
        * type of classifier as string
        * classifier
        * backtracked features
    """
    # backtracking features
    if type(model) is calibration.CalibratedClassifierCV:
        model = model.calibrated_classifiers_[0].base_estimator

    cls = type(model)
    if sklearn.pipeline.Pipeline in cls.mro():
        try:
            model, feature_names = pipeline_backtrack(model, feature_names)
        except FeatureDimredError as e:
            raise InspectionError("Cannot backtrack this pipeline") from e
        if hasattr(model, "choice"):
            model = model.choice.estimator
        cls = type(model)
    # detecting type of model
    if cls in [decision_tree.DecisionTree, tree.DecisionTreeClassifier]:
        ttt = "tree"
    elif cls in [
        sgd.SGD,
        liblinear_svc.LibLinear_SVC,
        lda.LDA,
        passive_aggressive.PassiveAggressive,
        LinearDiscriminantAnalysis,
        LinearSVC,
        RidgeClassifierCV,
        LogisticRegressionCV,
        # LogitNet,
        PassiveAggressiveClassifier,
        SGDClassifier,
    ] or cls.__module__ in [linear_model]:
        ttt = "linear"
    elif str(cls.__name__) == "DidoneClassifier":
        ttt = "didone"
    else:
        raise InspectionError(
            f"We don't have an inspection function for this type of model: {cls}"
        )
    return ttt, model, feature_names


class DidonePlotter:
    """
    This class stores a pool of models and, when `plot` method is called,
    visualizations for the pool are created.

    For each tree model, a graph is plotted.

    For each linear model, the features whose average absolute value is in the
    `top_k` percentage are plotted; only one plot is built for all the linear
    models, containing the distribution of the ablsolute weights for each features;
    while computing the importance of a feature, a model weight `M_w` is taken into
    account, so that the model's weights `w_i` are first multiplied by `M_w /
    sum of all the M_w`.

    If `output_dir` is not None, plots are saved in that directory.
    """

    def __init__(self, model=None, top_k=0.5, grid_resolution=100):
        self.top_k = top_k
        self.pool = []
        self.model = model
        self.grid_resolution = grid_resolution

    def push(self, model, weight=1.0, feature_names=None, classes=[]):
        """
        Model can be:
            * a Pipeline
            * an AutoSklearn/sklearn Decision Tree
            * an AutoSklearn/sklearn linear model

        For `DidoneClassifier` and `DidoneBagging`, use their method
        `get_plotter`
        """
        ttt, model, feature_names = get_classifier_type(model, feature_names)

        # appending it
        self.pool.append(
            dict(
                weight=weight,
                feature_names=feature_names,
                model=model,
                type=ttt,
                classes=classes,
            )
        )

    def __add__(self, obj):
        ret = DidonePlotter()
        ret.pool = self.pool + obj.pool
        return ret

    def __iadd__(self, obj):
        self.pool += obj.pool
        return self

    def select_coefs(self, X, y, coefs, output_dir):
        # counting the importance of each feature in average
        coefs = coefs.T
        coefs["mean_importance"] = coefs.mean(axis=1)
        coefs["abs_importance"] = coefs["mean_importance"].abs()
        coefs = coefs.sort_values("abs_importance", ascending=False)
        try:
            self._data_plot(X, y, coefs[:3], output_dir)
        except Exception as e:
            print(f"Cannot plot scatter 3d: {e}")
        # take the features that sum up to `top * total_sum`
        tot = coefs["abs_importance"].sum()
        idx = (np.cumsum(coefs["abs_importance"]) <= self.top_k * tot).argmin() + 1
        features = coefs.index[:idx]
        coefs = coefs.loc[features]
        # coefs = coefs.sort_values("importance")
        coefs = coefs.drop(columns=["mean_importance", "abs_importance"])
        coefs = coefs.T
        return coefs, features

    def get_dependence_coeff_sign(self, X, coefs, features, linear, y):
        dependence_coeff_sign = []
        try:
            angular_coeffs, classes = partial_dependence(
                self.model,
                X,
                y,
                features,
                linear=True,
                grid_resolution=self.grid_resolution,
            )
        except Exception as e:
            print(f"Cannot compute partial dependence: {e}")
            __import__("ipdb").set_trace()
            dependence_coeff_sign = {k: True for k in features}
        else:
            # modifying the coeffs so that positive coeffs correspond features
            # whose larger values increase the probability of classes[1]
            dependence_coeff_sign = {k: v > 0 for k, v in angular_coeffs.items()}
        return dependence_coeff_sign

    def plot(self, X, y, output_dir=None, prefix=""):
        # total sum of the weights
        # W = sum([m["weight"] for m in self.pool])

        # separate model types
        trees = [m for m in self.pool if m["type"] == "tree"]
        linears = [m for m in self.pool if m["type"] == "linear"]
        data_output = None

        # plotting trees
        figures = []
        for t in trees:
            fig = tree_to_graphviz(
                t["model"],
                feature_names=t["feature_names"],
                class_names=t["classes"],
                weight=t["weight"],  # / W,
            )
            figures.append(fig)

        # plotting linear models

        # collecting all the features
        if len(linears) > 0:
            coefs = []  # pd.DataFrame(columns=list(features))
            for linear in linears:
                this_coefs = pd.DataFrame(
                    linear["model"].coef_, columns=linear["feature_names"]
                )
                this_coefs *= linear["weight"]  # / W
                coefs.append(this_coefs)
            coefs = pd.concat(coefs, axis=0).reset_index()

            coefs, features = self.select_coefs(X, y, coefs, output_dir)

            # partial-dependence agular coefficients
            if self.model is not None:
                dependence_coeff_sign = self.get_dependence_coeff_sign(
                    X, coefs, features, linear, y
                )

            # generating the category of each feature
            color_scale = px.colors.qualitative.Plotly
            colors = {
                col: get_feature_category(col, color_scale) for col in coefs.columns
            }
            # box plots of the selected features
            fig = go.Figure(
                data=[
                    go.Box(
                        y=coefs[col],
                        x0=col,
                        name=colors[col][0],
                        marker_color=colors[col][1],
                        fillcolor="rgba(0, 0, 0, 0)"
                        if dependence_coeff_sign[col]
                        else None,
                    )
                    for col in coefs.columns
                ],
            )
            hide_duplicate_traces_from_legend(fig)
            fig.update_layout(
                xaxis=dict(tickmode="linear", tick0=0.5, dtick=0.75),
                title=f"Features for {prefix}, empty -> positive slope",
            )
            figures.append(fig)

        self._save_figs(figures, output_dir, prefix)

        return data_output

    def _data_plot(self, X, y, coefs, output_dir):
        """Plots points according to the passed features"""
        X = X[coefs.index]
        fig1 = px.scatter_3d(
            X,
            color=y,
            x=coefs.index[0],
            y=coefs.index[1],
            z=coefs.index[2],
        )
        fig2 = px.scatter_3d(
            X * coefs["mean_importance"],
            color=y,
            x=coefs.index[0],
            y=coefs.index[1],
            z=coefs.index[2],
        )
        if output_dir:
            plotly_save(fig1, output_dir / "scatter_plot_raw_features.svg")
            plotly_save(fig2, output_dir / "scatter_plot_transformed_features.svg")
        return fig1, fig2

    def _save_figs(self, figures, output_dir, prefix):
        print(">>>>> saving")
        if output_dir:
            for i, fig in enumerate(figures):
                if prefix:
                    fname = output_dir / f"{prefix}_{i:02d}"
                else:
                    fname = output_dir / f"{i:02d}"
                if type(fig) is gv.Source:
                    graphviz_save(fig, fname.with_suffix(".dot"))
                else:
                    plotly_save(fig, fname.with_suffix(".svg"))
