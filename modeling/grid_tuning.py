"""
A module to do grid-search based tuning of specified pipelines
"""

import pickle
import time
from collections import defaultdict, namedtuple
from copy import deepcopy
import gzip
from pathlib import Path

# import glmnet
import numpy as np
import sklearn
import sklearn.cross_decomposition
import sklearn.decomposition
import sklearn.linear_model
import sklearn.svm
from sklearn.base import BaseEstimator, TransformerMixin, clone, is_classifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.exceptions import NotFittedError
from sklearn.metrics import balanced_accuracy_score, make_scorer
from sklearn.model_selection import GridSearchCV, ParameterGrid
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from . import settings as S
from .model_wrappers import DidoneBagging
from .plotting import plotly_save
from .validation import crossvalidation

# glmnet.LogitNet._estimator_type = "classifier"


class CustomXySeparator(BaseEstimator, TransformerMixin):
    """
    Just an estimator that only returns the X of CCA/PLS and similar
    """

    def __init__(self, **params):
        super().__init__()
        self.set_params(**params)

    def fit(self, X, y):
        self.obj.fit(X, y)
        # computing th
        p = self.obj.predict(X)
        max = p.max()
        min = p.min()
        self.th_ = (max + min) / 2
        self.classes_ = 2
        self.coef_ = self.obj.x_rotations_.T
        return self

    def transform(self, X, y=None):
        if y is None:
            return self.obj.transform(X)
        return self.obj.transform(X, y)[0]

    def fit_transform(self, X, y=None):
        return self.obj.fit(X, y).transform(X, y)[0]

    def predict(self, X):
        pred = self.obj.predict(X)
        pred[pred > self.th_] = 1
        pred[pred < self.th_] = 0
        return pred

    def get_params(self, deep=True):
        obj_params = self.obj.get_params(deep)
        obj_params["obj"] = type(self.obj)
        return obj_params

    def set_params(self, **params):
        if "obj" in params:
            self.obj = params.pop("obj")()
        for k, v in params.items():
            setattr(self, k, v)
        return self.obj.set_params(**params)


def cca_get0(**kwargs):
    ret = CustomXySeparator(obj=sklearn.cross_decomposition.CCA)
    return ret.set_params(**kwargs)


def pls_get0(**kwargs):
    ret = CustomXySeparator(obj=sklearn.cross_decomposition.PLSCanonical)
    return ret.set_params(**kwargs)


def get_custom_grid(X, y=None):
    """
    Returns a grid to be used in `grid_tune_pipeline(X, y)`.

    The returned object must be a dictionary containing any number of steps.
    Each step has the name as key and a dictionary of methods as value.
    Each method has the sklearn-compliant class type as key and and a
    dictionary containing the hyper-parameters as value (see
    `sklearn.model_selection.ParameterGrid` for how to encode hyper-parameters)

    `X` and `y` can be used to set hyper-parameters values (e.g. the number of
    samples/features).
    """
    N, K = X.shape
    # n_classes = np.unique(y).shape[0]

    return {
        "data_preprocessor": [(StandardScaler(), {}), (None, {})],
        "feature_preprocessor": [
            (
                sklearn.decomposition.PCA(),
                {
                    "n_components": np.linspace(0.7, 1 - 1e-15, 10),
                },
            ),
            (
                sklearn.decomposition.TruncatedSVD(),
                {
                    "n_components": np.linspace(K // 10, K - 1, 10, dtype=int),
                    "random_state": [np.random.RandomState(543)],
                },
            ),
        ],
        "classifier": [
            # (
            #     glmnet.LogitNet(),
            #     {
            #         "alpha": np.linspace(0, 1, 4),
            #         "standardize": [False],
            #         "n_lambda": [10],
            #         "fit_intercept": [True, False],
            #         "n_splits": [5],
            #         "scoring": [glmnet.scorer.make_scorer(balanced_accuracy_score)],
            #         "random_state": [np.random.RandomState(543)],
            #     },
            # ),
            (
                sklearn.svm.LinearSVC(),
                {
                    "penalty": ["l2"],
                    "loss": ["hinge"],
                    "dual": [True],
                    "C": np.geomspace(1e-4, 1000, 10),
                    "random_state": [np.random.RandomState(543)],
                    "class_weight": ["balanced"],
                    "max_iter": [5000],
                },
            ),
            (
                sklearn.svm.LinearSVC(),
                {
                    "penalty": ["l2", "l1"],
                    "loss": ["squared_hinge"],
                    "dual": [False],
                    "C": np.geomspace(1e-4, 1000, 10),
                    "random_state": [np.random.RandomState(543)],
                    "class_weight": ["balanced"],
                    "max_iter": [5000],
                },
            ),
            (
                sklearn.linear_model.SGDClassifier(),
                {
                    "early_stopping": [True],
                    "fit_intercept": [False],
                    "average": [True, False],
                    "class_weight": ["balanced"],
                    "loss": [
                        "modified_huber",
                        "squared_hinge",
                        "perceptron",
                        "squared_loss",
                        "huber",
                        "epsilon_insensitive",
                        "squared_epsilon_insensitive",
                    ],
                    "penalty": ["l2", "l1"],
                    "random_state": [np.random.RandomState(543)],
                    "max_iter": [5000],
                },
            ),
            (
                sklearn.linear_model.PassiveAggressiveClassifier(),
                {
                    "loss": ["hinge", "squared_hinge"],
                    "C": np.geomspace(1e-4, 1000, 20),
                    "early_stopping": [True],
                    "average": [True, False],
                    "random_state": [np.random.RandomState(543)],
                    "class_weight": ["balanced"],
                },
            ),
            (
                sklearn.linear_model.LogisticRegressionCV(),
                {
                    "Cs": [10],
                    "cv": [5],
                    "fit_intercept": [False, True],
                    "solver": ["saga"],
                    "penalty": ["elasticnet"],
                    "l1_ratios": [np.linspace(0, 1, 4)],
                    "scoring": [make_scorer(balanced_accuracy_score)],
                    "class_weight": ["balanced"],
                    "max_iter": [5000],
                    "random_state": [np.random.RandomState(543)],
                },
            ),
            (
                sklearn.linear_model.RidgeClassifierCV(),
                {
                    "class_weight": ["balanced"],
                    "scoring": [make_scorer(balanced_accuracy_score)],
                    "cv": [5],
                    "fit_intercept": [False],
                },
            ),
            (
                sklearn.linear_model.RidgeClassifierCV(),
                {
                    "class_weight": ["balanced"],
                    "scoring": [make_scorer(balanced_accuracy_score)],
                    "fit_intercept": [True],
                    "cv": [5],
                    "normalize": [False, True],
                },
            ),
            (
                LinearDiscriminantAnalysis(),
                {
                    "shrinkage": np.linspace(0, 1, 10),
                    "solver": ["eigen"],
                },
            ),
            (cca_get0(n_components=1), {}),
            (pls_get0(n_components=1), {}),
        ],
    }


Result = namedtuple("Result", "best_estimator_ best_score_")


def model_fit(X, y, model, stdout=True):
    """
    Fits a model so that this can be done in parallel
    """
    try:
        try:
            model.fit(X, y)
        except Exception:
            raise NotFittedError
        if S.GRID_DEBUG:
            print(model)
            y_hat = model.predict(X)
            return Result(model, balanced_accuracy_score(y_hat, y))
        else:
            print(f"Best model: {model.best_score_:.2f}")
            return deepcopy(model)
    except NotFittedError:
        return None


def grid_tune_pipeline(X, y, splitter):
    """
    Performs various grid-searches, one for each combination of methods found
    in `get_custom_grid(X, y)`.

    `splitter` should be an sklearn-like splitter object which creates an
    iterator when calling `split()`

    Returns the best model found according to `balanced_accuracy_score` and the
    list of the grid-searches objects.
    """
    pipeline = []
    param_grid = {}
    options = []
    steps = ["data_preprocessor", "feature_preprocessor", "classifier"]
    grid = get_custom_grid(X, y)
    steps = list(grid.keys())
    delayed_fittings = []
    for step in steps:
        options.append(grid[step])
        pipeline.append(None)

    def iterate_step(step_idx):
        step = steps[step_idx]
        for obj, parameters in options[step_idx]:
            pipeline[step_idx] = step, obj

            # populating parameters
            for k, v in parameters.items():
                param_grid[step + "__" + k] = v
            if step_idx == len(options) - 1:
                model = Pipeline([x for x in pipeline if x[1] is not None])
                if not is_classifier(model):
                    continue

                if not S.GRID_DEBUG:
                    # tune the pipeline
                    model = GridSearchCV(
                        estimator=model,
                        param_grid=param_grid,
                        cv=splitter,
                        n_jobs=-1,
                        error_score=0.0,
                        scoring=make_scorer(balanced_accuracy_score),
                    )
                delayed_fittings.append((X, y, deepcopy(model), deepcopy(pipeline)))
            else:
                iterate_step(step_idx + 1)
            # cleaning parameters
            for k, v in parameters.items():
                del param_grid[step + "__" + k]

    # create the list of models
    iterate_step(0)
    # fit models in parallel:
    print(
        f"Evaluating {len(delayed_fittings)} pipelines (each with its hyper-parameter space)"
    )
    tot_hyperparams = sum(len(ParameterGrid(d[2].param_grid)) for d in delayed_fittings)
    print(f"Evaluating in total {tot_hyperparams} hyper-parameters.")
    results = []
    timings = defaultdict(lambda: (0, 0))
    for p in tqdm(delayed_fittings):
        ttt = time.time()
        try:
            results.append(model_fit(*p[:3]))
            ttt = time.time() - ttt
        except KeyboardInterrupt:
            import sys

            answer = input("Do you want to exit? ")
            if answer == "yes":
                sys.exit(1)
            else:
                ttt = np.inf

        for s in p[3]:
            key = str(s[1])
            timings[key] = timings[key][0] + ttt, timings[key][1] + 1

    # results = Parallel(n_jobs=-1, backend='multiprocessing')(tqdm(delayed_fittings))
    L_before = len(results)
    results = [r for r in results if r is not None]
    print(f"Model failed: {L_before - len(results)}")

    print("\n--------------------------")
    print("Computing average timings:")
    for k, v in timings.items():
        print(f"{k}: {v[0] / v[1]:.2e} sec")
    print("--------------------------\n")

    # look for the best pipeline
    best_idx = np.argmax(list(map(lambda x: x.best_score_, results)))
    best_model = clone(results[best_idx].best_estimator_)

    # refit the best pipeline on the whole dataset
    best_model.fit(X, y)

    # return it (with all the tuning data attached)
    return best_model, results


def gridsearch(data_x_y, splitter, output_dir, skipsearches=False,
               skipbagfitting=False):
    """
    1. Performs a grid_search on `get_custom_grid(X, y)` using train_idx.
    2. Saves the best model and the list of grid-search objects in `output_dir`
       (one grid-search object for each pipeline tested)
    3. Performs crossvalidation on test_idx.
    4. Saves crossvalidation figure in `output_dir`
    5. Performs linear feature analysis on the whole dataset
    6. Saves figures of the analysis in `output_dir`
    """
    X, y = data_x_y

    output_dir = Path(output_dir)

    if skipsearches:
        model = pickle.load(open(output_dir / 'best_model.pkl', 'rb'))
    else:
        print("Starting Grid-search")
        model, trajectory = grid_tune_pipeline(X, y, splitter)

        print("Best model found:")
        print(model)
        pickle.dump(model, open(output_dir / "best_model.pkl", "wb"))
        pickle.dump(trajectory, open(output_dir / "trajectory.pkl", "wb"))

    if not skipbagfitting:
        bag = DidoneBagging.default_init(model)
        print("Building best model using the whole dataset")
        bag.fit(X, y)
        pickle.dump(bag, gzip.open(output_dir / "bag.pkl", "wb"))

    scores, fig = crossvalidation(model, X, y, cv=splitter)

    if output_dir:
        plotly_save(fig, output_dir / "crossvalidation_bag.svg")
        plotly_save(fig, output_dir / "crossvalidation.svg")
        pickle.dump(scores, open(Path(output_dir) / "crossval_scores.pkl", "wb"))
