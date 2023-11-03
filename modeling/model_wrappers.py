import functools
import warnings

import numpy as np
import pandas as pd
import sklearn
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, clone
from sklearn.exceptions import FitFailedWarning
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, StandardScaler

from . import plotting
from . import settings as S


class DidoneClassifier(BaseEstimator):
    """
    This class wraps around an AutoSklearnClassifier to make it behave as a
    fully compliant sklearn classifier with a fixed structure (this class will
    never `fit` but only `refit` the AutoSklearnClassifier)

    This class also overloads the `predict` method so that it also returns an
    entropy score.
    """

    _estimator_type = "classifier"

    def __init__(
        self,
        models,
        ensemble,
        metric,
        x_encoder: OrdinalEncoder,
        y_encoder: LabelEncoder,
        standardize=True,
        n_jobs=-1,
    ):
        self.standardize = standardize
        self.x_encoder = x_encoder
        self.y_encoder = y_encoder
        self.models = models
        self.ensemble = ensemble
        self.metric = metric
        self.n_jobs = n_jobs

    def fit_models(self, X, y):
        """
        Fit the models but not the ensemble selection
        """
        try:
            X = self.x_encoder.fit_transform(X)
        except ValueError:
            pass
        y = self.y_encoder.fit_transform(y)
        self.classes_ = self.y_encoder.classes_

        if self.standardize:
            self.standardizer = StandardScaler()
            X = self.standardizer.fit_transform(X)

        # training the pipelines
        for k, model in self.models.items():
            self.models[k] = model.fit(X, y)
        return self

    def fit(self, X, y):
        """

        Fit the ensemble's pipelines. It does a full fit of the ensemble,
        without bayesian optimization.

        """
        self.fit_models(X, y)

        predictions = self._models_prediction(X)

        self.ensemble.metric = self.metric
        # one-hot encoding of targets
        y = pd.get_dummies(y).to_numpy()
        # one-hot encoding of predictions
        predictions = [
            pd.get_dummies(predictions[:, k]).to_numpy()
            for k in range(predictions.shape[1])
        ]
        self.ensemble.fit(predictions, y, list(self.models.keys()), None)
        return self

    def _models_prediction(self, X):
        predictions = []
        for model in self.models.values():
            predictions.append(model.predict(X))
        # predictions = Parallel(n_jobs=self.n_jobs)(
        #     delayed(model.predict)(X) for model in self.models.values())
        return np.asarray(predictions).T

    def _ensemble_prediction(self, predictions):
        predictions = np.asarray(predictions) @ self.weights
        predictions = np.round(predictions).astype(np.int32)
        predictions = self.y_encoder.inverse_transform(predictions)
        return predictions

    def predict(self, X):
        try:
            X = self.x_encoder.transform(X)
        except ValueError:
            pass
        if self.standardize:
            X = self.standardizer.transform(X)

        predictions = self._models_prediction(X)
        predictions = self._ensemble_prediction(predictions)
        return predictions

    def score(self, X, y, sample_weight=None):
        return balanced_accuracy_score(self.predict(X), y, sample_weight=sample_weight)

    def describe(self):
        """
        Prints all the hyper-parameters of the ensemble
        """
        models = self.ensemble.get_models_with_weights(self.models)
        for i, (w, model) in enumerate(models):
            print(f"Model number {i}")
            print(f"Weight: {w}")
            print(model.config)

    def has_decision_function(self):
        return any(
            hasattr(m["classifier"].choice.estimator, "decision_function")
            for m in self.models.values()
        )

    def decision_function(self, X):
        decisions = []
        for weight, model in self.models_with_weights:
            _X = X.copy()
            for _, step in model.steps[:-1]:
                _X = step.transform(_X)
            decisions.append(
                weight * model["classifier"].choice.estimator.decision_function(_X)
            )
        return np.sum(decisions, axis=0)

    def has_predict_proba(self):
        return any(hasattr(m, "predict_proba") for m in self.models.values())

    def predict_proba(self, X):
        probs = [w * m.predict_proba(X) for w, m in self.models_with_weights]
        return np.sum(probs, axis=0)

    @property
    def weights(self):
        return np.asarray([w for w, m in self.models_with_weights])

    @property
    def models_with_weights(self):
        return self.ensemble.get_models_with_weights(self.models)

    def get_plotter(self, top_k=0.5, feature_names=None):
        """
        Returns a DidonePlotter object with the models from this ensemble in
        it
        """

        plotter = plotting.DidonePlotter(model=self, top_k=0.5)
        for i, (w, m) in enumerate(self.models_with_weights):
            plotter.push(
                m, weight=w, feature_names=feature_names, classes=self.classes_
            )
        return plotter


class DidoneBagging(BaseEstimator):
    """
    Similar to sklearn `BaggingClassifier`, but circumvent possible errors
    connected with particular resamples. When this happens, it looks for
    another resample where this does not happens.
    After `n_trials` attempts, it raises an exception.

    If `replacement` is False, then `resample_size` must be < 1.0.
    """

    _estimator_type = "classifier"

    def __init__(
        self,
        estimator,
        *,
        n_resamples=100,
        resample_size=1.0,
        replacement=True,
        n_trials=20,
        random_state=None,
        n_jobs=-1,
    ):
        self.estimator = estimator
        self.n_resamples = n_resamples
        self.resample_size = resample_size
        self.replacement = replacement
        self.n_trials = n_trials
        self.random_state = random_state
        self.n_jobs = n_jobs

    def predict(self, X):
        """
        Return prediction using voting approach
        """
        predictions = self._predict(X)
        return predictions.columns[np.argmax(predictions.to_numpy(), axis=1)].to_numpy()

    def predict_proba(self, X):
        """
        Computes probabilities based on the ratio of the predictions for each
        label.
        """
        predictions = self._predict(X)
        L = len(self.bag)
        return (predictions / L).to_numpy()

    def _predict(self, X):
        classes = self.bag[0].classes_

        def predict_parallel(m, classes):
            model_preds = m.predict(X)
            predictions = {c: np.zeros(X.shape[0], dtype=np.int64) for c in classes}
            predictions = pd.DataFrame(predictions)
            for i, p in enumerate(model_preds):
                predictions.loc[i, p] = 1
            return predictions

        predictions = Parallel(n_jobs=self.n_jobs)(
            delayed(predict_parallel)(m, classes) for m in self.bag
        )
        predictions = functools.reduce(lambda x, y: x.add(y, fill_value=0), predictions)

        return predictions

    def fit(self, X, y):
        """
        Trains with bagging.
        If some resample causes errors, just resample it at least `n_trials`
        times
        """

        # here, N is a number much larger than n_resamples and n_trials
        # it is used to set the random_seed of each process, so that they do
        # not overlap
        N = max(self.n_resamples, self.n_trials) * 200
        resample_size = round(self.resample_size * X.shape[0])

        def fit_resample(i):
            if self.random_state is None:
                random_state = None
            else:
                random_state = (self.random_state + i + 1) * N
            k = 0  # this counts how many trials we do
            while True:
                k += 1
                if self.random_state is None:
                    random_state = None
                else:
                    random_state += 1
                X_r, y_r = sklearn.utils.resample(
                    X,
                    y,
                    replace=self.replacement,
                    n_samples=resample_size,
                    stratify=y,
                    random_state=random_state,
                )
                try:
                    m = clone(self.estimator).fit(X_r, y_r)
                except Exception as e:
                    if k > self.n_trials:
                        import traceback

                        traceback.print_exception(e)
                        raise Exception("Cannot found a resample without errors!")
                    else:
                        warnings.warn(
                            "This resample raised an exception, looking for another one!",
                            FitFailedWarning,
                        )
                else:
                    return m

        self.bag = Parallel(n_jobs=self.n_jobs)(
            delayed(fit_resample)(i) for i in range(self.n_resamples)
        )
        self.classes_ = np.unique(np.concatenate([b.classes_ for b in self.bag]))
        self.fitted_ = True
        return self

    def get_plotter(self, base_model=None, feature_names=None, top_k=0.5):
        """
        Returns a plotter to inspect the models in this bagging.

        If this is a bagging of trees or an ensemble of trees, the plotter will
        only plot the first element in the bag.

        If this is a bagging of linear models or of ensembles of linear
        models, a box plot will be created, where models in the ensembles are
        weighted with their weight.

        If `base_model` is not None, than it is passed to DidonePlotter , that will use
        it for further analysis, e.g. partial dependence analysis.
        """
        ttt, _, _ = plotting.get_classifier_type(self.bag[0], feature_names)
        plotter = plotting.DidonePlotter(model=base_model, top_k=top_k)
        if ttt == "tree":
            plotter.push(
                self.bag[0], feature_names=feature_names, classes=self.classes_
            )
        elif ttt == "linear":
            for model in self.bag:
                plotter.push(model, feature_names=feature_names, classes=self.classes_)
        elif ttt == "didone":
            ttt, _, _ = plotting.get_classifier_type(
                self.bag[0].models_with_weights[0][1], feature_names
            )
            if ttt == "tree":
                plotter = self.bag[0].get_plotter(
                    top_k=top_k, feature_names=feature_names
                )
            else:
                for model in self.bag:
                    plotter += model.get_plotter(
                        top_k=top_k, feature_names=feature_names
                    )
        else:
            raise plotting.InspectionError("Cannot plot this type of model!")
        return plotter

    @property
    def fitted(self):
        if hasattr(self, "fitted_") and self.fitted_:
            return True
        else:
            return False

    @staticmethod
    def default_init(estimator):
        return DidoneBagging(
            estimator,
            n_resamples=S.BAGGING_RESAMPLES,
            n_trials=20,
            resample_size=1.0,
            replacement=True,
            random_state=756,
            n_jobs=-1,
        )
