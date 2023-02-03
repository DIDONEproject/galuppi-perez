from typing import Dict, Optional, Tuple, Union

from ConfigSpace.configuration_space import ConfigurationSpace
import ConfigSpace.hyperparameters as CSH

import numpy as np

from autosklearn.pipeline.base import DATASET_PROPERTIES_TYPE
from autosklearn.pipeline.constants import DENSE, UNSIGNED_DATA, SIGNED_DATA, INPUT, SPARSE
from autosklearn.pipeline.components.base import AutoSklearnPreprocessingAlgorithm
from imblearn.under_sampling import ClusterCentroids
from imblearn.base import BaseSampler

from pyloras import ProWRAS


class OverAndUnderSampling(BaseSampler):
    """
    This class performs over-sampling of the minority class and under-sampling
    of the majority class.

    For multi-class problems, ``k_min`` and ``k_maj`` parameters are used to
    identify classes that should be, respectively, over-sampled and
    under-sampled. Since most of the algorithms work in the binary case, for
    multi-class problems, each minority class is threated versus the remaining
    dataset. The same happens for majority classes, even if this could
    transform a majority class into a minority class; note, however, that this
    is not a problem for the default method (`ClusterCentroids`) that has
    proven good performances even in respect to more complex algorithms [3].

    This class implements the imblearn API, but it implements ``fit()`` and
    ``transform()`` methods as sklearn and auto-sklearn like.

    Parameters
    ----------

    k_min: int, default=0.8
        A factor that defines the classes to be over-sampled and the number of
        samples to synthesize. For binary problems, ``k_min=1`` means that
        the number of samples drawn will be so that the number of data of the
        minority class will be th 50% of the dataset.

        In general, given $C$ classes and $N$ data, a factor $K = N/C$ is
        defined. The classes $c$ that will be over-sampled will have
        cardinality $N_c$ so that $N_c < K \times k_{min}$. The number of
        samples that will be synthesized will be $ K \times k_min - \times
        N_c$.

    k_maj: int, default=1.2
        A factor that defines the classes to be under-sampled and the number of
        samples to remove. For binary problems, ``k_maj=1`` means that the
        number of samples removed will be so that the number of data of the
        majority class will be th 50% of the dataset.

        In general, given $C$ classes and $N$ data, a factor $K = N/C$ is
        defined. The classes $c$ that will be under-sampled will have
        cardinality $N_c$ so that $N_c > K \times k_{maj}$. The number of
        samples that will be removed will be $ N_c - K \times k_maj$.

    over_sampling: type, default=ProWRAS
        A type implementing the Scikit-Learn interface, e.g. a type from
        ``imblearn.over_sampling``. The default is ProWRAS [1].

    under_sampling: type, default=ProWRAS
        A type implementing the Scikit-Learn interface, e.g. a type from
        ``imblearn.under_sampling``. The default is ClusterCentroids [2].

    over_sampling_kwargs: dict, default={}
        Dictionary conatining the parameters for instantiating the
        ``over_sampling`` type. The value ``None`` (default) means that the
        default parameters will be used. ``sampling_strategy`` is always
        overwritten by this class.

    under_sampling_kwargs: dict, default={}
        Dictionary conatining the parameters for instantiating the
        ``under_sampling`` type. The value ``None`` (default) means that the
        default parameters will be used. ``sampling_strategy`` is always
        overwritten by this class.

    References
    ----------

    .. [1] S. Bej, K. Schulz, P. Srivastava, M. Wolfien, and O. Wolkenhauer, “A
       Multi-Schematic Classifier-Independent Oversampling Approach for
       Imbalanced Datasets,” IEEE Access, vol. 9, pp. 123358–123374, 2021, doi:
       10.1109/ACCESS.2021.3108450.
    .. [2] “Cluster-based under-sampling approaches for imbalanced data
       distributions,” Expert Systems with Applications, vol. 36, no. 3, pp.
       5718–5727, Apr. 2009, doi: 10.1016/j.eswa.2008.06.108.
    .. [3] M. Koziarski, “Radial-Based Undersampling for imbalanced data
       classification,” Pattern Recognition, vol. 102, p. 107262, Jun. 2020,
       doi: 10.1016/j.patcog.2020.107262.

    """

    _sampling_type = "ensemble"

    def __init__(self,
                 k_min=0.8,
                 k_maj=1.2,
                 over_sampling=ProWRAS,
                 under_sampling=ClusterCentroids,
                 over_sampling_kwargs={},
                 under_sampling_kwargs={}):
        super().__init__()

        assert k_min < k_maj, "k_min must be < k_maj"
        self.k_min = k_min
        self.k_maj = k_maj
        self.over_sampling = over_sampling
        self.under_sampling = under_sampling
        self.over_sampling_kwargs = over_sampling_kwargs
        self.under_sampling_kwargs = under_sampling_kwargs

    def _fit_resample(self, X, y):
        """
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Matrix containing the data which have to be sampled.
        y : array-like of shape (n_samples,)
            Corresponding label for each sample in X.
        Returns
        -------
        X_resampled : {ndarray, sparse matrix} of shape \
                (n_samples_new, n_features)
            The array containing the resampled data.
        y_resampled : ndarray of shape (n_samples_new,)
            The corresponding label of `X_resampled`.
        """
        # setting sampling strategy
        classes, counts = np.unique(y, return_counts=True)
        K = np.sum(counts) / classes.shape[0]
        k_min = round(self.k_min * K)
        k_maj = round(self.k_maj * K)
        over_sampling_strategy = {
            c: k_min
            for i, c in enumerate(classes) if counts[i] < k_min
        }
        under_sampling_strategy = {
            c: k_maj
            for i, c in enumerate(classes) if counts[i] > k_maj
        }

        # instatiating samplers
        over_sampler = self.over_sampling(
            sampling_strategy=over_sampling_strategy,
            **self.over_sampling_kwargs)
        under_sampler = self.under_sampling(
            sampling_strategy=under_sampling_strategy,
            **self.under_sampling_kwargs)

        # resampling
        X, y = under_sampler.fit_resample(X, y)
        return over_sampler.fit_resample(X, y)

    # def fit(self, X, y):

    #     # just store X and y rebalanced
    #     self.X_, self.y_ = self.fit_resample(X, y)
    #     self.state = 'fitted'

    # def transform(self, X, y):
    #     assert self.state == 'fitted', "cannot transform without fitting"
    #     self.state = 'transformed'
    #     return self.X_, self.y_


class AutoSklearnCCProWras(AutoSklearnPreprocessingAlgorithm,
                           OverAndUnderSampling):
    """
    This class provides ``OverAndUnderSampling`` for ProWRAS and Cluster-based
    under-sampling to auto-sklearn.
    """

    def __init__(self,
                 k_min,
                 k_maj,
                 n_cluster_neighbors,
                 max_affine,
                 random_state=None,
                 n_jobs=None):

        AutoSklearnPreprocessingAlgorithm.__init__(self)

        OverAndUnderSampling.__init__(
            self,
            k_min,
            k_maj,
            over_sampling=ProWRAS,
            under_sampling=ClusterCentroids,
            under_sampling_kwargs={},
            over_sampling_kwargs=dict(max_clusters=5,
                                      n_neighbors_max=5,
                                      decay_rate=1.,
                                      n_cluster_neighbors=n_cluster_neighbors,
                                      max_affine=max_affine,
                                      n_shadow=100,
                                      std=1e-3))
        self.state = "unfitted"

    def fit(self, X, y):
        # remapping max_affine
        max_affine = 2
        if max_affine == 'full':
            max_affine = X.shape[0]
        elif max_affine == 'medium':
            max_affine = round(X.shape[0] / 2)
        self.over_sampling_kwargs['max_affine'] = max_affine

        # continuing with usual fit
        # Hack 1: store the result to return it on `transform`
        self.X_, y_ = OverAndUnderSampling.fit_resample(self, X, y)

        # Hack 2: change argument y in-place because we cannot return it
        # Hack 2.1: resize array in-place
        if type(y) is np.ndarray:
            y.resize(y_.shape, refcheck=False)
        else:
            raise RuntimeError("Error: received an object that is not numpy!")
        y[...] = y_[...]
        self.state = "fitted"
        return self

    def transform(self, X):
        # check if this was already fit-transformed and in this case return `X`
        # as it is
        if super(OverAndUnderSampling, self) == 'transformed':
            return X
        else:
            # the parameters of `transform` are discarded by
            self.state == 'transformed'
            # `OverAndUnderSampling`
            return self.X_

    @staticmethod
    def get_hyperparameter_search_space(
        dataset_properties: Optional[DATASET_PROPERTIES_TYPE] = None
    ) -> ConfigurationSpace:
        cs = ConfigurationSpace()
        cs.add_hyperparameter(
            CSH.UniformFloatHyperparameter(name='k_min', lower=0.0, upper=1.0))
        cs.add_hyperparameter(
            CSH.UniformFloatHyperparameter(name='k_maj', lower=1.0, upper=40.0))
        cs.add_hyperparameter(
            CSH.CategoricalHyperparameter(
                name='max_affine', choices=['minimum', 'medium', 'full']))
        cs.add_hyperparameter(
            CSH.UniformIntegerHyperparameter(name='n_cluster_neighbors',
                                             lower=5,
                                             upper=1000))
        return cs

    @staticmethod
    def get_properties(
        dataset_properties: Optional[DATASET_PROPERTIES_TYPE] = None
    ) -> Dict[str, Optional[Union[str, int, bool, Tuple]]]:
        return {
            'shortname': 'OverAndUnderSampling',
            'name': 'OverAndUnderSampling',
            'handles_regression': True,
            'handles_classification': True,
            'handles_multiclass': True,
            'handles_multilabel': True,
            'handles_multioutput': True,
            'is_deterministic': True,
            # TODO find out of this is right!
            'input': (DENSE, SPARSE, UNSIGNED_DATA, SIGNED_DATA),
            'output': (INPUT, )
        }


if __name__ == "__main__":
    print(f"Testing f{__file__}")

    # create random data
    X = np.random.rand(100, 45)
    y = np.random.randint(0, 12, 100)
    # make the data imbalanced
    y[y > 6] = 1
    y[y > 4] = 2

    print("############ OverAndUnderSampling ############")
    print("Starting histogram:")
    _, counts = np.unique(y, return_counts=True)
    print(counts)

    # re-sampling
    o = OverAndUnderSampling()
    X_new, y_new = o.fit_resample(X, y)

    print("Final histogram:")
    _, counts = np.unique(y_new, return_counts=True)
    print(counts)

    print("############ AutoSklearnCCProWras ############")
    print("Starting histogram:")
    _, counts = np.unique(y, return_counts=True)
    print(counts)

    o = AutoSklearnCCProWras(0.75, 1.25, 100, "medium")
    o.fit(X, y)
    X_new = o.transform(X)
    y_new = y

    print("Final histogram:")
    _, counts = np.unique(y_new, return_counts=True)
    print(counts)
