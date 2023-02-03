from pathlib import Path
from shutil import rmtree

import fire
import sklearn

from . import easy_tools
from . import settings as S
from .data import load_features


class Model(object):
    """
    Execute experiments on Galuppi-Perez discrimination

    Available commands:
        - blackbox automl
        - linear automl
        - tree automl

        - baseline gridsearch

        - baseline permutation
        - ensemble permutation
        - linear permutation
        - tree permutation

        - test
        - baseline test
        - ensemble test
        - linear test
        - tree test

    Options:
        --output_dir : directory for saving images and models
        --debug : if using debug mode (no metalearning and only 3 runs)
        --prehook : name of function in `easy_tool` to be used for
            preprocessing the dataframes before of the experiment
    """

    def __init__(
        self,
        *,
        output_dir=S.IMG_DIR,
        debug=S.AUTOML_DEBUG,
        skipsearches=S.SKIPSEARCHES,
        skipbagfitting=S.SKIPBAGFITTING,
        prehook="select_galuppi_perez",
    ):
        S.GRID_DEBUG = debug
        if debug:
            print("Entering debug mode, press 'c' to continue")
            # __import__('ipdb').set_trace()
            self._automl_debug = dict(
                initial_configurations_via_metalearning=0,
                smac_scenario_args={"runcount_limit": 3},
                delete_tmp_folder_after_terminate=False,
            )
            S.BAGGING_RESAMPLES = 3
            S.PERMUTATION_REPEATS = 3
        else:
            self._automl_debug = {}

        self.skipsearches = skipsearches
        self.skipbagfitting = skipbagfitting
        self.output_dir_original = Path(output_dir)
        self.output_dir = Path(output_dir)
        self.prehook = prehook
        self.tmp_folder = S.TMP_FOLDER

    def linear(self, *, nsize: int = 5):
        """
        Perform AutoML optimization and permutation-based feature analysis
        using linear models
        """
        self.output_dir = self.output_dir / "linear_automl"
        self.interpretable = "linear"
        self.nsize = nsize
        return self

    def tree(self, *, nsize: int = 5):
        """
        Perform AutoML optimization and permutation-based feature analysis
        using tree models
        """
        self.output_dir = self.output_dir / "tree_automl"
        self.interpretable = "tree"
        self.nsize = nsize
        return self

    def blackbox(self, *, nsize: int = 5):
        """
        Perform AutoML optimization and permutation-based feature analysis
        using an ensemble of any kind of models. The maximum size of the
        ensemble is set by `nsize`. Optional argument (deault: 4)
        """
        self.output_dir = self.output_dir / "blackbox_automl"
        self.interpretable = "none"
        self.nsize = nsize
        return self

    def baseline(self):
        """
        Perform grid-search optimization using selected methods. See
        `./modeling/grid_tuning.py` for models included.
        """
        self.output_dir = self.output_dir / "gridsearch"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        return self

    def __load_data(self):
        data, X, y = load_features(S.Y_VARIABLE)
        self.X, self.y = getattr(easy_tools, self.prehook)(data, X, y)
        self.data = data.iloc[self.X.index]
        self.splitter = sklearn.model_selection.StratifiedKFold(
            n_splits=S.FOLDS, shuffle=True, random_state=8734
        )

    def post_analysis(self):
        """
        Performs post-hoc analysis
        """
        from .easy_tools import post_analysis

        self.__load_data()

        post_analysis(self.data, self.X, self.y, self.output_dir)

    def inspection(self):
        """
        Performs some analysis such as:
            * t-sne plot of all the arias, with the 3 suspected arias
            highlighted
            * variance-inflation-factor for the input data and with the pre-processing
            of the linear models (without bagging)
            * partial dependence plots of the 10 most important
            features in the model (without bagging)
        """
        from .easy_tools import custom_inspection

        self.__load_data()
        self.output_dir /= "inspection"
        self.output_dir.mkdir(exist_ok=True)
        custom_inspection(self.output_dir_original, self.output_dir, self.X, self.y)

    def automl(self):
        """
        Performs AutoML tuning.
        """
        from . import automl

        self.__load_data()
        if Path(self.tmp_folder).exists():
            rmtree(self.tmp_folder)
        automl(
            data_x_y=(self.X, self.y),
            splitter=self.splitter,
            output_dir=self.output_dir,
            interpretable=self.interpretable,
            tmp_folder=self.tmp_folder,
            automl_time=S.AUTOML_TIME,
            autosklearn_kwargs=dict(
                ensemble_size=self.nsize,
                **self._automl_debug,
            ),
            skipsearches=self.skipsearches,
            skipbagfitting=self.skipbagfitting,
        )

    def gridsearch(self):
        """
        Perform grid-search optimization using selected methods. See
        `./modeling/grid_tuning.py` for models included.
        """
        from .grid_tuning import gridsearch

        self.__load_data()
        gridsearch(
            data_x_y=(self.X, self.y),
            splitter=self.splitter,
            output_dir=self.output_dir,
            skipsearches=self.skipsearches,
        )

    def test(self):
        """
        Perform the final test on the suspicious arias using all the models
        found in the most recent experiment folder or the type of models
        specified.
        """
        from .easy_tools import test_suspected_arias

        test_suspected_arias(self.output_dir)

    def novelty(self):
        """
        Perform novelty detection on the suspected arias
        """

        from .easy_tools import test_novelty_detection

        test_novelty_detection()

    def features(self):
        """
        Extract features using musif and cached data
        """
        from musif.extract.extract import FeaturesExtractor

        from .feature_extraction.processor_didone import DataProcessorDidone

        # this creates a large dataframe with all data in it (metadata, labels,
        # features)
        raw_df = FeaturesExtractor(
            S.DATA_DIR / "config_extractor.yml",
            cache_dir=S.DATA_DIR / "cache",
            metadata_dir=S.DATA_DIR / "metadata",
        ).extract()

        # raw_df.to_csv("temp.csv")
        # raw_df = __import__("pd").read_csv("temp.csv")

        # this post-processes the dataframe, removing some features used for computing
        # other features, removing nans, etc.
        p = DataProcessorDidone(
            raw_df, S.DATA_DIR / "config_processor.yml", merge_voices=True
        )
        p.process()

        # this saves the dataframes to three files (_metadata.csv, _labels.csv,
        # _features.csv)
        p.save(S.DATA_DIR / "extraction")

    def dummy(self):
        """
        Perform evaluation of dummy models
        """
        from .plotting import plotly_save
        from .validation import crossvalidation

        self.output_dir = self.output_dir / "dummy"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.__load_data()

        best_score = -99
        best_model = None

        print("Looking for the best dummy strategy")
        for strategy in ["most_frequent", "prior", "uniform", "stratified"]:
            model = sklearn.dummy.DummyClassifier(strategy=strategy, random_state=629)
            scores, _ = crossvalidation(
                model, self.X, self.y, cv=self.splitter, plotting=False
            )
            score = scores["test_balanced_accuracy_score"].mean()
            if score > best_score:
                best_score = score
                best_model = model

        print("Saving the plots for the best dummy strategy")
        _, fig = crossvalidation(
            best_model,
            self.X,
            self.y,
            cv=self.splitter,
            title=f"Results of {self.splitter}-fold cross-validation for {best_model.strategy} dummy strategy",
            plotting=True,
        )
        plotly_save(fig, Path(self.output_dir) / "crossvalidation.svg")


if __name__ == "__main__":
    fire.Fire(Model)
