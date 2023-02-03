"""
Some tools to make it easy work from terminal
"""


import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from . import settings as S


class C:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


def select_dummy(data, X, y):
    """
    Just a dummy function which only removes features that are `nan`
    and that have too few samples (> S.MIN_CARDINALITY)
    """
    # computing columns that are not nan
    X_columns = X.dropna(axis=1).select_dtypes([float]).columns
    X = X[X_columns]

    # computing classes > S.MIN_CLASS_CARDINALITY
    classes, counts = np.unique(y, return_counts=True)
    classes = classes[counts > S.MIN_CARDINALITY]
    idx = y.isin(classes)

    return X[idx], y[idx]


def select_galuppi_perez(data, X, y):
    """
    This takes care of keeping only the data that we want.

    We also remove the two questionable arias.

    Finally, we only keep numeric features and remove features containig
    `np.nan`
    """

    # the following returns a list of indices of the data that are inside a
    # list
    idx_to_keep = y.isin(["Galuppi", "Perez"])

    # computing indices of the suspected arias
    idx_aria_to_test = data.Id.isin(S.FINAL_TEST_IDS)
    X_train = X[~idx_aria_to_test]
    y_train = y[~idx_aria_to_test]
    X_train = X.loc[idx_to_keep]
    y_train = y.loc[idx_to_keep]
    X_test = X[idx_aria_to_test]
    y_test = y[idx_aria_to_test]

    # computing columns that are valid for both X_train and X_test
    X_columns = (
        pd.concat([X_test, X_train]).dropna(axis=1).select_dtypes([float]).columns
    )
    X_test = X_test[X_columns]
    X_train = X_train[X_columns]

    # print("Using {len(X_columns)} features:")
    # print(X_columns.to_list())
    # print(" --------- ")

    # Removing suspected arias from the dataframe and saving them to file
    # so that we can test later
    pickle.dump((data[idx_aria_to_test], X_test, y_test), open(S.FINAL_TEST_FILE, "wb"))

    print(f"{C.OKGREEN}Num of arias in train: {X_train.shape[0]}")
    print(f"    of which by Perez: {y_train[y_train == 'Perez'].shape[0]}")
    print(f"    of which by Galuppi: {y_train[y_train == 'Galuppi'].shape[0]}")
    print(f"Num of arias in test: {X_test.shape[0]}")
    print(f"Number of features: {X_train.shape[1]}{C.ENDC}")
    return X_train, y_train


def get_xy():
    from .data import load_features

    return select_galuppi_perez(*load_features("Composer"))


def get_automl():
    import pickle
    from pathlib import Path

    automls = sorted(Path("experiments/automl/").glob("automl-*.pkl"))
    return pickle.load(open(automls[-1], "rb"))


def test_novelty_detection():
    from sklearn.decomposition import PCA
    from sklearn.ensemble import IsolationForest
    from sklearn.neighbors import LocalOutlierFactor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    data_test, X_test, y_test = pickle.load(open(S.FINAL_TEST_FILE, "rb"))

    X, y = get_xy()
    for c in np.unique(y):
        X_ = X[y == c]
        detector = make_pipeline(
            StandardScaler(),
            PCA(n_components=0.9),
            # LocalOutlierFactor(
            #     n_neighbors=80,
            #     novelty=True,
            #     contamination=1e-32,
            #     n_jobs=-1,
            # ),
            IsolationForest(
                n_estimators=1000,
                random_state=723,
                contamination=1e-32,
                n_jobs=-1,
                max_samples=X_.shape[0],
                bootstrap=True,
            ),
        )
        detector.fit(X_)
        scores = detector.decision_function(X_test)
        scores = pd.DataFrame(scores, columns=["LOF Score"])
        scores["Labeled Composer"] = data_test["Composer"].to_numpy()
        scores["Id"] = data_test["Id"].to_numpy()
        scores["AriaLabel"] = data_test["AriaLabel"].to_numpy()
        scores["AriaName"] = data_test["AriaName"].to_numpy()
        print(f"{C.OKGREEN}Outlier scores for class {c} (< 0 is outlier){C.ENDC}")
        print(scores)
        print("\n---------\n")


def test_suspected_arias(experiments_dir, refit=True):

    data_test, X_test, y_test = pickle.load(open(S.FINAL_TEST_FILE, "rb"))
    if refit:
        X, y = get_xy()

    experiments_dir = Path(experiments_dir)
    models = list(experiments_dir.glob("**/ensemble.pkl"))
    models += list(experiments_dir.glob("**/best_model.pkl"))
    # models += list(experiments_dir.glob("**/bag.pkl"))
    for mfile in models:
        print(f"{C.OKGREEN}Predictions using {mfile}{C.ENDC}")
        try:
            m = pickle.load(open(mfile, "rb"))
            if refit or (hasattr(m, 'fitted') and not m.fitted):
                if 'X' not in vars():
                    # same as `if not refit`
                    X, y = get_xy()
                m.fit(X, y)
            pred = m.predict(X_test)
            pred = pd.DataFrame(pred, columns=["Pred"])
            if hasattr(m, "predict_proba"):
                preds = m.predict_proba(X_test)
                for label in range(preds.shape[1]):
                    pred[f"P({label})"] = preds[:, label]
            pred["Label"] = data_test["Composer"].to_numpy()
            pred["Id"] = data_test["Id"].to_numpy()
            pred["AriaLabel"] = data_test["AriaLabel"].to_numpy()
            pred["AriaName"] = data_test["AriaName"].to_numpy()
        except Exception as e:
            print(f"{C.FAIL}Cannot predict with this model due to the following error:")
            import traceback
            print(traceback.print_exception(e))
            print(C.ENDC)
        else:
            print(C.OKCYAN)
            print(pred)
            print(C.ENDC)
        finally:
            print("\n------\n")


def load_model(path):
    """
    Just load the model saved into `directory`
    """
    try:
        return pickle.load(open(Path(path) / "bag.pkl", "rb"))
    except FileNotFoundError:
        print(
            "Model not found! Have you trained and saved it? Try running `automl()` first!"
        )


def post_analysis(data, X, y, experiments_dir):

    experiments_dir = Path(experiments_dir)
    wrong_sets = []
    prediction_lists = []
    for mfile in experiments_dir.glob("**/crossval_scores.pkl"):
        print(f"{C.OKGREEN}Analyzing errors for {mfile}{C.ENDC}")
        s = pickle.load(open(mfile, "rb"))
        # m = pickle.load(open(mfile, "rb"))
        # m.fit(X, y)
        # if isinstance(m, DidoneClassifier) and not m.has_decision_function():
        if "tree" in str(mfile):
            print("no decision function available")
            # this skips trees
            continue

        # storing wrong predictions and whole predictions
        wrongs = []
        predictions = []
        for score in s["test_confusion_matrix"]:
            predictions.append(score.eval["predicted"])
            wrongs.append(score.eval["wrong"].to_list())
        wrongs = np.concatenate(wrongs).astype(np.int32)
        wrong_sets.append(set(wrongs))
        prediction_lists.append((mfile.parent.name, np.concatenate(predictions)))
        if len(wrongs) == 0:
            print("   There were no wrong predictions!")
            continue

        # Building a mini-report
        report = data.iloc[wrongs]
        report = report[["Id", "AriaLabel", "AriaName"]]
        report["Label"] = y.iloc[wrongs].to_list()
        # X_wrongs = X.iloc[wrongs]
        # report["Confidence"] = m.decision_function(X_wrongs)
        print(C.OKCYAN)
        print(report)
        print(C.ENDC)

    # compute McNemar tests
    print(f"{C.OKGREEN}Bonferroni-Holm corrected McNemar p-values:{C.ENDC}")
    mcnemar_corrected(*prediction_lists)

    # now compute the intersection of linear and blacbox automl
    wrong_ab_intersection = wrong_sets[0].intersection(wrong_sets[1])
    wrong_ab_union = wrong_sets[0].union(wrong_sets[1])
    print(
        f"{C.OKGREEN}Arias predicted wrongly by the first two but not from the third{C.ENDC}"
    )
    df = pd.DataFrame()
    for idx in wrong_ab_intersection.difference(wrong_sets[2]):
        aria = data.iloc[idx]
        df = pd.concat(
            [df, aria[["Id", "AriaLabel", "AriaName", S.Y_VARIABLE]]], axis=1
        )
    print(C.OKBLUE)
    print(df.T)
    print(C.ENDC)

    print(
        f"{C.OKGREEN}Arias predicted wrongly by the third but not from the first two{C.ENDC}"
    )
    df = pd.DataFrame()
    for idx in wrong_sets[2].difference(wrong_ab_union):
        aria = data.iloc[idx]
        df = pd.concat(
            [df, aria[["Id", "AriaLabel", "AriaName", S.Y_VARIABLE]]], axis=1
        )
    print(C.OKBLUE)
    print(df.T)
    print(C.ENDC)


def mcnemar_corrected(*predictions):
    from sklearn.metrics import confusion_matrix
    from statsmodels.stats.contingency_tables import StratifiedTable, mcnemar
    from statsmodels.stats.multitest import multipletests

    pvals = []
    mats = []
    print("p-value order:")
    for i, (name1, p1) in enumerate(predictions[:-1]):
        for name2, p2 in predictions[i + 1 :]:
            mat = confusion_matrix(p1, p2)
            mats.append(mat)
            # compute mcnemar
            res = mcnemar(mat)
            pvals.append(res.pvalue)
            print(name1, name2)

    pvals = multipletests(pvals, alpha=0.05, method="holm")[1]
    print(pvals)
    print("\nStatified table summary (statsmodels)")
    print(StratifiedTable(mats).summary())


def custom_inspection(experiments_dir, output_dir, X, y):
    """
    Performs some analysis such as:
        * t-sne plot of all the arias, with the 3 suspected arias
        highlighted
        * variance-inflation-factor for the input data and with the pre-processing
        of the linear models (without bagging)
        * partial dependence plots of the 10 most important
        features in the model (without bagging)
    """
    import pickle

    from sklearn.base import clone
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    from .inspection import umap_plot, variance_inflation_factor
    from .plotting import InspectionError

    print(C.BOLD + C.OKBLUE + "Computing feature importance and PDP" + C.ENDC)
    for m in experiments_dir.glob("**/bag.pkl"):
        # computing and plotting most important features
        dir_name = m.parent.name
        if 'blackbox' in dir_name:
            continue
        print(">>> " + dir_name)
        # loading the bag
        bag = pickle.load(open(m, "rb"))
        # loading the simpler model
        simple_model = m.parent / "ensemble.pkl"
        if not simple_model.exists():
            # gridsearch...
            simple_model = simple_model.with_stem("best_model")
        simple_model = pickle.load(open(simple_model, "rb"))
        simple_model.fit(X, y)
        try:
            plotter = bag.get_plotter(base_model=simple_model, top_k=0.5, feature_names=X.columns)
            plotter.plot(X, y, output_dir, prefix=dir_name)
            del bag, plotter
        except InspectionError:
            pass

    print(C.BOLD + C.OKBLUE + "Computing VIF on linear automl" + C.ENDC)
    m = experiments_dir / "linear_automl" / "ensemble.pkl"
    m = pickle.load(open(m, "rb"))
    for idx, (w, model) in enumerate(m.models_with_weights):
        preprocessor = clone(Pipeline(model.steps[:-1]))
        variance_inflation_factor(
            preprocessor.fit_transform(X.to_numpy()),
            prefix=f"linear_automl_{idx}",
            output_dir=output_dir,
        )

    print(C.BOLD + C.OKBLUE + "Computing VIF on linear gridsearch" + C.ENDC)
    m = experiments_dir / "gridsearch" / "best_model.pkl"
    m = pickle.load(open(m, "rb"))
    preprocessor = clone(Pipeline(m.steps[:-1]))
    variance_inflation_factor(
        preprocessor.fit_transform(X),
        prefix="baseline",
        output_dir=output_dir,
    )

    print(C.BOLD + C.OKBLUE + "Computing VIF on raw data" + C.ENDC)
    variance_inflation_factor(
        StandardScaler().fit_transform(X), prefix="raw", output_dir=output_dir
    )

    print(C.BOLD + C.OKBLUE + "Computing UMAP embedding" + C.ENDC)
    data_test, X_test, y_test = pickle.load(open(S.FINAL_TEST_FILE, "rb"))
    umap_plot(X, X_test, colors=y, output_dir=output_dir)
