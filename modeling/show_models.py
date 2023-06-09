from pprint import pp
import pickle
from pathlib import Path

from autosklearn.estimators import AutoSklearnEstimator
from sklearn.base import is_classifier
import pandas as pd

from . import settings as S


def main():
    pd.set_option('display.expand_frame_repr', False)
    for fname in Path(f"{S.IMG_DIR}-holdout_{S.HOLDOUT}").glob("**/*.pkl"):
        if fname.name == "bag.pkl":
            continue
        model = pickle.load(open(fname, "rb"))
        if is_classifier(model) and not isinstance(model, AutoSklearnEstimator):
            print("--------")
            print(f"Found a model in {fname.parent}: {fname.name}")
            if hasattr(model, "describe"):
                model.describe()
            else:
                pp(model.get_params())
            print("\n--------\n")


if __name__ == "__main__":
    main()
