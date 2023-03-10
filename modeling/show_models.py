import pickle
from pathlib import Path

from autosklearn.estimators import AutoSklearnEstimator
from sklearn.base import is_classifier
import pandas as pd

from . import settings as S


def main():
    pd.set_option('display.expand_frame_repr', False)
    for fname in Path(S.IMG_DIR).glob("**/*.pkl"):
        if fname.name == "bag.pkl":
            continue
        model = pickle.load(open(fname, "rb"))
        if is_classifier(model) and not isinstance(model, AutoSklearnEstimator):
            print("--------")
            print(f"Found a model in {fname.parent}: {fname.name}")
            if hasattr(model, "describe"):
                model.describe()
            else:
                print(model)
            print("\n--------\n")


if __name__ == "__main__":
    main()
