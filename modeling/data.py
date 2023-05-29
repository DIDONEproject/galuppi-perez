import pandas as pd
from pathlib import Path

import numpy as np

from .settings import DATA_DIR


LABELS_CSV = 'extraction_labels.csv'
FEATURES_CSV = 'extraction_features.csv'
METADATA_CSV = 'extraction_metadata.csv'


def normalize_strings(X):
    # Fixing composers
    X.Composer = X.Composer.str.split(' ').str[-1].str.title()
    return X


def __remove_string_cols(df):
    string_columns = (df.applymap(type) == str).all(0)
    return df[df.columns[string_columns]]


def load_features(label=None):
    """
    Load csv files, joins them, and returns them as ``pandas.DataFrame``
    objects.

    Parameters
    ----------

    `label` : str
        The name of a column that you want to use as `y` label. If it is set,
        it will be removed from the `X` dataframe.

    Returns
    -------

    `pandas.DataFrame` :
        All the available data coming from the set of features and from the
        passion labeling of each aria text

    `pandas.DataFrame` :
        The data that should be used as `X`; if ``label==True`` and ``label``
        is a column of `X`, the column with name `label` is removed from `X`.

        Also, `X` contains already encoded values for categorical data.

    `pandas.DataFrame` or `None` :
        The data that should be used as `Y`; if ``label==False``, `None` is
        returned.
    """
    metadata = pd.read_csv(Path(DATA_DIR) / METADATA_CSV)
    X = pd.read_csv(Path(DATA_DIR) / FEATURES_CSV)
    assert np.all(metadata['AriaId'] == X['AriaId']), 'Error in metadata.AriaId == X.AriaId'
    del X['AriaId']
    del X['WindowId'], metadata['WindowId']
    data = pd.concat([metadata, X], axis=1)
    data = data.sample(frac=1.0, random_state=987)

    data = normalize_strings(data).copy()
    # data has been shuffled, need to retake X
    X = data[X.columns].copy()

    # taking the label
    if label:
        y = data[label].copy()
        # make sure that X does not contains the label
        if label in X:
            del X[label]
    else:
        y = None

    # removing nans columns
    # data = data.dropna(axis=1)
    # X = X.dropna(axis=1)

    # only select numbers, categories, and strings
    # for now, only float (with encoded features something doesn't work)
    data_ = data.select_dtypes(include=['number', 'category'])
    X_ = X.select_dtypes(include=['number', 'category'])
    data = pd.concat([data_, __remove_string_cols(data)], axis=1)
    X = pd.concat([X_, __remove_string_cols(X)], axis=1)

    # # encoding X
    # feature_names = X.columns
    # X = OrdinalEncoder().fit_transform(X)
    # X = pd.DataFrame(X, columns=feature_names).copy()
    return data, X, y
