from .didone_automl import automl
from .plotting import tree_to_graphviz, plot_confusion_matrices, plot_time_performance, DidonePlotter, get_classifier_type
from .data import load_features
from .validation import crossvalidation
from .easy_tools import select_galuppi_perez, get_xy, get_automl, load_model
from .inspection import pipeline_backtrack
from .model_wrappers import DidoneClassifier, DidoneBagging

from warnings import simplefilter
import numpy as np
import pandas as pd
# fix missing retro-compatibility in numpy when pickling
old_fn_ = np.random._pickle.__randomstate_ctor


def new_fn(x, y=None):
    return old_fn_(x)


np.random._pickle.__randomstate_ctor = new_fn

# shut-down pandas performance warning
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)
