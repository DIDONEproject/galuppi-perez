from pathlib import Path

# DATA settings
DATA_DIR = Path('data')
TMP_FOLDER = './tmp_autosklearn'

# Experiments
Y_VARIABLE = 'Composer'
AUTOML_TIME = 60 # 4 * 3600
AUTOML_DEBUG = False
GRID_DEBUG = False
IMG_DIR = Path('experiments')
FINAL_TEST_FILE = "suspect_arias.pkl"
FINAL_TEST_IDS = [1239, 1241, 1242]
MIN_CARDINALITY = 20
FOLDS = 10
BAGGING_RESAMPLES = 1000
SKIPSEARCHES = False
SKIPBAGFITTING = False
HOLDOUT_FILE = "holdout.pkl"
HOLDOUT = 0.
