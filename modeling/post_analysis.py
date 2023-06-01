import pickle
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import kruskal, wilcoxon
from sklearn.metrics import confusion_matrix
from statsmodels.stats.contingency_tables import StratifiedTable, mcnemar
from statsmodels.stats.multitest import multipletests

from . import settings as S
from .easy_tools import C


def _collect_stats(data, mfile, prediction_lists, probs_lists, s, wrong_indices):
    wrong_index = []
    wrong_locations = []
    predictions = []
    probs = []
    indices = []
    conf_matrices = s["test_confusion_matrix"]
    proba_collector = s["test_proba_collector"]
    for idx in range(len(conf_matrices)):
        d = conf_matrices[idx].eval
        indices.append(d["indices"])
        predictions.append(d["predicted"])
        wrong_index.append(d["wrong"].to_list())
        wrong_locations.append(d["indices"].isin(d["wrong"]))
        probs.append(proba_collector[idx].eval["predicted"])
    wrong_index = np.concatenate(wrong_index).astype(np.int32)
    wrong_indices[mfile.parent.name] = wrong_index
    prediction_lists.append((mfile.parent.name, np.concatenate(predictions)))
    probs_lists.append((mfile.parent, np.concatenate(probs)))
    wrong_locations = np.concatenate(wrong_locations)
    data.iloc[indices][mfile.parent.name] = probs_lists[-1][1]
    return data, wrong_locations


def _analyze_errors(data, wrong_indices, keys_order):
    # now compute the intersection of linear and blacbox automl
    wrong_ab_intersection = set(wrong_indices[keys_order[0]]).intersection(
        set(wrong_indices[keys_order[1]])
    )
    wrong_abc_intersection = wrong_ab_intersection.intersection(
        set(wrong_indices[keys_order[2]])
    )
    print(
        f"{C.OKGREEN}Arias predicted wrongly by all of the three models{C.ENDC}"
    )
    df = pd.DataFrame()
    for idx in wrong_abc_intersection:
        aria = data.iloc[idx]
        df = pd.concat(
            [df, aria[["Id", "AriaLabel", "AriaName", S.Y_VARIABLE]]], axis=1
        )
    print(C.OKBLUE)
    print(df.T)
    print(C.ENDC)

    # wrong_ab_union = set(wrong_indices[keys_order[0]]).union(
    #     set(wrong_indices[keys_order[1]])
    # )
    # print(
    #     f"{C.OKGREEN}Arias predicted wrongly by the first two ({keys_order[0]}, {keys_order[1]}) but not from the third ({keys_order[2]}){C.ENDC}"
    # )
    # df = pd.DataFrame()
    # for idx in wrong_ab_intersection.difference(set(wrong_indices[keys_order[2]])):
    #     aria = data.iloc[idx]
    #     df = pd.concat(
    #         [df, aria[["Id", "AriaLabel", "AriaName", S.Y_VARIABLE]]], axis=1
    #     )
    # print(C.OKBLUE)
    # print(df.T)
    # print(C.ENDC)

    # print(
    #     f"{C.OKGREEN}Arias predicted wrongly by the third but not from the first two{C.ENDC}"
    # )
    # df = pd.DataFrame()
    # for idx in set(wrong_indices[keys_order[2]]).difference(wrong_ab_union):
    #     aria = data.iloc[idx]
    #     df = pd.concat(
    #         [df, aria[["Id", "AriaLabel", "AriaName", S.Y_VARIABLE]]], axis=1
    #     )
    # print(C.OKBLUE)
    # print(df.T)
    # print(C.ENDC)


def _statistical_significance(prediction_lists, probs_lists):
    print(f"{C.OKGREEN}Bonferroni-Holm corrected McNemar p-values:{C.ENDC}")
    mcnemar_corrected(prediction_lists)
    print(f"{C.OKGREEN}Kruskal-Wallis p-values:{C.ENDC}")
    kruskalwallis(prediction_lists)
    print(f"{C.OKGREEN}Bonferroni-Holm corrected Wilcoxon p-values:{C.ENDC}")
    wilcoxon_corrected(probs_lists)


def _load_outputs(data, experiments_dir, y):
    """
    Loads outputs from cross-validation experiments and collects statistics on wrong
    predictions.

    Returns:
    Tuple[pd.DataFrame, List[List[int]], List[List[float]], Dict[str, List[int]],
          Dict[str, Dict[str, List[int]]]]: A tuple containing:
        - The input data with additional columns for predicted values and probabilities.
        - A list of lists containing the predicted values for each cross-validation
          fold (concatenated).
        - A list of lists containing the predicted probabilities for each
          cross-validation fold (concatenated).
        - A dictionary mapping experiment directories to lists of indices of wrong
          predictions.
        - A dictionary mapping experiment directories to dictionaries mapping file paths
          to lists of indices of wrong predictions.

    Raises:
    FileNotFoundError: If a crossval_scores.pkl file is not found in the experiments
    directory.
    """
    data = data[["Id", "AriaLabel", "AriaName", S.Y_VARIABLE]].copy()
    experiments_dir = Path(experiments_dir)
    wrong_indices = {}
    prediction_lists = []
    probs_lists = []
    wrong_locations_dict = {}
    for mfile in experiments_dir.glob("**/crossval_scores.pkl"):
        print(f"{C.OKGREEN}Analyzing errors for {mfile}{C.ENDC}")
        s = pickle.load(open(mfile, "rb"))
        if "tree" in str(mfile):
            print("no decision function available")
            continue

        # storing wrong predictions, whole predictions, and probs
        data, wrong_locations = _collect_stats(
            data, mfile, prediction_lists, probs_lists, s, wrong_indices
        )

        wrong_index = wrong_indices[mfile.parent.name]
        if len(wrong_index) == 0:
            print("   There were no wrong predictions!")
            continue

        # Building a mini-report
        report = data.iloc[wrong_index][["AriaName", S.Y_VARIABLE]].copy()

        print(C.OKCYAN)
        print("Wrong arias:")
        print(report)
        print(C.ENDC)

        # Storing the wrong locations in a dictionary
        wrong_locations_dict[str(mfile.parent)] = wrong_locations

    return data, prediction_lists, probs_lists, wrong_indices, wrong_locations_dict


def _holdout_probability_histogram(X, y, experiments_dir, holdout_data):
    print("Testing hold-out set")
    holdout_X, holdout_y = holdout_data
    models = list(experiments_dir.glob("**/ensemble.pkl"))
    models += list(experiments_dir.glob("**/best_model.pkl"))
    holdout_probs_lists = []
    holdout_wrong_locations_dict = {}
    for mfile in models:
        model = pickle.load(open(mfile, 'rb'))
        model.fit(X, y)
        if hasattr(model, 'predict_proba'):
            probs = model.predict_proba(holdout_X)[:, 0]
            preds = model.predict(holdout_X)
            wrong_locations = preds != holdout_y
        holdout_probs_lists.append((mfile.parent, probs))
        holdout_wrong_locations_dict[str(mfile.parent)] = wrong_locations
    print("Computing histograms for hold-out set.")
    probability_histogram(holdout_probs_lists, holdout_wrong_locations_dict,
                          fname="prob_histogram_holdout.svg")


def post_analysis(data, X, y, experiments_dir, holdout):
    data = data.sort_index()
    X = X.sort_index()
    y = y.sort_index()
    (
        data,
        prediction_lists,
        probs_lists,
        wrong_indices,
        wrong_locations_dict,
    ) = _load_outputs(data, experiments_dir, y)

    column_names_probs = [p[0].name for p in probs_lists]
    find_most_typical_arias(data, column_names_probs, wrong_indices, percentile=90)

    if holdout > 0:
        holdout_data = pickle.load(open(S.HOLDOUT_FILE, 'rb'))
        _holdout_probability_histogram(X, y, experiments_dir, holdout_data)

    print("Computing histograms of probabilities.")
    probability_histogram(probs_lists, wrong_locations_dict)

    # compute statistical significance tests
    _statistical_significance(prediction_lists, probs_lists)

    _analyze_errors(data, wrong_indices, column_names_probs)


def find_most_typical_arias(data, names, wrong_indices, percentile=90):
    """
    Finds the most typical examples by finding the top and bottom percentile
    threshold values for each column in the given pandas DataFrame and then
    finding the samples that are in the top and bottom percentiles for all
    columns.

    Args:
    - data: A pandas DataFrame containing probability distributions.
    - names: A list of strings representing columns of data.
    - wrong_index: A dict of integers representing the indices to exclude from the
      result.
    Returns:
    - two list of integers representing the indices of the most extreme samples
      (top and bottom)
    """
    top_percentile = percentile
    bottom_percentile = 100 - top_percentile

    # Get the indices of the samples in the top and bottom percentiles for each column
    top_indices = []
    bottom_indices = []
    for name in names:
        top_threshold = np.percentile(data[name], top_percentile)
        bottom_threshold = np.percentile(data[name], bottom_percentile)
        top_indices.append(
            data.index[
                (data[name] >= top_threshold) & (~data.index.isin(wrong_indices[name]))
            ].tolist()
        )
        bottom_indices.append(
            data.index[
                (data[name] <= bottom_threshold)
                & (~data.index.isin(wrong_indices[name]))
            ].tolist()
        )

    # Find the samples that are in the top and bottom percentiles for *all* columns
    common_top_indices = set.intersection(*map(set, top_indices))
    common_bottom_indices = set.intersection(*map(set, bottom_indices))

    # Return the indices of the most extreme samples
    common_top_indices, common_bottom_indices = list(common_top_indices), list(
        common_bottom_indices
    )

    print(f"{C.OKGREEN}{C.BOLD}Most typical arias:")
    print(data.loc[common_top_indices, ["AriaName", S.Y_VARIABLE, *names]])
    print()
    print(data.loc[common_bottom_indices, ["AriaName", S.Y_VARIABLE, *names]])
    print(f"{C.ENDC}")


def probability_histogram(probs_lists, wrong_locations, fname="prob_histogram.svg"):
    from .plotting import plotly_save

    for probs in probs_lists:
        fig = go.Figure()

        fig.add_trace(
            go.Histogram(
                x=probs[1][wrong_locations[str(probs[0])]],
                nbinsx=30,
                name="Wrong predictions",
                marker=dict(color="red"),
            )
        )
        fig.add_trace(
            go.Histogram(
                x=probs[1],
                nbinsx=30,
                name="Predictions",
                marker=dict(color="blue"),
            )
        )

        fig.update_layout(
            yaxis_type="log",
            xaxis_title="Probability of being Galuppi",
            yaxis_title="Number of data (log)",
            xaxis=dict(
                tickmode="array",
            ),
        )

        plotly_save(fig, probs[0] / fname)


def kruskalwallis(probs_lists):
    kw_results = kruskal(*[prob_tuple[1] for prob_tuple in probs_lists])
    print(kw_results)


def wilcoxon_corrected(probs_lists):
    p_values = []
    for pair in combinations(probs_lists, 2):
        _, p_value = wilcoxon(pair[0][1], pair[1][1])
        p_values.append(p_value)
    corrected_p_values = multipletests(p_values, alpha=0.05, method="holm")[1]

    print("p-values:")
    for i, pairs in enumerate(combinations(probs_lists, 2)):
        print(f"{pairs[0][0]} vs {pairs[1][0]}", corrected_p_values[i])


def mcnemar_corrected(predictions):
    pvals = []
    mats = []
    print("p-value order:")
    for i, (name1, p1) in enumerate(predictions[:-1]):
        for name2, p2 in predictions[i + 1:]:
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
