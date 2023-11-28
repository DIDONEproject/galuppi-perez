import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import FunctionTransformer


class FeatureDimredError(RuntimeError):
    pass


def __remove_feature_dimred(pipeline):
    """
    Removes some dimensionality reductions from the pipeline by modifying the
    classifier weights. This is only possible if the classifier is a linear
    classifier and if the dimensionality reduction is one of:
    * pca
    * fastica
    * truncatedSVD

    In all the other cases, the pipeline remains untouched.

    Returns the pipeline
    """
    from .grid_tuning import CustomXySeparator

    classifier = pipeline["classifier"]
    preprocessor = pipeline["feature_preprocessor"]
    if hasattr(classifier, "choice"):
        classifier = classifier.choice.estimator
    if hasattr(preprocessor, "choice"):
        preprocessor = pipeline["feature_preprocessor"].choice.preprocessor
    if type(preprocessor) is CustomXySeparator:
        preprocessor = preprocessor.obj

    if hasattr(classifier, "coef_"):
        weight_field = "coef_"
    else:
        raise FeatureDimredError("Cannot find a 'coef_' field in the classifier")

    if hasattr(preprocessor, "components_"):
        reduction_coordinates_field = "components_"
    elif hasattr(preprocessor, "x_rotations_"):
        reduction_coordinates_field = "x_rotations_"
    elif isinstance(preprocessor, FunctionTransformer) or preprocessor == "passthrough":
        return
    else:
        raise FeatureDimredError(
            "Cannot find a 'components_' or 'x_rotations_' field in the preprocessor"
        )

    n_components = None  # by default, keep all the components
    if hasattr(preprocessor, "n_components_"):
        n_components = preprocessor.n_components_

    reduction = getattr(preprocessor, reduction_coordinates_field)[:n_components]
    weights = getattr(classifier, weight_field)
    if weights.shape[1] == reduction.shape[0]:
        transpose = False
    elif weights.shape[1] == reduction.shape[1]:
        transpose = True
    else:
        raise RuntimeError(
            f"Shape error while removing dimensionality reduction: weights have shape {weights.shape}, but reduction has shape {reduction.shape}. Are you sure that there are no other steps in the middle?"
        )
    if not transpose:
        setattr(classifier, weight_field, weights @ reduction)
    else:
        setattr(classifier, weight_field, weights @ reduction.T)

    for i, step in enumerate(pipeline.steps):
        if step[1] is pipeline["feature_preprocessor"]:
            pipeline.steps[i] = step[0], FunctionTransformer()
    # if hasattr(pipeline['feature_preprocessor'], 'choice'):
    #     pipeline['feature_preprocessor'].choice = NoPreprocessing(123)
    #     pipeline['feature_preprocessor'].choice.preprocessor = 'passthrough'
    return pipeline


def __reconstruct_feature_masking(pipeline, in_feature_names=None):
    """
    Returns the feature names remained in pipeline after feature selection
    methods

    If `in_features_names` is None, number are used as names
    """

    max_recursion_level = 30

    def get_supports_recursive(obj, recursion_level):
        supports = []
        if recursion_level > max_recursion_level:
            return supports
        elif hasattr(obj, "get_support"):
            return [obj.get_support()]
        elif hasattr(obj, "__dict__"):
            for k, v in vars(obj).items():
                if type(k) is str and not k.startswith("_") and type(v) is not str:
                    supports += get_supports_recursive(v, recursion_level + 1)
        elif type(obj) in [list, set, tuple]:
            for v in obj:
                supports += get_supports_recursive(v, recursion_level + 1)
        return supports

    supports = get_supports_recursive(pipeline.steps, 0)

    # sort supports by length (from the largest to the smaller)
    supports = sorted(supports, reverse=True, key=len)
    # apply all the support to the first one
    if in_feature_names is None:
        in_feature_names = [str(i) for i in range(len(supports[0]))]

    in_feature_names = np.asarray(in_feature_names)
    for s in supports:
        in_feature_names = in_feature_names[s]

    return in_feature_names


def __setattr_recursive(obj, field, value, recursion_level=0, max_recursion_level=30):
    if recursion_level > max_recursion_level:
        return
    elif hasattr(obj, "__dict__"):
        for k, v in vars(obj).items():
            if type(k) is str and not k.startswith("_") and type(v) is not str:
                if k == field:
                    setattr(obj, field, value)
                else:
                    __setattr_recursive(
                        v, field, value, recursion_level + 1, max_recursion_level
                    )
    elif type(obj) in [list, set, tuple]:
        for v in obj:
            __setattr_recursive(
                v, field, value, recursion_level + 1, max_recursion_level
            )


def pipeline_backtrack(pipeline, feature_names=None):
    """
    Perform backtracks of linear feature reduction and feature selections
    over the pipelines
    """

    # if possible, remove feature dimensionality reduction methods
    __remove_feature_dimred(pipeline)

    # get the mask for the features if needed
    selected_feature_names = __reconstruct_feature_masking(pipeline, feature_names)

    classifier = pipeline["classifier"]
    return classifier, selected_feature_names


def umap_plot(data, suspected_data, colors=None, output_dir=None):
    """
    Plot UMAP of data with colors and with suspected_data highlighted.
    Saves to output_dir if provided.
    """
    import warnings

    from sklearn.decomposition import PCA
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from tqdm import tqdm
    # from sklearn.manifold import TSNE
    from umap import UMAP

    from .plotting import plotly_save

    data = pd.concat([data, suspected_data])
    if colors is not None:
        colors = pd.concat(
            [colors, pd.Series(["Suspected"] * len(suspected_data))]
        ).to_numpy()

    max_iter = 10**10
    not_converged = 0
    min_loss = np.inf
    best_params = None
    figs = []
    # run block of code and catch warnings
    with warnings.catch_warnings():
        # ignore all caught warnings
        warnings.filterwarnings("ignore")
        for dist in tqdm(
            [
                "l1",
                "l2",
                "cosine",
                "correlation",
                "braycurtis",
                "chebyshev",
                "cityblock",
                "canberra",
            ]
        ):
            for min_dist in np.geomspace(0.01, 1.0, 10):
                for learning_rate in np.linspace(0.01, 800, 5):
                    try:
                        embedder = make_pipeline(
                            StandardScaler(),
                            PCA(0.9),
                            UMAP(
                                n_neighbors=15,
                                n_components=2,
                                min_dist=min_dist,
                                # larger is better
                                # perplexity=perplexity,
                                # early_exaggeration=12.0,
                                # n_iter=max_iter,
                                # n_iter_without_progress=10**3,
                                n_epochs=1000,
                                # lower is better
                                # min_grad_norm=1e-7,
                                # trade-off
                                learning_rate=learning_rate,
                                # other
                                init="spectral",
                                random_state=1993,
                                metric=dist,  # "braycurtis",
                                # method="exact",
                                n_jobs=-1,  # no effect with 'euclidean' and 'exact'
                            ),
                        )

                        embedded = embedder.fit_transform(data)
                        # print(f"loss: {embedder['tsne'].kl_divergence_:.2e}")
                        # print(f"n_iter: {embedder.n_iter_:.2e}")
                        # if embedder["tsne"].n_iter_ == max_iter:
                        #     print("Method did not converged!")
                        #     not_converged += 1
                        # if embedder["tsne"].kl_divergence_ < min_loss:
                        #     min_loss = embedder["tsne"].kl_divergence_
                        #     best_params = (dist, learning_rate, min_dist)
                        #     print(f"Best parameters: {best_params}")
                        #     print(f"Best loss: {min_loss}")
                        fig = px.scatter(
                            x=embedded[:, 0], y=embedded[:, 1], color=colors
                        )
                        figs.append(fig)
                        if output_dir is not None:
                            plotly_save(
                                fig,
                                output_dir
                                / f"umap_{dist}_{learning_rate}_{min_dist}.svg",
                            )
                    except Exception as e:
                        print(e)
                        continue

    print(f"\nNumber of methods not converged: {not_converged}")
    print(f"Best parameters: {best_params}")
    print(f"Best loss: {min_loss}")
    return figs


def variance_inflation_factor(data, prefix="", output_dir=None):
    """
    Compute the variance inflation factor for all the variables in data.
    Plot distribution of data in output_dir.
    Return average and variance of VIF
    """
    from statsmodels.stats import outliers_influence

    from .plotting import plotly_save

    vifs = []
    for i in range(data.shape[1]):
        vifs.append(outliers_influence.variance_inflation_factor(data, i))

    avg = np.nanmean(vifs)
    var = np.nanvar(vifs, ddof=1)
    min = np.nanmin(vifs)
    max = np.nanmax(vifs)

    print(f"VIF (avg, var, min, max): {avg:.2e}, {var:.2e}, {min:.2e}, {max:.2e}")

    try:
        fig = px.histogram(vifs, nbins=20)
        plotly_save(fig, output_dir / f"{prefix}_vif.svg")
    except Exception as e:
        print("Couldn't create plot")
        print(e)
    return avg, var


def partial_dependence(
    model,
    data,
    y,
    features,
    linear=True,
    prefix="",
    output_dir=None,
    grid_resolution=10,
):
    """
    Create partial dependence plots according to model on data for the features in
    features.

    If 'linear' is True, than the angular coefficient of each feature's dependence plot
    is computed and returned
    """

    from sklearn import inspection

    from .plotting import plotly_save

    print(">>> creating partial dependence")
    classes = model.classes_
    print(">>>>> P=0.0 -> " + classes[1])
    print(">>>>> P=1.0 -> " + classes[0])
    angular_coeffs = {}

    for feature in features:
        pdp = inspection.partial_dependence(
            model,
            data,
            [feature],
            grid_resolution=grid_resolution,
            kind="average",
            response_method="auto",
        )
        if linear:
            y = pdp["average"][0]
            x = pdp["values"][0]
            angular_coeff = (y[0] - y[-1]) / (x[0] - x[-1])
            angular_coeffs[feature] = angular_coeff

        if output_dir is not None:
            dfs = []
            for c in classes:
                avg_response_for_class_c = pdp["individual"][0, y == c].mean(axis=0)
                dfs.append(
                    pd.DataFrame(
                        {
                            "y": avg_response_for_class_c,
                            "class": c,
                            feature: pdp["values"][0],
                        }
                    )
                )
            dfs.append(
                pd.DataFrame(
                    {
                        "y": pdp["average"][0],
                        "class": "Average",
                        feature: pdp["values"][0],
                    }
                )
            )
            fig = px.line(pd.concat(dfs, axis=0), x=feature, y="y", color="class")
            plotly_save(fig, output_dir / f"{prefix}_pdp-{feature}.svg")

    return angular_coeffs, classes
