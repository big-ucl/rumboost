import numpy as np

from scipy.optimize import minimize
from scipy.interpolate import interp1d, PchipInterpolator, CubicSpline
from scipy.special import softmax, expit
from lightgbm import Dataset

from rumboost.metrics import cross_entropy, binary_cross_entropy, mse
from rumboost.nested_cross_nested import nest_probs, cross_nested_probs
from rumboost.ordinal import threshold_preds
from rumboost.utils import (
    data_leaf_value,
    map_x_knots,
)


class LinearExtrapolatorWrapper:
    """
    A wrapper class that adds linear extrapolation to a PchipInterpolator object.

    """

    def __init__(self, pchip):
        """
        Initialise the wrapper class.

        Parameters
        ----------
        pchip : scipy.interpolate.PchipInterpolator
            The scipy interpolator object from the monotonic splines.
        """
        self.pchip = pchip
        self.x = pchip.x

    def __call__(self, x):
        """
        Call the wrapper class.

        Parameters
        ----------
        x : numpy array
            The x values to interpolate.

        Returns
        -------
        numpy array
            The interpolated values.
        """
        return self.pchip_linear_extrapolator(x)

    def pchip_linear_extrapolator(self, x):
        return np.where(
            x < self.pchip.x[0],
            self.pchip(self.pchip.x[0])
            + (x - self.pchip.x[0]) * self.pchip.derivative()(self.pchip.x[0]),
            np.where(
                x > self.pchip.x[-1],
                self.pchip(self.pchip.x[-1])
                + (x - self.pchip.x[-1]) * self.pchip.derivative()(self.pchip.x[-1]),
                self.pchip(x),
            ),
        )


def monotone_spline(
    x_spline,
    weights,
    num_splines=5,
    x_knots=None,
    y_knots=None,
    linear_extrapolation=False,
    monotonic=0,
):
    """
    A function that apply monotonic spline interpolation on a given feature.

    Parameters
    ----------
    x_spline : numpy array
        Data from the interpolated feature.
    weights : dict
        The dictionary corresponding to the feature leaf values.
    num_splines : int, optional (default=5)
        The number of splines used for interpolation.
    x_knots : numpy array, optional (default=None)
        The positions of knots. If None, linearly spaced.
    y_knots : numpy array, optional (default=None)
        The value of the utility at knots. Need to be specified if x_knots is passed.
    linear_extrapolation : bool, optional (default=False)
        If True, the splines are linearly extrapolated.
    monotonic : int, optional (default=0)
        The monotonic nature of the feature. If -1, the feature is decreasing,
        if 1, the feature is increasing, if 0, the feature is not monotonic.

    Returns
    -------
    x_spline : numpy array
        A vector of x values used to plot the splines.
    y_spline : numpy array
        A vector of the spline values at x_spline.
    pchip : scipy.interpolate.PchipInterpolator
        The scipy interpolator object from the monotonic splines.
    x_knots : numpy array
        The positions of knots. If None, linearly spaced.
    y_knots : numpy array
        The value of the utility at knots.
    """

    # create knots if None
    if x_knots is None:
        x_knots = np.linspace(np.min(x_spline), np.max(x_spline), num_splines + 1)
        x_knots, y_knots = data_leaf_value(x_knots, weights)

    # sort knots in ascending order
    is_sorted = lambda a: np.all(a[:-1] <= a[1:])
    if not is_sorted(x_knots):
        x_knots = np.sort(x_knots)

    # untangle knots that have the same values
    is_equal = lambda a: np.any(a[:-1] == a[1:])
    if is_equal(x_knots):
        first_point = x_knots[0]
        last_point = x_knots[-1]
        x_knots = [
            x_ii + (i + 1) * 1e-11 if x_i == x_ii else x_ii
            for i, (x_i, x_ii) in enumerate(zip(x_knots[:-1], x_knots[1:]))
        ]
        if last_point < x_knots[-1]:
            diff = x_knots[-1] - last_point
            x_knots = [x - diff if x >= last_point else x for x in x_knots]
        x_knots.insert(0, first_point)

    # create spline object
    if monotonic == 0:
        pchip = CubicSpline(x_knots, y_knots, extrapolate=True)
    else:
        pchip = PchipInterpolator(x_knots, y_knots, extrapolate=True)

    if linear_extrapolation:
        pchip = LinearExtrapolatorWrapper(pchip)

    # compute utility values
    y_spline = pchip(x_spline)

    return x_spline, y_spline, pchip, x_knots, y_knots


def mean_monotone_spline(x_data, x_mean, y_data, y_mean, num_splines=15):
    """
    A function that apply monotonic spline interpolation on a given feature.
    The difference with monotone_spline, is that the knots are on the closest stairs mean.

    Parameters
    ----------
    x_data : numpy array
        Data from the interpolated feature.
    x_mean : numpy array
        The x coordinate of the vector of mean points at each stairs
    y_data : numpy array
        V(x_value), the values of the utility at x.
    y_mean : numpy array
        The y coordinate of the vector of mean points at each stairs

    Returns
    -------
    x_spline : numpy array
        A vector of x values used to plot the splines.
    y_spline : numpy array
        A vector of the spline values at x_spline.
    pchip : scipy.interpolate.PchipInterpolator
        The scipy interpolator object from the monotonic splines.
    """
    # case where there are more splines than mean data points
    if num_splines + 1 >= len(x_mean):
        x_knots = x_mean
        y_knots = y_mean

        # adding first and last point for extrapolation
        if x_knots[0] != x_data[0]:
            x_knots = np.insert(x_knots, 0, x_data[0])
            y_knots = np.insert(y_knots, 0, y_data[0])

        if x_knots[-1] != x_data[-1]:
            x_knots = np.append(x_knots, x_data[-1])
            y_knots = np.append(y_knots, y_data[-1])

        # create interpolator
        pchip = PchipInterpolator(x_knots, y_knots, extrapolate=True)

        # for plot
        x_spline = np.linspace(0, np.max(x_data) * 1.05, 10000)
        y_spline = pchip(x_spline)

        return x_spline, y_spline, pchip, x_knots, y_knots

    # candidate for knots
    x_candidates = np.linspace(
        np.min(x_mean) + 1e-10, np.max(x_mean) + 1e-10, num_splines + 1
    )

    # find closest mean point
    idx = np.unique(np.searchsorted(x_mean, x_candidates, side="left") - 1)

    x_knots = x_mean[idx]
    y_knots = y_mean[idx]

    # adding first and last point for extrapolation
    if x_knots[0] != x_data[0]:
        x_knots = np.insert(x_knots, 0, x_data[0])
        y_knots = np.insert(y_knots, 0, y_data[0])

    if x_knots[-1] != x_data[-1]:
        x_knots = np.append(x_knots, x_data[-1])
        y_knots = np.append(y_knots, y_data[-1])

    # create interpolator
    pchip = PchipInterpolator(x_knots, y_knots, extrapolate=True)

    # for plot
    x_spline = np.linspace(0, np.max(x_data) * 1.05, 10000)
    y_spline = pchip(x_spline)

    return x_spline, y_spline, pchip, x_knots, y_knots


def updated_utility_collection(
    weights,
    data,
    num_splines_feat,
    spline_utilities,
    mean_splines=False,
    x_knots=None,
    linear_extrapolation=False,
    monotonic_structure=None,
):
    """
    Create a dictionary that stores what type of utility (smoothed or not) should be used for smooth_predict.

    Parameters
    ----------
    weights : dict
        A dictionary containing all leaf values for all utilities and all features.
    data : pandas DataFrame
        The pandas DataFrame used for training.
    num_splines_feat : dict
        A dictionary of the same format than weights of features names for each utility that are interpolated with monotonic splines.
        The key is a spline interpolated feature name, and the value is the number of splines used for interpolation as an int.
        There should be a key for all features where splines are used.
    spline_utilities : dict
        A dictionary containing attributes where splines are applied. Must be in the form ]
        {utility_indx: [attributes1, attributes2, ...], ...}.
    mean_splines : bool, optional (default = False)
        If True, the splines are computed at the mean distribution of data for stairs.
    x_knots : dict
        A dictionary in the form of {utility: {attribute: x_knots}} where x_knots are the spline knots for the corresponding
        utility and attributes
    linear_extrapolation : bool, optional (default=False)
        If True, the splines are linearly extrapolated.
    monotonic_structure : dict[dict[int]], optional (default=None)
        A dictionary of the same format than weights of features names for each utility. The first key contains the utility index.
        The second key contains the feature name. The value is an int representing the monotonic nature of that feature. If -1,
        the feature is decreasing, if 1, the feature is increasing, if 0, the feature is not monotonic.

    Returns
    -------
    util_collection : dict
        A dictionary containing the type of utility to use for all features in all utilities.
    """
    # initialise utility collection
    util_collection = {}

    # for all utilities and features that have leaf values
    for u in weights:
        util_collection[u] = {}
        for f in weights[u]:
            # data points and their utilities
            x_dat, y_dat = data_leaf_value(data[f], weights[u][f])

            # if using splines
            if f in spline_utilities[u]:
                # if mean technique
                if mean_splines:
                    x_mean, y_mean = data_leaf_value(
                        data[f], weights[u][f], technique="mean_data"
                    )
                    _, _, func, _, _ = mean_monotone_spline(
                        x_dat, x_mean, y_dat, y_mean, num_splines=num_splines_feat[u][f]
                    )
                # else, i.e. linearly sampled points
                else:
                    x_spline = np.linspace(np.min(data[f]), np.max(data[f]), num=10000)
                    x_knots_temp, y_knots = data_leaf_value(
                        x_knots[u][f], weights[u][f]
                    )
                    _, _, func, _, _ = monotone_spline(
                        x_spline,
                        weights,
                        num_splines=num_splines_feat[u][f],
                        x_knots=x_knots_temp,
                        y_knots=y_knots,
                        linear_extrapolation=linear_extrapolation,
                        monotonic=monotonic_structure[u][f],
                    )
            # stairs functions
            else:
                func = interp1d(
                    x_dat,
                    y_dat,
                    kind="previous",
                    bounds_error=False,
                    fill_value=(y_dat[0], y_dat[-1]),
                )

            # save the utility function
            util_collection[u][f] = func

    return util_collection


def smooth_predict(
    data_test,
    util_collection,
    utilities=False,
    mu=None,
    nests=None,
    alphas=None,
    thresholds=None,
):
    """
    A prediction function that used monotonic spline interpolation on some features to predict their utilities.
    The function should be used with a trained model only.

    Parameters
    ----------
    data_test : pandas DataFrame
        A pandas DataFrame containing the observations that will be predicted.
    util_collection : dict
        A dictionary containing the type of utility to use for all features in all utilities.
    utilities : bool, optional (default = False)
        if True, return the raw utilities.
    mu : ndarray, optional (default=None)
        An array of mu values, the scaling parameters, for each nest.
        The first value of the array correspond to nest 0, and so on.
    nests : dict, optional (default=None)
        A dictionary representing the nesting structure.
        Keys are nests, and values are the the list of alternatives in the nest.
        For example {0: [0, 1], 1: [2, 3]} means that alternative 0 and 1
        are in nest 0, and alternative 2 and 3 are in nest 1.
    alphas : ndarray, optional (default=None)
        An array of J (alternatives) by M (nests).
        alpha_jn represents the degree of membership of alternative j to nest n
        By example, alpha_12 = 0.5 means that alternative one belongs 50% to nest 2.
    thresholds : ndarray, optional (default=None)
        An array of thresholds for ordinal regression.

    Returns
    -------
    preds : numpy array
        A numpy array containing the predictions for each class for each observation. Predictions are computed through the softmax function,
        unless the raw utilities are requested. A prediction for class j for observation n will be U[n, j].
    """
    raw_preds = np.array(np.zeros((data_test.shape[0], len(util_collection))))
    num_classes = 0
    for u in util_collection:
        num_classes += 1
        for f in util_collection[u]:
            raw_preds[:, int(u)] += util_collection[u][f](data_test[f])

    if not utilities:
        # compute nested probabilities. pred_i_m is predictions of choosing i knowing m, pred_m is prediction of choosing nest m and preds is pred_i_m * pred_m
        if nests:
            nest_alt = np.zeros(num_classes)
            for n, alts in nests.items():
                nest_alt[alts] = int(n)
            preds, _, _ = nest_probs(raw_preds, mu=mu, nests=nests, nest_alt=nest_alt)

            return preds

        # compute cross-nested probabilities. pred_i_m is predictions of choosing i knowing m, pred_m is prediction of choosing nest m and preds is pred_i_m * pred_m
        if alphas is not None:
            preds, _, _ = cross_nested_probs(raw_preds, mu=mu, alphas=alphas)

            return preds

        # ordinal preds
        if thresholds is not None:
            preds = threshold_preds(raw_preds, thresholds)

            return preds

        if num_classes == 1:  # binary classification
            preds = expit(raw_preds)

            return preds

        # softmax
        preds = softmax(raw_preds, axis=1)
        return preds

    return raw_preds


def optimise_splines(
    x_knots,
    weights,
    data_train,
    data_test,
    labels_test,
    spline_utilities,
    num_spline_range,
    deg_freedom=None,
    task="multiclass",
    criterion="BIC",
    linear_extrapolation=False,
    monotonic_structure=None,
    with_collection=False,
    mu=None,
    nests=None,
    alphas=None,
    thresholds=None,
):
    """
    Function wrapper to find the optimal position of knots for each feature. The optimal position is the one
    who minimises the CE loss.

    Parameters
    ----------
    x_knots ; 1d np.array
        The positions of knots in a 1d array, following this structure:
        np.array([x_att1_1, x_att1_2, ... x_att1_m, x_att2_1, ... x_attn_m]) where m is the number of knots
        and n the number of attributes that are interpolated with splines.
    weights : dict
        A dictionary containing all leaf values for all utilities and all features.
    data_train : pandas DataFrame
        The pandas DataFrame used for training.
    data_test : pandas DataFrame
        The pandas DataFrame used for testing.
    label_test : pandas Series or numpy array
        The labels of the dataset used for testing.
    spline_utilities : dict
        A dictionary containing attributes where splines are applied. Must be in the form ]
        {utility_indx: [attributes1, attributes2, ...], ...}.
    num_splines_range : dict
        A dictionary of the same format than weights of features names for each utility that are interpolated with monotonic splines.
        The key is a spline interpolated feature name, and the value is the number of splines used for interpolation as an int.
        There should be a key for all features where splines are used.
    deg_freedom : int, optional (default=None)
        The degree of freedom. If not specified, it is the number of knots to optimize.
    task : str, optional (default='multiclass')
        The task to perform. Can be 'multiclass', 'binary' or 'regression'.
    criterion : str, optional (default='BIC')
        The criterion to use for the optimisation. Can be 'BIC', 'AIC' or 'VAL'.
    linear_extrapolation : bool, optional (default=False)
        If True, the splines are linearly extrapolated.
    monotonic_structure : dict[dict[int]], optional (default=None)
        A dictionary of the same format than weights of features names for each utility. The first key contains the utility index.
        The second key contains the feature name. The value is an int representing the monotonic nature of that feature. If -1,
        the feature is decreasing, if 1, the feature is increasing, if 0, the feature is not monotonic.
    with_collection : bool, optional (default=False)
        If True, return the utility collection.
    mu : float, optional (default=None)
        The mean parameter for the utility functions.
    nests : list, optional (default=None)
        The nested structure for the utility functions.
    alphas : list, optional (default=None)
        The alpha parameters for the utility functions.
    thresholds : list, optional (default=None)
        The thresholds for the utility functions.

    Returns
    -------
    loss: float
        The final cross entropy or BIC on the test set.
    """
    x_knots_dict = map_x_knots(x_knots, num_spline_range)

    # compute new smoothed utility
    utility_collection = updated_utility_collection(
        weights,
        data_train,
        num_splines_feat=num_spline_range,
        spline_utilities=spline_utilities,
        x_knots=x_knots_dict,
        linear_extrapolation=linear_extrapolation,
        monotonic_structure=monotonic_structure,
    )
    if task == "regression":
        smooth_preds_final = smooth_predict(
            data_test, utility_collection, utilities=True
        )
        loss = mse(smooth_preds_final, labels_test)
    elif task == "binary":
        smooth_preds_final = smooth_predict(
            data_test,
            utility_collection,
        )
        loss = binary_cross_entropy(smooth_preds_final, labels_test)
    else:
        smooth_preds_final = smooth_predict(
            data_test,
            utility_collection,
            mu=mu,
            nests=nests,
            alphas=alphas,
            thresholds=thresholds,
        )
        loss = cross_entropy(smooth_preds_final, labels_test)
    if deg_freedom is not None:
        N = len(labels_test)
        if criterion == "BIC":
            loss = N * loss + np.log(N) * deg_freedom
        elif criterion == "AIC":
            loss = N * loss + 2 * deg_freedom
        elif criterion == "VAL":
            loss = loss
        else:
            raise ValueError("Criterion must be BIC, AIC or VAL")
    if with_collection:
        return loss, utility_collection
    return loss


def optimal_knots_position(
    weights,
    dataset_train,
    dataset_test,
    labels_test,
    spline_utilities,
    num_spline_range,
    monotonic_structure,
    optimisation_problem="local",
    max_iter=100,
    optimise=True,
    deg_freedom=None,
    n_iter=1,
    fix_first=False,
    fix_last=False,
    task="multiclass",
    x0="quantile",
    criterion="BIC",
    folds=None,
    linear_extrapolation=False,
    method="SLSQP",
    mu=None,
    nests=None,
    alphas=None,
    thresholds=None,
    edge_fraction=0.25,
    middle_quantile=(0.05, 0.95),
    jump_data_limitation=95,
    jump_weight=1.0,
):
    """
    Find the optimal position of knots for a given number of knots for given attributes.

    Parameters
    ----------
    weights : dict
        A dictionary containing all leaf values for all utilities and all features.
    dataset_train : pandas DataFrame
        The pandas DataFrame used for training.
    dataset_test : pandas DataFrame
        The pandas DataFrame used for testing.
    labels_test : pandas Series or numpy array
        The labels of the dataset used for testing.
    spline_utilities : dict
        A dictionary containing attributes where splines are applied. Must be in the form ]
        {utility_indx: [attributes1, attributes2, ...], ...}.
    num_splines_range : dict
        A dictionary of the same format than weights of features names for each utility that are interpolated with monotonic splines.
        The key is a spline interpolated feature name, and the value is the number of splines used for interpolation as an int.
        There should be a key for all features where splines are used.
    monotonic_structure : dict[dict[int]]
        A dictionary of the same format than weights of features names for each utility. The first key contains the utility index.
        The second key contains the feature name. The value is an int representing the monotonic nature of that feature. If -1,
        the feature is decreasing, if 1, the feature is increasing, if 0, the feature is not monotonic.
    optimisation_problem : str, optional (default='local')
        The optimisation problem to solve. Can be 'local' or 'global'.
        If 'local', the optimisation is performed independently for each feature,
        with objective to minimise the mean squared error between the smoothed and non-smoothed curves.
        If 'global', the optimisation is performed jointly for all features, with objective
        to minimise the cross entropy loss of the smoothed predictions.
    max_iter : int, optional (default=100)
        The maximum number of iterations from the solver
    optimise : bool, optional (default=True)
        If True, optimise the knots position with scipy.minimize
    deg_freedom : int, optional (default=None)
        The degree of freedom. If not specified, it is the number of knots to optimise.
    n_iter : int, optional (default=None)
        The number of iteration, to leverage the randomness induced by the local minimizer.
    fix_first : bool, optional (default=False)
        If True, the first knot is fixed at the minimum value of the feature.
    fix_last : bool, optional (default=False)
        If True, the last knot is fixed at the maximum value of the feature.
    task : str, optional (default='multiclass')
        The task to perform. Can be 'multiclass', 'binary' or 'regression'.
    x0 : str, optional (default='quantile')
        The initialisation of the knots. Can be 'quantile', 'quantile_random', 'linearly_spaced', 'optimised' and 'random'.
    criterion : str, optional (default='BIC')
        The criterion to use for the optimisation. Can be 'BIC', 'AIC' or 'VAL'.
        If 'BIC', the Bayesian Information Criterion is used.
        If 'AIC', the Akaike Information Criterion is used.
        If 'VAL', the Validation loss is used.
    linear_extrapolation : bool, optional (default=False)
        If True, the splines are linearly extrapolated.
    method : str, optional (default='SLSQP')
        The method to use for the optimization. Can be any scipy optimization method.
    mu : float, optional (default=None)
        The mean parameter for the utility functions.
    nests : list, optional (default=None)
        The nested structure for the utility functions.
    alphas : list, optional (default=None)
        The alpha parameters for the utility functions.
    thresholds : list, optional (default=None)
        The thresholds for the utility functions.

    Returns
    -------
    x_opt : OptimizeResult
        The result of scipy.minimize.
    """

    if optimisation_problem == "local":
        return independent_smoothing(
            weights,
            dataset_train,
            spline_utilities,
            num_spline_range,
            x0_method=x0,
            linear_extrapolation=linear_extrapolation,
            monotonic_structure=monotonic_structure,
            X_val=dataset_test,
            optimise_knot_position=optimise,
            fix_first=fix_first,
            fix_last=fix_last,
            method=method,
            n_iter=n_iter,
            max_iter=max_iter,
            deg_freedom=deg_freedom,
            criterion=criterion,
            edge_fraction=edge_fraction,
            middle_quantile=middle_quantile,
            jump_data_limitation=jump_data_limitation,
            jump_weight=jump_weight,
        )

    ce = np.inf
    for n in range(n_iter):
        x_0 = []
        all_cons = []
        all_bounds = []
        starter = 0
        for u in num_spline_range:
            for f in num_spline_range[u]:
                # get first and last data points
                first_point = np.min(dataset_train[f])
                last_point = np.max(dataset_train[f])
                first_split_point = weights[u][f]["Splitting points"][0] - 1e-10
                last_split_point = weights[u][f]["Splitting points"][-1] + 1e-10
                mid_first_interval = (first_point + first_split_point) / 2
                mid_last_interval = (last_split_point + last_point) / 2

                # initial knots position q/Nth quantile or random
                if x0 == "random":
                    x_0_random = list(
                        np.sort(
                            np.random.uniform(
                                first_point, last_point, num_spline_range[u][f] + 1
                            )
                        )
                    )
                    if fix_first:
                        x_0_random[0] = first_point
                    if fix_last:
                        x_0_random[-1] = last_point
                    x_0.extend(x_0_random)
                elif x0 == "quantile_random":
                    x_0_quantile = [
                        np.quantile(
                            dataset_train[f].unique(),
                            0.01 + 0.98 * (q / (num_spline_range[u][f])),
                        )
                        + np.random.normal(
                            0,
                            dataset_train[f].unique().std() / 10,
                        )
                        for q in range(0, num_spline_range[u][f] + 1)
                    ]
                    x_0.extend(x_0_quantile)
                elif x0 == "quantile_1_99":
                    x_0_quantile = [
                        np.quantile(
                            dataset_train[f].unique(),
                            0.01 + 0.98 * (q / (num_spline_range[u][f])),
                        )
                        for q in range(0, num_spline_range[u][f] + 1)
                    ]
                    x_0.extend(x_0_quantile)
                elif x0 == "quantile_5_95":
                    x_0_quantile = [
                        np.quantile(
                            dataset_train[f].unique(),
                            0.05 + 0.9 * (q / (num_spline_range[u][f])),
                        )
                        for q in range(0, num_spline_range[u][f] + 1)
                    ]
                    x_0.extend(x_0_quantile)
                elif x0 == "linearly_spaced":
                    x_0.extend(
                        np.linspace(first_point, last_point, num_spline_range[u][f] + 1)
                    )
                elif x0 == "linearly_spaced_sp":
                    x_0.extend(
                        np.linspace(
                            first_split_point,
                            last_split_point,
                            num_spline_range[u][f] + 1,
                        )
                    )
                elif x0 == "linearly_spaced_mid":
                    x_0.extend(
                        np.linspace(
                            mid_first_interval,
                            mid_last_interval,
                            num_spline_range[u][f] + 1,
                        )
                    )
                else:
                    x_0_quantile = [
                        np.quantile(
                            dataset_train[f].unique(), q / (num_spline_range[u][f])
                        )
                        for q in range(0, num_spline_range[u][f] + 1)
                    ]
                    x_0_quantile[0] += 1e-11
                    x_0_quantile[-1] -= 1e-11
                    x_0.extend(x_0_quantile)
                # knots must be greater than the previous one
                cons = [
                    {
                        "type": "ineq",
                        "fun": lambda x, i_plus=starter + j + 1, i_minus=starter + j: x[
                            i_plus
                        ]
                        - x[i_minus]
                        - 1e-6,
                        "keep_feasible": True,
                    }
                    for j in range(0, num_spline_range[u][f])
                ]

                # knots must be within the range of data points
                bounds = [
                    (
                        first_point + i * 1e-10,
                        last_point - (num_spline_range[u][f] - i) * 1e-10,
                    )
                    for i in range(0, num_spline_range[u][f] + 1)
                ]

                # first and last knots must be equal to first and last split points
                if fix_first:
                    bounds[0] = (first_point, first_point)
                if fix_last:
                    bounds[-1] = (last_point, last_point)
                # store all constraints and first and last points
                all_cons.extend(cons)
                all_bounds.extend(bounds)

                # count the number of knots until now
                starter += num_spline_range[u][f] + 1

        if deg_freedom is None:
            deg_freedom = starter

        x_opt_best = np.array(x_0)
        spline_collection_best = None

        if optimise:
            # optimise knot positions
            x_opt = minimize(
                optimise_splines,
                np.array(x_0),
                args=(
                    weights,
                    dataset_train,
                    dataset_test,
                    labels_test,
                    spline_utilities,
                    num_spline_range,
                    deg_freedom,
                    task,
                    criterion,
                    linear_extrapolation,
                    monotonic_structure,
                    False,
                    mu,
                    nests,
                    alphas,
                    thresholds,
                ),
                bounds=all_bounds,
                constraints=all_cons,
                method=method,
                options={"maxiter": max_iter, "disp": True},
            )

            # compute final negative cross-entropy with optimised knots
            ce_final, spline_collection = optimise_splines(
                x_opt.x,
                weights,
                dataset_train,
                dataset_test,
                labels_test,
                spline_utilities,
                num_spline_range,
                deg_freedom,
                task=task,
                criterion=criterion,
                linear_extrapolation=linear_extrapolation,
                monotonic_structure=monotonic_structure,
                with_collection=True,
                mu=mu,
                nests=nests,
                alphas=alphas,
                thresholds=thresholds,
            )

            # store best value
            if ce_final < ce:
                ce = ce_final
                x_opt_best = x_opt.x
                spline_collection_best = spline_collection
            # print(f"{n+1}/{n_iter}:{ce_final} with knots at: {x_opt.x}")
        else:
            # without optimisation
            final_loss, spline_collection = optimise_splines(
                np.array(x_0),
                weights,
                dataset_train,
                dataset_test,
                labels_test,
                spline_utilities,
                num_spline_range,
                deg_freedom,
                task=task,
                criterion=criterion,
                linear_extrapolation=linear_extrapolation,
                monotonic_structure=monotonic_structure,
                with_collection=True,
                mu=mu,
                nests=nests,
                alphas=alphas,
                thresholds=thresholds,
            )

            if final_loss < ce:
                ce = final_loss
                x_opt_best = x_0
                spline_collection_best = spline_collection
            # print(f"{n+1}/{n_iter}:{final_loss}")

    # return best x_opt and first and last points + score
    return x_opt_best, ce, spline_collection_best


def independent_smoothing(
    weights,
    dataset_train,
    spline_utilities,
    num_spline_range,
    x0_method="quantile",
    linear_extrapolation=False,
    monotonic_structure=None,
    X_val=None,
    optimise_knot_position=True,
    fix_first=False,
    fix_last=False,
    deg_freedom=None,
    n_iter=1,
    max_iter=200,
    method="SLSQP",
    criterion="BIC",
    edge_fraction=0.25,
    middle_quantile=(0.05, 0.95),
    jump_data_limitation=95,
    jump_weight=1.0,
):
    """
    A function that creates a new utility collection with independent smoothing.

    Parameters
    ----------
    weights : dict
        A dictionary containing all leaf values for all utilities and all features.
    dataset_train : pandas DataFrame
        The pandas DataFrame used for training.
    spline_utilities : dict
        A dictionary containing attributes where splines are applied. Must be in the form ]
        {utility_indx: [attributes1, attributes2, ...], ...}.
    num_spline_range : dict
        A dictionary of the same format than weights of features names for each utility that are interpolated with monotonic splines.
        The key is a spline interpolated feature name, and the value is a tuple with min and max number of splines used for interpolation.
        There should be a key for all features where splines are used.
    x0_method : str, optional (default='quantile')
        The method to use for the initial knots. Can be 'quantile', 'linearly_spaced', 'random' and 'optimised'.
        If optimised, the knots are optimised with the MSE on the curve.
    linear_extrapolation : bool, optional (default=False)
        If True, the splines are linearly extrapolated.
    monotonic_structure : dict[dict[int]], optional (default=None)
        A dictionary of the same format than weights of features names for each utility. The first key contains the utility index.
        The second key contains the feature name. The value is an int representing the monotonic nature of that feature. If -1,
        the feature is decreasing, if 1, the feature is increasing, if 0, the feature is not monotonic.
    X_val : pandas DataFrame, optional (default=None)
        The pandas DataFrame used for validation. If None, the validation is done on the training set.
    optimise_knot_position : bool, optional (default=True)
        If True, the knots position is optimised with scipy.minimize.
    fix_first : bool, optional (default=False)
        If True, the first knot is fixed at the minimum value of the feature.
    fix_last : bool, optional (default=False)
        If True, the last knot is fixed at the maximum value of the feature.
    deg_freedom : int, optional (default=None)
        The degrees of freedom for the smoothing splines.
    n_iter : int, optional (default=1)
        The number of iterations for the optimization.
    max_iter : int, optional (default=200)
        The maximum number of iterations for the optimization.
    method : str, optional (default='SLSQP')
        The optimization method to use.
    criterion : str, optional (default='BIC')
        The criterion to use for model selection.

    Returns
    -------
    new_util_collection : dict
        A dictionary containing the new utility collection with independent smoothing.

    """

    new_util_collection = independent_utility_collection(
        weights,
        dataset_train,
        num_splines_feat=num_spline_range,
        spline_utilities=spline_utilities,
        monotonic_structure=monotonic_structure,
        linear_extrapolation=linear_extrapolation,
        x0_method=x0_method,
        X_val=X_val,
        optimise_knot_position=optimise_knot_position,
        fix_first=fix_first,
        fix_last=fix_last,
        deg_freedom=deg_freedom,
        n_iter=n_iter,
        max_iter=max_iter,
        method=method,
        criterion=criterion,
        edge_fraction=edge_fraction,
        middle_quantile=middle_quantile,
        jump_data_limitation=jump_data_limitation,
        jump_weight=jump_weight,
    )

    return None, None, new_util_collection


def independent_utility_collection(
    weights,
    data,
    num_splines_feat,
    spline_utilities,
    x0_method=None,
    linear_extrapolation=False,
    monotonic_structure=None,
    X_val=None,
    optimise_knot_position=True,
    fix_first=False,
    fix_last=False,
    deg_freedom=None,
    n_iter=1,
    max_iter=200,
    method="SLSQP",
    criterion="BIC",
    edge_fraction=0.25,
    middle_quantile=(0.05, 0.95),
    jump_data_limitation=95,
    jump_weight=1.0,
):
    """
    Create a dictionary that stores what type of utility (smoothed or not) should be used for smooth_predict.

    Parameters
    ----------
    weights : dict
        A dictionary containing all leaf values for all utilities and all features.
    data : pandas DataFrame
        The pandas DataFrame used for training.
    num_splines_feat : dict
        A dictionary of the same format than weights of features names for each utility that are interpolated with monotonic splines.
        The key is a spline interpolated feature name, and the value is the number of splines used for interpolation as an int.
        There should be a key for all features where splines are used.
    spline_utilities : dict
        A dictionary containing attributes where splines are applied. Must be in the form ]
        {utility_indx: [attributes1, attributes2, ...], ...}.
    x0_method : str, optional (default=None)
        The method to use for the initial knots. Can be 'quantile', 'linearly_spaced', 'random' and 'optimised'.
    linear_extrapolation : bool, optional (default=False)
        If True, the splines are linearly extrapolated.
    monotonic_structure : dict[dict[int]], optional (default=None)
        A dictionary of the same format than weights of features names for each utility. The first key contains the utility index.
        The second key contains the feature name. The value is an int representing the monotonic nature of that feature. If -1,
        the feature is decreasing, if 1, the feature is increasing, if 0, the feature is not monotonic.
    X_val : pandas DataFrame, optional (default=None)
        The pandas DataFrame used for validation. If None, the validation is done on the training set.
    optimise_knot_position : bool, optional (default=True)
        If True, the knots position is optimised with scipy.minimize.
    fix_first : bool, optional (default=False)
        If True, the first knot is fixed at the minimum value of the feature.
    fix_last : bool, optional (default=False)
        If True, the last knot is fixed at the maximum value of the feature.
    deg_freedom : int, optional (default=None)
        The degrees of freedom for the smoothing splines.
    n_iter : int, optional (default=1)
        The number of iterations for the optimization.
    max_iter : int, optional (default=200)
        The maximum number of iterations for the optimization.
    method : str, optional (default='SLSQP')
        The optimization method to use.
    criterion : str, optional (default='BIC')
        The criterion to use for model selection.

    Returns
    -------
    util_collection : dict
        A dictionary containing the type of utility to use for all features in all utilities.
    """

    # initialise utility collection
    util_collection = {}

    # for all utilities and features that have leaf values
    for u in weights:
        util_collection[u] = {}
        for f in weights[u]:
            # data points and their utilitiesd
            x_dat, y_dat = data_leaf_value(data[f], weights[u][f])

            if X_val is not None:
                x_dat_val, y_dat_val = data_leaf_value(X_val[f], weights[u][f])
            else:
                x_dat_val, y_dat_val = None, None

            # if using splines
            if f in spline_utilities[u]:
                # if mean technique
                func = find_best_spline(
                    x_dat,
                    y_dat,
                    weights[u][f],
                    num_splines_feat[u][f],
                    monotonic=monotonic_structure[u][f],
                    linear_extrapolation=linear_extrapolation,
                    x0_method=x0_method,
                    x_data_val=x_dat_val,
                    y_data_val=y_dat_val,
                    optimise_knot_position=optimise_knot_position,
                    fix_first=fix_first,
                    fix_last=fix_last,
                    deg_freedom=deg_freedom,
                    n_iter=n_iter,
                    max_iter=max_iter,
                    method=method,
                    criterion=criterion,
                    edge_fraction=edge_fraction,
                    middle_quantile=middle_quantile,
                    jump_data_limitation=jump_data_limitation,
                    jump_weight=jump_weight,
                )
            # stairs functions
            else:
                func = interp1d(
                    x_dat,
                    y_dat,
                    kind="previous",
                    bounds_error=False,
                    fill_value=(y_dat[0], y_dat[-1]),
                )

            # save the utility function
            util_collection[u][f] = func

    return util_collection


def find_best_spline(
    x_spline,
    y_data,
    weights,
    num_splines,
    monotonic=0,
    linear_extrapolation=False,
    x0_method=None,
    optimise_knot_position=True,
    x_data_val=None,
    y_data_val=None,
    fix_first=False,
    fix_last=False,
    deg_freedom=None,
    n_iter=1,
    max_iter=200,
    method="SLSQP",
    criterion="BIC",
    edge_fraction=0.25,
    middle_quantile=(0.05, 0.95),
    jump_data_limitation=95,
    jump_weight=1.0,
):
    """
    A function that apply monotonic spline interpolation on a given feature.

    Parameters
    ----------
    x_data : numpy array
        Data from the interpolated feature.
    y_data : numpy array
        V(x_value), the values of the utility at x.
    num_splines : tuple(int, int)
        The number of splines to use for interpolation.
    monotonic : int, optional (default=0)
        If 0, the spline is not monotonic. If 1, the spline is increasing. If -1, the spline is decreasing.
    linear_extrapolation : bool, optional (default=False)
        If True, the splines are linearly extrapolated.
    x0_method : str, optional (default=None)
        The method to use for the initial knots. Can be 'quantile', 'linearly_spaced', 'random' and 'optimised'.
    optimise_knot_position : bool, optional (default=True)
        If True, the knots position is optimised with scipy.minimize.
    x_data_val : numpy array, optional (default=None)
        Data from the interpolated feature for validation.
    y_data_val : numpy array, optional (default=None)
        V(x_value), the values of the utility at x for validation.
    fix_first : bool, optional (default=False)
        If True, the first knot is fixed at the minimum value of the feature.
    fix_last : bool, optional (default=False)
        If True, the last knot is fixed at the maximum value of the feature.
    deg_freedom : int, optional (default=None)
        The degrees of freedom for the smoothing splines.
    n_iter : int, optional (default=1)
        The number of iterations for the optimization.
    max_iter : int, optional (default=200)
        The maximum number of iterations for the optimization.
    method : str, optional (default='SLSQP')
        The optimization method to use.
    criterion : str, optional (default='BIC')
        The criterion to use for model selection.

    Returns
    -------
    best_spline : scipy.interpolate.PchipInterpolator or scipy.interpolate.CubicSpline
        The best spline object.
    """

    min_knots, max_knots = num_splines

    best_fit = 1e10
    best_spline = None

    for _ in range(n_iter):
        for n_knot in range(min_knots, max_knots + 1):

            if not deg_freedom:
                deg_freedom = n_knot - fix_first - fix_last

            if x0_method == "random":
                x_knots = np.sort(np.random.random(n_knot) * np.max(x_spline))
            elif x0_method == "quantile":
                x_knots = np.quantile(np.unique(x_spline), np.linspace(0, 1, n_knot))
            elif x0_method == "linearly_spaced":
                x_knots = np.linspace(np.min(x_spline), np.max(x_spline), n_knot)
            elif x0_method == "optimised":
                x_knots = np.quantile(
                    np.unique(x_spline), np.linspace(0.01, 0.99, n_knot)
                )
            elif x0_method == "data_stabled":
                x_knots = edge_mutant_stable(
                    x_spline, y_data, n_knot,
                    edge_fraction=edge_fraction,
                    middle_quantile=middle_quantile,
                    jump_data_limitation=jump_data_limitation,
                    jump_weight=jump_weight)

            if optimise_knot_position:
                bounds = [
                    (
                        x_knots[0] + i * 1e-10,
                        x_knots[-1] - (n_knot - i) * 1e-10,
                    )
                    for i in range(0, n_knot)
                ]

                if fix_first:
                    bounds[0] = (np.min(x_spline), np.min(x_spline))
                if fix_last:
                    bounds[-1] = (np.max(x_spline), np.max(x_spline))

                if x_data_val is not None and y_data_val is not None:
                    x_knots_optimised = minimize(
                        optimise_single_spline,
                        x_knots,
                        args=(
                            weights,
                            monotonic,
                            linear_extrapolation,
                            x_data_val,
                            y_data_val,
                        ),
                        method=method,
                        bounds=bounds,
                        options={"maxiter": max_iter, "disp": True},
                    )
                else:
                    x_knots_optimised = minimize(
                        optimise_single_spline,
                        x_knots,
                        args=(
                            weights,
                            monotonic,
                            linear_extrapolation,
                            x_spline,
                            y_data,
                        ),
                        method=method,
                        bounds=bounds,
                        options={"maxiter": max_iter, "disp": True},
                    )

                x_knots = x_knots_optimised.x

            x_knots, y_knots = data_leaf_value(x_knots, weights)
            # create spline object
            if monotonic == 0:
                sp_interp = CubicSpline(x_knots, y_knots, extrapolate=True)
            else:
                sp_interp = PchipInterpolator(x_knots, y_knots, extrapolate=True)

            if linear_extrapolation:
                sp_interp = LinearExtrapolatorWrapper(sp_interp)

            # compute utility values
            if x_data_val is not None and y_data_val is not None:
                x = x_data_val
                y = y_data_val
            else:
                x = x_spline
                y = y_data

            y_spline = sp_interp(x)

            # compute loss
            if criterion == "BIC":
                loss = (
                    np.mean((y - y_spline) ** 2)
                    + n_knot * np.log(y.shape[0]) / y.shape[0]
                )
            elif criterion == "AIC":
                loss = (
                    np.mean((y - y_spline) ** 2)
                    + 2 * n_knot / y.shape[0]
                )
            elif criterion == "VAL":
                loss = (
                    np.mean((y - y_spline) ** 2)
                )

            if loss < best_fit:
                best_fit = loss
                best_spline = sp_interp

    return best_spline


def optimise_single_spline(
    x_knots, weights, monotonic, linear_extrapolation, x_spline, y_data
):
    """
    A function that apply monotonic spline interpolation on a given feature.

    Parameters
    ----------
    x_knots : numpy array
        Knots for the spline.
    weights : numpy array
        Weights learnt by RUMBoost.
    monotonic : int
        If 0, the spline is not monotonic. If 1, the spline is increasing. If -1, the spline is decreasing.
    linear_extrapolation : bool
        If True, the splines are linearly extrapolated.
    x_spline : numpy array
        Data from the interpolated feature.
    y_data : numpy array
        V(x_value), the values of the utility at x.
    """

    x_knots, y_knots = data_leaf_value(x_knots, weights)
    # create spline object
    if monotonic == 0:
        sp_interp = CubicSpline(x_knots, y_knots, extrapolate=True)

    else:
        sp_interp = PchipInterpolator(x_knots, y_knots, extrapolate=True)

    if linear_extrapolation:
        sp_interp = LinearExtrapolatorWrapper(sp_interp)

    # compute utility values
    y_spline = sp_interp(x_spline)

    # compute loss
    loss = np.mean((y_data - y_spline) ** 2)

    return loss


def edge_mutant_stable(
    x_spline,
    y_data,
    n_knot,
    edge_fraction=0.25,
    middle_quantile=(0.05, 0.95),
    jump_data_limitation=95, # or None --- no cap
    jump_weight=1.0
):
    """
    A function that initialise spline knots for classification with the stability of edge and mutant data.
    
    Paramaters
    ----------
    x_data : numpy array
        Data from the interpolated feature.
    y_data : numpy array
        V(x_value), the values of the utility at x.
    x_spline : numpy array
        A vector of x values used to plot the splines.
    n_knot : int
        Total number of knots to place.
    edge_fraction : float
        Fraction of number of knots reserved for each edge. Evenly distribute these knots at each edge. Aim to improve the stability of edge interpolation.
    middle_quantile : tuple(float, float)
        Quantile range on Cumulative Distribution Function (CDF) used to distribute knots (a range for middle knots to distribute). Aim to improve the stability of the middle knots and reduce the risk of extrapolation.
    jump_data_limitation : int or None
        Apply percentile limitation to the mutant data and keep the robustness. Aim to avoid the extreme data jump and avoid the knots accumulate at the jump data points
    jump_weight : float
        Decide how middle knots will distribute. The data with big changes has larger weight and the flat data has less weight. Aim to distribute knots on the useful place to get a better smooth spline. It will combine with the jump_data_limitation.

    Returns
    -------
    x_knots : Monotonically increasing knot positions
    
    """

    # Uniquely ordered x-axis because of the requirements of spline interpolation
    x_order = np.unique(x_spline).astype(float, copy=False)
    
     # If number of knots is less than 2, the spline can't distribute these knots. So directly return the 1d array of knots.
  #  if x_order.size < 2:
   #     return x_order.copy()

    # Spline interpolation needs at least 2 knots to form an interval. So when there is only one value, constructing a strictly increasing sequence of length n_knot to ensure the sucessful caluculation and avoid the duplicated knots
    if x_order.size == 1:
        only_value = float(x_order[0])
        eps_1 = 1e-9 * max(1.0, abs(only_value)) # a very small value -- epsilon
        return only_value + eps_1 * np.arange(int(n_knot), dtype=float)
        #Return a sequence based on only_value, with gradually increasing small offsets


   # n_knot = min(n_knot, x_order.size - 1) # guarantee strict progressive increase and no duplicate knots at the same x value point.
    n_knot = int(n_knot)

    # 1. Edge Strategy
    edge_knot = max(2, int(round(n_knot * edge_fraction))) # At least 2 knots. Aim to calculate the number of knots should be put evenly at each edge.
    #If edge_fraction is really large, mid_knot can be protect by setting the minimum
    edge_knot = min(edge_knot, (n_knot - 2) // 2)  # guarantee the left-right symmetry. Make sure there is at least 2 knots in the middle.
    mid_knot  = n_knot - 2 * edge_knot # calculate the number of knots should be distributed in the middle range.

    quantile_low, quantile_high = middle_quantile
    # (start---min value of x, stop, num of knots, endpoint)
    low_loc = np.linspace(x_order[0],  np.quantile(x_order, quantile_low), edge_knot, endpoint=False)
    # (start, stop---max value of x, num of knots, endpoint)
    high_loc = np.linspace(np.quantile(x_order, quantile_high), x_order[-1],  edge_knot, endpoint=True)
    # Without border, thereby avoiding repetition. Then the middle range will have the border


    # 2. Mutant Strategy
    if mid_knot > 0:
        y_order = np.interp(x_order, x_spline, y_data)   # 1d linear interpolation
        dy = np.abs(np.diff(y_order))    # calculate |deta y| - The amplitude of change between two adjacent y values
        if dy.size == 0: # no jump data -- evenly distribute according to the place
            mid = np.linspace(np.quantile(x_order, quantile_low), np.quantile(x_order, quantile_high), mid_knot) # evenly
        else: # if there is jump data
            if jump_data_limitation is not None:
                cap = np.percentile(dy, jump_data_limitation) # find the target less than 95% dy (default value) -- improve robustness
                dy = np.minimum(dy, cap) # get the new dy range - maximum dy from original value to 95% value
            w = (dy ** jump_weight) + 1e-16  # weight and avoid 0
            w = w / w.sum()   # Normalised to probability weights (summing to 1).
            w   = np.r_[w, w[-1]] # Align the lengths of w and x_order.
            cdf = np.cumsum(w)
            cdf = cdf / cdf[-1] # Make the last one is 1. Rescale the CDF to the interval [0, 1].
            pb_knot  = np.linspace(quantile_low, quantile_high, mid_knot) # get the middle quantiles of cdf, which are distributed uniformly according to probability
            mid = np.interp(pb_knot, cdf, x_order) # Map quantile positions (qs) via CDF to x_order to obtain middle-knot locations.
    else:
        mid = np.array([])

    # 3. Combine and guarantee the strict increase of x-axis and not duplicated
    x_knots = np.sort(np.r_[low_loc, mid, high_loc]).astype(float, copy=False) # merge the data - order
    # x_knots.sort()
    
    range_x = float(x_order[-1] - x_order[0]) if x_order.size > 1 else 1.0
    eps_2 = 1e-9 * max(range_x, 1.0)
    # When more then 2 knots, checking is working.
    if x_knots.size >= 2:
        basis = x_knots[0] # use the first knot as the first basis
        for i in range(1, x_knots.size):
            if x_knots[i] <= basis:
                x_knots[i] = basis + eps_2  # make sure strict increase
            basis = x_knots[i]  # new basis
        # both edge: move inside a little bit to avoid edge interpolation instability
        x_knots[0]  = max(x_knots[0],  x_order[0]  + eps_2)
        x_knots[-1] = min(x_knots[-1], x_order[-1] - eps_2)

    # 4. Guarantee the length
    m_knot = x_knots.size  # current number of knots
    nk = int(n_knot)  # total number of knots to place (get the whole number)
    if m_knot != nk:
        if m_knot > nk:  # more knots - use linspace to get nk evenly
            idx = np.linspace(0, m_knot - 1, nk).round().astype(int)
            x_knots = x_knots[idx]
        else:       # m_knots are not enough - need to align the length
            point  = nk - m_knot  # need to add how many points
            new_point = np.linspace(x_knots[0], x_knots[-1], point + 2, endpoint=True)[1:-1] # remove edge point and get the middle point
            x_knots = np.sort(np.r_[x_knots, new_point])
        # move inside again because there is a new ordered sequence
        if x_knots.size >= 2:
            basis = x_knots[0]
            for i in range(1, x_knots.size):
                if x_knots[i] <= basis:
                    x_knots[i] = basis + eps_2
                basis = x_knots[i]
    
    return x_knots
