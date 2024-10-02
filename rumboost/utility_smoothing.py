import numpy as np

from scipy.optimize import minimize
from scipy.interpolate import interp1d, PchipInterpolator
from scipy.special import softmax
from lightgbm import Dataset

from rumboost.metrics import cross_entropy
from rumboost.nested_cross_nested import nest_probs, cross_nested_probs
from rumboost.utils import (
    data_leaf_value,
    map_x_knots,
)


def linear_extrapolator_wrapper(pchip):
    """
    A wrapper function that adds linear extrapolation to a PchipInterpolator object.

    Parameters
    ----------
    pchip : scipy.interpolate.PchipInterpolator
        The scipy interpolator object from the monotonic splines.

    Returns
    -------
    pchip : scipy.interpolate.PchipInterpolator
        The scipy interpolator object from the monotonic splines with linear extrapolation.
    """

    def pchip_linear_extrapolator(x):
        return np.where(
            x < pchip.x[0],
            pchip(pchip.x[0]) + (x - pchip.x[0]) * pchip.derivative()(pchip.x[0]),
            np.where(
                x > pchip.x[-1],
                pchip(pchip.x[-1])
                + (x - pchip.x[-1]) * pchip.derivative()(pchip.x[-1]),
                pchip(x),
            ),
        )

    return pchip_linear_extrapolator


def monotone_spline(
    x_spline,
    weights,
    num_splines=5,
    x_knots=None,
    y_knots=None,
    linear_extrapolation=False,
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
        If True, the splines are linearly extrapolated before and after the first and last knots.


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
        x_knots = [
            x_ii + 1e-10 if x_i == x_ii else x_ii
            for x_i, x_ii in zip(x_knots[:-1], x_knots[1:])
        ]
        x_knots.insert(0, first_point)

    # create spline object
    pchip = PchipInterpolator(x_knots, y_knots, extrapolate=True)

    if linear_extrapolation:
        pchip = linear_extrapolator_wrapper(pchip)

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
        If True, the splines are linearly extrapolated before and after the first and last knots.

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
    fe_model=None,
    target="choice",
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
    mu : list, optional (default=None)
        Only used, and required, if nests is True. It is the list of mu values for each nest.
        The first value correspond to the first nest and so on.
    nests : dict, optional (default=False)
        If not none, compute predictions with the nested probability function. The dictionary keys are alternatives number and their values are
        their nest number. By example {0:0, 1:1, 2:0} means that alt 0 and 2 are in nest 0 and alt 1 is in nest 1.
    alphas : numpy.ndarray, optional (default=None)
        An array of J (alternatives) by M (nests).
        alpha_jn represents the degree of membership of alternative j to nest n.
    fe_model : RUMBoost, optional (default=None)
        The socio-economic characteristics part of the functional effect model.

    Returns
    -------
    preds : numpy array
        A numpy array containing the predictions for each class for each observation. Predictions are computed through the softmax function,
        unless the raw utilities are requested. A prediction for class j for observation n will be U[n, j].
    """
    raw_preds = np.array(np.zeros((data_test.shape[0], len(util_collection))))
    for u in util_collection:
        for f in util_collection[u]:
            raw_preds[:, int(u)] += util_collection[u][f](data_test[f])

    # adding the socio-economic constant
    if fe_model is not None:
        raw_preds += fe_model.predict(
            Dataset(data_test, label=data_test[target], free_raw_data=False),
            utilities=True,
        )

    # probabilities
    if alphas is not None:
        preds, _, _ = cross_nested_probs(raw_preds, mu, alphas)
        return preds

    if mu is not None:
        preds, _, _ = nest_probs(raw_preds, mu, nests)
        return preds

    if not utilities:
        preds = softmax(raw_preds, axis=1)
        return preds

    return raw_preds


def optimise_splines(
    x_knots,
    weights,
    data_train,
    data_test,
    label_test,
    spline_utilities,
    num_spline_range,
    x_first=None,
    x_last=None,
    deg_freedom=None,
    mu=None,
    nests=None,
    alphas=None,
    fe_model=None,
    linear_extrapolation=False,
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
    x_first : list, optional (default=None)
        A list of all first knots in the order of the attributes from spline_utilities and num_splines_range.
    x_last : list, optional (default=None)
        A list of all last knots in the order of the attributes from spline_utilities and num_splines_range.
    mu : list, optional (default=None)
        Only used, and required, if nests is True. It is the list of mu values for each nest.
        The first value correspond to the first nest and so on.
    nests : dict, optional (default=False)
        If not none, compute predictions with the nested probability function. The dictionary keys are alternatives number and their values are
        their nest number. By example {0:0, 1:1, 2:0} means that alt 0 and 2 are in nest 0 and alt 1 is in nest 1.
    alphas : numpy.ndarray, optional (default=None)
        An array of J (alternatives) by M (nests).
        alpha_jn represents the degree of membership of alternative j to nest n.
    fe_model : RUMBoost, optional (default=None)
        The socio-economic characteristics part of the functional effect model.
    linear_extrapolation : bool, optional (default=False)
        If True, the splines are linearly extrapolated before and after the first and last knots.

    Returns
    -------
    loss: float
        The final cross entropy or BIC on the test set.
    """
    x_knots_dict = map_x_knots(x_knots, num_spline_range, x_first, x_last)

    # compute new CE
    utility_collection = updated_utility_collection(
        weights,
        data_train,
        num_splines_feat=num_spline_range,
        spline_utilities=spline_utilities,
        x_knots=x_knots_dict,
        linear_extrapolation=linear_extrapolation,
    )
    smooth_preds_final = smooth_predict(
        data_test,
        utility_collection,
        mu=mu,
        nests=nests,
        alphas=alphas,
        fe_model=fe_model,
    )
    loss = cross_entropy(smooth_preds_final, label_test)
    # BIC
    if deg_freedom is not None:
        N = len(label_test)
        loss = 2 * N * loss + np.log(N) * deg_freedom
    return loss


def optimal_knots_position(
    weights,
    dataset_train,
    dataset_test,
    labels_test,
    spline_utilities,
    num_spline_range,
    max_iter=100,
    optimize=True,
    deg_freedom=None,
    n_iter=1,
    x_first=None,
    x_last=None,
    mu=None,
    nests=None,
    alphas=None,
    fe_model=None,
    linear_extrapolation=False,
):
    """
    Find the optimal position of knots for a given number of knots for given attributes.
    Smoothing is not implemented yet for shared ensembles.

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
    max_iter : int, optional (default=100)
        The maximum number of iterations from the solver
    optimize : bool, optional (default=True)
        If True, optimize the knots position with scipy.minimize
    deg_freedom : int, optional (default=None)
        The degree of freedom. If not specified, it is the number of knots to optimize.
    n_iter : int, optional (default=None)
        The number of iteration, to leverage the randomness induced by the local minimizer.
    x_first : list, optional (default=None)
        A list of all first knots in the order of the attributes from spline_utilities and num_splines_range.
    x_last : list, optional (default=None)
        A list of all last knots in the order of the attributes from spline_utilities and num_splines_range.
    mu : numpy.ndarray, optional (default=None)
        Only used, and required, if nests is True. It is the array of mu values for each nest.
        The first value correspond to the first nest and so on.
    nests : dict, optional (default=None)
        If not None, compute predictions with the nested probability function. The dictionary keys are nests id and their values are
        the alternatives belonging to the nest. By example {0:[0, 2], 1:[1]} means that alt 0 and 2 are in nest 0 and alt 1 is in nest 1.
    alphas : numpy.ndarray, optional (default=None)
        An array of J (alternatives) by N (nests).
        alpha_jn represents the degree of membership of alternative j to nest n.
    fe_model : RUMBoost, optional (default=None)
        The socio-economic characteristics part of the functional effect model.
    linear_extrapolation : bool, optional (default=False)
        If True, the splines are linearly extrapolated before and after th first and last knots.

    Returns
    -------
    x_opt : OptimizeResult
        The result of scipy.minimize.
    """

    ce = 1e10
    for n in range(n_iter):
        x_0 = []
        x_first = []
        x_last = []
        all_cons = []
        all_bounds = []
        starter = 0
        for u in num_spline_range:
            for f in num_spline_range[u]:
                # last split points
                last_split_point = weights[u][f]["Splitting points"][-1]

                # get first and last data points
                first_point = np.min(dataset_train[f])
                last_point = np.max(dataset_train[f])

                if x_first:
                    # initial knots position q/Nth quantile
                    x_0.extend(
                        [
                            np.quantile(
                                dataset_train[f].unique(), q / (num_spline_range[u][f])
                            )
                            for q in range(1, num_spline_range[u][f])
                        ]
                    )

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
                        for j in range(0, num_spline_range[u][f] - 2)
                    ]
                    # last knots must be greater than last split point
                    cons.append(
                        {
                            "type": "ineq",
                            "fun": lambda x, i_knot=starter + num_spline_range[u][
                                f
                            ] - 2, lsp=last_split_point: x[i_knot]
                            - lsp
                            - 1e-6,
                            "keep_feasible": True,
                        }
                    )
                    # knots must be within the range of data points
                    bounds = [
                        (
                            first_point + q * 1e-7,
                            last_point + (-num_spline_range[u][f] + 1 + q) * 1e-7,
                        )
                        for q in range(1, num_spline_range[u][f])
                    ]
                    x_first.append(first_point)
                    x_last.append(last_point)
                else:
                    x_0.extend(
                        [
                            np.quantile(
                                dataset_train[f].unique(),
                                0.95 * q / (num_spline_range[u][f]),
                            )
                            for q in range(0, num_spline_range[u][f] + 1)
                        ]
                    )
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
                    bounds = [
                        (
                            first_point + q * 1e-7,
                            last_point + (-num_spline_range[u][f] - 1 + q) * 1e-7,
                        )
                        for q in range(0, num_spline_range[u][f] + 1)
                    ]
                    starter += num_spline_range[u][f] + 1

                # store all constraints and first and last points
                all_cons.extend(cons)
                all_bounds.extend(bounds)

                # count the number of knots until now
                starter += num_spline_range[u][f] - 1

        if deg_freedom is not None:
            deg_freedom = starter

        if optimize:
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
                    x_first,
                    x_last,
                    deg_freedom,
                    mu,
                    nests,
                    alphas,
                    fe_model,
                    linear_extrapolation,
                ),
                bounds=all_bounds,
                constraints=all_cons,
                method="SLSQP",
                options={"maxiter": max_iter, "disp": True},
            )

            # compute final negative cross-entropy with optimised knots
            ce_final = optimise_splines(
                x_opt.x,
                weights,
                dataset_train,
                dataset_test,
                labels_test,
                spline_utilities,
                num_spline_range,
                x_first,
                x_last,
                deg_freedom,
                mu=mu,
                nests=nests,
                alphas=alphas,
                fe_model=fe_model,
                linear_extrapolation=linear_extrapolation,
            )

            # store best value
            if ce_final < ce:
                ce = ce_final
                x_opt_best = x_opt
                x_first_best = x_first
                x_last_best = x_last
            print(f"{n+1}/{n_iter}:{ce_final} with knots at: {x_opt.x}")
        else:
            # without optimisation
            final_loss = optimise_splines(
                np.array(x_0),
                weights,
                dataset_train,
                dataset_test,
                labels_test,
                spline_utilities,
                num_spline_range,
                x_first,
                x_last,
                deg_freedom,
                mu=mu,
                nests=nests,
                alphas=alphas,
                fe_model=fe_model,
                linear_extrapolation=linear_extrapolation,
            )

            if final_loss < ce:
                ce = final_loss
                x_opt_best = x_0
            print(f"{n+1}/{n_iter}:{final_loss}")

    # return best x_opt and first and last points + score
    if optimize:
        return x_opt_best, x_first_best, x_last_best, ce

    return x_opt_best, x_first, x_last, ce
