import numpy as np
from rumboost.metrics import cross_entropy
from scipy.special import softmax

def optimise_asc(asc, raw_preds, labels):
    """
    Optimise the ASC parameters of the model.

    Parameters
    ----------
    asc : np.array
        The array of ASC parameters.
    raw_preds : np.array
        The raw predictions of the model.
    labels : np.array
        The labels of the dataset.

    Returns
    -------
    asc : np.array
        The optimised ASC parameters.
    """
    raw_preds_asc = raw_preds + asc
    new_preds = softmax(raw_preds_asc, axis=1)
    new_ce = cross_entropy(new_preds, labels)

    return new_ce
    

def process_parent(parent, pairs):
    """
    Dig into the biogeme expression to retrieve name of variable and beta parameter. Work only with simple utility specification (beta * variable).
    """
    # final expression to be stored
    if parent.getClassName() == "Times":
        pairs.append(get_pair(parent))
    else:  # if not final
        try:  # dig into the expression
            left = parent.left
            right = parent.right
        except:  # if no left and right children
            return pairs
        else:  # dig further left and right
            process_parent(left, pairs)
            process_parent(right, pairs)
    return pairs


def get_pair(parent):
    """
    Return beta and variable names on a tupple from a parent expression.
    """
    left = parent.left
    right = parent.right
    beta = None
    variable = None
    for exp in [left, right]:
        if exp.getClassName() == "Beta":
            beta = exp.name
        elif exp.getClassName() == "Variable":
            variable = exp.name
    if beta and variable:
        return (beta, variable)
    else:
        raise ValueError("Parent does not contain beta and variable")


def bio_to_rumboost(
    model,
    all_columns=False,
    monotonic_constraints=True,
    interaction_contraints=True,
    fct_effect_variables=[],
):
    """
    Converts a biogeme model to a rumboost dict.

    Parameters
    ----------
    model : a BIOGEME object
        The model used to create the rumboost structure dictionary.
    all_columns : bool, optional (default = False)
        If True, do not consider alternative-specific features.
    monotonic_constraints : bool, optional (default = True)
        If False, do not consider monotonic constraints.
    interaction_contraints : bool, optional (default = True)
        If False, do not consider feature interactions constraints.
    fct_effect_variables : list, optional (default = [])
        The list of variables in the functional effect part of the model

    Returns
    -------
    rum_structure : dict
        A dictionary specifying the structure of a RUMBoost object.

    """
    utilities = model.loglike.util  # biogeme expression
    rum_structure = []

    # for all utilities
    for k, v in utilities.items():
        rum_structure.append(
            {
                "columns": [],
                "monotone_constraints": [],
                "interaction_constraints": [],
                "betas": [],
                "categorical_feature": [],
            }
        )
        if len(fct_effect_variables) > 0:
            rum_structure_re = {
                "columns": [],
                "monotone_constraints": [],
                "interaction_constraints": [],
                "betas": [],
                "categorical_feature": [],
            }
        for i, pair in enumerate(
            process_parent(v, [])
        ):  # get all the pairs of the utility

            if pair[1] in fct_effect_variables:
                rum_structure_re["columns"].append(pair[1])  # append variable name
                rum_structure_re["betas"].append(pair[0])  # append beta name
                if interaction_contraints:
                    rum_structure_re["interaction_constraints"].append(
                        len(rum_structure_re["interaction_constraints"])
                    )  # no interaction between features
                if monotonic_constraints:
                    bounds = model.getBoundsOnBeta(
                        pair[0]
                    )  # get bounds on beta parameter for monotonic constraint
                    if (bounds[0] is not None) and (bounds[1] is not None):
                        raise ValueError("Only one bound can be not None")
                    if bounds[0] is not None:
                        if bounds[0] >= 0:
                            rum_structure_re["monotone_constraints"].append(
                                1
                            )  # register positive monotonic constraint
                    elif bounds[1] is not None:
                        if bounds[1] <= 0:
                            rum_structure_re["monotone_constraints"].append(
                                -1
                            )  # register negative monotonic constraint
                    else:
                        rum_structure_re["monotone_constraints"].append(0)  # none

            else:
                rum_structure[-1]["columns"].append(pair[1])  # append variable name
                rum_structure[-1]["betas"].append(pair[0])  # append beta name
                if interaction_contraints:
                    if len(fct_effect_variables) > 0:
                        rum_structure[-1]["interaction_constraints"].append(
                            [len(rum_structure[-1]["interaction_constraints"])]
                        )  # no interaction between features
                    else:
                        rum_structure[-1]["interaction_constraints"].append(
                            [i]
                        )  # no interaction between features
                if monotonic_constraints:
                    bounds = model.getBoundsOnBeta(
                        pair[0]
                    )  # get bounds on beta parameter for monotonic constraint
                    if (bounds[0] is not None) and (bounds[1] is not None):
                        raise ValueError("Only one bound can be not None")
                    if bounds[0] is not None:
                        if bounds[0] >= 0:
                            rum_structure[-1]["monotone_constraints"].append(
                                1
                            )  # register positive monotonic constraint
                    elif bounds[1] is not None:
                        if bounds[1] <= 0:
                            rum_structure[-1]["monotone_constraints"].append(
                                -1
                            )  # register negative monotonic constraint
                    else:
                        rum_structure[k]["monotone_constraints"].append(0)  # none
        if all_columns:
            rum_structure[-1]["columns"] = [
                col
                for col in model.database.data.drop(
                    ["choice"], axis=1
                ).columns.values.tolist()
            ]
        if len(fct_effect_variables) > 0:
            rum_structure.append(rum_structure_re)

    return rum_structure


def get_mid_pos(data, split_points, end="data"):
    """
    Return the mid point in-between two split points for a specific feature (used in pw linear predict).

    Parameters
    ----------
    data: pandas Series
        The column of the dataframe associated with the feature.
    split_points : list
        The list of split points for that feature.
    end : str
        How to compute the mid position of the first and last point, it can be:
            -'data': add min and max values of data
            -'split point': add first and last split points
            -'mean_data': add the mean of data before the first split point, and after the last split point

    Returns
    -------

    mid_pos : list
        A list of points in the middle of every consecutive split points.
    """
    # getting position in the middle of splitting points intervals
    if len(split_points) > 1:
        mid_pos = [
            (sp2 + sp1) / 2 for sp2, sp1 in zip(split_points[:-1], split_points[1:])
        ]
    else:
        mid_pos = []

    if end == "data":
        mid_pos.insert(0, min(data))  # adding first point
        mid_pos.append(max(data))  # adding last point
    elif end == "split point":
        mid_pos.insert(0, min(split_points))  # adding first point
        mid_pos.append(max(split_points))  # adding last point
    elif end == "mean_data":
        mid_pos.insert(0, data[data < split_points[0]].mean())  # adding first point
        mid_pos.append(data[data > split_points[-1]].mean())  # adding last point

    return mid_pos


def get_mean_pos(data, split_points):
    """
    Return the mean point in-between two split points for a specific feature (used in smoothing).
    At end points, it is the mean of data before the first split point, and after the last split point.

    Parameters
    ----------
    data : pandas.Series
        The column of the dataframe associated with the feature.
    split_points : list
        The list of split points for that feature.

    Returns
    -------

    mean_data : list
        A list of points in the mean of every consecutive split points.
    """
    # getting the mean of data of splitting points intervals
    mean_data = [
        np.mean(data[(data < s_ii) & (data > s_i)])
        for s_i, s_ii in zip(split_points[:-1], split_points[1:])
    ]
    mean_data.insert(0, np.mean(data[data < split_points[0]]))  # adding first point
    mean_data.append(np.mean(data[data > split_points[-1]]))  # adding last point

    return mean_data


def data_leaf_value(data, weights_feature, technique="data_weighted"):
    """
    Computes the utility values of given data, according to the prespecified technique.

    Parameters
    ----------
    data : pandas.Series
        The column of the dataframe associated with the feature.
    weight_feature : dict
        The dictionary corresponding to the feature leaf values.
    technique : str, optional (default = weight_data)
        The technique used to compute data values. It can be:

            data_weighted : feature data and its utility values.
            mid_point : the mid point in between all splitting points.
            mean_data : the mean of data in between all splitting points.
            mid_point_weighted : the mid points in between all splitting points, weighted by the number of data points in the interval.
            mean_data_weighted : the mean of data in between all splitting points, weighted by the number of data points in the interval.

    Returns
    -------
    data_ordered : numpy array
        X coordinates of the data, or feature data point values.
    data_values : numpy array
        Y coordinates of the data, or utility values

    """
    if technique == "data_weighted":
        data_ordered = np.sort(data)
        idx = np.searchsorted(
            np.array(weights_feature["Splitting points"]), data_ordered
        )
        data_values = np.array(weights_feature["Histogram values"])[idx]

        return np.array(data_ordered), data_values

    if technique == "mid_point":
        mid_points = np.array(get_mid_pos(data, weights_feature["Splitting points"]))
        return mid_points, np.array(weights_feature["Histogram values"])
    elif technique == "mean_data":
        mean_data = np.array(get_mean_pos(data, weights_feature["Splitting points"]))
        return mean_data, np.array(weights_feature["Histogram values"])

    data_ordered = data.copy().sort_values()
    data_values = [weights_feature["Histogram values"][0]] * sum(
        data_ordered < weights_feature["Splitting points"][0]
    )

    if technique == "mid_point_weighted":
        mid_points = get_mid_pos(data, weights_feature["Splitting points"])
        mid_points_weighted = [mid_points[0]] * sum(
            data_ordered < weights_feature["Splitting points"][0]
        )
    elif technique == "mean_data_weighted":
        mean_data = get_mean_pos(data, weights_feature["Splitting points"])
        mean_data_weighted = [mean_data[0]] * sum(
            data_ordered < weights_feature["Splitting points"][0]
        )

    for i, (s_i, s_ii) in enumerate(
        zip(
            weights_feature["Splitting points"][:-1],
            weights_feature["Splitting points"][1:],
        )
    ):
        data_values += [weights_feature["Histogram values"][i + 1]] * sum(
            (data_ordered < s_ii) & (data_ordered > s_i)
        )
        if technique == "mid_point_weighted":
            mid_points_weighted += [mid_points[i + 1]] * sum(
                (data_ordered < s_ii) & (data_ordered > s_i)
            )
        elif technique == "mean_data_weighted":
            mean_data_weighted += [mean_data[i + 1]] * sum(
                (data_ordered < s_ii) & (data_ordered > s_i)
            )

    data_values += [weights_feature["Histogram values"][-1]] * sum(
        data_ordered > weights_feature["Splitting points"][-1]
    )
    if technique == "mid_point_weighted":
        mid_points_weighted += [mid_points[-1]] * sum(
            data_ordered > weights_feature["Splitting points"][-1]
        )
        return np.array(mid_points_weighted), np.array(data_values)
    elif technique == "mean_data_weighted":
        mean_data_weighted += [mean_data[-1]] * sum(
            data_ordered > weights_feature["Splitting points"][-1]
        )
        return np.array(mean_data_weighted), np.array(data_values)

    return np.array(data_ordered), np.array(data_values)


def map_x_knots(x_knots, num_splines_range, x_first=None, x_last=None):
    """
    Map the 1d array of x_knots into a dictionary with utility and attributes as keys.

    Parameters
    ----------
    x_knots : 1d np.array
        The positions of knots in a 1d array, following this structure:
        np.array([x_att1_1, x_att1_2, ... x_att1_m, x_att2_1, ... x_attn_m]) where m is the number of knots
        and n the number of attributes that are interpolated with splines.
    num_splines_range: dict
        A dictionary of the same format than weights of features names for each utility that are interpolated with monotonic splines.
        The key is a spline interpolated feature name, and the value is the number of splines used for interpolation as an int.
        There should be a key for all features where splines are used.
    x_first : list, optional (default=None)
        A list of all first knots in the order of the attributes from spline_utilities and num_splines_range.
    x_last : list, optional (default=None)
        A list of all last knots in the order of the attributes from spline_utilities and num_splines_range.

    Returns
    -------
    x_knots_dict : dict
        A dictionary in the form of {utility: {attribute: x_knots}} where x_knots are the spline knots for the corresponding
        utility and attributes
    """
    x_knots_dict = {}
    starter = 0
    i = 0
    for u in num_splines_range:
        num_splines_sorted = sort_dict(num_splines_range[u])
        x_knots_dict[u] = {}
        for f in num_splines_sorted:
            if x_first is not None:
                x_knots_dict[u][f] = [x_first[i]]
                x_knots_dict[u][f].extend(
                    x_knots[starter : starter + num_splines_range[u][f] - 1]
                )
                x_knots_dict[u][f].append(x_last[i])
                x_knots_dict[u][f] = np.array(x_knots_dict[u][f])
                starter += num_splines_range[u][f] - 1
                i += 1
            else:
                x_knots_dict[u][f] = x_knots[
                    starter : starter + num_splines_range[u][f] + 1
                ]
                starter += num_splines_range[u][f] + 1

    return x_knots_dict


def sort_dict(dict_to_sort):
    """
    Sort a dictionary by its keys.

    Parameters
    ----------
    dict_to_sort : dict
        A dictionary to sort.

    Returns
    -------
    dict_sorted : dict
        The sorted dictionary.
    """
    dict_sorted = {}
    for k in sorted(dict_to_sort.keys()):
        dict_sorted[k] = dict_to_sort[k]

    return dict_sorted
    
def _check_rum_structure(rum_structure):
    """ Check that rum_structure, a list of dictionaries, is of the correct format. """

    if not isinstance(rum_structure, list):
        raise ValueError("rum_structure must be a list")

    for i, rum_struct in enumerate(rum_structure):
        if "utility" not in rum_struct:
            raise ValueError(
                f"rum_structure {i} must contain utility key with the list of alternatives"
            )
        if "variables" not in rum_struct:
            raise ValueError(
                f"rum_structure {i} must contain variables key with the list of variables"
            )
        if "boosting_params" not in rum_struct:
            raise ValueError(
                f"rum_structure {i} must contain boosting_params key with the boosting parameters"
            )
        if "shared" not in rum_struct:
            raise ValueError(
                f"rum_structure {i} must contain shared key with a boolean value"
            )
        if len(rum_struct["utility"]) > 1 and not rum_struct["shared"]:
            raise ValueError(
                f"rum_structure {i} must be shared if the parameter is used in more than one utility function"
            )
        if rum_struct["shared"] and len(rum_struct["utility"]) != len(rum_struct["variables"]):
            raise ValueError(
                f"rum_structure {i} must have the same number of variables and utility functions if shared is True"
            )