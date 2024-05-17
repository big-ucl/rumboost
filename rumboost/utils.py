import numpy as np
import pandas as pd
import random
import torch

from collections import Counter, defaultdict
from scipy.special import softmax
from rumboost.torch_functions import cross_entropy_torch, cross_entropy_torch_compiled

def process_parent(parent, pairs):
    '''
    Dig into the biogeme expression to retrieve name of variable and beta parameter. Work only with simple utility specification (beta * variable).
    '''
    # final expression to be stored
    if parent.getClassName() == 'Times':
        pairs.append(get_pair(parent))
    else: #if not final
        try: #dig into the expression
            left = parent.left
            right = parent.right
        except: #if no left and right children
            return pairs 
        else: #dig further left and right
            process_parent(left, pairs)
            process_parent(right, pairs)
    return pairs

def get_pair(parent):
    '''
    Return beta and variable names on a tupple from a parent expression.
    '''
    left = parent.left
    right = parent.right
    beta = None
    variable = None
    for exp in [left, right]:
        if exp.getClassName() == 'Beta':
            beta = exp.name
        elif exp.getClassName() == 'Variable':
            variable = exp.name
    if beta and variable:
        return (beta, variable)
    else:
        raise ValueError("Parent does not contain beta and variable")
    
def bio_to_rumboost(model, all_columns = False, monotonic_constraints = True, interaction_contraints = True, fct_effect_variables = []):
    '''
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

    '''
    utilities = model.loglike.util #biogeme expression
    rum_structure = []

    #for all utilities
    for k, v in utilities.items():
        rum_structure.append({'columns': [], 'monotone_constraints': [], 'interaction_constraints': [], 'betas': [], 'categorical_feature': []})
        if len(fct_effect_variables) > 0:
            rum_structure_re = {'columns': [], 'monotone_constraints': [], 'interaction_constraints': [], 'betas': [], 'categorical_feature': []}
        for i, pair in enumerate(process_parent(v, [])): # get all the pairs of the utility
            
            if pair[1] in fct_effect_variables:
                rum_structure_re['columns'].append(pair[1]) #append variable name
                rum_structure_re['betas'].append(pair[0]) #append beta name
                if interaction_contraints:
                    rum_structure_re['interaction_constraints'].append(len(rum_structure_re['interaction_constraints'])) #no interaction between features
                if monotonic_constraints:
                    bounds = model.getBoundsOnBeta(pair[0]) #get bounds on beta parameter for monotonic constraint
                    if (bounds[0] is not None) and (bounds[1] is not None):
                        raise ValueError("Only one bound can be not None")
                    if bounds[0] is not None:
                        if bounds[0] >= 0:
                            rum_structure_re['monotone_constraints'].append(1) #register positive monotonic constraint
                    elif bounds[1] is not None:
                        if bounds[1] <= 0:
                            rum_structure_re['monotone_constraints'].append(-1) #register negative monotonic constraint
                    else:
                        rum_structure_re['monotone_constraints'].append(0) #none
            
            else:
                rum_structure[-1]['columns'].append(pair[1]) #append variable name
                rum_structure[-1]['betas'].append(pair[0]) #append beta name
                if interaction_contraints:
                    if len(fct_effect_variables) > 0:
                        rum_structure[-1]['interaction_constraints'].append([len(rum_structure[-1]['interaction_constraints'])]) #no interaction between features
                    else:
                        rum_structure[-1]['interaction_constraints'].append([i]) #no interaction between features
                if monotonic_constraints:
                    bounds = model.getBoundsOnBeta(pair[0]) #get bounds on beta parameter for monotonic constraint
                    if (bounds[0] is not None) and (bounds[1] is not None):
                        raise ValueError("Only one bound can be not None")
                    if bounds[0] is not None:
                        if bounds[0] >= 0:
                            rum_structure[-1]['monotone_constraints'].append(1) #register positive monotonic constraint
                    elif bounds[1] is not None:
                        if bounds[1] <= 0:
                            rum_structure[-1]['monotone_constraints'].append(-1) #register negative monotonic constraint
                    else:
                        rum_structure[k]['monotone_constraints'].append(0) #none      
        if all_columns:
            rum_structure[-1]['columns'] = [col for col in model.database.data.drop(['choice'], axis=1).columns.values.tolist()]
        if len(fct_effect_variables) > 0:
            rum_structure.append(rum_structure_re)
        
    return rum_structure
    
def get_mid_pos(data, split_points, end='data'):
    '''
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
    '''
    #getting position in the middle of splitting points intervals
    if len(split_points) > 1:
        mid_pos = [(sp2 + sp1)/2 for sp2, sp1 in zip(split_points[:-1], split_points[1:])]
    else:
        mid_pos = []
    
    if end == 'data':
        mid_pos.insert(0, min(data)) #adding first point
        mid_pos.append(max(data)) #adding last point
    elif end == 'split point':
        mid_pos.insert(0, min(split_points)) #adding first point
        mid_pos.append(max(split_points)) #adding last point
    elif end == 'mean_data':
        mid_pos.insert(0, data[data<split_points[0]].mean()) #adding first point
        mid_pos.append(data[data>split_points[-1]].mean()) #adding last point

    return mid_pos

def get_mean_pos(data, split_points):
    '''
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
    '''
    #getting the mean of data of splitting points intervals
    mean_data = [np.mean(data[(data < s_ii) & (data > s_i)]) for s_i, s_ii in zip(split_points[:-1], split_points[1:])]
    mean_data.insert(0, np.mean(data[data<split_points[0]])) #adding first point
    mean_data.append(np.mean(data[data>split_points[-1]])) #adding last point

    return mean_data

def data_leaf_value(data, weights_feature, technique='data_weighted'):
    '''
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

    '''
    if technique == 'data_weighted':
        data_ordered = np.sort(data)
        idx = np.searchsorted(np.array(weights_feature['Splitting points']), data_ordered)
        data_values = np.array(weights_feature['Histogram values'])[idx]

        return np.array(data_ordered), data_values

    if technique == 'mid_point':
        mid_points = np.array(get_mid_pos(data, weights_feature['Splitting points']))
        return mid_points, np.array(weights_feature['Histogram values'])
    elif technique == 'mean_data':
        mean_data = np.array(get_mean_pos(data, weights_feature['Splitting points']))
        return mean_data, np.array(weights_feature['Histogram values'])

    data_ordered = data.copy().sort_values()
    data_values = [weights_feature['Histogram values'][0]]*sum(data_ordered < weights_feature['Splitting points'][0])

    if technique == 'mid_point_weighted':
        mid_points = get_mid_pos(data, weights_feature['Splitting points'])
        mid_points_weighted = [mid_points[0]]*sum(data_ordered < weights_feature['Splitting points'][0])
    elif technique == 'mean_data_weighted':
        mean_data = get_mean_pos(data, weights_feature['Splitting points'])
        mean_data_weighted = [mean_data[0]]*sum(data_ordered < weights_feature['Splitting points'][0])

    for i, (s_i, s_ii) in enumerate(zip(weights_feature['Splitting points'][:-1], weights_feature['Splitting points'][1:])):
        data_values += [weights_feature['Histogram values'][i+1]]*sum((data_ordered < s_ii) & (data_ordered > s_i))
        if technique == 'mid_point_weighted':
            mid_points_weighted += [mid_points[i+1]]*sum((data_ordered < s_ii) & (data_ordered > s_i))
        elif technique == 'mean_data_weighted':
            mean_data_weighted += [mean_data[i+1]]*sum((data_ordered < s_ii) & (data_ordered > s_i))

    data_values += [weights_feature['Histogram values'][-1]]*sum(data_ordered > weights_feature['Splitting points'][-1])
    if technique == 'mid_point_weighted':
        mid_points_weighted += [mid_points[-1]]*sum(data_ordered > weights_feature['Splitting points'][-1])
        return np.array(mid_points_weighted), np.array(data_values)
    elif technique == 'mean_data_weighted':
        mean_data_weighted += [mean_data[-1]]*sum(data_ordered > weights_feature['Splitting points'][-1])
        return np.array(mean_data_weighted), np.array(data_values)

    return np.array(data_ordered), np.array(data_values)

def utility_ranking(weights, spline_utilities):
    """
    Rank attributes utility importance by their utility range. The first rank is the attribute having the largest
    max(V(x)) - min(V(x)).

    Parameters
    ----------
    weights : dict
        A dictionary containing all the split points and leaf values for all attributes, for all utilities.
    spline_utilities : dict
        A dictionary containing attributes where splines are applied. Must be in the form ]
        {utility_indx: [attributes1, attributes2, ...], ...}.

    Returns
    -------
    util_ranks_ascend : list of tupple
        A list of tupple where the first tupple is the one having the largest utility range. Tupples are composed of 
        their utility and the name of their attributes.
    """
    util_ranks = []
    util_ranges = []
    for u in spline_utilities:
        for f in spline_utilities[u]:
            #compute range
            util_ranges.append(np.max(weights[u][f]['Histogram values']) - np.min(weights[u][f]['Histogram values']))
            util_ranks.append((u, f))

    sort_idx = np.argsort(util_ranges)
    util_ranks = np.array(util_ranks)
    util_ranks_ascend = util_ranks[np.flip(sort_idx)]
    
    return util_ranks_ascend

def map_x_knots(x_knots, num_splines_range, x_first = None, x_last = None):
    '''
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
    '''
    x_knots_dict = {}
    starter = 0
    i=0
    for u in num_splines_range:
        num_splines_sorted = sort_dict(num_splines_range[u])
        x_knots_dict[u]={}
        for f in num_splines_sorted:
            if x_first is not None:
                x_knots_dict[u][f] = [x_first[i]]
                x_knots_dict[u][f].extend(x_knots[starter:starter+num_splines_range[u][f]-1])
                x_knots_dict[u][f].append(x_last[i])
                x_knots_dict[u][f] = np.array(x_knots_dict[u][f])
                starter += num_splines_range[u][f]-1
                i +=1
            else:
                x_knots_dict[u][f] = x_knots[starter:starter+num_splines_range[u][f]+1]
                starter += num_splines_range[u][f]+1

    return x_knots_dict

def sort_dict(dict_to_sort):
    '''
    Sort a dictionary by its keys.

    Parameters
    ----------
    dict_to_sort : dict
        A dictionary to sort.

    Returns
    -------
    dict_sorted : dict
        The sorted dictionary.
    '''
    dict_sorted = {}
    for k in sorted(dict_to_sort.keys()):
        dict_sorted[k] = dict_to_sort[k]

    return dict_sorted

def compute_VoT(util_collection, u, f1, f2):
    '''
    The function compute the Value of Time of the attributes specified in attribute_VoT.

    Parameters
    ----------
    util_collection : dict
        A dictionary containing the type of utility to use for all features in all utilities.
    u : str
        The utility number, as a str (e.g. '0', '1', ...).
    f1 : str
        The time related attribtue name.
    f2 : str
        The cost related attribtue name.

    Return
    ------
    VoT : lamda function
        The function calculating value of time for attribute1 and attribute2. 
    '''

    VoT = lambda x1, x2, u1 = util_collection[u][f1], u2 = util_collection[u][f2]: u1.derivative()(x1) / u2.derivative()(x2)

    return VoT

def accuracy(preds, labels):
    """
    Compute accuracy of the model.

    Parameters
    ----------
    preds : numpy array
        Predictions for all data points and each classes from a softmax function. preds[i, j] correspond
        to the prediction of data point i to belong to class j.
    labels : numpy array
        The labels of the original dataset, as int.

    Returns
    -------
    Accuracy: float
        The computed accuracy, as a float.
    """
    return np.mean(np.argmax(preds, axis=1) == labels)

def cross_entropy(preds, labels):
    """
    Compute negative cross entropy for given predictions and data.
    
    Parameters
    ----------
    preds: numpy array
        Predictions for all data points and each classes from a softmax function. preds[i, j] correspond
        to the prediction of data point i to belong to class j.
    labels: numpy array
        The labels of the original dataset, as int.

    Returns
    -------
    Cross entropy : float
        The negative cross-entropy, as float.
    """
    num_data = len(labels)
    data_idx = np.arange(num_data)
    return - np.mean(np.log(preds[data_idx, labels]))

def nest_probs(raw_preds, mu, nests, nest_alt):
    """compute nested predictions.
    
    Parameters
    ----------

    raw_preds :
        The raw predictions from the booster
    mu :
        The list of mu values for each nest.
        The first value correspond to the first nest and so on.
    nests :
        The dictionary keys are alternatives number and their values are their nest number. 
        By example, {0:0, 1:1, 2:0} means that alt 0 and 2 are in nest 0 and alt 1 is in nest 1.
    nest_alt :
        The nest of each alternative. By example, [0, 1, 0] means that alt 0 and 2 are in nest 0 and alt 1 is in nest 1.

    Returns
    -------

    preds.T :
        The nested predictions
    pred_i_m :
        The prediction of choosing alt i knowing nest m
    pred_m :
        The prediction of choosing nest m
    """
    #initialisation
    n_obs, n_alt = raw_preds.shape
    mu_obs = mu[nest_alt]
    nests_array = np.array(list(nests.keys()))
    n_mu = nests_array.shape[0]

    mu_raw_preds_3d = np.exp(mu_obs * raw_preds)[:, :, None]
    #sum_in_nest = np.sum(mu_raw_preds_3d * mask_3d, axis=1)
    mask_3d = (nest_alt[:,None]==nests_array[None, :])[None, :, :]
    sum_in_nest = np.sum(mu_raw_preds_3d * mask_3d, axis=1)

    pred_i_m = np.exp(mu_obs * raw_preds) / np.sum(sum_in_nest[:, None, :] * mask_3d, axis=2)

    V_tilde_m = 1 / mu * np.log(sum_in_nest)

    # Pred of choosing nest m
    pred_m = softmax(V_tilde_m, axis=1)

    # Final predictions for choosing i
    preds = pred_i_m * np.sum(pred_m[:, None, :] * mask_3d, axis=2)

    return preds, pred_i_m, pred_m

def cross_nested_probs(raw_preds, mu, alphas):
    """Compute nested predictions.
    
    Parameters
    ----------
    raw_preds : numpy.ndarray
        The raw predictions from the booster
    mu : list
        The list of mu values for each nest.
        The first value corresponds to the first nest and so on.
    alphas : numpy.ndarray
        An array of J (alternatives) by M (nests).
        alpha_jn represents the degree of membership of alternative j to nest n.
        For example, alpha_12 = 0.5 means that alternative one belongs 50% to nest 2.

    Returns
    -------
    preds : numpy.ndarray
        The cross nested predictions
    pred_i_m : numpy.ndarray
        The prediction of choosing alt i knowing nest m
    pred_m : numpy.ndarray
        The prediction of choosing nest m
    """
    #scaling and exponential of raw_preds, following by degree of memberships
    raw_preds_mu_alpha_3d = (alphas** mu)[None,:,:] * np.exp(mu[None, None, :] * raw_preds[:, :, None])
    #storing sum of utilities in nests
    sum_in_nest = np.sum(raw_preds_mu_alpha_3d, axis=1) ** (1/mu)[None, :]

    #pred of choosing i knowing m.
    pred_i_m = raw_preds_mu_alpha_3d / np.sum(raw_preds_mu_alpha_3d, axis=1, keepdims=True)

    #pred of choosing m
    pred_m = sum_in_nest / np.sum(sum_in_nest, axis=1, keepdims=True)

    #final predictions for choosing i
    preds = np.sum(pred_i_m * pred_m[:, None, :], axis=2)

    return preds, pred_i_m, pred_m

def optimise_mu_or_alpha(params_to_optimise, labels, rumb, optimise_mu, optimise_alpha, alpha_shape, data_idx):
    """
    Optimize mu or alpha values for a given dataset.

    Parameters
    ----------
    params_to_optimise : list
        The list of mu or alpha values to optimize.
    labels : numpy.ndarray, optional (default=None)
        The labels of the original dataset, as int.
    rumb : RUMBoost, optional (default=None)
        A trained RUMBoost object.

    Returns
    -------
    loss : int
        The loss according to the optimization of mu or alpha values.
    """
    if rumb.device is not None:
        if optimise_mu:
            rumb.mu = torch.from_numpy(params_to_optimise[:rumb.mu.shape[0]]).to(rumb.device)
            if optimise_alpha:
                rumb.alpha = torch.from_numpy(params_to_optimise[rumb.mu.shape[0]]).view(alpha_shape).to(rumb.device)
                rumb.alpha = rumb.alpha / rumb.alpha.sum(dim=1, keepdim=True)
        elif optimise_alpha:
            rumb.alpha = torch.from_numpy(params_to_optimise).view(alpha_shape).to(rumb.device)
            rumb.alpha = rumb.alpha / rumb.alpha.sum(dim=1, keepdim=True)
    else:
        if optimise_mu:
            rumb.mu = params_to_optimise[:rumb.mu.shape[0]]
            if optimise_alpha:
                rumb.alpha = params_to_optimise[rumb.mu.shape[0]].reshape(alpha_shape)
                rumb.alpha = rumb.alpha / rumb.alpha.sum(axis=1, keepdims=True)
        elif optimise_alpha:
            rumb.alpha = params_to_optimise.reshape(alpha_shape)
            rumb.alpha = rumb.alpha / rumb.alpha.sum(axis=1, keepdims=True)

    new_preds, _, _ = rumb._inner_predict(data_idx)
    if rumb.device is not None:
        if rumb.torch_compile:
            loss = cross_entropy_torch_compiled(new_preds, labels).cpu().numpy()
        else:
            loss = cross_entropy_torch(new_preds, labels).cpu().numpy()
    else:
        loss = cross_entropy(new_preds, labels)

    return loss

def create_name(features):
    """Create new feature names from a list of feature names"""
    new_name = features[0]
    for f_name in features[1:]:
        new_name += '-'+f_name
    return new_name

def get_child(model, weights, weights_2d, weights_market, tree, split_points, features, feature_names, i, market_segm, direction = None):
    """Dig into the tree to get splitting points, features, left and right leaves values"""
    min_r = 0
    max_r = 10000

    if feature_names[tree['split_feature']] not in features:
        features.append(feature_names[tree['split_feature']])

    split_points.append(tree['threshold'])

    if 'leaf_value' in tree['left_child'] and 'leaf_value' in tree['right_child']:
        if direction is None:
            weights.append([feature_names[tree['split_feature']], tree['threshold'], tree['left_child']['leaf_value'], tree['right_child']['leaf_value'], i])
        elif direction == 'left':
            if len(features) == 1:
                weights.append([feature_names[tree['split_feature']], tree['threshold'], tree['left_child']['leaf_value'], tree['right_child']['leaf_value'], i])
                weights.append([feature_names[tree['split_feature']], split_points[0], 0, -tree['right_child']['leaf_value'], i])
            elif market_segm:
                feature_name = create_name(features)
                if features[0] in model.rum_structure[i]['categorical_feature']:
                    weights_market.append([features[-1]+'-0', tree['threshold'], tree['left_child']['leaf_value'], tree['right_child']['leaf_value'], i])
                else:
                    weights_market.append([features[0]+'-0', split_points[0], tree['left_child']['leaf_value'], 0, i])
                    weights_market.append([features[0]+'-1', split_points[0], tree['right_child']['leaf_value'], 0, i])
            else:
                feature_name = create_name(features)
                weights_2d.append([feature_name, (min_r, split_points[0]), (min_r, tree['threshold']), tree['left_child']['leaf_value'], i])
                weights_2d.append([feature_name, (min_r, split_points[0]), (tree['threshold'], max_r), tree['right_child']['leaf_value'], i])
                if len(features) > 1:
                    features.pop(-1)
                    split_points.pop(-1)
        elif direction == 'right':
            if len(features) == 1:
                weights.append([feature_names[tree['split_feature']], tree['threshold'], tree['left_child']['leaf_value'], tree['right_child']['leaf_value'], i])
                weights.append([feature_names[tree['split_feature']], split_points[0], -tree['left_child']['leaf_value'], 0, i])
            elif market_segm:
                feature_name = create_name(features)
                if features[0] in model.rum_structure[i]['categorical_feature']:
                    weights_market.append([features[-1]+'-1', tree['threshold'], tree['left_child']['leaf_value'], tree['right_child']['leaf_value'], i])
                else:
                    weights_market.append([features[0]+'-0', split_points[0], 0, tree['left_child']['leaf_value'], i])
                    weights_market.append([features[0]+'-1', split_points[0], 0, tree['right_child']['leaf_value'], i])
            else:
                feature_name = create_name(features)
                weights_2d.append([feature_name, (split_points[0], max_r), (min_r, tree['threshold']), tree['left_child']['leaf_value'], i])
                weights_2d.append([feature_name, (split_points[0], max_r), (tree['threshold'], max_r), tree['right_child']['leaf_value'], i])
    elif 'leaf_value' in tree['left_child']:
        weights.append([feature_names[tree['split_feature']], tree['threshold'], tree['left_child']['leaf_value'], 0, i])
        get_child(model, weights, weights_2d, weights_market, tree['right_child'], split_points, features, feature_names, i, market_segm, direction='right')
    elif 'leaf_value' in tree['right_child']:
        weights.append([feature_names[tree['split_feature']], tree['threshold'], 0, tree['right_child']['leaf_value'], i])
        get_child(model, weights, weights_2d, weights_market, tree['left_child'], split_points, features, feature_names, i, market_segm, direction='left')
    else:
        get_child(model, weights, weights_2d, weights_market, tree['left_child'], split_points, features, feature_names, i, market_segm, direction='left')
        get_child(model, weights, weights_2d, weights_market, tree['right_child'], split_points, features, feature_names, i, market_segm, direction='right') 

def get_weights(model):
    """
    Get leaf values from a RUMBoost model.

    Parameters
    ----------
    model : RUMBoost
        A trained RUMBoost object.

    Returns
    -------
    weights_df : pandas DataFrame
        DataFrame containing all split points and their corresponding left and right leaves value, 
        for all features.
    weights_2d_df : pandas DataFrame
        Dataframe with weights arranged for a 2d plot, used in the case of 2d feature interaction.
    weights_market : pandas DataFrame
        Dataframe with weights arranged for market segmentation, used in the case of market segmentation.
    
    """
    #using self object or a given model
    model_json = model.dump_model()

    weights = []
    weights_2d = []
    weights_market = []

    for i, b in enumerate(model_json):
        feature_names = b['feature_names']
        for trees in b['tree_info']:
            features = []
            split_points = []
            market_segm = False

            get_child(model, weights, weights_2d, weights_market, trees['tree_structure'], split_points, features, feature_names, i, market_segm)

    weights_df = pd.DataFrame(weights, columns= ['Feature', 'Split point', 'Left leaf value', 'Right leaf value', 'Utility'])
    weights_2d_df = pd.DataFrame(weights_2d, columns=['Feature', 'higher_lvl_range', 'lower_lvl_range', 'area_value', 'Utility'])
    weights_market_df = pd.DataFrame(weights_market, columns= ['Feature', 'Cat value', 'Split point', 'Left leaf value', 'Right leaf value', 'Utility'])
    return weights_df, weights_2d_df, weights_market_df

def weights_to_plot_v2(model, market_segm=False):
    """
    Arrange weights by ascending splitting points and cumulative sum of weights.

    Parameters
    ----------
    model : RUMBoost
        A trained RUMBoost object.

    Returns
    -------
    weights_for_plot : dict
        Dictionary containing splitting points and corresponding cumulative weights value for all features.

    """

    #get raw weights
    if market_segm:
        _, _, weights= get_weights(model)
    else:
        weights, _, _ = get_weights(model)

    weights_for_plot = {}
    #for all features
    for i in weights.Utility.unique():
        weights_for_plot[str(i)] = {}
        
        for f in weights[weights.Utility == i].Feature.unique():
            
            split_points = []
            function_value = [0]

            #getting values related to the corresponding utility
            weights_util = weights[weights.Utility == i]
            
            #sort by ascending order
            feature_data = weights_util[weights_util.Feature == f]
            ordered_data = feature_data.sort_values(by = ['Split point'], ignore_index = True)
            for j, s in enumerate(ordered_data['Split point']):
                #new split point
                if s not in split_points:
                    split_points.append(s)
                    #add a new right leaf value to the current right side value
                    function_value.append(function_value[-1] + float(ordered_data.loc[j, 'Right leaf value']))
                    #add left leaf value to all other current left leaf values
                    function_value[:-1] = [h + float(ordered_data.loc[j, 'Left leaf value']) for h in function_value[:-1]]
                else:
                    #add right leaf value to the current right side value
                    function_value[-1] += float(ordered_data.loc[j, 'Right leaf value'])
                    #add left leaf value to all other current left leaf values
                    function_value[:-1] = [h + float(ordered_data.loc[j, 'Left leaf value']) for h in function_value[:-1]]
                    
            weights_for_plot[str(i)][f] = {'Splitting points': split_points,
                                           'Histogram values': function_value}
                
    return weights_for_plot

def non_lin_function(weights_ordered, x_min, x_max, num_points):
    """
    Create the nonlinear function for parameters, from weights ordered by ascending splitting points.

    Parameters
    ----------
    weights_ordered : dict
        Dictionary containing splitting points and corresponding cumulative weights value for a specific 
        feature's parameter.
    x_min : float, int
        Minimum x value for which the nonlinear function is computed.
    x_max : float, int
        Maximum x value for which the nonlinear function is computed.
    num_points : int
        Number of points used to draw the nonlinear function line.

    Returns
    -------
    x_values : list
        X values for which the function will be plotted.
    nonlin_function : list
        Values of the function at the corresponding x points.
    """
    #create x points
    x_values = np.linspace(x_min, x_max, num_points)
    nonlin_function = []
    i = 0
    max_i = len(weights_ordered['Splitting points']) #all splitting points

    #handling no split points
    if max_i == 0:
        return x_values, float(weights_ordered['Histogram values'][i])*x_values

    for x in x_values:
        #compute the value of the function at x according to the weights value in between splitting points
        if x < float(weights_ordered['Splitting points'][i]):
            nonlin_function += [float(weights_ordered['Histogram values'][i])]
        else:
            nonlin_function += [float(weights_ordered['Histogram values'][i+1])]
            #go to next splitting points
            if i < max_i-1:
                i+=1
    
    return x_values, nonlin_function

def get_asc(weights, alt_to_normalise = 'Driving', alternatives = {'Walking':'0', 'Cycling':'1', 'Public Transport':'2', 'Driving':'3'}):
    '''Retrieve ASCs from a dictionary of all values from a dictionary of leaves values per alternative per feature'''
    ASCs = []
    for k, alt in alternatives.items():
        asc_temp = 0
        for feat in weights[alt]:
            asc_temp += weights[alt][feat]['Histogram values'][0]
        ASCs.append(asc_temp)

    return [a - ASCs[int(alternatives[alt_to_normalise])] for a in ASCs]

def function_2d(weights_2d, x_vect, y_vect):
    """
    Create the nonlinear contour plot for parameters, from weights gathered in getweights_v2

    Parameters
    ----------
    weights_2d : dict
        Pandas DataFrame containing all possible rectangles with their corresponding area values, for the given feature and utility.
    x_vect : numpy array
        Vector of higher level feature.
    y_vect : numpy array
        Vector of lower level feature.

    Returns
    -------
    contour_plot_values : numpy array
        Array with values at (x,y) points.
    """
    contour_plot_values = np.zeros(shape=(len(x_vect), len(y_vect)))

    for k in range(len(weights_2d.index)):
        if (weights_2d['lower_lvl_range'].iloc[k][1] == 10000) and (weights_2d['higher_lvl_range'].iloc[k][1] == 10000):
            i_x = np.searchsorted(x_vect, weights_2d['higher_lvl_range'].iloc[k][0])
            i_y = np.searchsorted(y_vect, weights_2d['lower_lvl_range'].iloc[k][0])

            contour_plot_values[i_x:, i_y:] += weights_2d['area_value'].iloc[k]

        elif (weights_2d['lower_lvl_range'].iloc[k][1] == 10000):
            i_x = np.searchsorted(x_vect, weights_2d['higher_lvl_range'].iloc[k][1])
            i_y = np.searchsorted(y_vect, weights_2d['lower_lvl_range'].iloc[k][0])

            contour_plot_values[:i_x, i_y:] += weights_2d['area_value'].iloc[k]

        elif (weights_2d['higher_lvl_range'].iloc[k][1] == 10000):
            i_x = np.searchsorted(x_vect, weights_2d['higher_lvl_range'].iloc[k][0])
            i_y = np.searchsorted(y_vect, weights_2d['lower_lvl_range'].iloc[k][1])
            
            contour_plot_values[i_x:, :i_y] += weights_2d['area_value'].iloc[k]

        else:
            i_x = np.searchsorted(x_vect, weights_2d['higher_lvl_range'].iloc[k][1])
            i_y = np.searchsorted(y_vect, weights_2d['lower_lvl_range'].iloc[k][1])
            
            contour_plot_values[:i_x, :i_y] += weights_2d['area_value'].iloc[k]

    return contour_plot_values

# Sample a dataset grouped by `groups` and stratified by `y`
# Source: https://www.kaggle.com/jakubwasikowski/stratified-group-k-fold-cross-validation
def stratified_group_k_fold(X, y, groups, k, seed=None):
    labels_num = np.max(y) + 1
    y_counts_per_group = defaultdict(lambda: np.zeros(labels_num))
    y_distr = Counter()
    for label, g in zip(y, groups):
        y_counts_per_group[g][label] += 1
        y_distr[label] += 1

    y_counts_per_fold = defaultdict(lambda: np.zeros(labels_num))
    groups_per_fold = defaultdict(set)

    def eval_y_counts_per_fold(y_counts, fold):
        y_counts_per_fold[fold] += y_counts
        std_per_label = []
        for label in range(labels_num):
            label_std = np.std([y_counts_per_fold[i][label] / y_distr[label] for i in range(k)])
            std_per_label.append(label_std)
        y_counts_per_fold[fold] -= y_counts
        return np.mean(std_per_label)
    
    groups_and_y_counts = list(y_counts_per_group.items())
    random.Random(seed).shuffle(groups_and_y_counts)

    for g, y_counts in sorted(groups_and_y_counts, key=lambda x: -np.std(x[1])):
        best_fold = None
        min_eval = None
        for i in range(k):
            fold_eval = eval_y_counts_per_fold(y_counts, i)
            if min_eval is None or fold_eval < min_eval:
                min_eval = fold_eval
                best_fold = i
        y_counts_per_fold[best_fold] += y_counts
        groups_per_fold[best_fold].add(g)

    all_groups = set(groups)
    for i in range(k):
        train_groups = all_groups - groups_per_fold[i]
        test_groups = groups_per_fold[i]

        train_indices = [i for i, g in enumerate(groups) if g in train_groups]
        test_indices = [i for i, g in enumerate(groups) if g in test_groups]

        yield train_indices, test_indices
