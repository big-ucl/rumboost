import numpy as np
import pandas as pd

import random
from collections import Counter, defaultdict

# from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
# from scipy.special import softmax
# from fit_functions import func_wrapper, logistic
# import matplotlib.pyplot as plt
# import seaborn as sns

def process_parent(parent, pairs):
    '''
    dig into the biogeme expression to retrieve name of variable and beta parameter. Work only with simple utility specification (beta * variable)
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
    return beta and variable names on a tupple from a parent expression
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
    
def bio_to_rumboost(model, all_columns = False, monotonic_constraints = True, interaction_contraints = True, max_depth=1):
    '''
    Converts a biogeme model to a rumboost dict
    '''
    utilities = model.loglike.util #biogeme expression
    rum_structure = []

    #for all utilities
    for k, v in utilities.items():
        rum_structure.append({'columns': [], 'monotone_constraints': [], 'interaction_constraints': [], 'betas': [], 'categorical_feature': []})
        interac_2d = []
        for i, pair in enumerate(process_parent(v, [])): # get all the pairs of the utility
            rum_structure[-1]['columns'].append(pair[1]) #append variable name
            rum_structure[-1]['betas'].append(pair[0]) #append beta name
            if interaction_contraints:
                if (max_depth > 1) and (('weekend'in pair[0])|('dur_driving' in pair[0])|('dur_walking' in pair[0])|('dur_cycling' in pair[0])|('dur_pt_rail' in pair[0])): #('distance' in pair[0])): |('dur_pt_bus' in pair[0]))
                    interac_2d.append(i) #in the case of interaction constraint, append only the relevant continous features to be interacted
                else:             
                    rum_structure[-1]['interaction_constraints'].append([i]) #no interaction between features
            if ('fueltype' in pair[0]) | ('female' in pair[0]) | ('purpose' in pair[0]) | ('license' in pair[0]) | ('week' in pair[0]):
                rum_structure[-1]['categorical_feature'].append(i) #register categorical features
            bounds = model.getBoundsOnBeta(pair[0]) #get bounds on beta parameter for monotonic constraint
            if monotonic_constraints:
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
            rum_structure[-1]['columns'] = [col for col in model.database.data.drop(['household_id', 'choice', 'travel_month','driving_traffic_percent'], axis=1).columns.values.tolist()]
        if max_depth > 1:
            rum_structure[-1]['interaction_constraints'].append(interac_2d)
        return rum_structure
    
def get_mid_pos(data, split_points, end='data'):
    '''
    Return the mid point in-between two split points for a specific feature (used in pw linear predict)

    Parameters
    ----------
    data: pandas.Series
        The column of the dataframe associated with the feature
    split_points: list
        The list of split points for that feature
    end: str
        How to compute the mid position of the first and last point, it can be:
            -'data': add min and max values of data
            -'split point': add first and last split points
            -'mean_data': add the mean of data before the first split point, and after the last split point

    Returns
    -------

    mid_pos: list
        A list of points in the middle of every consecutive split points
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
    data: pandas.Series
        The column of the dataframe associated with the feature
    split_points: list
        The list of split points for that feature

    Returns
    -------

    mid_pos: list
        A list of points in the middle of every consecutive split points
    '''
    #getting the mean of data of splitting points intervals
    mean_data = [np.mean(data[(data < s_ii) & (data > s_i)]) for s_i, s_ii in zip(split_points[:-1], split_points[1:])]
    mean_data.insert(0, np.mean(data[data<split_points[0]])) #adding first point
    mean_data.append(np.mean(data[data>split_points[-1]])) #adding last point

    return mean_data

def data_leaf_value(data, weights_feature, technique='weighted_data'):

    if technique == 'mid_point':
        mid_points = np.array(get_mid_pos(data, weights_feature['Splitting points']))
        return mid_points, weights_feature['Histogram values']
    elif technique == 'mean_data':
        mean_data = np.array(get_mean_pos(data, weights_feature['Splitting points']))
        return mean_data, weights_feature['Histogram values']  

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
        return np.array(mid_points_weighted), data_values
    elif technique == 'mean_data_weighted':
        mean_data_weighted += [mean_data[-1]]*sum(data_ordered > weights_feature['Splitting points'][-1])
        return np.array(mean_data_weighted), data_values

    return data_ordered, data_values

def get_grad(x, y, technique='slope', sample_points=30, normalise = False):

    if len(y) <= 1:
        return 0
    
    x_values = x
    y_values = y

    if normalise:
        x_values = (x - np.min(x))/(np.max(x) - np.min(x))
        y_values = (y - np.min(y))/(np.max(y) - np.min(y))

    if technique == 'slope'  :
        grad = [(y_values[i+1]-y_values[i])/(x_values[i+1]-x_values[i]) for i in range(0, len(x_values)-1)]
        #grad.insert(0, 0) #adding first slope
        grad.append(0) #adding last slope
    elif technique == 'sample_data':
        x_sample = np.linspace(np.min(x_values), np.max(x_values), sample_points)
        f = interp1d(x_values, y_values, kind='previous')
        y_sample = f(x_sample)
        grad = [(y_sample[i+1]-y_sample[i])/(x_sample[i+1]-x_sample[i]) for i in range(0, len(x_sample)-1)]
        #grad.insert(0, 0) #adding first slope
        grad.append(0) #adding last slope

        if normalise:
            x_sample = x_sample*(np.max(x) - np.min(x)) + np.min(x)
            y_sample = y_sample*(np.max(y) - np.min(y)) + np.min(y)

        return grad, x_sample, y_sample

    return grad

def get_angle_diff(x_values, y_values):

    slope = get_grad(x_values, y_values, normalise = True)
    angle = np.arctan(slope)
    diff_angle = [np.pi - np.abs(angle[0])]
    diff_angle += [np.pi - np.abs(a_1-a_0) for (a_1, a_0) in zip(angle[1:], angle[:-1])]

    return diff_angle

def find_disc(x_values, grad):
    
    diff_angle = get_angle_diff(x_values, grad)

    is_disc = [True if (angle < 0.2) and (np.abs(g) > 5) else False for angle, g in zip(diff_angle, grad)]

    disc = x_values[is_disc]
    disc_idx = np.nonzero(is_disc)[0]
    num_disc = np.sum(is_disc)

    return disc, disc_idx, num_disc

def accuracy(preds, labels):
    """
    Compute accuracy of the model

    Parameters
    ----------
    preds: ndarray
        Predictions for all data points and each classes from a softmax function. preds[i, j] correspond
        to the prediction of data point i to belong to class j
    labels: ndarray
        The labels of the original dataset, as int

    Returns
    -------
    Accuracy: float
    """
    return np.mean(np.argmax(preds, axis=1) == labels)

def cross_entropy(preds, labels):
    """
    Compute negative cross entropy for given predictions and data
    
    Parameters
    ----------
    preds: ndarray
        Predictions for all data points and each classes from a softmax function. preds[i, j] correspond
        to the prediction of data point i to belong to class j
    labels: ndarray
        The labels of the original dataset, as int

    Returns
    -------
    Cross entropy : float
    """
    num_data = len(labels)
    data_idx = np.arange(num_data)

    return - np.mean(np.log(preds[data_idx, labels]))

# def get_weights(model):
#     """
#     get leaf values from a RUMBoost model

#     Parameters
#     ----------
#     model: lightGBM model

#     Returns
#     -------
#     weights_df: DataFrame
#         DataFrame containing all split points and their corresponding left and right leaves value, 
#         for all features
#     """
#     #using self object or a given model
#     model_json = [model.dump_model()]

#     weights = []

#     for i, b in enumerate(model_json):
#         feature_names = b['feature_names']
#         for trees in b['tree_info']:
#             feature = feature_names[trees['tree_structure']['split_feature']]
#             split_point = trees['tree_structure']['threshold']
#             left_leaf_value = trees['tree_structure']['left_child']['leaf_value']
#             right_leaf_value = trees['tree_structure']['right_child']['leaf_value']
#             weights.append([feature, split_point, left_leaf_value, right_leaf_value, i])

#     weights_df = pd.DataFrame(weights, columns= ['Feature', 'Split point', 'Left leaf value', 'Right leaf value', 'Utility'])
#     return weights_df

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
    get leaf values from a RUMBoost model

    Parameters
    ----------
    model: RUMBoost

    Returns
    -------
    weights_df: pandas.DataFrame
        DataFrame containing all split points and their corresponding left and right leaves value, 
        for all features
    weights_2d_df: pandas.DataFrame
        Dataframe with weights arranged for a 2d plot, used in the case of 2d feature interaction.
    weights_market: pandas.DataFrame
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
    Arrange weights by ascending splitting points and cumulative sum of weights

    Parameters
    ----------
    model: lightGBM model

    Returns
    -------
    weights_for_plot: dict
        Dictionary containing splitting points and corresponding cumulative weights value for all features
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

# def weights_to_plot(self, model = None):
#     """
#     Arrange weights by ascending splitting points and cumulative sum of weights

#     Parameters
#     ----------
#     model: lightGBM model

#     Returns
#     -------
#     weights_for_plot: dict
#         Dictionary containing splitting points and corresponding cumulative weights value for all features
#     """

#     #get raw weights
#     if model is None:
#         weights = self.getweights()
#     else:
#         weights = self.getweights(model=model)

#     weights_for_plot = {}
#     weights_for_plot_double = {}
#     #for all features
#     for i in weights.Utility.unique():
#         weights_for_plot[str(i)] = {}
#         weights_for_plot_double[str(i)] = {}
        
#         for f in weights[weights.Utility == i].Feature.unique():
            
#             split_points = []
#             function_value = [0]

#             #getting values related to the corresponding utility
#             weights_util = weights[weights.Utility == i]
            
#             #sort by ascending order
#             feature_data = weights_util[weights_util.Feature == f]
#             ordered_data = feature_data.sort_values(by = ['Split point'], ignore_index = True)
#             for j, s in enumerate(ordered_data['Split point']):
#                 #new split point
#                 if s not in split_points:
#                     split_points.append(s)
#                     #add a new right leaf value to the current right side value
#                     function_value.append(function_value[-1] + float(ordered_data.loc[j, 'Right leaf value']))
#                     #add left leaf value to all other current left leaf values
#                     function_value[:-1] = [h + float(ordered_data.loc[j, 'Left leaf value']) for h in function_value[:-1]]
#                 else:
#                     #add right leaf value to the current right side value
#                     function_value[-1] += float(ordered_data.loc[j, 'Right leaf value'])
#                     #add left leaf value to all other current left leaf values
#                     function_value[:-1] = [h + float(ordered_data.loc[j, 'Left leaf value']) for h in function_value[:-1]]
                    
#             weights_for_plot[str(i)][f] = {'Splitting points': split_points,
#                                             'Histogram values': function_value}
                
#     return weights_for_plot

def non_lin_function(weights_ordered, x_min, x_max, num_points):
    """
    Create the nonlinear function for parameters, from weights ordered by ascending splitting points

    Parameters
    ----------
    weights_ordered : dict
        Dictionary containing splitting points and corresponding cumulative weights value for a specific 
        feature's parameter
    x_min : float, int
        Minimum x value for which the nonlinear function is computed
    x_max : float, int
        Maximum x value for which the nonlinear function is computed
    num_points: int
        Number of points used to draw the nonlinear function line

    Returns
    -------
    x_values: list
        X values for which the function will be plotted
    nonlin_function: list
        Values of the function at the corresponding x points
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
        Pandas DataFrame containing all possible rectangles with their corresponding area values, for the given feature and utility
    x_vect : np.linspace
        Vector of higher level feature
    y_vect : np.linspace
        Vector of lower level feature

    Returns
    -------
    contour_plot_values: np.darray
        Array with values at (x,y) points
    """
    contour_plot_values = np.zeros(shape=(len(x_vect), len(y_vect)))

    for k in range(len(weights_2d.index)):
        y_check = True
        for i in range(len(x_vect)):
            if (x_vect[i] >= weights_2d['higher_lvl_range'].iloc[k][1]) | (not y_check):
                break

            if x_vect[i] >= weights_2d['higher_lvl_range'].iloc[k][0]:
                for j in range(len(y_vect)):
                    if y_vect[j] >= weights_2d['lower_lvl_range'].iloc[k][1]:
                        y_check = False
                        break
                    if y_vect[j] >= weights_2d['lower_lvl_range'].iloc[k][0]:
                        if (weights_2d['lower_lvl_range'].iloc[k][1] == 10000) and (weights_2d['higher_lvl_range'].iloc[k][1] == 10000):
                            contour_plot_values[i:, j:] += weights_2d['area_value'].iloc[k]
                            y_check = False
                            break
                        else:
                            contour_plot_values[i, j] += weights_2d['area_value'].iloc[k]

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
