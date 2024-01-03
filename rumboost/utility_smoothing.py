import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.optimize import curve_fit, minimize
from scipy.interpolate import interp1d, PchipInterpolator
from scipy.special import softmax
from copy import deepcopy
from lightgbm import Dataset

from rumboost.utils import data_leaf_value, get_grad, get_mid_pos, cross_entropy, utility_ranking, nest_probs, map_x_knots

def fit_func(data, weight, technique='weighted_data'):
    '''
    Fit a function that minimises the least-squares.

    Parameter
    ---------
    data : pandas Series
        The pandas Series containing data about the feature that is being fitted.
    weight : dict
        The dictionary containing weights ordered for the feature being fitted.
    technique : str, optional (default = 'weighted_data)
        The technique used to approximate the stair utility in data_leaf_values.

    Returns
    -------
    func_fitted : dict
        A dictionary with the name of the fitted function as key and its parameters as value.
    fit_score : int
        The corresponding sum of least squares of the fit.
    '''

    data_ordered, data_values = data_leaf_value(data, weight, technique)

    best_fit = np.inf
    func_fitted = {}
    fit_score = {}

    func_for_fit = func_wrapper()

    for n, f in func_for_fit.items():
        try:
            param_opt, _, info, _, _ = curve_fit(f, data_ordered, data_values, full_output=True)
        except:
            continue
        func_fitted[n] = param_opt
        fit_score[n] = np.sum(info['fvec']**2)
        if np.sum(info['fvec']**2) < best_fit:
            best_fit = np.sum(info['fvec']**2)
            best_func = n
        print('Fitting residuals for the function {} is: {}'.format(n, np.sum(info['fvec']**2)))

    print('Best fit ({}) for function {}'.format(best_fit, best_func))

    return func_fitted, fit_score

def find_feat_best_fit(model, data, technique='weighted_data'):
    '''
    Find the best fit among several functions according to the least-squares for all features.

    Parameter
    ---------
    model : RUMBoost
        A RUMBoost object.
    data : pandas DataFrame
        The pandas DataFrame used for training.
    technique : str, optional (default = 'weighted_data)
        The technique used to approximate the stair utility in data_leaf_values.

    Returns
    -------
    best_fit : dict
        A dictionary used to store the best fitting functions for all utilities and all features.
        For each utility and feature, the dictionary contains three keys:

            best_func : the name of the best function
            best_params : the parameters associated with the best function
            best_score : the sum of the least squares score

    '''
    
    weights = model.weights_to_plot_v2()
    best_fit = {}
    for u in weights:
        best_fit[u] = {}
        for f in weights[u]:
            if model.rum_structure[int(u)]['columns'].index(f) in model.rum_structure[int(u)]['categorical_feature']:
                continue #add a function that is basic tree
            func_fitted, fit_score = fit_func(data[f], weights[u][f], technique=technique)
            best_fit[u][f] = {'best_func': min(fit_score, key = fit_score.get), 'best_params': func_fitted[min(fit_score, key = fit_score.get)], 'best_score': min(fit_score)}

    return best_fit

def monotone_spline(x_spline, weights, num_splines=5, x_knots=None, y_knots=None):
    '''
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
    '''

    #create knots if None
    if x_knots is None:
        x_knots = np.linspace(np.min(x_spline), np.max(x_spline), num_splines+1)
        x_knots, y_knots = data_leaf_value(x_knots, weights)

    #sort knots in ascending order
    is_sorted = lambda a: np.all(a[:-1] <= a[1:])
    if not is_sorted(x_knots):
        x_knots = np.sort(x_knots)

    #create spline object
    pchip = PchipInterpolator(x_knots, y_knots, extrapolate=True)

    #compute utility values
    y_spline = pchip(x_spline)

    return x_spline, y_spline, pchip, x_knots, y_knots

def mean_monotone_spline(x_data, x_mean, y_data, y_mean, num_splines=15):
    '''
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
    '''
    #case where there are more splines than mean data points
    if num_splines + 1 >= len(x_mean):
        x_knots = x_mean
        y_knots = y_mean

        #adding first and last point for extrapolation
        if x_knots[0] != x_data[0]:
            x_knots = np.insert(x_knots,0,x_data[0])
            y_knots = np.insert(y_knots,0,y_data[0])

        if x_knots[-1] != x_data[-1]:
            x_knots = np.append(x_knots,x_data[-1])
            y_knots = np.append(y_knots,y_data[-1])

        #create interpolator
        pchip = PchipInterpolator(x_knots, y_knots, extrapolate=True)

        #for plot
        x_spline = np.linspace(0, np.max(x_data)*1.05, 10000)
        y_spline = pchip(x_spline)

        return x_spline, y_spline, pchip, x_knots, y_knots

    #candidate for knots
    x_candidates = np.linspace(np.min(x_mean)+1e-10, np.max(x_mean)+1e-10, num_splines+1)

    #find closest mean point
    idx = np.unique(np.searchsorted(x_mean, x_candidates, side='left') - 1)

    x_knots = x_mean[idx]
    y_knots = y_mean[idx]

    #adding first and last point for extrapolation
    if x_knots[0] != x_data[0]:
        x_knots = np.insert(x_knots,0,x_data[0])
        y_knots = np.insert(y_knots,0,y_data[0])

    if x_knots[-1] != x_data[-1]:
        x_knots = np.append(x_knots,x_data[-1])
        y_knots = np.append(y_knots,y_data[-1])

    #create interpolator
    pchip = PchipInterpolator(x_knots, y_knots, extrapolate=True)

    #for plot
    x_spline = np.linspace(0, np.max(x_data)*1.05, 10000)
    y_spline = pchip(x_spline)

    return x_spline, y_spline, pchip, x_knots, y_knots

def updated_utility_collection(weights, data, num_splines_feat, spline_utilities, mean_splines=False, x_knots = None):
    '''
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

    Returns
    -------
    util_collection : dict
        A dictionary containing the type of utility to use for all features in all utilities.
    '''
    #initialise utility collection
    util_collection = {}

    #for all utilities and features that have leaf values
    for u in weights:
        util_collection[u] = {}
        for f in weights[u]:
            #data points and their utilities
            x_dat, y_dat = data_leaf_value(data[f], weights[u][f])

            #if using splines
            if f in spline_utilities[u]:
                #if mean technique
                if mean_splines:
                    x_mean, y_mean = data_leaf_value(data[f], weights[u][f], technique='mean_data')
                    _, _, func, _, _ = mean_monotone_spline(x_dat, x_mean, y_dat, y_mean, num_splines=num_splines_feat[u][f])
                #else, i.e. linearly sampled points
                else:       
                    x_spline = np.linspace(np.min(data[f]), np.max(data[f]), num=10000)
                    x_knots_temp, y_knots = data_leaf_value(x_knots[u][f], weights[u][f])
                    _, _, func, _, _ = monotone_spline(x_spline, weights, num_splines=num_splines_feat[u][f], x_knots=x_knots_temp, y_knots=y_knots)
            #stairs functions
            else:
                func = interp1d(x_dat, y_dat, kind='previous', bounds_error=False, fill_value=(y_dat[0],y_dat[-1]))

            #save the utility function
            util_collection[u][f] = func
                
    return util_collection

def smooth_predict(data_test, util_collection, utilities=False, mu=None, nests=None, fe_model = None, target='choice'):
    '''
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
    fe_model : RUMBoost, optional (default=None)
        The socio-economic characteristics part of the functional effect model.
        
    Returns
    -------
    preds : numpy array
        A numpy array containing the predictions for each class for each observation. Predictions are computed through the softmax function,
        unless the raw utilities are requested. A prediction for class j for observation n will be U[n, j].
    '''
    raw_preds = np.array(np.zeros((data_test.shape[0], len(util_collection))))
    for u in util_collection:
        for f in util_collection[u]:
            raw_preds[:, int(u)] += util_collection[u][f](data_test[f])

    #adding the socio-economic constant
    if fe_model is not None:
        raw_preds += fe_model.predict(Dataset(data_test, label=data_test[target], free_raw_data=False), utilities = True)

    #softmax
    if mu is not None:
        preds, _, _ = nest_probs(raw_preds, mu, nests)
        return preds
    
    if not utilities:
        preds = softmax(raw_preds, axis=1)
        return preds

    return raw_preds

def optimise_splines(x_knots, weights, data_train, data_test, label_test, spline_utilities, num_spline_range, x_first = None, x_last = None, deg_freedom = None, mu=None, nests=None, fe_model=None):
    '''
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
    fe_model : RUMBoost, optional (default=None)
        The socio-economic characteristics part of the functional effect model.

    Returns
    -------
    loss: float
        The final cross entropy or BIC on the test set.
    '''
    x_knots_dict = map_x_knots(x_knots, num_spline_range, x_first, x_last)

    #compute new CE
    utility_collection = updated_utility_collection(weights, data_train, num_splines_feat=num_spline_range, spline_utilities=spline_utilities, x_knots=x_knots_dict)
    smooth_preds_final = smooth_predict(data_test, utility_collection, mu=mu, nests=nests, fe_model=fe_model)
    loss = cross_entropy(smooth_preds_final, label_test)
    #BIC
    if deg_freedom is not None:
        N = len(label_test)
        loss = 2 * N * loss + np.log(N) * deg_freedom
    return loss

def optimal_knots_position(weights, dataset_train, dataset_test, labels_test, spline_utilities, num_spline_range, max_iter = 100, 
                           optimize = True, deg_freedom = None, n_iter=1, x_first=None, x_last=None, mu=None, nests=None, fe_model=None):
    '''
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
    mu : list, optional (default=None)
        Only used, and required, if nests is True. It is the list of mu values for each nest.
        The first value correspond to the first nest and so on.
    nests : dict, optional (default=False)
        If not none, compute predictions with the nested probability function. The dictionary keys are alternatives number and their values are
        their nest number. By example {0:0, 1:1, 2:0} means that alt 0 and 2 are in nest 0 and alt 1 is in nest 1.
    fe_model : RUMBoost, optional (default=None)
        The socio-economic characteristics part of the functional effect model.

    Returns
    -------
    x_opt : OptimizeResult
        The result of scipy.minimize.
    '''
    
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
                #get first and last split points
                first_split_point = weights[u][f]['Splitting points'][0]
                last_split_point = weights[u][f]['Splitting points'][-1]

                #get first and last data points
                first_point = np.min(dataset_train[f])
                last_point = np.max(dataset_train[f])

                #initial knots position q/Nth quantile
                x_0.extend([np.quantile(dataset_train[f].unique(), q/(num_spline_range[u][f])) for q in range(1,num_spline_range[u][f])])
                
                #knots must be greater than the previous one
                cons = [{'type':'ineq', 'fun': lambda x, i_plus = starter + j+1, i_minus = starter + j: x[i_plus] - x[i_minus] - 1e-6, 'keep_feasible':True} for j in range(0,num_spline_range[u][f]-2)]
                #last knots must be greater than last split point
                cons.append({'type':'ineq', 'fun': lambda x, i_knot = starter + num_spline_range[u][f] - 2, lsp = last_split_point: x[i_knot] - lsp - 1e-6, 'keep_feasible':True})
                #knots must be within the range of data points
                bounds = [(first_point + q*1e-7, last_point - q*1e-7) for q in range(1, num_spline_range[u][f])]
                
                #store all constraints and first and last points
                all_cons.extend(cons)
                all_bounds.extend(bounds)
                x_first.append(first_point)
                x_last.append(last_point)

                #count the number of knots until now
                starter += num_spline_range[u][f]-1

        if deg_freedom is not None:
            deg_freedom = starter

        if optimize:
            #optimise knot positions
            x_opt = minimize(optimise_splines, np.array(x_0), args = (weights, dataset_train, dataset_test, labels_test, spline_utilities, num_spline_range, x_first, x_last, deg_freedom, mu, nests, fe_model), bounds=all_bounds, constraints=all_cons, method='SLSQP', options={'maxiter':max_iter, 'disp':True})

            #compute final negative cross-entropy with optimised knots
            ce_final = optimise_splines(x_opt.x, weights, dataset_train, dataset_test, labels_test, spline_utilities, num_spline_range, x_first, x_last, deg_freedom, mu=mu, nests=nests, fe_model=fe_model)
            
            #store best value
            if ce_final < ce:
                ce = ce_final
                x_opt_best = x_opt
                x_first_best = x_first
                x_last_best = x_last
            print(f'{n+1}/{n_iter}:{ce_final} with knots at: {x_opt.x}')
        else:
            #without optimisation
            final_loss = optimise_splines(np.array(x_0), weights, dataset_train, dataset_test, labels_test, spline_utilities, num_spline_range, x_first, x_last, deg_freedom, mu=mu, nests=nests, fe_model=fe_model)

            if final_loss < ce:
                ce = final_loss
                x_opt_best = x_0
            print(f'{n+1}/{n_iter}:{final_loss}')

    #return best x_opt and first and last points + score
    if optimize:
        return x_opt_best, x_first_best, x_last_best, ce

    return x_opt_best, x_first, x_last, ce

def find_best_num_splines(weights, data_train, data_test, label_test, spline_utilities, mean_splines=False, search_technique='greedy'):
    '''
    DEPRECATED
    Find the best number of splines fro each features prespecified.

    Parameters
    ----------
    weights : dict
        A dictionary containing all leaf values for all utilities and all features.
    data_train : pandas DataFrame
        The pandas DataFrame used for training.
    data_test : pandas DataFrame
        The pandas DataFrame used for testing.
    label_test : pandas Series or numpy array
        The labels of the dataset used for testing.
    spline_utilities : dict[list[str]]
        A dictionary of lists. The dictionary should contain the index of alternatives as a str (i.e., '0', '1', ...).
        The list contains features where splines will be applied.
    mean_splines : bool, optional (default = False)
        If True, the splines are computed at the mean distribution of data for stairs.
    search_technique : str, optional (default = 'greedy')
        The technique used to search for the best number of splines. It can be 'greedy' (i.e., optimise one feature after each other, while storing the feature value),
        'greedy_ranked' (i.e., same as 'greedy' but starts with the feature with the largest utility range) or 'feature_independant'.

    Returns
    -------
    best_splines : dict
        A dictionary containing the optimal number of splines for each feature interpolated of each utility
    ce : int
        The negative cross-entropy on the test set
    '''
    #initialisation
    spline_range = np.arange(3, 50)
    num_splines = {}
    best_splines = {}
    ce=1000

    #'greedy_ranked' search
    if search_technique == 'greedy_ranked':
        util_ranked = utility_ranking(weights, spline_utilities)
        for rank in util_ranked:
            if num_splines.get(rank[0], None) is None:
                num_splines[rank[0]] = {}
                best_splines[rank[0]] = {}
            for s in spline_range:
                num_splines[rank[0]][rank[1]] = s
                #compute new utility collection 
                utility_collection = updated_utility_collection(weights, data_train, num_splines_feat=num_splines, mean_splines=mean_splines)

                #get new predictions
                smooth_preds = smooth_predict(data_test, utility_collection)

                #compute new CE
                ce_knot = cross_entropy(smooth_preds, label_test)
                
                #store best one
                if ce_knot < ce:
                    ce = ce_knot
                    best_splines[rank[0]][rank[1]] = s
                
                print("CE = {} at iteration {} for feature {} ---- best CE = {} with best knots: {}".format(ce_knot, s-2, rank[1], ce, best_splines))

            #keep best values for next features
            num_splines = deepcopy(best_splines)
        
        return best_splines, ce
    #'greedy' search
    elif search_technique == 'greedy':
        for u in spline_utilities:
            best_splines[u] = {}
            num_splines[u] = {}
            for f in spline_utilities[u]:
                for s in spline_range:
                    num_splines[u][f] = s
                    #compute new utility collection 
                    utility_collection = updated_utility_collection(weights, data_train, num_splines_feat=num_splines, mean_splines=mean_splines)

                    #get new predictions
                    smooth_preds = smooth_predict(data_test, utility_collection)

                    #compute new CE
                    ce_knot = cross_entropy(smooth_preds, label_test)
                    
                    #store best one
                    if ce_knot < ce:
                        ce = ce_knot
                        best_splines[u][f] = s
                    
                    print("CE = {} at iteration {} for feature {} ---- best CE = {} with best knots: {}".format(ce_knot, s-2, f, ce, best_splines))
                
                #keep best values for next features
                num_splines = deepcopy(best_splines)
            
        return best_splines, ce
    
    #'feature_independant' search
    elif search_technique == 'feature_independant':
        for u in spline_utilities:
            best_splines[u] = {}
            for f in spline_utilities[u]:
                ce = 1000
                for s in spline_range:
                    temp_num_splines = {u:{f:s}}
                    #compute new utility collection
                    utility_collection = updated_utility_collection(weights, data_train, num_splines_feat=temp_num_splines, mean_splines=mean_splines)

                    #get new predictions
                    smooth_preds = smooth_predict(data_test, utility_collection)

                    #compute new CE
                    ce_knot = cross_entropy(smooth_preds, label_test)

                    #store best one
                    if ce_knot < ce:
                        ce = ce_knot
                        best_splines[u][f] = s
                    
                    print("CE = {} at iteration {} for feature {} ---- best CE = {} with best knots: {}".format(ce_knot, s-2, f, ce, best_splines))

        #computation of final cross entropy
        utility_collection_final = updated_utility_collection(weights, data_train, num_splines_feat=best_splines, mean_splines=mean_splines)
        smooth_preds_final = smooth_predict(data_test, utility_collection_final)
        ce_final = cross_entropy(smooth_preds_final, label_test)

        return best_splines, ce_final
    
    else:
        raise ValueError('search_technique must be greedy, greedy_ranked, or feature_independant.')

def stairs_to_pw(model, train_data, data_to_transform = None, util_for_plot = False):
    '''
    DEPRECATED
    Transform a stair output to a piecewise linear prediction.

    Parameters
    ----------
    model : RUMBoost
        A trained RUMBoost object.
    train_data : pandas DataFrame
        The full dataset used for training.
    data_to_transform : pandas DataFrame, optional (default = None)
        The data that need to be transform for prediction. If None, the training dataset is used.
    util_for_plot : bool, optional (default = False)
        If True, the output is formatted for plotting.

    Returns
    -------
    pw_utility : numpy array or list
        The piece-wise output. It is usually a numpy array, but can be a list if util_for_plot is True.
    '''
    if type(train_data) is list:
        new_train_data = train_data[0].get_data()
        for data in train_data[1:]:
            new_train_data = new_train_data.join(data.get_data(), lsuffix='DROP').filter(regex="^(?!.*DROP)")

        train_data = new_train_data

    if data_to_transform is None:
        data_to_transform = train_data
    weights = model.weights_to_plot()
    pw_utility = []
    for u in weights:
        if util_for_plot:
            pw_util = []
        else:
            pw_util = np.zeros(data_to_transform.iloc[:, 0].shape)

        for f in weights[u]:
            leaf_values = weights[u][f]['Histogram values']
            split_points =  weights[u][f]['Splitting points']
            transf_data_arr = np.array(data_to_transform[f])

            if len(split_points) < 1:
                break
            
            if model.rum_structure[int(u)]['columns'].index(f) not in model.rum_structure[int(u)]['categorical_feature']:

                mid_pos = get_mid_pos(model.train_set[int(u)].get_data()[f],split_points)
                
                slope = get_grad(mid_pos, leaf_values, technique='slope')

                conds = [(mp1 <= transf_data_arr) & (transf_data_arr < mp2) for mp1, mp2 in zip(mid_pos[:-1], mid_pos[1:])]
                conds.insert(0, transf_data_arr<mid_pos[0])
                conds.append(transf_data_arr >= mid_pos[-1])

                values = [lambda x, j=j: leaf_values[j] + slope[j+1] * (x - mid_pos[j]) for j in range(0, len(leaf_values))]
                values.insert(0, leaf_values[0])
            else:
                conds = [(sp1 <= transf_data_arr) & (transf_data_arr < sp2) for sp1, sp2 in zip(split_points[:-1], split_points[1:])]
                conds.insert(0, transf_data_arr < split_points[0])
                conds.append(transf_data_arr >= split_points[-1])
                values = leaf_values
            
            if util_for_plot:
                pw_util.append(np.piecewise(transf_data_arr, conds, values))
            else:
                pw_util = np.add(pw_util, np.piecewise(transf_data_arr, conds, values))
        pw_utility.append(pw_util)

    return pw_utility