import numpy as np
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d, PchipInterpolator
from scipy.special import softmax
from fit_functions import func_wrapper, logistic
import matplotlib.pyplot as plt
import seaborn as sns
from utils import data_leaf_value, get_grad, find_disc, get_mid_pos

def fit_func(data, weight, technique='weighted_data'):

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

def monotone_spline(x, y, num_knots=15):
    '''
    x: data from the interpolated feature
    y: V(x_value)
    '''

    # x_knots = np.array(x.iloc[::2000])
    # x_knots = np.append(x_knots, x.iloc[-1])

    x_knots = np.linspace(np.min(x), np.max(x), num_knots)

    f = interp1d(x, y, kind='previous')
    y_knots = f(x_knots)

    pchip = PchipInterpolator(x_knots, y_knots, extrapolate=True)

    x_spline = np.linspace(0, np.max(x)*1.05, 10000)
    y_spline = pchip(x_spline)

    return x_spline, y_spline, pchip

def updated_utility_collection(weights, data, spline_utilities, num_knots=40):

    util_collection = {}
    for u in weights:
        util_collection[u] = {}
        for f in weights[u]:
            x_dat, y_dat = data_leaf_value(data[f], weights[u][f], technique='data_weighted')

            if f in spline_utilities:
                _, _, func = monotone_spline(x_dat, y_dat, num_knots=num_knots)
                util_collection[u][f] = func
            else:
                func = interp1d(x_dat, y_dat, kind='previous')
                util_collection[u][f] = func

    return util_collection

def smooth_predict(data_test, util_collection, utilities=False):
    U = np.array(np.zeros((data_test.shape[0], len(util_collection))))
    for u in util_collection:
        for f in util_collection[u]:
            U[:, int(u)] += util_collection[u][f](data_test[f])

        #softmax
    if not utilities:
        U = softmax(U, axis=1)

    return U


# def all_func(x, a, b, c, d, e, f, g, h, i, j):
#     return a*x**3 + b*x**2 + c*x + d*np.exp(-e*x) + f*np.log(g*x) + h/(x + i) + j

# def penalty_neg(x, a, b, c, d, e, f, g, h, i, j):
#     return (np.sign(np.grad(all_func(x, a, b, c, d, e, f, g, h, i, j))) + 1)*1000000

# def penalty_pos(x, a, b, c, d, e, f, g, h, i, j):
#     return (np.sign(np.grad(all_func(x, a, b, c, d, e, f, g, h, i, j))) - 1)*1000000

# def all_func_neg(x, a, b, c, d, e, f, g, h, i, j):
#     return all_func(x, a, b, c, d, e, f, g, h, i, j) + penalty_neg(x, a, b, c, d, e, f, g, h, i, j)

# def all_func_pos(x, a, b, c, d, e, f, g, h, i, j):
#     return all_func(x, a, b, c, d, e, f, g, h, i, j) + penalty_pos(x, a, b, c, d, e, f, g, h, i, j)

# def all_func_1step(x, a, b, c, d, e, f, g, h, i, j, k, l, m):
#     return all_func(x, a, b, c, d, e, f, g, h, i, j) + logistic(x, k, l, m)

# def all_func_neg_1step(x, a, b, c, d, e, f, g, h, i, j, k, l, m):
#     return all_func(x, a, b, c, d, e, f, g, h, i, j) + penalty_neg(x, a, b, c, d, e, f, g, h, i, j) + logistic(x, k, l, m)

# def all_func_pos_1step(x, a, b, c, d, e, f, g, h, i, j, k, l, m):
#     return all_func(x, a, b, c, d, e, f, g, h, i, j) + penalty_pos(x, a, b, c, d, e, f, g, h, i, j) + logistic(x, k, l, m)

# def all_func_2step(x, a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p):
#     return all_func(x, a, b, c, d, e, f, g, h, i, j) + logistic(x, k, l, m) + logistic(x, n, o, p)

# def all_func_neg_2step(x, a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p):
#     return all_func(x, a, b, c, d, e, f, g, h, i, j) + penalty_neg(x, a, b, c, d, e, f, g, h, i, j) + logistic(x, k, l, m) + logistic(x, n, o, p)

# def all_func_pos_2step(x, a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p):
#     return all_func(x, a, b, c, d, e, f, g, h, i, j) + penalty_pos(x, a, b, c, d, e, f, g, h, i, j) + logistic(x, k, l, m) + logistic(x, n, o, p)

# def all_func_3step(x, a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s):
#     return all_func(x, a, b, c, d, e, f, g, h, i, j) + logistic(x, k, l, m) + logistic(x, n, o, p) + logistic(x, q, r, s)

# def all_func_neg_3step(x, a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s):
#     return all_func(x, a, b, c, d, e, f, g, h, i, j) + penalty_neg(x, a, b, c, d, e, f, g, h, i, j) + logistic(x, k, l, m) + logistic(x, n, o, p) + logistic(x, q, r, s)

# def all_func_pos_3step(x, a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s):
#     return all_func(x, a, b, c, d, e, f, g, h, i, j) + penalty_pos(x, a, b, c, d, e, f, g, h, i, j) + logistic(x, k, l, m) + logistic(x, n, o, p) + logistic(x, q, r, s)

def cub(x, a, b, c, d):
    return a*x**3 + b*x**2 + c*x + d 

def cub_step(x, a, b, c, d, e, f, g):
    return cub(x, a, b, c, d) + logistic(x, e, f, g)

def penalty_neg(x, a, b, c, d, e, f, g):
    return (np.sign(np.grad(cub_step(x, a, b, c, d, e, f, g))) + 1)*1000000

def cub_step_neg(x, a, b, c, d, e, f, g):
    return cub(x, a, b, c, d) + logistic(x, e, f, g) + penalty_neg(x, a, b, c, d, e, f, g)

def parametric_output(data, weight, monotonic_constraint):

    data_ordered, data_values = data_leaf_value(data, weight, technique = 'weighted_data')

    grad, x_sample, y_sample = get_grad(data_ordered, data_values, technique='sample_data', normalise=True)

    disc, disc_idx, num_disc = find_disc(x_sample, grad)

    try:
        popt, _, info, _, _ = curve_fit(cub_step_neg, data_ordered, data_values, full_output=True)
    except:
        return
    best_score = np.sum(info['fvec']**2)
    print('Best fit: {} with monotonic {} and {} step(s)'.format(best_score, monotonic_constraint, num_disc))
    func_used = cub_step_neg

    # if monotonic_constraint == 0:
    #     if num_disc == 0:
    #         try:
    #             popt, _, info, _, _ = curve_fit(all_func, data_ordered, data_values, p0=[1, 1, 1, 1, 1, 1, 1, 1, 1 ,1], full_output=True)
    #         except:
    #             return
    #         best_score = np.sum(info['fvec']**2)
    #         print('Best fit: {} with monotonic {} and {} step(s)'.format(best_score, monotonic_constraint, num_disc))
    #         func_used = all_func
    #     elif num_disc == 1:
    #         try:
    #             popt, _, info, _, _ = curve_fit(all_func_1step, data_ordered, data_values, p0=[1, 1, 1, 1, 1, 1, 1, 1, 1 ,1, 1, 1, 1], full_output=True)
    #         except:
    #             return
    #         best_score = np.sum(info['fvec']**2)
    #         print('Best fit: {} with monotonic {} and {} step(s)'.format(best_score, monotonic_constraint, num_disc))
    #         func_used = all_func_1step
    #     elif num_disc == 2:
    #         try:
    #             popt, _, info, _, _ = curve_fit(all_func_2step, data_ordered, data_values, p0=[1, 1, 1, 1, 1, 1, 1, 1, 1 ,1, 1, 1, 1, 1, 1, 1], full_output=True)
    #         except:
    #             return
    #         best_score = np.sum(info['fvec']**2)
    #         print('Best fit: {} with monotonic {} and {} step(s)'.format(best_score, monotonic_constraint, num_disc))
    #         func_used = all_func_2step
    #     else:
    #         try:
    #             popt, _, info, _, _ = curve_fit(all_func_3step, data_ordered, data_values, p0=[1, 1, 1, 1, 1, 1, 1, 1, 1 ,1, 1, 1, 1, 1, 1, 1, 1, 1, 1], full_output=True)
    #         except:
    #             return
    #         best_score = np.sum(info['fvec']**2)
    #         print('Best fit: {} with monotonic {} and {} step(s)'.format(best_score, monotonic_constraint, num_disc))
    #         func_used = all_func_3step
    # elif monotonic_constraint == -1:
    #     if num_disc == 0:
    #         try:
    #             popt, _, info, _, _ = curve_fit(all_func_neg, data_ordered, data_values, p0=[1, 1, 1, 1, 1, 1, 1, 1, 1 ,1], full_output=True)
    #         except:
    #             return
    #         best_score = np.sum(info['fvec']**2)
    #         print('Best fit: {} with monotonic {} and {} step(s)'.format(best_score, monotonic_constraint, num_disc))
    #         func_used = all_func_neg
    #     elif num_disc == 1:
    #         try:
    #             popt, _, info, _, _ = curve_fit(all_func_neg_1step, data_ordered, data_values, p0=[1, 1, 1, 1, 1, 1, 1, 1, 1 ,1, 1, 1, 1], full_output=True)
    #         except:
    #             return
    #         best_score = np.sum(info['fvec']**2)
    #         print('Best fit: {} with monotonic {} and {} step(s)'.format(best_score, monotonic_constraint, num_disc))
    #         func_used = all_func_neg_1step
    #     elif num_disc == 2:
    #         try:
    #             popt, _, info, _, _ = curve_fit(all_func_neg_2step, data_ordered, data_values, p0=[1, 1, 1, 1, 1, 1, 1, 1, 1 ,1, 1, 1, 1, 1, 1, 1], full_output=True)
    #         except:
    #             return
    #         best_score = np.sum(info['fvec']**2)
    #         print('Best fit: {} with monotonic {} and {} step(s)'.format(best_score, monotonic_constraint, num_disc))
    #         func_used = all_func_neg_2step
    #     else:
    #         try:
    #             popt, _, info, _, _ = curve_fit(all_func_neg_3step, data_ordered, data_values, p0=[1, 1, 1, 1, 1, 1, 1, 1, 1 ,1, 1, 1, 1, 1, 1, 1, 1, 1, 1], full_output=True)
    #         except:
    #             return
    #         best_score = np.sum(info['fvec']**2)
    #         print('Best fit: {} with monotonic {} and {} step(s)'.format(best_score, monotonic_constraint, num_disc))
    #         func_used = all_func_neg_3step
    # elif monotonic_constraint == 1:
    #     if num_disc == 0:
    #         try:
    #             popt, _, info, _, _ = curve_fit(all_func_pos, data_ordered, data_values, p0=[1, 1, 1, 1, 1, 1, 1, 1, 1 ,1], full_output=True)
    #         except:
    #             return
    #         best_score = np.sum(info['fvec']**2)
    #         print('Best fit: {} with monotonic {} and {} step(s)'.format(best_score, monotonic_constraint, num_disc))
    #         func_used = all_func_pos
    #     elif num_disc == 1:
    #         try:
    #             popt, _, info, _, _ = curve_fit(all_func_pos_1step, data_ordered, data_values, p0=[1, 1, 1, 1, 1, 1, 1, 1, 1 ,1, 1, 1, 1], full_output=True)
    #         except:
    #             return
    #         best_score = np.sum(info['fvec']**2)
    #         print('Best fit: {} with monotonic {} and {} step(s)'.format(best_score, monotonic_constraint, num_disc))
    #         func_used = all_func_pos_1step
    #     elif num_disc == 2:
    #         try:
    #             popt, _, info, _, _ = curve_fit(all_func_pos_2step, data_ordered, data_values, p0=[1, 1, 1, 1, 1, 1, 1, 1, 1 ,1, 1, 1, 1, 1, 1, 1], full_output=True)
    #         except:
    #             return
    #         best_score = np.sum(info['fvec']**2)
    #         print('Best fit: {} with monotonic {} and {} step(s)'.format(best_score, monotonic_constraint, num_disc))
    #         func_used = all_func_pos_2step
    #     else:
    #         try:
    #             popt, _, info, _, _ = curve_fit(all_func_pos_3step, data_ordered, data_values, p0=[1, 1, 1, 1, 1, 1, 1, 1, 1 ,1, 1, 1, 1, 1, 1, 1, 1, 1, 1], full_output=True)
    #         except:
    #             return
    #         best_score = np.sum(info['fvec']**2)
    #         print('Best fit: {} with monotonic {} and {} step(s)'.format(best_score, monotonic_constraint, num_disc))
    #         func_used = all_func_pos_3step

    return popt, best_score, func_used
                
def features_param_output(model, data):
    
    weights = model.rumb_model.weights_to_plot_v2()
    best_fit = {}
    for u in weights:
        best_fit[u] = {}
        for i, f in enumerate(weights[u]):
            if model.rumb_model.rum_structure[int(u)]['columns'].index(f) in model.rumb_model.rum_structure[int(u)]['categorical_feature']:
                continue 
            try:
                params, fit_score, func_used = parametric_output(data[f], weights[u][f], model.rumb_model.rum_structure[int(u)]['monotone_constraints'][model.rumb_model.rum_structure[int(u)]['columns'].index(f)])
            except:
                continue
            best_fit[u][f] = {'best_func': func_used, 'best_params': params, 'best_score': fit_score}

    return best_fit

def plot_fit(model, data):

    best_fit = features_param_output(model, data)
    weights = model.rumb_model.weights_to_plot_v2()

    sns.set_theme()

    for u in best_fit:
        for f in best_fit[u]:
            x_dat, y_dat = data_leaf_value(model.dataset_train[f], weights[u][f], technique='data_weighted')

            plt.figure(figsize=(10, 7))

            plt.scatter(x_dat, y_dat)

            plt.plot(x_dat, best_fit[u][f]['best_func'](x_dat, *best_fit[u][f]['best_params']))
            plt.show()

def data_split_to_fit(model, data, weights_feat, weight_w_data = False):
    split_points = weights_feat['Splitting points']
    leaves_values = weights_feat['Histogram values']
    mid_points = model._get_mid_pos(data, weights_feat['Splitting points'], end ='split point')

    split_range = np.max(split_points) - np.min(split_points)
    leaves_range = np.max(leaves_values) - np.min(leaves_values)

    new_func_idx = []
    start = 0

    for i, (s_1, s_2) in enumerate(zip(mid_points[:-1], mid_points[1:])):
        if (s_2-s_1) > 0.2*split_range:
            stop = i
            new_func_idx.append((start, stop))
            start = i+1
        elif np.abs(leaves_values[i] - leaves_values[i+1]) > 0.2*leaves_range:
            stop = i
            new_func_idx.append((start, stop))
            start = i+1

    return [{'Mid points':mid_points[i[0]:i[1]+1], 'Histogram values':leaves_values[i[0]:i[1]+1]} for i in new_func_idx]

def fit_sev_functions(model, data, weight, technique='mid_point'):

    func_for_fit = func_wrapper()
    
    data_list = data_split_to_fit(model, data, weight, technique)
    best_funcs = []
    x_range = []
    for d in data_list:
        best_fit = np.inf
        for n, f in func_for_fit.items():
            try:
                param_opt, _, info, _, _ = curve_fit(f, d['Mid points'], d['Histogram values'], full_output=True)
            except:
                continue
            if np.sum(info['fvec']**2) < best_fit:
                best_fit = np.sum(info['fvec']**2)
                best_func = n
                best_params = param_opt
            print('Fitting residuals for the function {} is: {}'.format(n, np.sum(info['fvec']**2)))

        if best_fit == np.inf:
            continue
        print('Best fit ({}) for function {}'.format(best_fit, best_func))
        best_funcs.append({best_func:best_params})
        x_range.append(np.linspace(np.min(d['Mid points']), np.max(d['Mid points']), 10000))

    return best_funcs, x_range

def stairs_to_pw(model, train_data, data_to_transform = None, util_for_plot = False):
    '''
    Transform a stair output to a piecewise linear prediction
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