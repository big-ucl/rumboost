import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils import function_2d, get_weights, weights_to_plot_v2, get_asc, non_lin_function
from basic_functions import func_wrapper
from utility_smoothing import stairs_to_pw, find_feat_best_fit, fit_func, fit_sev_functions

def plot_2d(model, feature1: str, feature2: str, max1: int, max2: int, save_figure: bool = False, utility_names: list[str] = ['Walking', 'Cycling', 'Public Transport', 'Driving']):
    '''
    Plot a 2nd order feature interaction as a contour plot.

    Parameters
    ----------
    model: RUMBoost
    feature1: str
        Name of feature 1.
    feature2: str
        Name of feature 2.
    max1: int
        Maximum value of feature 1.
    max2: int
        Maximum value of feature 2.
    save_figure: bool, optional (defalut = False)
        If true, save the figure as a png file
    utility_names: list[str]
        List of the alternative names
    '''
    #get the good weights
    _, weights_2d, _ = get_weights(model)

    #prepare name and vectors
    name1 = feature1 + "-" + feature2
    name2 = feature2 + "-" + feature1
    x_vect = np.linspace(0, max1, 200)
    y_vect = np.linspace(0, max2, 200)

    sns.set_theme()

    for u in weights_2d.Utility.unique():
        #compute and aggregate contour plot values
        weights_2d_util = weights_2d[weights_2d.Utility==u]
        contour_plot1 = function_2d(weights_2d_util[weights_2d_util.Feature==name1], x_vect, y_vect)
        contour_plot2 = function_2d(weights_2d_util[weights_2d_util.Feature==name2], y_vect, x_vect)
        contour_plot = contour_plot1 + contour_plot2.T

        #draw figure
        X, Y = np.meshgrid(x_vect, y_vect)
        fig, axes = plt.subplots(figsize=(10,8), layout='constrained')

        fig.suptitle('Impact of {} and {} on the utility function'.format(feature1, feature2))

        res = 100

        c_plot = axes.contourf(X, Y, contour_plot, levels=res, linewidths=0, cmap=sns.cubehelix_palette(light=1, as_cmap=True, reverse=True))

        axes.set_title('{}'.format(utility_names[int(u)]))
        axes.set_xlabel('{}'.format(feature1))
        axes.set_ylabel('{}'.format(feature2))

        cbar = fig.colorbar(c_plot, ax = axes)
        cbar.ax.set_ylabel('Utility value')

        if save_figure:
            plt.savefig('Figures/rumboost_2d_{}_{}_{}_res{}.png'.format(feature1,feature2,utility_names[int(u)],res))

        plt.show()

def plot_parameters(model, X, utility_names, Betas = None,  model_unconstrained = None, 
                    with_pw = False, save_figure=False, asc_normalised = False, with_asc = False, 
                    with_cat = True, only_tt = False, only_1d = False, with_fit = False, 
                    fit_all = True, technique = 'weighted_data', data_sep = False):
    """
    Plot the non linear impact of parameters on the utility function. When specified, unconstrained parameters
    and parameters from a RUM model can be added to the plot.

    Parameters
    ----------
    model : RUMBoost
    X : pandas dataframe
        Features used to train the model, in a pandas dataframe.
    utility_name : dict
        Dictionary mapping utilities indices to their names.
    Betas : list, optional (default = None)
        List of beta parameters value from a RUM. They should be listed in the same order as 
        in the RUMBoost model.
    model_unconstrained : LightGBM model, optional (default = None)
        The unconstrained model. Must be trained and compatible with dump_model().
    with_pw : bool, optional (default = False)
        If the piece-wise function should be included in the graph.
    save_figure : bool, optional (default = False)
        If True, save the plot as a png file.
    asc_normalised : bool, optional (default = False)
        If True, scale down utilities to be zero at the y axis.
    with_asc : bool, optional (default = False)
        If True, add the ASCs to all graphs (one is normalised, and asc_normalised must be True).
    with_cat : bool, optional (default = True)
        If False, categorical features are not plotted.
    only_tt : bool, optional (default = False)
        If True, plot only travel time and distance.
    only_1d : bool, optional (default = False)
        If True, plot only the features separately.
    with_fit : bool, optional (default = False)
        If True, fit the data with simple functions to approximate the step functions.
    fit_all : bool, optional (default = True)
        If False, plot only the best fitting function.
    technique : str, optional (default = 'weighted_data')
        The technique for data sampling in the function fitting.
    data_sep : bool, optional (default = False)
        If True, split the data to fit subsets of data
    """

    #get and prepare weights
    weights_arranged = weights_to_plot_v2(model)

    #load the functions to fit
    if with_fit | data_sep:
        func_for_fit = func_wrapper()

    #piece-wise linear plot
    if with_pw:
        pw_func = plot_util_pw(model, X)
    
    #unconstrained model plot
    if model_unconstrained is not None:
        weights_arranged_unc = weights_to_plot_v2(model=model_unconstrained)

    #compute ASCs
    if with_asc:
        ASCs = get_asc(weights_arranged)

    #find the best fit
    if (not fit_all) and (with_fit):
        best_funcs = find_feat_best_fit(model, X, technique = technique, func_for_fit=func_for_fit)

    sns.set_theme()

    if not only_1d:
        #plot for travel time on one figure
        plt.figure(figsize=(10, 10), dpi=300)
        x_w, non_lin_func_walk = non_lin_function(weights_arranged['0']['dur_walking'], 0, 1.05*max(X['dur_driving']), 10000)
        if asc_normalised:
            non_lin_func_walk = [n - non_lin_func_walk[0] for n in non_lin_func_walk]
        if with_asc:
            non_lin_func_walk = [n + ASCs[0] for n in non_lin_func_walk]

        x_c, non_lin_func_cycle = non_lin_function(weights_arranged['1']['dur_cycling'], 0, 1.05*max(X['dur_driving']), 10000)
        if asc_normalised:
            non_lin_func_cycle = [n - non_lin_func_cycle[0] for n in non_lin_func_cycle]
        if with_asc:
            non_lin_func_cycle = [n + ASCs[1] for n in non_lin_func_cycle]

        x_ptb, non_lin_func_pt_bus = non_lin_function(weights_arranged['2']['dur_pt_bus'], 0, 1.05*max(X['dur_driving']), 10000)
        if asc_normalised:
            non_lin_func_pt_bus = [n - non_lin_func_pt_bus[0] for n in non_lin_func_pt_bus]
        if with_asc:
            non_lin_func_pt_bus = [n + ASCs[2] for n in non_lin_func_pt_bus]

        x_ptr, non_lin_func_pt_rail = non_lin_function(weights_arranged['2']['dur_pt_rail'], 0, 1.05*max(X['dur_driving']), 10000)
        if asc_normalised:
            non_lin_func_pt_rail = [n - non_lin_func_pt_rail[0] for n in non_lin_func_pt_rail]
        if with_asc:
            non_lin_func_pt_rail = [n + ASCs[2] for n in non_lin_func_pt_rail]

        x_d, non_lin_func_driving = non_lin_function(weights_arranged['3']['dur_driving'], 0, 1.05*max(X['dur_driving']), 10000)
        if asc_normalised:
            non_lin_func_driving = [n - non_lin_func_driving[0] for n in non_lin_func_driving]
        if with_asc:
            non_lin_func_driving = [n + ASCs[3] for n in non_lin_func_driving]

        sns.lineplot(x=x_w, y=non_lin_func_walk, lw=2, color='#DB7E8F', label='Walking')
        sns.lineplot(x=x_c, y=non_lin_func_cycle, lw=2, color='#ECC692', label='Cycling')
        sns.lineplot(x=x_ptb, y=non_lin_func_pt_bus, lw=2, color='#8ED7DF', label='PT Bus')
        sns.lineplot(x=x_ptr, y=non_lin_func_pt_rail, lw=2, color='#8EB8DF', label='PT Rail')
        sns.lineplot(x=x_d, y=non_lin_func_driving, lw=2, color='#A3EC97', label='Driving')


        #plt.title('Influence of alternative travel time on the utility function', fontdict={'fontsize':  16})
        plt.xlabel('time [h]')
        plt.ylabel('Utility')

        if save_figure:
            plt.savefig('Figures/rumbooster_travel_time_WoTitle_asc_{}_norm_{}.png'.format(with_asc, asc_normalised))
        
        #plot for distance on one figure
        plt.figure(figsize=(10, 10), dpi=300)
        x_w, non_lin_func_walk = non_lin_function(weights_arranged['0']['distance'], 0, 1.05*max(X['distance']), 10000)
        if asc_normalised:
            non_lin_func_walk = [n - non_lin_func_walk[0] for n in non_lin_func_walk]
        if with_asc:
            non_lin_func_walk = [n + ASCs[0] for n in non_lin_func_walk]

        x_c, non_lin_func_cycle = non_lin_function(weights_arranged['1']['distance'], 0, 1.05*max(X['distance']), 10000)
        if asc_normalised:
            non_lin_func_cycle = [n - non_lin_func_cycle[0] for n in non_lin_func_cycle]
        if with_asc:
            non_lin_func_cycle = [n + ASCs[1] for n in non_lin_func_cycle]

        x_pt, non_lin_func_pt = non_lin_function(weights_arranged['2']['distance'], 0, 1.05*max(X['distance']), 10000)
        if asc_normalised:
            non_lin_func_pt = [n - non_lin_func_pt[0] for n in non_lin_func_pt]
        if with_asc:
            non_lin_func_pt = [n + ASCs[2] for n in non_lin_func_pt]

        x_d, non_lin_func_driving = non_lin_function(weights_arranged['3']['distance'], 0, 1.05*max(X['distance']), 10000)
        if asc_normalised:
            non_lin_func_driving = [n - non_lin_func_driving[0] for n in non_lin_func_driving]
        if with_asc:
            non_lin_func_driving = [n + ASCs[3] for n in non_lin_func_driving]

        sns.lineplot(x=x_w, y=non_lin_func_walk, lw=2, color='#DB7E8F', label='Walking')
        sns.lineplot(x=x_c, y=non_lin_func_cycle, lw=2, color='#ECC692', label='Cycling')
        sns.lineplot(x=x_pt, y=non_lin_func_pt, lw=2, color='#8ED7DF', label='PT')
        sns.lineplot(x=x_d, y=non_lin_func_driving, lw=2, color='#A3EC97', label='Driving')


        #plt.title('Influence of straight line distance on the utility function', fontdict={'fontsize':  16})
        plt.xlabel('distance [km]')
        plt.ylabel('Utility')

        
        if save_figure:
            plt.savefig('Figures/rumbooster_distance_WoTitle_asc_{}_norm_{}.png'.format(with_asc, asc_normalised))

        plt.show()

    #for all features parameters
    if not only_tt:
        for u in weights_arranged:
            for i, f in enumerate(weights_arranged[u]):
                
                #find if the feature is categorical or not
                if not with_cat:
                    is_cat = model.rum_structure[int(u)]['columns'].index(f) in model.rum_structure[int(u)]['categorical_feature']
                    if is_cat:
                        continue
                else:
                    is_cat = False

                #create nonlinear plot
                x, non_lin_func = non_lin_function(weights_arranged[u][f], 0, 1.05*max(X[f]), 10000)
                
                #normalised without ascs
                if asc_normalised:
                    val_0 = non_lin_func[0]
                    non_lin_func = [n - val_0 for n in non_lin_func]

                #add full ASCs
                if with_asc:
                    non_lin_func = [n + ASCs[int(u)] for n in non_lin_func]
                
                #plot parameters
                plt.figure(figsize=(10, 6))
                plt.title('Influence of {} on the predictive function ({} utility)'.format(f, utility_names[u]), fontdict={'fontsize':  16})
                plt.ylabel('{} utility'.format(utility_names[u]))

                                    
                if 'dur' in f:
                    plt.xlabel('{} [h]'.format(f))
                elif 'time' in f:
                    plt.xlabel('{} [h]'.format(f))
                elif 'cost' in f:
                    plt.xlabel('{} [£]'.format(f))
                elif 'distance' in f:
                    plt.xlabel('{} [m]'.format(f))
                else:
                    plt.xlabel('{}'.format(f))

                #plot non categorical features
                if not is_cat:
                    sns.lineplot(x=x, y=non_lin_func, lw=2, color='k', label='RUMBoost')

                #plot smoothed functions
                if (with_fit) and (not is_cat):
                    if fit_all:
                        opt_params, _ = fit_func(X[f], weights_arranged[u][f], technique = technique, func_for_fit=func_for_fit)
                        for func, p in opt_params.items():
                            y_smooth = func_for_fit[func](x, *p)
                            if asc_normalised:
                                y_smooth += -val_0
                            if with_asc:
                                y_smooth += ASCs[int(u)]
                            sns.lineplot(x=x, y=y_smooth, lw=2, label=func)
                    else:
                        y_smooth = func_for_fit[best_funcs[u][f]['best_func']](x, *best_funcs[u][f]['best_params'])
                        if asc_normalised:
                            y_smooth += -val_0
                        if with_asc:
                            y_smooth += ASCs[int(u)]
                        sns.lineplot(x=x, y=y_smooth, lw=2, label=best_funcs[u][f]['best_func'])

                #plot smoothed functions on subset of features
                if (not is_cat) and (data_sep):
                    funcs, x_range = fit_sev_functions(model, X[f], weights_arranged[u][f])
                    for i, func in enumerate(funcs):
                        y_smooth = func_for_fit[list(func)[0]](x_range[i], *func[list(func)[0]])
                        if asc_normalised:
                            y_smooth += -val_0
                        if with_asc:
                            y_smooth += ASCs[int(u)]
                        sns.lineplot(x=x_range[i], y=y_smooth, lw=2, label=list(func)[0])

                #plot unconstrained model parameters
                if model_unconstrained is not None:
                    _, non_lin_func_unc = non_lin_function(weights_arranged_unc[u][f], 0, 1.05*max(X[f]), 10000)
                    sns.lineplot(x=x, y=non_lin_func_unc, lw=2)

                #plot piece-wise linear
                if (with_pw) & (model.rum_structure[int(u)]['columns'].index(f)  not in model.rum_structure[int(u)]['categorical_feature']):
                    sns.lineplot(x=x, y=pw_func[int(u)][i], lw=2)
                
                #plot RUM parameters
                if Betas is not None:
                    sns.lineplot(x=x, y=Betas[i]*x)
                
                plt.xlim([0-0.05*np.max(X[f]), np.max(X[f])*1.05])
                plt.ylim([np.min(non_lin_func) - 0.05*(np.max(non_lin_func)-np.min(non_lin_func)), np.max(non_lin_func) + 0.05*(np.max(non_lin_func)-np.min(non_lin_func))])

                #legend
                # if Betas is not None:
                #     if model_unconstrained is not None:
                #         if withPointDist:
                #             plt.legend(labels = ['With GBM constrained', 'With GBM unconstrained', 'With RUM', 'Data'])
                #         else:
                #             plt.legend(labels = ['With GBM constrained', 'With GBM unconstrained', 'With RUM'])
                #     else:
                #         if withPointDist:
                #             plt.legend(labels = ['With GBM constrained', 'With RUM', 'Data'])
                #         else:
                #             plt.legend(labels = ['With GBM constrained', 'With RUM'])
                # else:
                #     if model_unconstrained is not None:
                #         if withPointDist:
                #             plt.legend(labels = ['With GBM constrained', 'With GBM unconstrained', 'Data'])
                #         else:
                #             plt.legend(labels = ['With GBM constrained', 'With GBM unconstrained'])
                #     else:
                #         if withPointDist:
                #             plt.legend(labels = ['With GBM constrained', 'Data'])
                #         elif with_pw:
                #             plt.legend(labels = ['With GBM constrained', 'With piece-wise linear function'])
                #         else:
                #             plt.legend(labels = ['RUMBooster'])
                    
                if save_figure:
                    if with_fit:
                        plt.savefig('Figures/rumbooster_vfinal_lr3e-1 {} utility, {} feature {} technique.png'.format(utility_names[u], f, technique))
                    else:
                        plt.savefig('Figures/rumbooster_vfinal_lr3e-1 {} utility, {} feature.png'.format(utility_names[u], f))

                plt.show()

def plot_market_segm(model, X, asc_normalised: bool = True, utility_names: list[str] = ['Walking', 'Cycling', 'Public Transport', 'Driving']):
    '''
    Plot the market segmentation.

    Parameters
    ----------
    model : RUMBoost
        The RUMBoost object used for market segmentation
    X : pandas DataFrame
    asc_normalised : bool, optional (default = False)
        If True, scale down utilities to be zero at the y axis.
    utility_names : list[str], optional (default = ['Walking', 'Cycling', 'Public Transport', 'Driving'])
    '''
    
    sns.set_theme()

    weights_arranged = weights_to_plot_v2(model, market_segm=True)
    label = {0:'Weekdays',1:'Weekends'}
    color = ['r', 'b']

    for u in weights_arranged:
        plt.figure(figsize=(10, 6))

        for i, f in enumerate(weights_arranged[u]):

            #create nonlinear plot
            x, non_lin_func = non_lin_function(weights_arranged[u][f], 0, 1.05*max(X[f]), 10000)

            if asc_normalised:
                val_0 = non_lin_func[0]
                non_lin_func = [n - val_0 for n in non_lin_func]
            
            sns.lineplot(x=x, y=non_lin_func, lw=2, color=color[i], label=label[i])

        plt.title('Impact of travel time in weekdays and weekends on {} utility'.format(utility_names[u]), fontdict={'fontsize':  16})
        plt.ylabel('{} utility'.format(utility_names[u]))
        plt.xlabel('Travel time [h]')
        plt.show()
#               plt.savefig('Figures/rumbooster_vfinal_lr3e-1 {} utility, {} feature.png'.format(utility_names[u], f))          

def plot_util(model, data_train, points=10000):
    '''
    plot the raw utility functions of all features
    '''
    sns.set_theme()
    for j, struct in enumerate(model.rum_structure):
        booster = model.boosters[j]
        for i, f in enumerate(struct['columns']):
            xin = np.zeros(shape = (points, len(struct['columns'])))
            xin[:, i] = np.linspace(0,1.05*max(data_train[f]),points)
            
            ypred = booster.predict(xin)
            plt.figure()
            plt.plot(np.linspace(0,1.05*max(data_train[f]),points), ypred)
            plt.title(f)

def plot_util_pw(model, data_train, points = 10000):
    '''
    plot the piece-wise utility function
    '''
    features = data_train.columns
    data_to_transform = {}
    for f in features:
        xi = np.linspace(0, 1.05*max(data_train[f]), points)
        data_to_transform[f] = xi

    data_to_transform = pd.DataFrame(data_to_transform)

    pw_func = stairs_to_pw(model, data_train, data_to_transform, util_for_plot = True)

    return pw_func