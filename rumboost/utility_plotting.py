import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from rumboost.utils import function_2d, weights_to_plot_v2, get_asc, non_lin_function, data_leaf_value, get_weights
from rumboost.utility_smoothing import monotone_spline, mean_monotone_spline

def plot_2d(model, feature1: str, feature2: str, min1: int, max1: int, min2: int, max2: int, save_figure: bool = False, utility_names: list[str] = ['Walking', 'Cycling', 'Public Transport', 'Driving'], num_points = 1000):
    '''
    Plot a 2nd order feature interaction as a contour plot.

    Parameters
    ----------
    model : RUMBoost
        A RUMBoost object.
    feature1 : str
        Name of feature 1.
    feature2 : str
        Name of feature 2.
    min1 : int
        Minimum value of feature 1.
    max1 : int
        Maximum value of feature 1.
    min2 : int
        Minimum value of feature 2.
    max2 : int
        Maximum value of feature 2.
    save_figure : bool, optional (default = False)
        If true, save the figure as a png file
    utility_names : list[str], optional (default=['Walking', 'Cycling', 'Public Transport', 'Driving'])
        List of the alternative names
    num_points : int, optional (default=1000)
        The number of points per axis. The total number of points is num_points**2.

    '''
    _, weights_2d, _ = get_weights(model = model)
    weights_ordered = weights_to_plot_v2(model=model)

    name1 = feature1 + "-" + feature2
    name2 = feature2 + "-" + feature1

    x_vect = np.linspace(min1, max1, num_points)
    y_vect = np.linspace(min2, max2, num_points)

    #to generalise
    utility_names = ['Walking', 'Cycling', 'PT', 'Driving']
    tex_fonts = {
            # Use LaTeX to write all text
            # "text.usetex": True, 
            # "font.family": "serif",
            # "font.serif": "Computer Modern Roman",
            # Use 14pt font in plots, to match 10pt font in document
            "axes.labelsize": 7,
            "axes.linewidth":0.5,
            "axes.labelpad": 1,
            "font.size": 7,
            # Make the legend/label fonts a little smaller
            "legend.fontsize": 6,
            "legend.fancybox": False,
            "legend.edgecolor": "inherit",
            'legend.borderaxespad': 0.4,
            'legend.borderpad': 0.4,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "xtick.major.pad": 0.5,
            "ytick.major.pad": 0.5,
            "grid.linewidth": 0.5,
            "lines.linewidth": 0.8
        }
    sns.set_theme(font_scale=1, rc=tex_fonts)
    #sns.set_context(tex_fonts)
    sns.set_style("whitegrid")
    # plt.rcParams.update({
    #     # "text.usetex": True,
    #     "font.family": "serif"
    #     #"font.sans-serif": "Computer Modern Roman",
    # })

    for u in weights_2d.Utility.unique():
        weights_2d_util = weights_2d[weights_2d.Utility==u]
        contour_plot1 = function_2d(weights_2d_util[weights_2d_util.Feature==name1], x_vect, y_vect)
        contour_plot2 = function_2d(weights_2d_util[weights_2d_util.Feature==name2], y_vect, x_vect)

        contour_plot = contour_plot1 + contour_plot2.T

        if np.sum(contour_plot) == 0:
            continue

        if (feature1 in weights_ordered[str(u)].keys()) and (feature2 in weights_ordered[str(u)].keys()):
            _, feature1_alone = non_lin_function(weights_ordered[str(u)][feature1], min1, max1, num_points)
            feature1_grid = np.repeat(feature1_alone, num_points).reshape((num_points, num_points))
            contour_plot += feature1_grid

            _, feature2_alone = non_lin_function(weights_ordered[str(u)][feature2], min2, max2, num_points)
            feature2_grid = np.repeat(feature2_alone, num_points).reshape((num_points, num_points)).T
            contour_plot += feature2_grid

        contour_plot -= contour_plot.max()

        colors = ['#F5E5E2', '#DF7057', '#A31D04']
        customPalette = sns.set_palette(sns.color_palette(colors, as_cmap=True))

        if np.sum(contour_plot) != 0:
            X, Y = np.meshgrid(x_vect, y_vect)
            fig, axes = plt.subplots(figsize=(3.49,3), layout='constrained', dpi=1000)

            res = num_points

            c_plot = axes.contourf(X, Y, contour_plot.T, levels=res, linewidths=0, cmap=customPalette, vmin=-12, vmax=0)

            #axes.set_title(f'{utility_names[int(u)]}')
            axes.set_xlabel(f'{feature1} [h]')
            axes.set_ylabel(f'{feature2}')

            cbar = fig.colorbar(c_plot, ax = axes, ticks=[-10, -8, -6, -4, -2, 0])
            cbar.ax.set_ylabel('Utility')

            if save_figure:
                plt.savefig('Figures/FI RUMBoost/age_travel_time_{}.png'.format(utility_names[int(u)]))

            plt.show()

def plot_parameters(model, X, utility_names, save_figure=False, asc_normalised = False, with_asc = False, 
                    only_tt = False, only_1d = False, with_fit = False, sm_tt_cost=False, save_file=''):
    """
    Plot the non linear impact of parameters on the utility function.

    Parameters
    ----------
    model : RUMBoost
        A RUMBoost object.
    X : pandas dataframe
        Features used to train the model, in a pandas dataframe.
    utility_name : dict
        Dictionary mapping utilities indices to their names.
    save_figure : bool, optional (default = False)
        If True, save the plot as a png file.
    asc_normalised : bool, optional (default = False)
        If True, scale down utilities to be zero at the y axis.
    with_asc : bool, optional (default = False)
        If True, add the ASCs to all graphs (one is normalised, and asc_normalised must be True).
    only_tt : bool, optional (default = False)
        If True, plot only travel time and distance.
    only_1d : bool, optional (default = False)
        If True, plot only the features separately.
    sm_tt_cost : bool, optional (default = False)
        If True, plot only the swissmetro travel time and cost on the same figure.
    save_file : str, optional (default='')
        The name to save the figure with.
    """
    weights_arranged = weights_to_plot_v2(model)

    if with_asc:
        ASCs = get_asc(weights_arranged)

    tex_fonts = {
        # Use LaTeX to write all text
        # "text.usetex": True, 
        # "font.family": "serif",
        # "font.serif": "Computer Modern Roman",
        # Use 14pt font in plots, to match 10pt font in document
        "axes.labelsize": 7,
        "axes.linewidth":0.5,
        "axes.labelpad": 1,
        "font.size": 7,
        # Make the legend/label fonts a little smaller
        "legend.fontsize": 6,
        "legend.fancybox": False,
        "legend.edgecolor": "inherit",
        'legend.borderaxespad': 0.4,
        'legend.borderpad': 0.4,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "xtick.major.pad": 0.5,
        "ytick.major.pad": 0.5,
        "grid.linewidth": 0.5,
        "lines.linewidth": 0.8
    }
    sns.set_theme(font_scale=1, rc=tex_fonts)
    #sns.set_context(tex_fonts)
    sns.set_style("whitegrid")
    # plt.rcParams.update({
    #     # "text.usetex": True,
    #     "font.family": "serif"
    #     #"font.sans-serif": "Computer Modern Roman",
    # })

    if sm_tt_cost:
        #plot for travel time on one figure
        plt.figure(figsize=(3.49, 3.49), dpi=1000)
        x_w, non_lin_func_rail = non_lin_function(weights_arranged['0']['TRAIN_TT'], 0, 600, 10000)
        if asc_normalised:
            non_lin_func_rail = [n - non_lin_func_rail[0] for n in non_lin_func_rail]
        if with_asc:
            non_lin_func_rail = [n + ASCs[0] for n in non_lin_func_rail]

        x_c, non_lin_func_SM = non_lin_function(weights_arranged['1']['SM_TT'], 0, 600, 10000)
        if asc_normalised:
            non_lin_func_SM = [n - non_lin_func_SM[0] for n in non_lin_func_SM]
        if with_asc:
            non_lin_func_SM = [n + ASCs[1] for n in non_lin_func_SM]

        x_d, non_lin_func_driving = non_lin_function(weights_arranged['2']['CAR_TT'], 0, 600, 10000)
        if asc_normalised:
            non_lin_func_driving = [n - non_lin_func_driving[0] for n in non_lin_func_driving]
        if with_asc:
            non_lin_func_driving = [n + ASCs[3] for n in non_lin_func_driving]

        sns.lineplot(x=x_w/60, y=non_lin_func_rail, color='g', label='Rail')
        sns.lineplot(x=x_c/60, y=non_lin_func_SM, color='#6b8ba4', label='Swissmetro')
        sns.lineplot(x=x_d/60, y=non_lin_func_driving, color='orange', label='Driving')

        #plt.title('Influence of alternative travel time on the utility function', fontdict={'fontsize':  16})
        plt.xlabel('Travel time [h]')
        plt.ylabel('Utility')

        plt.tight_layout()

        if save_figure:
            plt.savefig('Figures/RUMBoost/SwissMetro/travel_time.png')

        #plot for travel time on one figure
        plt.figure(figsize=(3.49, 3.49), dpi=1000)
        x_w, non_lin_func_rail = non_lin_function(weights_arranged['0']['TRAIN_COST'], 0, 500, 10000)
        if asc_normalised:
            non_lin_func_rail = [n - non_lin_func_rail[0] for n in non_lin_func_rail]
        if with_asc:
            non_lin_func_rail = [n + ASCs[0] for n in non_lin_func_rail]

        x_c, non_lin_func_SM = non_lin_function(weights_arranged['1']['SM_COST'], 0, 500, 10000)
        if asc_normalised:
            non_lin_func_SM = [n - non_lin_func_SM[0] for n in non_lin_func_SM]
        if with_asc:
            non_lin_func_SM = [n + ASCs[1] for n in non_lin_func_SM]

        x_d, non_lin_func_driving = non_lin_function(weights_arranged['2']['CAR_CO'], 0, 500, 10000)
        if asc_normalised:
            non_lin_func_driving = [n - non_lin_func_driving[0] for n in non_lin_func_driving]
        if with_asc:
            non_lin_func_driving = [n + ASCs[3] for n in non_lin_func_driving]

        sns.lineplot(x=x_w, y=non_lin_func_rail, color='g', label='Rail')
        sns.lineplot(x=x_c, y=non_lin_func_SM, color='#6b8ba4', label='Swissmetro')
        sns.lineplot(x=x_d, y=non_lin_func_driving, color='orange', label='Driving')

        #plt.title('Influence of alternative cost on the utility function', fontdict={'fontsize':  16})

        plt.xlabel('Cost [chf]')
        plt.ylabel('Utility')

        plt.tight_layout()

        if save_figure:
            plt.savefig('Figures/RUMBoost/SwissMetro/cost.png')

    if not only_1d:
        #plot for travel time on one figure
        plt.figure(figsize=(3.49, 3.49), dpi=1000)
        x_w, non_lin_func_walk = non_lin_function(weights_arranged['0']['dur_walking'], 0, 2.5, 10000)
        if asc_normalised:
            non_lin_func_walk = [n - non_lin_func_walk[0] for n in non_lin_func_walk]
        if with_asc:
            non_lin_func_walk = [n + ASCs[0] for n in non_lin_func_walk]

        x_c, non_lin_func_cycle = non_lin_function(weights_arranged['1']['dur_cycling'], 0, 2.5, 10000)
        if asc_normalised:
            non_lin_func_cycle = [n - non_lin_func_cycle[0] for n in non_lin_func_cycle]
        if with_asc:
            non_lin_func_cycle = [n + ASCs[1] for n in non_lin_func_cycle]

        x_ptb, non_lin_func_pt_bus = non_lin_function(weights_arranged['2']['dur_pt_bus'], 0, 2.5, 10000)
        if asc_normalised:
            non_lin_func_pt_bus = [n - non_lin_func_pt_bus[0] for n in non_lin_func_pt_bus]
        if with_asc:
            non_lin_func_pt_bus = [n + ASCs[2] for n in non_lin_func_pt_bus]

        x_ptr, non_lin_func_pt_rail = non_lin_function(weights_arranged['2']['dur_pt_rail'], 0, 2.5, 10000)
        if asc_normalised:
            non_lin_func_pt_rail = [n - non_lin_func_pt_rail[0] for n in non_lin_func_pt_rail]
        if with_asc:
            non_lin_func_pt_rail = [n + ASCs[2] for n in non_lin_func_pt_rail]

        x_d, non_lin_func_driving = non_lin_function(weights_arranged['3']['dur_driving'], 0, 2.5, 10000)
        if asc_normalised:
            non_lin_func_driving = [n - non_lin_func_driving[0] for n in non_lin_func_driving]
        if with_asc:
            non_lin_func_driving = [n + ASCs[3] for n in non_lin_func_driving]

        sns.lineplot(x=x_w, y=non_lin_func_walk, color='b', label='Walking')
        sns.lineplot(x=x_c, y=non_lin_func_cycle, color='r', label='Cycling')
        sns.lineplot(x=x_ptb, y=non_lin_func_pt_bus, color='#02590f', label='PT Bus')
        sns.lineplot(x=x_ptr, y=non_lin_func_pt_rail, color='g', label='PT Rail')
        sns.lineplot(x=x_d, y=non_lin_func_driving, color='orange', label='Driving')


        #plt.title('Influence of alternative travel time on the utility function', fontdict={'fontsize':  16})
        plt.xlabel('Travel time [h]')
        plt.ylabel('Utility')

        plt.tight_layout()

        if save_figure:
            plt.savefig('Figures/RUMBoost/LPMC/travel_time.png')
        
        #plot for distance on one figure
        plt.figure(figsize=(3.49, 3.49), dpi=1000)

        x_pt, non_lin_func_pt = non_lin_function(weights_arranged['2']['cost_transit'], 0, 10, 10000)
        if asc_normalised:
            non_lin_func_pt = [n - non_lin_func_pt[0] for n in non_lin_func_pt]
        if with_asc:
            non_lin_func_pt = [n + ASCs[2] for n in non_lin_func_pt]

        x_d, non_lin_func_driving = non_lin_function(weights_arranged['3']['cost_driving_fuel'], 0, 10, 10000)
        if asc_normalised:
            non_lin_func_driving = [n - non_lin_func_driving[0] for n in non_lin_func_driving]
        if with_asc:
            non_lin_func_driving = [n + ASCs[3] for n in non_lin_func_driving]

        # sns.lineplot(x=x_w, y=non_lin_func_walk, lw=2, color='#fab9a5', label='Walking')
        # sns.lineplot(x=x_c, y=non_lin_func_cycle, lw=2, color='#B65FCF', label='Cycling')
        sns.lineplot(x=x_pt, y=non_lin_func_pt, color='g', label='PT')
        sns.lineplot(x=x_d, y=non_lin_func_driving, color='orange', label='Driving')


        #plt.title('Influence of straight line distance on the utility function', fontdict={'fontsize':  16})
        plt.xlabel('Cost [£]')
        plt.ylabel('Utility')

        plt.tight_layout()
        
        if save_figure:
            plt.savefig('Figures/RUMBoost/LPMC/cost.png')

        plt.show()

        plt.figure(figsize=(3.49, 3.49), dpi=1000)
        x_w, non_lin_func_walk = non_lin_function(weights_arranged['0']['age'], 0, 100, 10000)
        if asc_normalised:
            non_lin_func_walk = [n - non_lin_func_walk[0] for n in non_lin_func_walk]
        if with_asc:
            non_lin_func_walk = [n + ASCs[0] for n in non_lin_func_walk]

        x_c, non_lin_func_cycle = non_lin_function(weights_arranged['1']['age'], 0, 100, 10000)
        if asc_normalised:
            non_lin_func_cycle = [n - non_lin_func_cycle[0] for n in non_lin_func_cycle]
        if with_asc:
            non_lin_func_cycle = [n + ASCs[1] for n in non_lin_func_cycle]

        x_pt, non_lin_func_pt = non_lin_function(weights_arranged['2']['age'], 0, 100, 10000)
        if asc_normalised:
            non_lin_func_pt = [n - non_lin_func_pt[0] for n in non_lin_func_pt]
        if with_asc:
            non_lin_func_pt = [n + ASCs[2] for n in non_lin_func_pt]

        x_d, non_lin_func_driving = non_lin_function(weights_arranged['3']['age'], 0, 100, 10000)
        if asc_normalised:
            non_lin_func_driving = [n - non_lin_func_driving[0] for n in non_lin_func_driving]
        if with_asc:
            non_lin_func_driving = [n + ASCs[3] for n in non_lin_func_driving]

        sns.lineplot(x=x_w, y=non_lin_func_walk, color='b', label='Walking')
        sns.lineplot(x=x_c, y=non_lin_func_cycle, color='r', label='Cycling')
        sns.lineplot(x=x_pt, y=non_lin_func_pt, color='g', label='PT')
        sns.lineplot(x=x_d, y=non_lin_func_driving, color='orange', label='Driving')


        #plt.title('Influence of straight line distance on the utility function', fontdict={'fontsize':  16})
        plt.xlabel('Age')
        plt.ylabel('Utility')

        plt.tight_layout()

        if save_figure:
            plt.savefig('Figures/RUMBoost/LPMC/age.png')

        plt.show()

        plt.figure(figsize=(3.49, 3.49), dpi=1000)
        x_w, non_lin_func_walk = non_lin_function(weights_arranged['0']['start_time_linear'], 0, 24, 10000)
        if asc_normalised:
            non_lin_func_walk = [n - non_lin_func_walk[0] for n in non_lin_func_walk]
        if with_asc:
            non_lin_func_walk = [n + ASCs[0] for n in non_lin_func_walk]

        x_c, non_lin_func_cycle = non_lin_function(weights_arranged['1']['start_time_linear'], 0, 24, 10000)
        if asc_normalised:
            non_lin_func_cycle = [n - non_lin_func_cycle[0] for n in non_lin_func_cycle]
        if with_asc:
            non_lin_func_cycle = [n + ASCs[1] for n in non_lin_func_cycle]

        x_pt, non_lin_func_pt = non_lin_function(weights_arranged['2']['start_time_linear'], 0, 24, 10000)
        if asc_normalised:
            non_lin_func_pt = [n - non_lin_func_pt[0] for n in non_lin_func_pt]
        if with_asc:
            non_lin_func_pt = [n + ASCs[2] for n in non_lin_func_pt]

        x_d, non_lin_func_driving = non_lin_function(weights_arranged['3']['start_time_linear'], 0, 24, 10000)
        if asc_normalised:
            non_lin_func_driving = [n - non_lin_func_driving[0] for n in non_lin_func_driving]
        if with_asc:
            non_lin_func_driving = [n + ASCs[3] for n in non_lin_func_driving]

        sns.lineplot(x=x_w, y=non_lin_func_walk, color='b', label='Walking')
        sns.lineplot(x=x_c, y=non_lin_func_cycle, color='r', label='Cycling')
        sns.lineplot(x=x_pt, y=non_lin_func_pt, color='g', label='PT')
        sns.lineplot(x=x_d, y=non_lin_func_driving, color='orange', label='Driving')


        #plt.title('Influence of straight line distance on the utility function', fontdict={'fontsize':  16})
        plt.xlabel('Departure time')
        plt.ylabel('Utility')

        plt.tight_layout()

        if save_figure:
            plt.savefig('Figures/RUMBoost/LPMC/departure_time.png')

        plt.show()

    #for all features parameters
    if not only_tt:
        for u in weights_arranged:
            for i, f in enumerate(weights_arranged[u]):

                #create nonlinear plot
                if f in list(X.columns):
                    x, non_lin_func = non_lin_function(weights_arranged[u][f], 0, 1.05*max(X[f]), 10000)
                else:
                    x, non_lin_func = non_lin_function(weights_arranged[u][f], 0, 10, 10000)

                if asc_normalised:
                    val_0 = non_lin_func[0]
                    non_lin_func = [n - val_0 for n in non_lin_func]

                if with_asc:
                    non_lin_func = [n + ASCs[int(u)] for n in non_lin_func]
                
                #plot parameters
                plt.figure(figsize=(3.49, 2.09), dpi=1000)
                #plt.title('Influence of {} on the predictive function ({} utility)'.format(f, utility_names[u]), fontdict={'fontsize':  16})
                plt.ylabel('{} utility'.format(utility_names[u]))

                                    
                if 'dur' in f:
                    plt.xlabel('{} [h]'.format(f))
                elif 'TIME' in f:
                    plt.xlabel('{} [min]'.format(f))
                elif 'cost' in f:
                    plt.xlabel('{} [£]'.format(f))
                elif 'distance' in f:
                    plt.xlabel('{} [km]'.format(f))
                elif 'CO' in f:
                    plt.xlabel('{} [chf]'.format(f))
                else:
                    plt.xlabel('{}'.format(f))

                sns.lineplot(x=x, y=non_lin_func, color='k', label='RUMBoost')
                

                if f in list(X.columns):
                    plt.xlim([0-0.05*np.max(X[f]), np.max(X[f])*1.05])
                else:
                    plt.xlim([0-0.05*10, 10*1.05])
                plt.ylim([np.min(non_lin_func) - 0.05*(np.max(non_lin_func)-np.min(non_lin_func)), np.max(non_lin_func) + 0.05*(np.max(non_lin_func)-np.min(non_lin_func))])

                plt.tight_layout()
                    
                if save_figure:
                    plt.savefig('Figures/{}{} utility, {} feature.png'.format(save_file, utility_names[u], f))

                plt.show()

def plot_market_segm(model, X, asc_normalised: bool = True, utility_names: list[str] = ['Walking', 'Cycling', 'Public Transport', 'Driving']):
    '''
    Plot the market segmentation.

    Parameters
    ----------
    model : RUMBoost
        A RUMBoost object.
    X : pandas DataFrame
        Training data.
    asc_normalised : bool, optional (default = False)
        If True, scale down utilities to be zero at the y axis.
    utility_names : list[str], optional (default = ['Walking', 'Cycling', 'Public Transport', 'Driving'])
        Names of utilities.
    
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

def plot_util(model, data_train, points=10000):
    '''
    Plot the raw utility functions of all features. This is done directly from the predict attribute of lightgbm.Boosters.

    Parameters
    ----------
    model : RUMBoost
        A RUMBoost object.
    data_train : pandas Dataframe
        The full training dataset.
    points : int, optional (default = 10000)
        The number of points used to draw the line plot.

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

def plot_spline(model, data_train, spline_collection, utility_names, mean_splines = False, x_knots_dict = None, save_fig = False, lpmc_tt_cost=False, sm_tt_cost=False, save_file=''):
    '''
    Plot the spline interpolation for all utilities interpolated.

    Parameters
    ----------
    model : RUMBoost
        A RUMBoost object.
    data_train : pandas Dataframe
        The full training dataset.
    spline_collection : dict
        A dictionary containing the optimal number of splines for each feature interpolated of each utility
    mean_splines : bool, optional (default = False)
        Must be True if the splines are computed at the mean distribution of data for stairs.
    x_knots_dict : dict, optional (default = None)
        A dictionary in the form of {utility: {attribute: x_knots}} where x_knots are the spline knots for the corresponding 
        utility and attributes
    save_fig : bool, optional (default = False)
        If True, save the plot as a png file.
    lpmc_tt_cost : bool, optional (default = False)
        If True, plot only the LPMC travel time and cost on the same figure.
    sm_tt_cost : bool, optional (default = False)
        If True, plot only the swissmetro travel time and cost on the same figure.
    save_file : str, optional (default='')
        The name to save the figure with.
    '''
    #get weights ordered by features
    weights = weights_to_plot_v2(model)
    tex_fonts = {
            # Use LaTeX to write all text
            # "text.usetex": True, 
            # "font.family": "serif",
            # "font.serif": "Computer Modern Roman",
            # Use 14pt font in plots, to match 10pt font in document
            "axes.labelsize": 7,
            "axes.linewidth":0.5,
            "axes.labelpad": 1,
            "font.size": 7,
            # Make the legend/label fonts a little smaller
            "legend.fontsize": 6,
            "legend.fancybox": False,
            "legend.edgecolor": "inherit",
            'legend.borderaxespad': 0.4,
            'legend.borderpad': 0.4,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "xtick.major.pad": 0.5,
            "ytick.major.pad": 0.5,
            "grid.linewidth": 0.5,
            "lines.linewidth": 0.8,
            'scatter.edgecolors': 'none'
        }
    sns.set_theme(font_scale=1, rc=tex_fonts)
    #sns.set_context(tex_fonts)
    sns.set_style("whitegrid")
    # plt.rcParams.update({
    #     # "text.usetex": True,
    #     "font.family": "serif"
    #     #"font.sans-serif": "Computer Modern Roman",
    # })

    if lpmc_tt_cost:
        x_plot_w, y_plot_w = data_leaf_value(data_train['dur_walking'], weights['0']['dur_walking'], 'data_weighted')
        y_plot_norm_w = [y - y_plot_w[0] for y in y_plot_w]
        x_spline_w = np.linspace(np.min(data_train['dur_walking']), np.max(data_train['dur_walking']), num=10000)
        x_knots_temp_w, y_knots_w = data_leaf_value(x_knots_dict['0']['dur_walking'], weights['0']['dur_walking'])
        _, y_spline_w, _, x_knot_w, y_knot_w = monotone_spline(x_spline_w, weights['0']['dur_walking'], num_splines=spline_collection['0']['dur_walking'], x_knots=x_knots_temp_w, y_knots=y_knots_w)
        y_spline_norm_w = [y - y_plot_w[0] for y in y_spline_w]
        y_knot_norm_w = [y - y_plot_w[0] for y in y_knot_w]
        
        plt.figure(figsize=(3.49, 2.09), dpi=1000)

        #data
        plt.scatter(x_plot_w, y_plot_norm_w, color='b', s=0.3, alpha=1, edgecolors='none')

        #splines
        plt.plot(x_spline_w, y_spline_norm_w, color='b', label=f'Walking travel time ({spline_collection["0"]["dur_walking"]} splines)')

        #knots position
        plt.scatter(x_knot_w, y_knot_norm_w, color='k', s=1)

        x_plot_c, y_plot_c = data_leaf_value(data_train['dur_cycling'], weights['1']['dur_cycling'], 'data_weighted')
        y_plot_norm_c = [y - y_plot_c[0] for y in y_plot_c]
        x_spline_c = np.linspace(np.min(data_train['dur_cycling']), np.max(data_train['dur_cycling']), num=10000)
        x_knots_temp_c, y_knots_c = data_leaf_value(x_knots_dict['1']['dur_cycling'], weights['1']['dur_cycling'])
        _, y_spline_c, _, x_knot_c, y_knot_c = monotone_spline(x_spline_c, weights['1']['dur_cycling'], num_splines=spline_collection['1']['dur_cycling'], x_knots=x_knots_temp_c, y_knots=y_knots_c)
        y_spline_norm_c = [y - y_plot_c[0] for y in y_spline_c]
        y_knot_norm_c = [y - y_plot_c[0] for y in y_knot_c]

        #data
        plt.scatter(x_plot_c, y_plot_norm_c, color='r', s=0.3, alpha=1, edgecolors='none')

        #splines
        plt.plot(x_spline_c, y_spline_norm_c, color='r', label=f'Cycling travel time ({spline_collection["1"]["dur_cycling"]} splines)')

        #knots position
        plt.scatter(x_knot_c, y_knot_norm_c, color='k', s=1)

        x_plot_p, y_plot_p = data_leaf_value(data_train['dur_pt_rail'], weights['2']['dur_pt_rail'], 'data_weighted')
        y_plot_norm_p = [y - y_plot_p[0] for y in y_plot_p]
        x_spline_p = np.linspace(np.min(data_train['dur_pt_rail']), np.max(data_train['dur_pt_rail']), num=10000)
        x_knots_temp_p, y_knots_p = data_leaf_value(x_knots_dict['2']['dur_pt_rail'], weights['2']['dur_pt_rail'])
        _, y_spline_p, _, x_knot_p, y_knot_p = monotone_spline(x_spline_p, weights['2']['dur_pt_rail'], num_splines=spline_collection['2']['dur_pt_rail'], x_knots=x_knots_temp_p, y_knots=y_knots_p)
        y_spline_norm_p = [y - y_plot_p[0] for y in y_spline_p]
        y_knot_norm_p = [y - y_plot_p[0] for y in y_knot_p]

        #data
        plt.scatter(x_plot_p, y_plot_norm_p, color='g', s=0.3, alpha=1, edgecolors='none')

        #splines
        plt.plot(x_spline_p, y_spline_norm_p, color='g', label=f'Rail travel time ({spline_collection["2"]["dur_pt_rail"]} splines)')

        #knots position
        plt.scatter(x_knot_p, y_knot_norm_p, color='k', s=1)

        x_plot_d, y_plot_d = data_leaf_value(data_train['dur_driving'], weights['3']['dur_driving'], 'data_weighted')
        y_plot_norm_d = [y - y_plot_d[0] for y in y_plot_d]
        x_spline_d = np.linspace(np.min(data_train['dur_driving']), np.max(data_train['dur_driving']), num=10000)
        x_knots_temp_d, y_knots_d = data_leaf_value(x_knots_dict['3']['dur_driving'], weights['3']['dur_driving'])
        _, y_spline_d, _, x_knot_d, y_knot_d = monotone_spline(x_spline_d, weights['3']['dur_driving'], num_splines=spline_collection['3']['dur_driving'], x_knots=x_knots_temp_d, y_knots=y_knots_d)
        y_spline_norm_d = [y - y_plot_d[0] for y in y_spline_d]
        y_knot_norm_d = [y - y_plot_d[0] for y in y_knot_d]

        #data
        plt.scatter(x_plot_d, y_plot_norm_d, color='orange', s=0.3, alpha=1, edgecolors='none')

        #splines
        plt.plot(x_spline_d, y_spline_norm_d, color='orange', label=f'Driving travel time ({spline_collection["3"]["dur_driving"]} splines)')

        #knots position
        plt.scatter(x_knot_d, y_knot_norm_d, color='k', s=1, label='Knots')

        #plt.title('Spline interpolation of {}'.format(f))
        plt.ylabel('Utility')
        plt.xlim([0, 5])
        plt.xlabel('Travel time  [h]')
        plt.legend()
        plt.tight_layout()
        if save_fig:
            plt.savefig("Figures/RUMBoost/LPMC/splines_travel_time.png")
        plt.show()

        plt.figure(figsize=(3.49, 2.09), dpi=1000)

        x_plot_p, y_plot_p = data_leaf_value(data_train['cost_transit'], weights['2']['cost_transit'], 'data_weighted')
        y_plot_norm_p = [y - y_plot_p[0] for y in y_plot_p]
        x_spline_p = np.linspace(np.min(data_train['cost_transit']), np.max(data_train['cost_transit']), num=10000)
        x_knots_temp_p, y_knots_p = data_leaf_value(x_knots_dict['2']['cost_transit'], weights['2']['cost_transit'])
        _, y_spline_p, _, x_knot_p, y_knot_p = monotone_spline(x_spline_p, weights['2']['cost_transit'], num_splines=spline_collection['2']['cost_transit'], x_knots=x_knots_temp_p, y_knots=y_knots_p)
        y_spline_norm_p = [y - y_plot_p[0] for y in y_spline_p]
        y_knot_norm_p = [y - y_plot_p[0] for y in y_knot_p]

        #data
        plt.scatter(x_plot_p, y_plot_norm_p, color='g', s=0.3, alpha=1, edgecolors='none')

        #splines
        plt.plot(x_spline_p, y_spline_norm_p, color='g', label=f'PT cost ({spline_collection["2"]["cost_transit"]} splines)')

        #knots position
        plt.scatter(x_knot_p, y_knot_norm_p, color='k', s=1)

        x_plot_d, y_plot_d = data_leaf_value(data_train['cost_driving_fuel'], weights['3']['cost_driving_fuel'], 'data_weighted')
        y_plot_norm_d = [y - y_plot_d[0] for y in y_plot_d]
        x_spline_d = np.linspace(np.min(data_train['cost_driving_fuel']), np.max(data_train['cost_driving_fuel']), num=10000)
        x_knots_temp_d, y_knots_d = data_leaf_value(x_knots_dict['3']['cost_driving_fuel'], weights['3']['cost_driving_fuel'])
        _, y_spline_d, _, x_knot_d, y_knot_d = monotone_spline(x_spline_d, weights['3']['cost_driving_fuel'], num_splines=spline_collection['3']['cost_driving_fuel'], x_knots=x_knots_temp_d, y_knots=y_knots_d)
        y_spline_norm_d = [y - y_plot_d[0] for y in y_spline_d]
        y_knot_norm_d = [y - y_plot_d[0] for y in y_knot_d]

        #data
        plt.scatter(x_plot_d, y_plot_norm_d, color='orange', s=0.3, alpha=1, edgecolors='none')

        #splines
        plt.plot(x_spline_d, y_spline_norm_d, color='orange', label=f'Driving cost ({spline_collection["3"]["cost_driving_fuel"]} splines)')

        #knots position
        plt.scatter(x_knot_d, y_knot_norm_d, color='k', s=1, label='Knots')

        #plt.title('Spline interpolation of {}'.format(f))
        plt.ylabel('Utility')
        plt.xlim([0, 10])
        plt.xlabel('Cost [£]')
        plt.legend()
        plt.tight_layout()
        if save_fig:
            plt.savefig("Figures/RUMBoost/LPMC/splines_cost.png")
        plt.show()

    if sm_tt_cost:

        x_plot_p, y_plot_p = data_leaf_value(data_train['TRAIN_TT'], weights['0']['TRAIN_TT'], 'data_weighted')
        y_plot_norm_p = [y - y_plot_p[0] for y in y_plot_p]
        x_spline_p = np.linspace(np.min(data_train['TRAIN_TT']), np.max(data_train['TRAIN_TT']), num=10000)
        x_knots_temp_p, y_knots_p = data_leaf_value(x_knots_dict['0']['TRAIN_TT'], weights['0']['TRAIN_TT'])
        _, y_spline_p, _, x_knot_p, y_knot_p = monotone_spline(x_spline_p, weights['0']['TRAIN_TT'], num_splines=spline_collection['0']['TRAIN_TT'], x_knots=x_knots_temp_p, y_knots=y_knots_p)
        y_spline_norm_p = [y - y_plot_p[0] for y in y_spline_p]
        y_knot_norm_p = [y - y_plot_p[0] for y in y_knot_p]
        
        plt.figure(figsize=(3.49, 2.09), dpi=1000)
        #data
        plt.scatter(x_plot_p/60, y_plot_norm_p, color='g', s=0.3, alpha=1, edgecolors='none')

        #splines
        plt.plot(x_spline_p/60, y_spline_norm_p, color='g', label=f'Rail travel time ({spline_collection["0"]["TRAIN_TT"]} splines)')

        #knots position
        plt.scatter(x_knot_p/60, y_knot_norm_p, color='k', s=1)

        x_plot_s, y_plot_s = data_leaf_value(data_train['SM_TT'], weights['1']['SM_TT'], 'data_weighted')
        y_plot_norm_s = [y - y_plot_s[0] for y in y_plot_s]
        x_spline_s = np.linspace(np.min(data_train['SM_TT']), np.max(data_train['SM_TT']), num=10000)
        x_knots_temp_s, y_knots_s = data_leaf_value(x_knots_dict['1']['SM_TT'], weights['1']['SM_TT'])
        _, y_spline_s, _, x_knot_s, y_knot_s = monotone_spline(x_spline_s, weights['1']['SM_TT'], num_splines=spline_collection['1']['SM_TT'], x_knots=x_knots_temp_s, y_knots=y_knots_s)
        y_spline_norm_s = [y - y_plot_s[0] for y in y_spline_s]
        y_knot_norm_s = [y - y_plot_s[0] for y in y_knot_s]
        
        #data
        plt.scatter(x_plot_s/60, y_plot_norm_s, color='#6b8ba4', s=0.3, alpha=1, edgecolors='none')

        #splines
        plt.plot(x_spline_s/60, y_spline_norm_s, color='#6b8ba4', label=f'SwissMetro travel time ({spline_collection["1"]["SM_TT"]} splines)')

        #knots position
        plt.scatter(x_knot_s/60, y_knot_norm_s, color='k', s=1)

        x_plot_d, y_plot_d = data_leaf_value(data_train['CAR_TT'], weights['2']['CAR_TT'], 'data_weighted')
        y_plot_norm_d = [y - y_plot_d[0] for y in y_plot_d]
        x_spline_d = np.linspace(np.min(data_train['CAR_TT']), np.max(data_train['CAR_TT']), num=10000)
        x_knots_temp_d, y_knots_d = data_leaf_value(x_knots_dict['2']['CAR_TT'], weights['2']['CAR_TT'])
        _, y_spline_d, _, x_knot_d, y_knot_d = monotone_spline(x_spline_d, weights['2']['CAR_TT'], num_splines=spline_collection['2']['CAR_TT'], x_knots=x_knots_temp_d, y_knots=y_knots_d)
        y_spline_norm_d = [y - y_plot_d[0] for y in y_spline_d]
        y_knot_norm_d = [y - y_plot_d[0] for y in y_knot_d]

        #data
        plt.scatter(x_plot_d/60, y_plot_norm_d, color='orange', s=0.3, alpha=1, edgecolors='none')

        #splines
        plt.plot(x_spline_d/60, y_spline_norm_d, color='orange', label=f'Driving travel time ({spline_collection["2"]["CAR_TT"]} splines)')

        #knots position
        plt.scatter(x_knot_d/60, y_knot_norm_d, color='k', s=1, label='Knots')

        #plt.title('Spline interpolation of {}'.format(f))
        plt.ylabel('Utility')
        plt.xlim([0, 10])
        plt.xlabel('Travel time [h]')
        plt.legend()
        plt.tight_layout()
        if save_fig:
            plt.savefig("Figures/RUMBoost/SwissMetro/splines_travel_time.png")
        plt.show()


        plt.figure(figsize=(3.49, 2.09), dpi=1000)
        x_plot_p, y_plot_p = data_leaf_value(data_train['TRAIN_COST'], weights['0']['TRAIN_COST'], 'data_weighted')
        y_plot_norm_p = [y - y_plot_p[0] for y in y_plot_p]
        x_spline_p = np.linspace(np.min(data_train['TRAIN_COST']), np.max(data_train['TRAIN_COST']), num=10000)
        x_knots_temp_p, y_knots_p = data_leaf_value(x_knots_dict['0']['TRAIN_COST'], weights['0']['TRAIN_COST'])
        _, y_spline_p, _, x_knot_p, y_knot_p = monotone_spline(x_spline_p, weights['0']['TRAIN_COST'], num_splines=spline_collection['0']['TRAIN_COST'], x_knots=x_knots_temp_p, y_knots=y_knots_p)
        y_spline_norm_p = [y - y_plot_p[0] for y in y_spline_p]
        y_knot_norm_p = [y - y_plot_p[0] for y in y_knot_p]

        #data
        plt.scatter(x_plot_p, y_plot_norm_p, color='g', s=0.3, alpha=1, edgecolors='none')

        #splines
        plt.plot(x_spline_p, y_spline_norm_p, color='g', label=f'Rail cost ({spline_collection["0"]["TRAIN_COST"]} splines)')

        #knots position
        plt.scatter(x_knot_p, y_knot_norm_p, color='k', s=1)

        x_plot_s, y_plot_s = data_leaf_value(data_train['SM_COST'], weights['1']['SM_COST'], 'data_weighted')
        y_plot_norm_s = [y - y_plot_s[0] for y in y_plot_s]
        x_spline_s = np.linspace(np.min(data_train['SM_COST']), np.max(data_train['SM_COST']), num=10000)
        x_knots_temp_s, y_knots_s = data_leaf_value(x_knots_dict['1']['SM_COST'], weights['1']['SM_COST'])
        _, y_spline_s, _, x_knot_s, y_knot_s = monotone_spline(x_spline_s, weights['1']['SM_COST'], num_splines=spline_collection['1']['SM_COST'], x_knots=x_knots_temp_s, y_knots=y_knots_s)
        y_spline_norm_s = [y - y_plot_s[0] for y in y_spline_s]
        y_knot_norm_s = [y - y_plot_s[0] for y in y_knot_s]

        #data
        plt.scatter(x_plot_s, y_plot_norm_s, color='#6b8ba4', s=0.3, alpha=1, edgecolors='none')

        #splines
        plt.plot(x_spline_s, y_spline_norm_s, color='#6b8ba4', label=f'SwissMetro cost ({spline_collection["1"]["SM_COST"]} splines)')

        #knots position
        plt.scatter(x_knot_s, y_knot_norm_s, color='k', s=1)

        x_plot_d, y_plot_d = data_leaf_value(data_train['CAR_CO'], weights['2']['CAR_CO'], 'data_weighted')
        y_plot_norm_d = [y - y_plot_d[0] for y in y_plot_d]
        x_spline_d = np.linspace(np.min(data_train['CAR_CO']), np.max(data_train['CAR_CO']), num=10000)
        x_knots_temp_d, y_knots_d = data_leaf_value(x_knots_dict['2']['CAR_CO'], weights['2']['CAR_CO'])
        _, y_spline_d, _, x_knot_d, y_knot_d = monotone_spline(x_spline_d, weights['2']['CAR_CO'], num_splines=spline_collection['2']['CAR_CO'], x_knots=x_knots_temp_d, y_knots=y_knots_d)
        y_spline_norm_d = [y - y_plot_d[0] for y in y_spline_d]
        y_knot_norm_d = [y - y_plot_d[0] for y in y_knot_d]

        #data
        plt.scatter(x_plot_d, y_plot_norm_d, color='orange', s=0.3, alpha=1, edgecolors='none')

        #splines
        plt.plot(x_spline_d, y_spline_norm_d, color='orange', label=f'Driving cost ({spline_collection["2"]["CAR_CO"]} splines)')

        #knots position
        plt.scatter(x_knot_d, y_knot_norm_d, color='k', s=1, label='Knots')

        #plt.title('Spline interpolation of {}'.format(f))
        plt.ylabel('Utility')
        plt.xlim([0, 500])
        plt.xlabel('Cost [chf]')
        plt.legend()
        plt.tight_layout()
        if save_fig:
            plt.savefig("Figures/RUMBoost/SwissMetro/splines_cost.png")
        plt.show()


    for u in spline_collection:
        for f in spline_collection[u]:
            #data points and their utilities
            x_plot, y_plot = data_leaf_value(data_train[f], weights[u][f], 'data_weighted')
            y_plot_norm = [y - y_plot[0] for y in y_plot]
            x_spline = np.linspace(np.min(data_train[f]), np.max(data_train[f]), num=10000)

            #if using splines
            #if mean technique
            if mean_splines:
                x_mean, y_mean = data_leaf_value(data_train[f], weights[u][f], technique='mean_data')
                x_spline, y_spline, _, x_knot, y_knot = mean_monotone_spline(x_plot, x_mean, y_plot, y_mean, num_splines=spline_collection[u][f])
            #else, i.e. linearly sampled points
            else:
                if x_knots_dict is not None:
                    x_knots_temp, y_knots = data_leaf_value(x_knots_dict[u][f], weights[u][f])
                    _, y_spline, _, x_knot, y_knot = monotone_spline(x_spline, weights[u][f], num_splines=spline_collection[u][f], x_knots=x_knots_temp, y_knots=y_knots)
                else:
                    x_spline, y_spline, _, x_knot, y_knot = monotone_spline(x_plot, y_plot, num_splines=spline_collection[u][f])
            y_spline_norm = [y - y_plot[0] for y in y_spline]
            y_knot_norm = [y - y_plot[0] for y in y_knot]

            
            plt.figure(figsize=(3.49, 2.09), dpi=1000)

            #data
            plt.scatter(x_plot, y_plot_norm, color='k', s=0.3)

            #splines
            plt.plot(x_spline, y_spline_norm, color='#5badc7')

            #knots position
            plt.scatter(x_knot, y_knot_norm, color='#CC5500', s=1)

            plt.legend(['Data', 'Splines ({})'.format(spline_collection[u][f]), 'Knots'])
            #plt.title('Spline interpolation of {}'.format(f))
            plt.ylabel('{} utility'.format(utility_names[u]))     
            plt.tight_layout()        
            if 'dur' in f:
                plt.xlabel('{} [h]'.format(f))
            elif 'TIME' in f:
                plt.xlabel('{} [h]'.format(f))
            elif 'cost' in f:
                plt.xlabel('{} [£]'.format(f))
            elif 'CO' in f:
                plt.xlabel('{} [chf]'.format(f))
            elif 'distance' in f:
                plt.xlabel('{} [km]'.format(f))
            else:
                plt.xlabel('{}'.format(f))
            if save_fig:
                plt.savefig(save_file + "{} utility, {} feature.png".format(u, f))
            plt.show()

def plot_VoT(data_train, util_collection, attribute_VoT, utility_names, draw_range, save_figure = False, num_points = 1000):
    '''
    The function plot the Value of Time of the attributes specified in attribute_VoT.

    Parameters
    ----------
    util_collection : dict
        A dictionary containing the type of utility to use for all features in all utilities.
    attribute_VoT : dict
        A dictionary with keys being the utility number (as string) and values being a tuple of the attributes to compute the VoT on.
        The structure follows this form: {utility: (attribute1, attribute2)}
    utility_names : dict
        A dictionary containing the names of the utilities.
        The structure of the dictionary follows this form: {utility: names}
    draw_range : dict
        A dictionary containing the range of the attributes to draw the VoT.
        The structure of the dictionary follows this form: {utility: {attribute: (min, max)}}
    save_figure : bool, optional (default = False)
        If True, save the plot as a png file.
    num_points : int, optional (default = 1000)
        The number of points used to draw the contour plot.  
    '''

    tex_fonts = {
            # Use LaTeX to write all text
            # "text.usetex": True, 
            # "font.family": "serif",
            # "font.serif": "Computer Modern Roman",
            # Use 14pt font in plots, to match 10pt font in document
            "axes.labelsize": 7,
            "axes.linewidth":0.5,
            "axes.labelpad": 1,
            "font.size": 7,
            # Make the legend/label fonts a little smaller
            "legend.fontsize": 6,
            "legend.fancybox": False,
            "legend.edgecolor": "inherit",
            'legend.borderaxespad': 0.4,
            'legend.borderpad': 0.4,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "xtick.major.pad": 0.1,
            "ytick.major.pad": 0.1,
            "grid.linewidth": 0.5,
            "lines.linewidth": 0.8
        }
    sns.set_theme(font_scale=1, rc=tex_fonts)
    #sns.set_context(tex_fonts)
    sns.set_style("whitegrid")
    # plt.rcParams.update({
    #     # "text.usetex": True,
    #     "font.family": "serif"
    #     #"font.sans-serif": "Computer Modern Roman",
    # })
    
    for u in attribute_VoT: 
        f1, f2 = attribute_VoT[u]
        x_vect = np.linspace(draw_range[u][f1][0], draw_range[u][f1][1], num_points)
        y_vect = np.linspace(draw_range[u][f2][0], draw_range[u][f2][1], num_points)
        d_f1 = util_collection[u][f1].derivative()
        d_f2 = util_collection[u][f2].derivative()
        VoT = lambda x1, x2, df1 = d_f1, df2 = d_f2: df1(x1) / df2(x2)
        VoT_contour_plot = np.array(np.zeros((len(x_vect), len(y_vect))))
        X, Y = np.meshgrid(x_vect, y_vect, indexing='ij')
        for i in range(len(x_vect)):
            for j in range(len(y_vect)):
                if d_f2(Y[i, j]) == 0:
                    VoT_contour_plot[i, j] = 100
                elif VoT(X[i, j], Y[i, j]) > 100:
                    VoT_contour_plot[i, j] = 100
                elif VoT(X[i, j], Y[i, j]) < 0.1:
                    VoT_contour_plot[i, j] = 0.1
                else:
                    VoT_contour_plot[i, j] = VoT(X[i, j], Y[i, j])

        fig, axes = plt.subplots(figsize=(3.49,3.49), dpi=1000)

        #fig.suptitle(f'VoT ({f1} and {f2}) of {utility_names[u]}')

        res = 100

        c_plot = axes.contourf(X, Y, np.log(VoT_contour_plot)/np.log(10), levels=res, linewidths=0, cmap=sns.color_palette("Blues", as_cmap=True), vmin = -1, vmax = 2)

        #axes.set_title(f'{utility_names[u]}')
        axes.set_xlabel(f'{f1} [h]')
        axes.set_ylabel(f'{f2} [£]')

        cbar = fig.colorbar(c_plot, ax = axes, ticks=[-1, 0, 1, 2])
        cbar.set_ticklabels([0.1, 1, 10, 100])
        cbar.ax.set_ylabel('VoT [£/h]')
        cbar.ax.set_ylim([-1, 2])

        #plt.tight_layout()

        if save_figure:
            plt.savefig('Figures/RUMBoost/LPMC/VoT_{}.png'.format(utility_names[u]))

        plt.show()


def plot_pop_VoT(data_test, util_collection, attribute_VoT, save_figure = False):
    '''
    Plot the Value of Time for the given observations.

    Parameters
    ----------
    data_test : pd.DataFrame
        The test dataset.
    util_collection : dict
        A dictionary containing the utility function (spline or tree) to use for all features in all utilities where the VoT is computed. it follows this structure {utility: {feature: tree/spline function}}
    attribute_VoT : dict
        A dictionary with keys being the utility number (as string) and values being a tuple of the attributes to compute the VoT on.
        The structure follows this form: {utility: (attribute1, attribute2)}
    save_figure : bool, optional (default = False)
        If True, save the plot as a png file.
    '''

    tex_fonts = {
            # Use LaTeX to write all text
            # "text.usetex": True, 
            # "font.family": "serif",
            # "font.serif": "Computer Modern Roman",
            # Use 14pt font in plots, to match 10pt font in document
            "axes.labelsize": 7,
            "axes.linewidth":0.5,
            "axes.labelpad": 1,
            "font.size": 7,
            # Make the legend/label fonts a little smaller
            "legend.fontsize": 6,
            "legend.fancybox": False,
            "legend.edgecolor": "inherit",
            'legend.borderaxespad': 0.4,
            'legend.borderpad': 0.4,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "xtick.major.pad": 0.5,
            "ytick.major.pad": 0.5,
            "grid.linewidth": 0.5,
            "lines.linewidth": 0.8
        }
    sns.set_theme(font_scale=1, rc=tex_fonts)
    #sns.set_context(tex_fonts)
    sns.set_style("whitegrid")
    # plt.rcParams.update({
    #     # "text.usetex": True,
    #     "font.family": "serif"
    #     #"font.sans-serif": "Computer Modern Roman",
    # })

    for u in attribute_VoT: 
        f1, f2 = attribute_VoT[u]
        d_f1 = util_collection[u][f1].derivative()
        d_f2 = util_collection[u][f2].derivative()

        VoT_pop = d_f1(data_test[f1])/d_f2(data_test[f2])

        filtered_VoT_pop = VoT_pop[~np.isnan(VoT_pop)]

        limited_VoT_pop = filtered_VoT_pop[(filtered_VoT_pop>0) & (filtered_VoT_pop < np.quantile(filtered_VoT_pop, 0.99))]

        #fig, axes = plt.subplots(figsize=(10,8), layout='constrained')

        plt.figure(figsize=(3.49, 2.09), dpi=1000)
        sns.histplot(limited_VoT_pop, color='b', alpha = 0.5, kde=True, bins=50)
        plt.xlabel("VoT [£/h]")
        plt.tight_layout()
        plt.show()

        if save_figure:
           plt.savefig('Figures/RUMBoost/SwissMetro/pop_VoT_{}.png'.format(u))

def plot_ind_spec_constant(socec_model, dataset_train, alternatives: list[str]):
    '''
    Plot a histogram of all alternatives individual specific constant of a functional effect model.

    Parameters
    ----------

    socec_model:
        The part of the functional effect model with full interactions of socio-economic characteristics.
    dataset_train: 
        The dataset used to train the model. It must be a lightGBM Dataset object.
    alternatives: list[str]
        The list of alternatives name.
    '''

    tex_fonts = {
        # Use LaTeX to write all text
        # "text.usetex": True, 
        # "font.family": "serif",
        # "font.serif": "Computer Modern Roman",
        # Use 14pt font in plots, to match 10pt font in document
        "axes.labelsize": 7,
        "axes.linewidth": 0.5,
        "axes.labelpad": 1,
        "font.size": 7,
        # Make the legend/label fonts a little smaller
        "legend.fontsize": 6,
        "legend.fancybox": False,
        "legend.edgecolor": "inherit",
        'legend.borderaxespad': 0.4,
        'legend.borderpad': 0.4,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "xtick.major.pad": 0.5,
        "ytick.major.pad": 0.5,
        "grid.linewidth": 0.5,
        "lines.linewidth": 0.8
    }
    sns.set_theme(font_scale=1, rc=tex_fonts)
    #sns.set_context(tex_fonts)
    sns.set_style("whitegrid")
    # plt.rcParams.update({
    #     # "text.usetex": True,
    #     "font.family": "serif"
    #     #"font.sans-serif": "Computer Modern Roman",
    # })

    ind_spec_constants = socec_model.predict(dataset_train, utilities=True)

    bins=np.histogram(ind_spec_constants, bins=50)[1]
    sns.set_theme()
    f, axes = plt.subplots(2, 2, figsize=(12, 10), tight_layout=True)
    colors = ['b', 'r', 'g', 'orange']

    for i, axs in enumerate(axes.flatten()):
        sns.histplot(ind_spec_constants[:, i], bins=bins, alpha=0.5, ax=axs, kde=True, color=colors[i])
        axs.set_title(f'{alternatives[i]}')

    # Defining custom 'xlim' and 'ylim' values.
    xlim = (-3.5, 3.5)
    ylim = (0, 5250)

    # Setting the values for all axes.
    plt.setp(axes, xlim=xlim, ylim=ylim)

    plt.show()

def plot_bootstrap(models: list, dataset: pd.DataFrame, features: dict[list[str]]):
    '''
    Plot the bootstrap sampling.

    Parameters
    ----------
    models: list
        A list containing all the trained mdoels of the bootstrap sampling
    dataset: pd.DataFrame
        The full dataset used for training
    features: dict[list[str]]
        A dictionary of lists of strings contaning the number of alternatives, and the features for that alternative, 
        e.g. {'0':['feature_1', ...], '1': [], ...]
    '''
    tex_fonts = {
        # Use LaTeX to write all text
        # "text.usetex": True, 
        # "font.family": "serif",
        # "font.serif": "Computer Modern Roman",
        # Use 14pt font in plots, to match 10pt font in document
        "axes.labelsize": 7,
        "axes.linewidth":0.5,
        "axes.labelpad": 1,
        "font.size": 7,
        # Make the legend/label fonts a little smaller
        "legend.fontsize": 6,
        "legend.fancybox": False,
        "legend.edgecolor": "inherit",
        'legend.borderaxespad': 0.4,
        'legend.borderpad': 0.4,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "xtick.major.pad": 0.5,
        "ytick.major.pad": 0.5,
        "grid.linewidth": 0.5,
        "lines.linewidth": 0.8
    }
    sns.set_theme(font_scale=1, rc=tex_fonts)
    sns.set_style("whitegrid")
    # plt.rcParams.update({
    #     # "text.usetex": True,
    #     "font.family": "serif"
    #     #"font.sans-serif": "Computer Modern Roman",
    # })

    ufs_dict = {}
    for u in features:
        ufs_dict[u] = {}
        for f in features[u]:
            ufs_dict[u][f] = {'xplot': np.linspace(0, dataset[f].max(), 1000), 'yarr': np.array([]), 'yav': []}
            yi = []
            for model in models:
                vals = weights_to_plot_v2(model)
                _, y = non_lin_function(vals[u][f], 0, dataset[f].max(), 1000)
                yi.append([yii-y[0] for yii in y])
            ufs_dict[u][f]['yarr'] = np.array(yi)
            ufs_dict[u][f]['yav'] = ufs_dict[u][f]['yarr'].mean(axis=0)
            
            g = sns.JointGrid(xlim=(0, np.max(dataset[f])), height=3.89)
            g.figure.set_dpi(1000)
            x, y = ufs_dict[u][f]['xplot'],ufs_dict[u][f]['yav']
            sns.lineplot(x=x, y=y, ax=g.ax_joint,color='orange', linewidth=1, label='Average')
            sns.histplot(x=dataset[f], ax=g.ax_marg_x, bins=100, color = 'orange', alpha=0.5)
            for i in range(len(models)):
                sns.lineplot(x=x, y=ufs_dict[u][f]['yarr'][i, :].T, color = 'orange', alpha = 0.1, ax=g.ax_joint, linewidth=0.5)
            g.ax_joint.set(xlabel=f'{f}', ylabel='Utility')



