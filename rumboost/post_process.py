import pandas as pd
import numpy as np
import lightgbm as lgb
from rumboost.rumboost import RUMBoost, rum_train
from rumboost.utility_plotting import weights_to_plot_v2

from biogeme.expressions import Beta, Variable, bioMultSum
from biogeme.models import piecewise_formula, loglogit
from biogeme.biogeme import BIOGEME
import biogeme.database as db


def split_fe_model(model: RUMBoost):
    """
    Split a functional effect model and returns its two parts

    Parameters
    ----------

    model: RUMBoost
        A functional effect RUMBoost model with rum_structure

    Returns
    -------

    attributes_model: RUMBoost
        The part of the functional effect model with trip attributes without interaction
    socio_economic_model: RUMBoost
        The part of the model leading to the individual-specific constant, where socio-economic characteristics fully interact.
    """
    if not isinstance(model.rum_structure, list):
        raise ValueError(
            "Please add a rum_structure to your model by setting model.rum_structure. A rum_structure must be a list of 2*n_alt dictionaries in this function"
        )

    attributes_model = RUMBoost()
    socio_economic_model = RUMBoost()

    attributes_model.boosters = [b for i, b in enumerate(model.boosters) if i % 2 == 0]
    attributes_model.rum_structure = model.rum_structure[::2]
    attributes_model.num_classes = model.num_classes
    attributes_model.device = model.device
    attributes_model.nests = model.nests
    attributes_model.alphas = model.alphas

    socio_economic_model.boosters = [
        b for i, b in enumerate(model.boosters) if i % 2 == 1
    ]
    socio_economic_model.rum_structure = model.rum_structure[1::2]
    socio_economic_model.num_classes = model.num_classes
    socio_economic_model.device = model.device
    socio_economic_model.nests = model.nests
    socio_economic_model.alphas = model.alphas

    return attributes_model, socio_economic_model


def bootstrap(
    dataset: pd.DataFrame,
    model_specification: dict,
    num_it: int = 100,
    seed: int = 42,
):
    """
    Performs bootstrapping, with given dataset, parameters and rum_structure. For now, only a basic rumboost can be used.

    Parameters
    ----------
    dataset: pd.DataFrame
        A dataset used to train RUMBoost
    model_specification: dict
        A dictionary containing the model specification used to train the model.
        It should follow the same structure than in the rum_train() function.
    num_it: int, optional (default=100)
        The number of bootstrapping iterations
    seed: int, optional (default=42)
        The seed used to randomly sample the dataset.

    Returns
    -------
    models: list
        Return a list containing all trained models.
    """
    np.random.seed(seed)

    N = dataset.shape[0]
    models = []
    for _ in range(num_it):
        ids = np.random.choice(dataset.index, size=N, replace=True)
        ids2 = np.setdiff1d(dataset.index, ids)

        df_train = dataset.loc[ids]
        df_test = dataset.loc[ids2]

        dataset_train = lgb.Dataset(
            df_train.drop("choice", axis=1), label=df_train.choice, free_raw_data=False
        )

        valid_set = lgb.Dataset(
            df_test.drop("choice", axis=1), label=df_test.choice, free_raw_data=False
        )

        models.append(
            rum_train(dataset_train, model_specification, valid_sets=[valid_set])
        )

    return models


def assist_model_spec(model, dataset, choice, alt_to_normalise=0):
    """
    Provide a piece-wise linear model spcification based on a pre-trained rumboost model.

    Parameters
    ----------

    model: RUMBoost
        A trained rumboost model.
    dataset: pd.DataFrame
        A dataset used to train the model
    choice: pd.Series
        A series containing the choices
    alt_to_normalise: int, optional (default=0)
        The variables of that alternative will be normalised when needed (socio-economic characteristics, ascs, ...).

    Returns
    -------
    model_spec: dict
        A dictionary containing the model specification used to train a biogeme model.
    """
    dataset["choice"] = choice
    database = db.Database("rumboost", dataset)
    globals().update(database.variables)

    # define ascs, with one normalised to zero
    ascs = {
        f"asc_{i}": Beta(f"asc_{i}", 0, None, None, 1 if i == alt_to_normalise else 0)
        for i in range(model.num_classes)
    }

    # prepare variables to normalise
    vars_in_utility = {v: [] for v in dataset.columns}
    unique_betas = {}
    for rum in model.rum_structure:
        for v in rum["variables"]:
            vars_in_utility[v].extend(rum["utility"])
            unique_betas[v] = Beta(f"{v}_0", 0, None, None, 0)


    vars_to_normalise = []
    for variables, utilities in vars_in_utility.items():
        if len(np.unique(utilities)) == model.num_classes:
            vars_to_normalise.append(variables) 

    # get aggregated split points and leaf values by ensembles and variables
    weights = weights_to_plot_v2(model)

    # initialise utility specification with ascs
    utility_spec = {i: ascs[f"asc_{i}"] for i in range(model.num_classes)}

    # loop over the ensembles
    for i, weight in weights.items():
        # loop over the variables within an ensemble
        for name, tree_info in weight.items():
            # if linear
            if model.boost_from_parameter_space[int(i)]:
                split_points = tree_info["Splitting points"]
                init_beta = tree_info["Histogram values"]
                split_points.insert(0, dataset[name].min())
                split_points.append(dataset[name].max())
                # monotonicity constraints
                lowerbound = (
                    0
                    if model.rum_structure[int(i)]["boosting_params"][
                        "monotone_constraints"
                    ][0]
                    == 1
                    else None
                )
                upperbound = (
                    0
                    if model.rum_structure[int(i)]["boosting_params"][
                        "monotone_constraints"
                    ][0]
                    == -1
                    else None
                )
                # define betas
                betas = [
                    Beta(f"{name}_{i}_{j}", init_beta[j], lowerbound, upperbound, 0)
                    for j in range(len(split_points) - 1)
                ]
                # add piecewise linear variables to the proper utility function
                for u in model.rum_structure[int(i)]["utility"]:
                    if u == alt_to_normalise and name in vars_to_normalise:
                        continue
                    utility_spec[u] = utility_spec[u] + piecewise_formula(
                        name, split_points, betas
                    )
            else:
                # if piece-wise constant
                split_points = tree_info["Splitting points"]
                init_beta = tree_info["Histogram values"]
                beta_0 = init_beta[0]
                init_beta = [i - beta_0 for i in init_beta]
                # monotonicity constraints
                lowerbound = (
                    0
                    if model.rum_structure[int(i)]["boosting_params"][
                        "monotone_constraints"
                    ][0]
                    == 1
                    else None
                )
                upperbound = (
                    0
                    if model.rum_structure[int(i)]["boosting_params"][
                        "monotone_constraints"
                    ][0]
                    == -1
                    else None
                )
                # define betas
                if len(split_points) == 1: # if already binary
                    if len(vars_in_utility[name]) > 1:
                        beta_dict = {
                            f"{name}_{i}_{0}": unique_betas[name]
                        }
                        vars = [Variable(name)]
                    else:
                        beta_dict = {
                            f"{name}_{i}_{0}": Beta(f"{name}_{i}_0", init_beta[0], lowerbound, upperbound, 0)
                        }
                        vars = [Variable(name)]
                else:
                    #if non binary
                    split_points.insert(0, dataset[name].min())
                    split_points.append(dataset[name].max())
                    # we normalise to zero the first beta
                    beta_dict = {
                        f"{name}_{i}_0": Beta(f"{name}_{i}_0", 0, None, None, 1)
                    }
                    # if monotonicity constraint, we use previous beta as lower/upper bound
                    for j in range(1, len(split_points) - 1):
                        beta_dict[f"{name}_{i}_{j}"] = Beta(
                            f"{name}_{i}_{j}",
                            init_beta[j],
                            beta_dict[f"{name}_{i}_{j-1}"] * (lowerbound == 0), 
                            beta_dict[f"{name}_{i}_{j-1}"] * (upperbound == 0),
                            int(j == 0),
                        )
                    vars = [
                        database.define_variable(
                            f"{name}_{i}_{j}",
                            (Variable(name) - split_points[j])
                            * (Variable(name) - split_points[j + 1] <= 0),
                        )
                        for j in range(len(split_points) - 1)
                    ]
                for u in model.rum_structure[int(i)]["utility"]:
                    if u == alt_to_normalise and name in vars_to_normalise:
                        continue
                    utility_spec[u] = utility_spec[u] + bioMultSum(
                        [b * v for b, v in zip(beta_dict.values(), vars)]
                    )

    availability = {i: 1 for i in range(model.num_classes)}

    model_name = "assisted_model"

    logprob = loglogit(utility_spec, availability, Variable("choice"))

    the_biogeme = BIOGEME(database, logprob)
    the_biogeme.modelName = model_name
    
    the_biogeme.calculateNullLoglikelihood(availability)

    return the_biogeme

def estimate_dcm_with_assisted_spec(
    dataset: pd.DataFrame,
    choice: pd.Series,
    model: RUMBoost,
):
    """
    Estimate a Discrete Choice Model (currently only logit) with a piece-wise linear model specification based on a pre-trained rumboost model.

    Parameters
    ----------
    dataset: pd.DataFrame
        A dataset used to train the model
    choice: pd.Series
        A series containing the choices
    model: RUMBoost
        A trained rumboost model.

    Returns
    -------
    estimated_model: biogeme.results.bioResults
    """
    the_biogeme = assist_model_spec(model, dataset, choice)

    results = the_biogeme.estimate(recycle=True)

    return results

def predict_with_assisted_spec(
    dataset: pd.DataFrame,
    choice: pd.Series,
    model: RUMBoost,
    beta_values: dict,
):
    """
    Predict choices with a piece-wise linear model specification based on a pre-trained rumboost model.

    Parameters
    ----------
    dataset: pd.DataFrame
        A dataset used to predict the choices
    choice: pd.Series
        A series containing the choices
    model: RUMBoost
        A trained rumboost model.
    beta_values: dict
        A dictionary containing the beta values of the model, estimated on the train set.

    Returns
    -------
    prediction_results: biogeme.results.bioResults
    """
    the_biogeme = assist_model_spec(model, dataset, choice)

    prediction_results = the_biogeme.simulate(beta_values)

    return prediction_results