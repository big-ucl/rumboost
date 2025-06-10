import pandas as pd
import numpy as np
import lightgbm as lgb
import os
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
    attributes_model.boost_from_parameter_space = model.boost_from_parameter_space[::2]
    attributes_model.asc = model.asc

    socio_economic_model.boosters = [
        b for i, b in enumerate(model.boosters) if i % 2 == 1
    ]
    socio_economic_model.rum_structure = model.rum_structure[1::2]
    socio_economic_model.num_classes = model.num_classes
    socio_economic_model.device = model.device
    socio_economic_model.nests = model.nests
    socio_economic_model.alphas = model.alphas
    socio_economic_model.boost_from_parameter_space = model.boost_from_parameter_space[
        1::2
    ]
    socio_economic_model.asc = model.asc

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


def assist_model_spec(
    model: RUMBoost,
    dataset: pd.DataFrame,
    choice: pd.Series,
    alt_to_normalise: int = 0,
    return_utilities: bool = False,
    dataset_test: pd.DataFrame = None,
    choice_test: pd.Series = None,
):
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
    utilities: bool, optional (default=False)
        If True, the model will return the utility values, otherwise it will return the loglogit values.
    dataset_test: pd.DataFrame, optional (default=None)
        Only for predictions. If None, the dataset used to train the model will be used.
    choice_test: pd.Series, optional (default=None)
        A series containing the choices for the test dataset

    Returns
    -------
    model_spec: dict
        A dictionary containing the model specification used to train a biogeme model.
    """
    dataset["choice"] = choice
    if dataset_test is not None and choice_test is not None:
        dataset_test["choice"] = choice_test

    database = db.Database("rumboost", dataset)
    globals().update(database.variables)

    # define ascs, with one normalised to zero
    ascs = {
        f"asc_{i}": Beta(f"asc_{i}", 0, None, None, 1 if i == alt_to_normalise else 0)
        for i in range(model.num_classes)
    }

    # prepare variables to normalise
    vars_in_utility = {v: [] for v in dataset.columns}
    for rum in model.rum_structure:
        for v in rum["variables"]:
            vars_in_utility[v].extend(rum["utility"])

    vars_to_normalise = []
    for variables, utilities in vars_in_utility.items():
        if len(np.unique(utilities)) == model.num_classes:
            vars_to_normalise.append(variables)

    # get aggregated split points and leaf values by ensembles and variables
    weights = weights_to_plot_v2(model)

    # initialise utility specification with ascs
    utility_spec = {i: ascs[f"asc_{i}"] for i in range(model.num_classes)}

    # store new variables created and split_points
    variables_created = {}

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
                    0.0
                    if model.rum_structure[int(i)]["boosting_params"][
                        "monotone_constraints"
                    ][0]
                    == 1
                    else None
                )
                upperbound = (
                    0.0
                    if model.rum_structure[int(i)]["boosting_params"][
                        "monotone_constraints"
                    ][0]
                    == -1
                    else None
                )
                # define betas
                if (
                    alt_to_normalise == model.rum_structure[int(i)]["utility"][0]
                    and name in vars_to_normalise
                ):
                    beta_fixed = 1
                else:
                    beta_fixed = 0
                betas = [
                    Beta(
                        f"b_{name}_{i}_{j}",
                        init_beta[j],
                        lowerbound,
                        upperbound,
                        beta_fixed,
                    )
                    for j in range(len(split_points) - 1)
                ]
                # add piecewise linear variables to the proper utility function
                for u in model.rum_structure[int(i)]["utility"]:
                    utility_spec[u] = utility_spec[u] + piecewise_formula(
                        name, split_points, betas
                    )
            else:
                # if piece-wise constant
                split_points = tree_info["Splitting points"]
                init_beta = tree_info["Histogram values"]
                beta_0 = init_beta[0]
                init_beta = [i - beta_0 for i in init_beta]
                if (
                    alt_to_normalise == model.rum_structure[int(i)]["utility"][0]
                    and name in vars_to_normalise
                ):
                    beta_fixed = 1
                else:
                    beta_fixed = 0
                # monotonicity constraints
                lowerbound = (
                    0.0
                    if model.rum_structure[int(i)]["boosting_params"][
                        "monotone_constraints"
                    ][0]
                    == 1
                    else None
                )
                upperbound = (
                    0.0
                    if model.rum_structure[int(i)]["boosting_params"][
                        "monotone_constraints"
                    ][0]
                    == -1
                    else None
                )
                # define betas
                if len(split_points) == 1:  # if already binary
                    beta_dict = {
                        f"b_{name}_{i}_{0}": Beta(
                            f"b_{name}_{i}_0",
                            beta_0,
                            lowerbound,
                            upperbound,
                            beta_fixed,
                        )
                    }
                    vars = [Variable(name)]
                else:
                    # if non binary
                    split_points.insert(0, dataset[name].min())
                    split_points.append(dataset[name].max())
                    # we normalise to zero the first beta
                    beta_dict = {
                        f"b_{name}_{i}_0": Beta(f"b_{name}_{i}_0", 0, None, None, 1)
                    }
                    # if monotonicity constraint, we use previous beta as lower/upper bound
                    vars = []
                    for j in range(1, len(split_points) - 1):
                        beta_dict[f"b_{name}_{i}_{j}"] = (
                            Beta(
                                f"delta_{name}_{i}_{j}",
                                init_beta[j] - init_beta[j - 1],
                                lowerbound,
                                upperbound,
                                beta_fixed,
                            )
                            + beta_dict[f"b_{name}_{i}_{j-1}"]
                        )
                        if f"{name}_{i}_{j}" not in database.variables:
                            database.define_variable(
                                f"{name}_{i}_{j}",
                                (
                                    (Variable(name) - split_points[j])
                                    * (Variable(name) - split_points[j + 1])
                                )
                                <= 0,
                            )
                            variables_created[f"{name}_{i}_{j}"] = (
                                split_points[j],
                                split_points[j + 1],
                            )
                        vars.append(Variable(f"{name}_{i}_{j}"))
                for u in model.rum_structure[int(i)]["utility"]:
                    utility_spec[u] = utility_spec[u] + bioMultSum(
                        [b * v for b, v in zip(beta_dict.values(), vars)]
                    )

    availability = {i: 1 for i in range(model.num_classes)}

    if not return_utilities:
        logprob = loglogit(utility_spec, availability, Variable("choice"))
        # if dataset_test is provided, we use it to define the variables
        if dataset_test is not None:
            test_database = db.Database("rumboost_test", dataset_test)
            globals().update(test_database.variables)
            # we need to define the variables in the test database
            for var, sp in variables_created.items():
                if var not in test_database.variables:
                    test_database.define_variable(
                        var,
                        (
                            (Variable(var.split("_")[0]) - sp[0])
                            * (Variable(var.split("_")[0]) - sp[1])
                        )
                        <= 0,
                    )

            # we use the test database to create the biogeme object
            the_biogeme = BIOGEME(test_database, logprob)
        else:
            the_biogeme = BIOGEME(database, logprob)

        model_name = "assisted_model_pwlinear_lpmc"
        the_biogeme.modelName = model_name

        the_biogeme.calculateNullLoglikelihood(availability)

        return the_biogeme

    else:
        model_name = "assisted_model_utilities_pwlinear"

        utilities_expr = {str(i): utility_spec[i] for i in range(model.num_classes)}

        # if dataset_test is provided, we use it to define the variables
        if dataset_test is not None:
            test_database = db.Database("rumboost_test", dataset_test)
            globals().update(test_database.variables)
            # we need to define the variables in the test database
            for var, sp in variables_created.items():
                if var not in test_database.variables:
                    test_database.define_variable(
                        var,
                        (
                            (Variable(var.split("_")[0]) - sp[0])
                            * (Variable(var.split("_")[0]) - sp[1])
                        )
                        <= 0,
                    )

            # we use the test database to create the biogeme object
            the_biogeme = BIOGEME(test_database, utilities_expr)
        else:
            the_biogeme = BIOGEME(database, utilities_expr)
        the_biogeme.modelName = model_name

        the_biogeme.calculateNullLoglikelihood(availability)

        return the_biogeme


def estimate_dcm_with_assisted_spec(
    dataset: pd.DataFrame,
    choice: pd.Series,
    model: RUMBoost,
    dataset_name: str = "SwissMetro",
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
    dataset_name: str, optional (default="SwissMetro")
        The dataset name

    Returns
    -------
    estimated_model: biogeme.results.bioResults
    """
    the_biogeme = assist_model_spec(model, dataset, choice)

    current_directory = os.getcwd()

    os.chdir(current_directory + f"/results/{dataset_name}/assisted_specification/")

    # results = the_biogeme.estimate(recycle=True)
    results = the_biogeme.estimate()

    os.chdir(current_directory)

    return results


def predict_with_assisted_spec(
    dataset_train: pd.DataFrame,
    dataset_test: pd.DataFrame,
    choice_train: pd.Series,
    choice_test: pd.Series,
    model: RUMBoost,
    beta_values: dict,
    utilities: bool = False,
):
    """
    Predict choices with a piece-wise linear model specification based on a pre-trained rumboost model.

    Parameters
    ----------
    dataset_train: pd.DataFrame
        A dataset used for estimation
    dataset_test: pd.DataFrame
        A dataset used for prediction
    choice_train: pd.Series
        A series containing the training set choices
    choice_test: pd.Series
        A series containing the test set choices
    model: RUMBoost
        A trained rumboost model.
    beta_values: dict
        A dictionary containing the beta values of the model, estimated on the train set.
    utilities: bool, optional (default=False)
        If True, the model will return the utilities instead of the log-probs.

    Returns
    -------
    prediction_results: biogeme.results.bioResults
    """
    the_biogeme = assist_model_spec(
        model,
        dataset_train,
        choice_train,
        return_utilities=utilities,
        dataset_test=dataset_test,
        choice_test=choice_test,
    )

    prediction_results = the_biogeme.simulate(beta_values)

    return prediction_results
