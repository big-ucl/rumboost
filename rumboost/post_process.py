import pandas as pd
import numpy as np
import lightgbm as lgb
from rumboost.rumboost import RUMBoost, rum_train

def split_fe_model(model: RUMBoost):
    '''
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
    '''
    if not isinstance(model.rum_structure, list):
        raise ValueError('Please add a rum_structure to your model by setting model.rum_structure. A rum_structure must be a list of 2*n_alt dictionaries in this function')

    attributes_model = RUMBoost()
    socio_economic_model = RUMBoost()

    attributes_model.boosters = [b for i, b in enumerate(model.boosters) if i%2 == 0]
    attributes_model.rum_structure = model.rum_structure[::2]

    socio_economic_model.boosters = [b for i, b in enumerate(model.boosters) if i%2 == 1]
    socio_economic_model.rum_structure = model.rum_structure[1::2]

    return attributes_model, socio_economic_model

def bootstrap(dataset: pd.DataFrame, params: dict, rum_structure: list[dict], num_it: int = 100, seed: int = 42):
    '''
    Performs bootstrapping, with given dataset, parameters and rum_structure. For now, only a basic rumboost can be used.

    Parameters
    ----------
    dataset: pd.DataFrame
        A dataset used to train RUMBoost
    params: dict
        A dictionary used to train RUMBoost
    rum_structure: list[dict]
        A list of dictionaries used to specify the structure of RUMBoost
    num_it: int, optional (default=100)
        The number of bootstrapping iterations
    seed: int, optional (default=42)
        The seed used to randomly sample the dataset.

    Returns
    -------
    models: list
        Return a list containing all trained models.
    '''
    np.random.seed(seed)

    N = dataset.shape[0]
    models = []
    for _ in range(num_it):
        ids = np.random.choice(dataset.index, size=N, replace=True)
        ids2 = np.setdiff1d(dataset.index, ids)

        df_train = dataset.loc[ids]
        df_test = dataset.loc[ids2]
        
        dataset_train = lgb.Dataset(df_train.drop('choice', axis=1), label=df_train.choice, free_raw_data=False)
        
        valid_set = lgb.Dataset(df_test.drop('choice', axis=1), label=df_test.choice, free_raw_data=False)

        models.append(rum_train(params, dataset_train, rum_structure, valid_sets=[valid_set]))

    return models