import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from utils import stratified_group_k_fold


def load_preprocess_LPMC():
    '''
    Load and preprocess the LPMC dataset.

    Returns
    -------
    dataset_train : pandas Dataframe
        The training dataset ready to use.
    dataset_test : pandas Dataframe
        The training dataset ready to use.
    folds : zip(list, list)
        5 folds of indices grouped by household for CV.
    '''
    #source: https://github.com/JoseAngelMartinB/prediction-behavioural-analysis-ml-travel-mode-choice
    data_train = pd.read_csv('Data/LTDS_train.csv')
    data_test = pd.read_csv('Data/LTDS_test.csv')

    #distance in km
    data_train['distance'] = data_train['distance']/1000
    data_test['distance'] = data_test['distance']/1000
    
    #cost of driving only for people having a car
    data_train['cost_driving_total'] = data_train['cost_driving_total'] * (data_train['car_ownership'] > 0) 
    data_test['cost_driving_total'] = data_test['cost_driving_total'] * (data_test['car_ownership'] > 0)

    #for market segmentation
    # data_train['weekend'] = (data_train['day_of_week'] > 5).apply(int)  
    # data_test['weekend'] = (data_test['day_of_week'] > 5).apply(int)  

    #rename label
    label_name = {'travel_mode': 'choice'}
    dataset_train = data_train.rename(columns = label_name)
    dataset_test = data_test.rename(columns = label_name)

    #get all features
    target = 'choice'
    features = [f for f in dataset_test.columns if f != target]

    #get household ids
    hh_id = np.array(data_train['household_id'].values)

    #k folds sampled by households for cross validation
    train_idx = []
    test_idx = []
    try:
        train_idx, test_idx = pickle.load(open('strat_group_k_fold.pickle', "rb"))
    except FileNotFoundError:
        for (train_i, test_i) in stratified_group_k_fold(data_train[features], data_train['travel_mode'], hh_id, k=5):
            train_idx.append(train_i)
            test_idx.append(test_i)
        pickle.dump([train_idx, test_idx], open('strat_group_k_fold.pickle', "wb"))

    folds = zip(train_idx, test_idx)

    return dataset_train, dataset_test, folds

def load_preprocess_SwissMetro(test_size: float = 0.3, random_state: int = 42):
    '''
    Load and preprocess the SwissMetro dataset.

    Parameters
    ----------
    test_size : float, optional (default = 0.3)
        The proportion of data used for test set.
    random_state : int, optional (default = 42)
        For reproducibility in the train-test split

    Returns
    -------
    dataset_train : pandas Dataframe
        The training dataset ready to use.
    dataset_test : pandas Dataframe
        The training dataset ready to use.
    '''
    df = pd.read_csv('Data/swissmetro.dat', sep='\t')

    #remove irrelevant choices and purposes
    keep = ((df['PURPOSE']!=1)*(df['PURPOSE']!=3)+(df['CHOICE']==0)) == 0
    df = df[keep]

    #apply cost to people without GA
    df.loc[:, 'TRAIN_COST'] = df['TRAIN_CO'] * (df['GA']==0)
    df.loc[:, 'SM_COST'] = df['SM_CO'] * (df['GA']==0)

    #final dataset
    df_final = df[['TRAIN_TT', 'TRAIN_COST', 'TRAIN_HE', 'SM_TT', 'SM_COST', 'SM_HE', 'CAR_TT', 'CAR_CO', 'CHOICE']]

    #split dataset
    df_train, df_test  = train_test_split(df_final, test_size=test_size, random_state=random_state)
    
    return df_train, df_test