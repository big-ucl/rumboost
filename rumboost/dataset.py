import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from rumboost.utils import stratified_group_k_fold
import sys
sys.path.append('../')

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
    data_train = pd.read_csv('Data/LPMC_train.csv')
    data_test = pd.read_csv('Data/LPMC_test.csv')

    # data_train_2 = pd.read_csv('Data/LTDS_train.csv')
    # data_test_2 = pd.read_csv('Data/LTDS_test.csv')

    #distance in km
    data_train['distance'] = data_train['distance']/1000
    data_test['distance'] = data_test['distance']/1000
    
    # #cyclical start time
    # data_train['start_time_linear_cos'] = np.cos(data_train['start_time_linear']*(2.*np.pi/24))
    # data_train['start_time_linear_sin'] = np.sin(data_train['start_time_linear']*(2.*np.pi/24))
    # data_test['start_time_linear_cos'] = np.cos(data_test['start_time_linear']*(2.*np.pi/24))
    # data_test['start_time_linear_sin'] = np.sin(data_test['start_time_linear']*(2.*np.pi/24))

    # #cyclical travel month
    # data_train['travel_month_cos'] = np.cos(data_train_2['travel_month']*(2.*np.pi/12))
    # data_train['travel_month_sin'] = np.sin(data_train_2['travel_month']*(2.*np.pi/12))
    # data_test['travel_month_cos'] = np.cos(data_test_2['travel_month']*(2.*np.pi/12))
    # data_test['travel_month_sin'] = np.sin(data_test_2['travel_month']*(2.*np.pi/12))

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
        train_idx, test_idx = pickle.load(open('Data/strat_group_k_fold_london.pickle', "rb"))
    except FileNotFoundError:
        for (train_i, test_i) in stratified_group_k_fold(data_train[features], data_train['travel_mode'], hh_id, k=5):
            train_idx.append(train_i)
            test_idx.append(test_i)
        pickle.dump([train_idx, test_idx], open('Data/strat_group_k_fold_london.pickle', "wb"))

    folds = zip(train_idx, test_idx)

    return dataset_train, dataset_test, folds

def load_preprocess_SwissMetro(test_size: float = 0.3, random_state: int = 42, full_data=False):
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

    label_name = {'CHOICE': 'choice'}

    #remove irrelevant choices and purposes
    keep = ((df['PURPOSE']!=1)*(df['PURPOSE']!=3)+(df['CHOICE']==0)) == 0
    df = df[keep]

    #apply cost to people without GA
    df.loc[:, 'TRAIN_COST'] = df['TRAIN_CO'] * (df['GA']==0)
    df.loc[:, 'SM_COST'] = df['SM_CO'] * (df['GA']==0)

    #rescale choice from 0 to 2
    df.loc[:,'CHOICE'] = df['CHOICE'] - 1

    #age dummies
    df.loc[:, 'SEV_LUGGAGES'] = (df['LUGGAGE']==3).astype(int)

    #origin
    df.loc[:, 'ORIG_ROM'] = df['ORIGIN'].apply(lambda x: 1 if x in [10, 22, 23, 24, 25, 26] else 0)
    df.loc[:, 'ORIG_TIC'] = df['ORIGIN'].apply(lambda x: 1 if x in [21] else 0)

    #dest
    df.loc[:, 'DEST_ROM'] = df['DEST'].apply(lambda x: 1 if x in [10, 22, 23, 24, 25, 26] else 0)
    df.loc[:, 'DEST_TIC'] = df['DEST'].apply(lambda x: 1 if x in [21] else 0)

    #final dataset
    df_final = df[['ID', 'TRAIN_TT', 'TRAIN_COST', 'TRAIN_HE', 'SM_TT', 'SM_COST', 'SM_HE', 'CAR_TT', 'CAR_CO', 'MALE', 'SM_SEATS', 'SEV_LUGGAGES', 'FIRST', 'ORIG_ROM', 'ORIG_TIC', 'DEST_ROM', 'DEST_TIC', 'CHOICE']]

    df_final = df_final.rename(columns= label_name)

    if full_data:
        return df_final
    #split dataset
    df_train, df_test  = train_test_split(df_final, test_size=test_size, random_state=random_state)

    hh_id = df_train.index.tolist()

    #k folds sampled by households for cross validation
    train_idx = []
    test_idx = []
    try:
        train_idx, test_idx = pickle.load(open('Data/strat_group_k_fold_swissmetro.pickle', "rb"))
    except FileNotFoundError:
        for (train_i, test_i) in stratified_group_k_fold(df_train[['TRAIN_TT', 'TRAIN_COST', 'TRAIN_HE', 'SM_TT', 'SM_COST', 'SM_HE', 'CAR_TT', 'CAR_CO']], df_train['choice'], hh_id, k=5):
            train_idx.append(train_i)
            test_idx.append(test_i)
        pickle.dump([train_idx, test_idx], open('Data/strat_group_k_fold_swissmetro.pickle', "wb"))

    folds = zip(train_idx, test_idx)
    
    return df_train, df_test, folds

def load_preprocess_Optima():
    '''
    Load and preprocess the Optima dataset.

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
    data_train = pd.read_csv('Data/optima_ext_train.csv')
    data_test = pd.read_csv('Data/optima_ext_test.csv')

    #get household ids
    hh_id = np.array(data_train['ID'].values)

    #rename label and drop IDs
    label_name = {'Choice': 'choice'}
    data_train = data_train.rename(columns = label_name)
    data_test = data_test.rename(columns = label_name)
    dataset_train = data_train.drop('ID', axis=1)
    dataset_test = data_test.drop('ID', axis=1)

    #get all features
    target = 'choice'
    features = [f for f in dataset_train.columns if f != target]

    #k folds sampled by households for cross validation
    train_idx = []
    test_idx = []
    try:
        train_idx, test_idx = pickle.load(open('Data/strat_group_k_fold_optima.pickle', "rb"))
    except FileNotFoundError:
        for (train_i, test_i) in stratified_group_k_fold(dataset_train[features], dataset_train[target], hh_id, k=5):
            train_idx.append(train_i)
            test_idx.append(test_i)
        pickle.dump([train_idx, test_idx], open('Data/strat_group_k_fold_optima.pickle', "wb"))

    folds = zip(train_idx, test_idx)

    return dataset_train, dataset_test, folds

def load_preprocess_Netherlands(test_size: float = 0.3, random_state: int = 42):

    pandas = pd.read_table("Data/netherlands.dat")

    pandas_rp = pandas[pandas['rp']==1]

    pandas_rp.loc[:, 'rail_time'] = pandas_rp.loc[:,'rail_ivtt'] + pandas_rp.loc[:,'rp_rail_ovt']
    pandas_rp.loc[:, 'car_time'] = pandas_rp.loc[:,'car_ivtt'] + pandas_rp.loc[:,'rp_car_ovt']
    pandas_rp.loc[:, 'car_cost_euro'] = pandas_rp.loc[:,'car_cost'] * 0.44378022
    pandas_rp.loc[:, 'rail_cost_euro'] = pandas_rp.loc[:,'rail_cost'] * 0.44378022
    
    pandas_rp = pandas_rp.drop(['rp', 'sp', 'rail_comfort', 'rail_ivtt', 'rail_cost', 'rail_acc_time', 'rail_egr_time', 'rail_transfers', 'car_ivtt', 'car_cost', 'car_walk_time', 'rp_choice', 'rp_rail_ovt', 'rp_car_ovt'], axis = 1)
    #database = db.Database("netherlands",pandas)
    df_train, df_test = train_test_split(pandas_rp, test_size=test_size, random_state = random_state)

    #get all features
    target = 'choice'
    features = [f for f in df_train.columns if f != target]

    #get household ids
    hh_id = np.array(df_train['id'].values)

    #k folds sampled by households for cross validation
    train_idx = []
    test_idx = []
    try:
        train_idx, test_idx = pickle.load(open('Data/strat_group_k_fold_netherlands.pickle', "rb"))
    except FileNotFoundError:
        for (train_i, test_i) in stratified_group_k_fold(df_train[features], df_train[target], hh_id, k=5):
            train_idx.append(train_i)
            test_idx.append(test_i)
        pickle.dump([train_idx, test_idx], open('Data/strat_group_k_fold_netherlands.pickle', "wb"))

    folds = zip(train_idx, test_idx)

    return df_train, df_test, folds

def load_preprocess_Airplane(test_size: float = 0.3, random_state: int = 42):

    pandas = pd.read_table("Data/airline.dat")
    pandas['choice'] = (pandas['BestAlternative_2'] == 1) + 2*(pandas['BestAlternative_3'] == 1)
    pandas = pandas.drop(['BestAlternative_1', 'BestAlternative_2', 'BestAlternative_3'], axis=1)
    pandas.loc[:, 'Fare_1_scaled'] = pandas['Fare_1'] / 100
    pandas.loc[:, 'Fare_2_scaled'] = pandas['Fare_2'] / 100
    pandas.loc[:, 'Fare_3_scaled'] = pandas['Fare_3'] / 100
    pandas.loc[:, 'TTDIFF_TRANSFER'] = pandas['TripTimeHours_2'] - pandas['TripTimeHours_1']
    pandas.loc[:, 'TTDIFF_TRANSFER_TWOAIRLINES'] = pandas['TripTimeHours_3'] - pandas['TripTimeHours_1']
    df = pandas[['DepartureTimeHours_1','DepartureTimeHours_2','DepartureTimeHours_3', 'ArrivalTimeHours_1','ArrivalTimeHours_2','ArrivalTimeHours_3','TTDIFF_TRANSFER', 'TTDIFF_TRANSFER_TWOAIRLINES', 'Legroom_1', 'Legroom_2','Legroom_3','Fare_1_scaled','Fare_2_scaled','Fare_3_scaled', 'choice']]
    #database = db.Database("netherlands",pandas)
    df_train, df_test = train_test_split(df, test_size=test_size, random_state = random_state)

    #get all features
    target = 'choice'
    features = [f for f in df_train.columns if f != target]

    #get household ids
    hh_id = df_train.index.tolist()

    #k folds sampled by households for cross validation
    train_idx = []
    test_idx = []
    try:
        train_idx, test_idx = pickle.load(open('Data/strat_group_k_fold_airplane.pickle', "rb"))
    except FileNotFoundError:
        for (train_i, test_i) in stratified_group_k_fold(df_train[features], df_train[target], hh_id, k=5):
            train_idx.append(train_i)
            test_idx.append(test_i)
        pickle.dump([train_idx, test_idx], open('Data/strat_group_k_fold_airplane.pickle', "wb"))

    folds = zip(train_idx, test_idx)

    return df_train, df_test, folds

def load_preprocess_Telephone(test_size: float = 0.3, random_state: int = 3):

    pandas = pd.read_table("Data/telephone.dat")
    pandas.loc[:,'choice'] = pandas['choice'] - 1

    pandas.loc[:, 'cost1_scaled'] = pandas['cost1'] / 10
    pandas.loc[:, 'cost2_scaled'] = pandas['cost2'] / 10
    pandas.loc[:, 'cost3_scaled'] = pandas['cost3'] / 10
    pandas.loc[:, 'cost4_scaled'] = pandas['cost4'] / 10
    pandas.loc[:, 'cost5_scaled'] = pandas['cost5'] / 10
    #database = db.Database("netherlands",pandas)
    df_train, df_test = train_test_split(pandas, test_size=test_size, random_state = random_state)

    #get all features
    target = 'choice'
    features = [f for f in df_train.columns if f != target]

    #get household ids
    hh_id = df_train.index.tolist()

    #k folds sampled by households for cross validation
    train_idx = []
    test_idx = []
    try:
        train_idx, test_idx = pickle.load(open('Data/strat_group_k_fold_telephone.pickle', "rb"))
    except FileNotFoundError:
        for (train_i, test_i) in stratified_group_k_fold(df_train[features], df_train[target], hh_id, k=5):
            train_idx.append(train_i)
            test_idx.append(test_i)
        pickle.dump([train_idx, test_idx], open('Data/strat_group_k_fold_telephone.pickle', "wb"))

    folds = zip(train_idx, test_idx)

    return df_train, df_test, folds

def load_preprocess_Parking(test_size: float = 0.3, random_state: int = 42):

    pandas = pd.read_table("Data/parking.dat")
    pandas.loc[:,'CHOICE'] = pandas['CHOICE'] - 1
    pandas = pandas.drop(['ID', 'OBSID', 'SCENARIO'], axis=1)
    label_name = {'CHOICE': 'choice'}
    pandas = pandas.rename(columns= label_name)
    #database = db.Database("netherlands",pandas)
    df_train, df_test = train_test_split(pandas, test_size=test_size, random_state = random_state)

    #get all features
    target = 'choice'
    features = [f for f in df_train.columns if f != target]

    #get household ids
    hh_id = df_train.index.tolist()

    #k folds sampled by households for cross validation
    train_idx = []
    test_idx = []
    try:
        train_idx, test_idx = pickle.load(open('Data/strat_group_k_fold_parking.pickle', "rb"))
    except FileNotFoundError:
        for (train_i, test_i) in stratified_group_k_fold(df_train[features], df_train[target], hh_id, k=5):
            train_idx.append(train_i)
            test_idx.append(test_i)
        pickle.dump([train_idx, test_idx], open('Data/strat_group_k_fold_parking.pickle', "wb"))

    folds = zip(train_idx, test_idx)

    return df_train, df_test, folds

def load_preprocess_Vaccines():

    pandas = pd.read_csv("Data/vaccinechoiceMar12.csv")
    #pandas.drop()
    pandas.loc[:,'choice'] = pandas['vaccinechoice'] - 1
    new_names = {'cost.1':'cost1','effectiveness.1':'effectiveness1','protection.1':'protection1','incubation.1':'incubation1','severe.1':'severe1','mild.1':'mild1','doses.1':'doses1','booster.1':'booster1','USA.1':'USA1','UK.1':'UK1','Germany.1':'Germany1','China.1':'China1','Russia.1':'Russia1','media.1':'media1','CDC.1':'CDC1','WHO.1':'WHO1','months.1':'months1','cost.3':'cost3','effectiveness.3':'effectiveness3','protection.3':'protection3','incubation.3':'incubation3','severe.3':'severe3','mild.3':'mild3','doses.3':'doses3','booster.3':'booster3','USA.3':'USA3','UK.3':'UK3','Germany.3':'Germany3','China.3':'China3','Russia.3':'Russia3','media.3':'media3','CDC.3':'CDC3','WHO.3':'WHO3','months.3':'months3'}
    pandas = pandas.rename(columns=new_names) 
	
    pandas_cleaned = pandas.drop(['ID', 'ZIP', 'state'], axis=1)
    #pandas_cleaned = pandas[['IDnum','choice','cost1','effectiveness1','protection1','incubation1','severe1','mild1','doses1','booster1','USA1','UK1','Germany1','China1','Russia1','media1','CDC1','WHO1','months1','cost3','effectiveness3','protection3','incubation3','severe3','mild3','doses3','booster3','USA3','UK3','Germany3','China3','Russia3','media3','CDC3','WHO3','Male','Black','Democrat','covidpos','FluShot','babyboomolder','HHInc10K','BSc','PostGrad','Underlying','Wave4']]
    #pandas_cl_sampled = pandas_cleaned.groupby('IDnum').sample(n=1, random_state=2)

    df_train = pandas_cleaned[pandas_cleaned['Wave4']!=1]
    df_test = pandas_cleaned[pandas_cleaned['Wave4']==1]

    #get all features
    target = 'choice'
    features = [f for f in df_train.columns if f != target]

    #get household ids
    hh_id = df_train['IDnum']

    #drop irrelevant features
    # df_train = df_train.drop(['IDnum', 'Wave4'], axis=1)
    # df_test = df_test.drop(['IDnum', 'Wave4'], axis=1)

    #get all features
    target = 'choice'
    features = [f for f in df_train.columns if f != target]

    #k folds sampled by households for cross validation
    train_idx = []
    test_idx = []
    try:
        train_idx, test_idx = pickle.load(open('Data/strat_group_k_fold_vaccine.pickle', "rb"))
    except FileNotFoundError:
        for (train_i, test_i) in stratified_group_k_fold(df_train[features], df_train[target], hh_id, k=5):
            train_idx.append(train_i)
            test_idx.append(test_i)
        pickle.dump([train_idx, test_idx], open('Data/strat_group_k_fold_vaccine.pickle', "wb"))

    folds = zip(train_idx, test_idx)

    return df_train, df_test, folds
