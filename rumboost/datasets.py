import numpy as np
import pandas as pd
import pickle
import random
import sys
import gc

from lightgbm import Dataset
from collections import Counter, defaultdict

try:
    from sklearn.model_selection import train_test_split, GroupShuffleSplit, GroupKFold

    sklearn_installed = True
except ImportError:
    sklearn_installed = False

sys.path.append("../")


def load_preprocess_LPMC(
    path="/media/nicolas-salvade/Windows/Users/DAF1/OneDrive - University College London/Documents/PhD - UCL/rumboost/Data/",
):
    """
    Load and preprocess the LPMC dataset.

    Returns
    -------
    dataset_train : pandas Dataframe
        The training dataset ready to use.
    dataset_test : pandas Dataframe
        The training dataset ready to use.
    folds : zip(list, list)
        5 folds of indices grouped by household for CV.
    """
    # source: https://github.com/JoseAngelMartinB/prediction-behavioural-analysis-ml-travel-mode-choice
    data_train = pd.read_csv(path + "LPMC_train.csv")
    data_test = pd.read_csv(path + "LPMC_test.csv")

    # data_train_2 = pd.read_csv('Data/LTDS_train.csv')
    # data_test_2 = pd.read_csv('Data/LTDS_test.csv')

    # distance in km
    data_train["distance"] = data_train["distance"] / 1000
    data_test["distance"] = data_test["distance"] / 1000

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

    # for market segmentation
    # data_train['weekend'] = (data_train['day_of_week'] > 5).apply(int)
    # data_test['weekend'] = (data_test['day_of_week'] > 5).apply(int)

    # rename label
    label_name = {"travel_mode": "choice"}
    dataset_train = data_train.rename(columns=label_name)
    dataset_test = data_test.rename(columns=label_name)

    # get all features
    target = "choice"
    features = [f for f in dataset_test.columns if f != target]

    # get household ids
    hh_id = np.array(data_train["household_id"].values)

    # k folds sampled by households for cross validation
    train_idx = []
    test_idx = []
    try:
        train_idx, test_idx = pickle.load(
            open(
                path + "strat_group_k_fold_london.pickle",
                "rb",
            )
        )
    except FileNotFoundError:
        for train_i, test_i in stratified_group_k_fold(
            data_train[features], data_train["travel_mode"], hh_id, k=5
        ):
            train_idx.append(train_i)
            test_idx.append(test_i)
        pickle.dump(
            [train_idx, test_idx],
            open(
                path + "strat_group_k_fold_london.pickle",
                "wb",
            ),
        )

    folds = zip(train_idx, test_idx)

    return dataset_train, dataset_test, folds


def load_preprocess_SwissMetro(
    test_size: float = 0.3, random_state: int = 42, full_data=False
):
    """
    Load and preprocess the SwissMetro dataset. See Biogeme website for data.

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
    """
    if not sklearn_installed:
        raise ImportError("scikit-learn is required for this function.")
    df = pd.read_csv("../Data/swissmetro.dat", sep="\t")

    label_name = {"CHOICE": "choice"}

    # remove irrelevant choices and purposes
    keep = ((df["PURPOSE"] != 1) * (df["PURPOSE"] != 3) + (df["CHOICE"] == 0)) == 0
    df = df[keep]

    # apply cost to people without GA
    df.loc[:, "TRAIN_COST"] = df["TRAIN_CO"] * (df["GA"] == 0)
    df.loc[:, "SM_COST"] = df["SM_CO"] * (df["GA"] == 0)

    # rescale choice from 0 to 2
    df.loc[:, "CHOICE"] = df["CHOICE"] - 1

    # age dummies
    df.loc[:, "SEV_LUGGAGES"] = (df["LUGGAGE"] == 3).astype(int)

    # origin
    df.loc[:, "ORIG_ROM"] = df["ORIGIN"].apply(
        lambda x: 1 if x in [10, 22, 23, 24, 25, 26] else 0
    )
    df.loc[:, "ORIG_TIC"] = df["ORIGIN"].apply(lambda x: 1 if x in [21] else 0)

    # dest
    df.loc[:, "DEST_ROM"] = df["DEST"].apply(
        lambda x: 1 if x in [10, 22, 23, 24, 25, 26] else 0
    )
    df.loc[:, "DEST_TIC"] = df["DEST"].apply(lambda x: 1 if x in [21] else 0)

    # final dataset
    df_final = df[
        [
            "ID",
            "TRAIN_TT",
            "TRAIN_COST",
            "TRAIN_HE",
            "SM_TT",
            "SM_COST",
            "SM_HE",
            "CAR_TT",
            "CAR_CO",
            "MALE",
            "SM_SEATS",
            "SEV_LUGGAGES",
            "FIRST",
            "ORIG_ROM",
            "ORIG_TIC",
            "DEST_ROM",
            "DEST_TIC",
            "CHOICE",
        ]
    ]

    df_final = df_final.rename(columns=label_name)

    if full_data:
        return df_final
    # split dataset
    df_train, df_test = train_test_split(
        df_final, test_size=test_size, random_state=random_state
    )

    hh_id = df_train.index.tolist()

    # k folds sampled by households for cross validation
    train_idx = []
    test_idx = []
    try:
        train_idx, test_idx = pickle.load(
            open("../Data/strat_group_k_fold_swissmetro.pickle", "rb")
        )
    except FileNotFoundError:
        for train_i, test_i in stratified_group_k_fold(
            df_train[
                [
                    "TRAIN_TT",
                    "TRAIN_COST",
                    "TRAIN_HE",
                    "SM_TT",
                    "SM_COST",
                    "SM_HE",
                    "CAR_TT",
                    "CAR_CO",
                ]
            ],
            df_train["choice"],
            hh_id,
            k=5,
        ):
            train_idx.append(train_i)
            test_idx.append(test_i)
        pickle.dump(
            [train_idx, test_idx],
            open("../Data/strat_group_k_fold_swissmetro.pickle", "wb"),
        )

    folds = zip(train_idx, test_idx)

    return df_train, df_test, folds


def load_preprocess_Optima():
    """
    Load and preprocess the Optima dataset. See Biogeme website for data.

    Returns
    -------
    dataset_train : pandas Dataframe
        The training dataset ready to use.
    dataset_test : pandas Dataframe
        The training dataset ready to use.
    folds : zip(list, list)
        5 folds of indices grouped by household for CV.
    """
    if not sklearn_installed:
        raise ImportError("scikit-learn is required for this function.")
    # source: https://github.com/JoseAngelMartinB/prediction-behavioural-analysis-ml-travel-mode-choice
    data_train = pd.read_csv("../Data/optima_ext_train.csv")
    data_test = pd.read_csv("../Data/optima_ext_test.csv")

    # get household ids
    hh_id = np.array(data_train["ID"].values)

    # rename label and drop IDs
    label_name = {"Choice": "choice"}
    data_train = data_train.rename(columns=label_name)
    data_test = data_test.rename(columns=label_name)
    dataset_train = data_train.drop("ID", axis=1)
    dataset_test = data_test.drop("ID", axis=1)

    # get all features
    target = "choice"
    features = [f for f in dataset_train.columns if f != target]

    # k folds sampled by households for cross validation
    train_idx = []
    test_idx = []
    try:
        train_idx, test_idx = pickle.load(
            open("../Data/strat_group_k_fold_optima.pickle", "rb")
        )
    except FileNotFoundError:
        for train_i, test_i in stratified_group_k_fold(
            dataset_train[features], dataset_train[target], hh_id, k=5
        ):
            train_idx.append(train_i)
            test_idx.append(test_i)
        pickle.dump(
            [train_idx, test_idx],
            open("../Data/strat_group_k_fold_optima.pickle", "wb"),
        )

    folds = zip(train_idx, test_idx)

    return dataset_train, dataset_test, folds


def load_preprocess_Netherlands(test_size: float = 0.3, random_state: int = 42):
    """Load and preprocess the Netherlands dataset. See Biogeme website for data."""
    if not sklearn_installed:
        raise ImportError("scikit-learn is required for this function.")
    pandas = pd.read_table("../Data/netherlands.dat")

    pandas_rp = pandas[pandas["rp"] == 1]

    pandas_rp.loc[:, "rail_time"] = (
        pandas_rp.loc[:, "rail_ivtt"] + pandas_rp.loc[:, "rp_rail_ovt"]
    )
    pandas_rp.loc[:, "car_time"] = (
        pandas_rp.loc[:, "car_ivtt"] + pandas_rp.loc[:, "rp_car_ovt"]
    )
    pandas_rp.loc[:, "car_cost_euro"] = pandas_rp.loc[:, "car_cost"] * 0.44378022
    pandas_rp.loc[:, "rail_cost_euro"] = pandas_rp.loc[:, "rail_cost"] * 0.44378022

    pandas_rp = pandas_rp.drop(
        [
            "rp",
            "sp",
            "rail_comfort",
            "rail_ivtt",
            "rail_cost",
            "rail_acc_time",
            "rail_egr_time",
            "rail_transfers",
            "car_ivtt",
            "car_cost",
            "car_walk_time",
            "rp_choice",
            "rp_rail_ovt",
            "rp_car_ovt",
        ],
        axis=1,
    )
    # database = db.Database("netherlands",pandas)
    df_train, df_test = train_test_split(
        pandas_rp, test_size=test_size, random_state=random_state
    )

    # get all features
    target = "choice"
    features = [f for f in df_train.columns if f != target]

    # get household ids
    hh_id = np.array(df_train["id"].values)

    # k folds sampled by households for cross validation
    train_idx = []
    test_idx = []
    try:
        train_idx, test_idx = pickle.load(
            open("../Data/strat_group_k_fold_netherlands.pickle", "rb")
        )
    except FileNotFoundError:
        for train_i, test_i in stratified_group_k_fold(
            df_train[features], df_train[target], hh_id, k=5
        ):
            train_idx.append(train_i)
            test_idx.append(test_i)
        pickle.dump(
            [train_idx, test_idx],
            open("../Data/strat_group_k_fold_netherlands.pickle", "wb"),
        )

    folds = zip(train_idx, test_idx)

    return df_train, df_test, folds


def load_preprocess_Airplane(test_size: float = 0.3, random_state: int = 42):
    """Load and preprocess the Airplane dataset. See Biogeme website for data."""
    if not sklearn_installed:
        raise ImportError("scikit-learn is required for this function.")
    pandas = pd.read_table("Data/airline.dat")
    pandas["choice"] = (pandas["BestAlternative_2"] == 1) + 2 * (
        pandas["BestAlternative_3"] == 1
    )
    pandas = pandas.drop(
        ["BestAlternative_1", "BestAlternative_2", "BestAlternative_3"], axis=1
    )
    pandas.loc[:, "Fare_1_scaled"] = pandas["Fare_1"] / 100
    pandas.loc[:, "Fare_2_scaled"] = pandas["Fare_2"] / 100
    pandas.loc[:, "Fare_3_scaled"] = pandas["Fare_3"] / 100
    pandas.loc[:, "TTDIFF_TRANSFER"] = (
        pandas["TripTimeHours_2"] - pandas["TripTimeHours_1"]
    )
    pandas.loc[:, "TTDIFF_TRANSFER_TWOAIRLINES"] = (
        pandas["TripTimeHours_3"] - pandas["TripTimeHours_1"]
    )
    df = pandas[
        [
            "DepartureTimeHours_1",
            "DepartureTimeHours_2",
            "DepartureTimeHours_3",
            "ArrivalTimeHours_1",
            "ArrivalTimeHours_2",
            "ArrivalTimeHours_3",
            "TTDIFF_TRANSFER",
            "TTDIFF_TRANSFER_TWOAIRLINES",
            "Legroom_1",
            "Legroom_2",
            "Legroom_3",
            "Fare_1_scaled",
            "Fare_2_scaled",
            "Fare_3_scaled",
            "choice",
        ]
    ]
    # database = db.Database("netherlands",pandas)
    df_train, df_test = train_test_split(
        df, test_size=test_size, random_state=random_state
    )

    # get all features
    target = "choice"
    features = [f for f in df_train.columns if f != target]

    # get household ids
    hh_id = df_train.index.tolist()

    # k folds sampled by households for cross validation
    train_idx = []
    test_idx = []
    try:
        train_idx, test_idx = pickle.load(
            open("../Data/strat_group_k_fold_airplane.pickle", "rb")
        )
    except FileNotFoundError:
        for train_i, test_i in stratified_group_k_fold(
            df_train[features], df_train[target], hh_id, k=5
        ):
            train_idx.append(train_i)
            test_idx.append(test_i)
        pickle.dump(
            [train_idx, test_idx],
            open("../Data/strat_group_k_fold_airplane.pickle", "wb"),
        )

    folds = zip(train_idx, test_idx)

    return df_train, df_test, folds


def load_preprocess_Telephone(test_size: float = 0.3, random_state: int = 3):
    """Load and preprocess the Telephone dataset. See Biogeme website for data."""
    if not sklearn_installed:
        raise ImportError("scikit-learn is required for this function.")
    pandas = pd.read_table("Data/telephone.dat")
    pandas.loc[:, "choice"] = pandas["choice"] - 1

    pandas.loc[:, "cost1_scaled"] = pandas["cost1"] / 10
    pandas.loc[:, "cost2_scaled"] = pandas["cost2"] / 10
    pandas.loc[:, "cost3_scaled"] = pandas["cost3"] / 10
    pandas.loc[:, "cost4_scaled"] = pandas["cost4"] / 10
    pandas.loc[:, "cost5_scaled"] = pandas["cost5"] / 10
    # database = db.Database("netherlands",pandas)
    df_train, df_test = train_test_split(
        pandas, test_size=test_size, random_state=random_state
    )

    # get all features
    target = "choice"
    features = [f for f in df_train.columns if f != target]

    # get household ids
    hh_id = df_train.index.tolist()

    # k folds sampled by households for cross validation
    train_idx = []
    test_idx = []
    try:
        train_idx, test_idx = pickle.load(
            open("../Data/strat_group_k_fold_telephone.pickle", "rb")
        )
    except FileNotFoundError:
        for train_i, test_i in stratified_group_k_fold(
            df_train[features], df_train[target], hh_id, k=5
        ):
            train_idx.append(train_i)
            test_idx.append(test_i)
        pickle.dump(
            [train_idx, test_idx],
            open("../Data/strat_group_k_fold_telephone.pickle", "wb"),
        )

    folds = zip(train_idx, test_idx)

    return df_train, df_test, folds


def load_preprocess_Parking(test_size: float = 0.3, random_state: int = 42):
    """Load and preprocess the Parking dataset. See Biogeme website for data."""
    if not sklearn_installed:
        raise ImportError("scikit-learn is required for this function.")
    pandas = pd.read_table("Data/parking.dat")
    pandas.loc[:, "CHOICE"] = pandas["CHOICE"] - 1
    pandas = pandas.drop(["ID", "OBSID", "SCENARIO"], axis=1)
    label_name = {"CHOICE": "choice"}
    pandas = pandas.rename(columns=label_name)
    # database = db.Database("netherlands",pandas)
    df_train, df_test = train_test_split(
        pandas, test_size=test_size, random_state=random_state
    )

    # get all features
    target = "choice"
    features = [f for f in df_train.columns if f != target]

    # get household ids
    hh_id = df_train.index.tolist()

    # k folds sampled by households for cross validation
    train_idx = []
    test_idx = []
    try:
        train_idx, test_idx = pickle.load(
            open("../Data/strat_group_k_fold_parking.pickle", "rb")
        )
    except FileNotFoundError:
        for train_i, test_i in stratified_group_k_fold(
            df_train[features], df_train[target], hh_id, k=5
        ):
            train_idx.append(train_i)
            test_idx.append(test_i)
        pickle.dump(
            [train_idx, test_idx],
            open("../Data/strat_group_k_fold_parking.pickle", "wb"),
        )

    folds = zip(train_idx, test_idx)

    return df_train, df_test, folds


def load_preprocess_Vaccines():
    """Load and preprocess the Vaccines dataset."""

    pandas = pd.read_csv("../Data/vaccinechoiceMar12.csv")
    # pandas.drop()
    pandas.loc[:, "choice"] = pandas["vaccinechoice"] - 1
    new_names = {
        "cost.1": "cost1",
        "effectiveness.1": "effectiveness1",
        "protection.1": "protection1",
        "incubation.1": "incubation1",
        "severe.1": "severe1",
        "mild.1": "mild1",
        "doses.1": "doses1",
        "booster.1": "booster1",
        "USA.1": "USA1",
        "UK.1": "UK1",
        "Germany.1": "Germany1",
        "China.1": "China1",
        "Russia.1": "Russia1",
        "media.1": "media1",
        "CDC.1": "CDC1",
        "WHO.1": "WHO1",
        "months.1": "months1",
        "cost.3": "cost3",
        "effectiveness.3": "effectiveness3",
        "protection.3": "protection3",
        "incubation.3": "incubation3",
        "severe.3": "severe3",
        "mild.3": "mild3",
        "doses.3": "doses3",
        "booster.3": "booster3",
        "USA.3": "USA3",
        "UK.3": "UK3",
        "Germany.3": "Germany3",
        "China.3": "China3",
        "Russia.3": "Russia3",
        "media.3": "media3",
        "CDC.3": "CDC3",
        "WHO.3": "WHO3",
        "months.3": "months3",
    }
    pandas = pandas.rename(columns=new_names)

    pandas_cleaned = pandas.drop(["ID", "ZIP", "state"], axis=1)
    # pandas_cleaned = pandas[['IDnum','choice','cost1','effectiveness1','protection1','incubation1','severe1','mild1','doses1','booster1','USA1','UK1','Germany1','China1','Russia1','media1','CDC1','WHO1','months1','cost3','effectiveness3','protection3','incubation3','severe3','mild3','doses3','booster3','USA3','UK3','Germany3','China3','Russia3','media3','CDC3','WHO3','Male','Black','Democrat','covidpos','FluShot','babyboomolder','HHInc10K','BSc','PostGrad','Underlying','Wave4']]
    # pandas_cl_sampled = pandas_cleaned.groupby('IDnum').sample(n=1, random_state=2)

    df_train = pandas_cleaned[pandas_cleaned["Wave4"] != 1]
    df_test = pandas_cleaned[pandas_cleaned["Wave4"] == 1]

    # get all features
    target = "choice"
    features = [f for f in df_train.columns if f != target]

    # get household ids
    hh_id = df_train["IDnum"]

    # drop irrelevant features
    # df_train = df_train.drop(['IDnum', 'Wave4'], axis=1)
    # df_test = df_test.drop(['IDnum', 'Wave4'], axis=1)

    # get all features
    target = "choice"
    features = [f for f in df_train.columns if f != target]

    # k folds sampled by households for cross validation
    train_idx = []
    test_idx = []
    try:
        train_idx, test_idx = pickle.load(
            open("../Data/strat_group_k_fold_vaccine.pickle", "rb")
        )
    except FileNotFoundError:
        for train_i, test_i in stratified_group_k_fold(
            df_train[features], df_train[target], hh_id, k=5
        ):
            train_idx.append(train_i)
            test_idx.append(test_i)
        pickle.dump(
            [train_idx, test_idx],
            open("../Data/strat_group_k_fold_vaccine.pickle", "wb"),
        )

    folds = zip(train_idx, test_idx)

    return df_train, df_test, folds


def load_preprocess_MTMC(
    test_size: float = 0.2,
    random_state: int = 1,
    path="/media/nicolas-salvade/Windows/Users/DAF1/OneDrive - University College London/Documents/PhD - UCL/rumboost/Data/",
):
    """
    Load and preprocess the MTMC dataset.
    """
    if not sklearn_installed:
        raise ImportError("scikit-learn is required for this function.")
    # load data
    data = pd.read_csv(path + "data_laus_trips_prep_attractions_allalt.csv")

    # load destination zones
    z_idx = list(np.loadtxt(path + "z_idx.csv"))

    # split by household
    gsp = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(gsp.split(data, groups=data["HHNR"]))
    df_train, df_test = data.iloc[train_idx], data.iloc[test_idx]

    # get all features
    target = "choice"
    features = [f for f in df_train.columns if f != target]

    hh_id = df_train["HHNR"]

    # k folds sampled by households for cross validation
    train_idx = []
    test_idx = []
    try:
        train_idx, test_idx = pickle.load(
            open(
                path + "strat_group_k_fold_mtmc.pickle",
                "rb",
            )
        )
    except FileNotFoundError:
        gkf = GroupKFold()
        for train_i, test_i in gkf.split(df_train[features], df_train[target], hh_id):
            train_idx.append(train_i)
            test_idx.append(test_i)
        pickle.dump(
            [train_idx, test_idx],
            open(
                path + "strat_group_k_fold_mtmc.pickle",
                "wb",
            ),
        )

    folds = zip(train_idx, test_idx)

    return df_train, df_test, folds, z_idx


def load_preprocess_MTMC_all(
    test_size: float = 0.2,
    random_state: int = 1,
    path="/media/nicolas-salvade/Windows/Users/DAF1/OneDrive - University College London/Documents/PhD - UCL/rumboost/Data/",
):
    """
    Load and preprocess the MTMC dataset for all swiss zones.
    """
    if not sklearn_installed:
        raise ImportError("scikit-learn is required for this function.")
    try:
        z_idx = list(np.loadtxt(path + "z_idx_all_wo_alps.csv"))
        with open(
            path + "train_set_switzerland.pkl",
            "rb",
        ) as f:
            df_train = pickle.load(f)
        with open(
            path + "test_set_switzerland.pkl",
            "rb",
        ) as f:
            df_test = pickle.load(f)
    except FileNotFoundError:
        # load data
        with open(
            path
            + "/media/nicolas-salvade/Windows/Users/DAF1/OneDrive - University College London/Documents/PhD - UCL/choice_set_location_travelmode/Data/input/data_switzerland_trips_preprocessed.pkl",
            "rb",
        ) as f:
            data = pickle.load(f)

        # load destination zones
        z_idx = list(np.loadtxt(path + "z_idx_all_wo_alps.csv"))

        zone_to_drop = [i for i in range(7965, 7978)]
        mask = ~data["d_idx"].isin(zone_to_drop) & ~data["o_idx"].isin(zone_to_drop)
        data = data[mask]

        # split by household
        gsp = GroupShuffleSplit(
            n_splits=1, test_size=test_size, random_state=random_state
        )
        train_idx, test_idx = next(gsp.split(data, groups=data["HHNR"]))
        df_train, df_test = data.iloc[train_idx], data.iloc[test_idx]
        pickle.dump(
            df_train,
            open(
                path + "train_set_switzerland.pkl",
                "wb",
            ),
        )
        pickle.dump(
            df_test,
            open(
                path + "test_set_switzerland.pkl",
                "wb",
            ),
        )

    # try:
    #     with open(
    #         path + "strat_group_k_fold_mtmc_all.pickle",
    #         "rb",
    #     ) as f:
    #         train_idx, test_idx = pickle.load(f)
    # except FileNotFoundError:
    #     # get all features
    #     target = "choice"
    #     features = [f for f in df_train.columns if f != target]

    #     hh_id = df_train["HHNR"]

    #     # k folds sampled by households for cross validation
    #     train_idx = []
    #     test_idx = []
    #     gkf = GroupKFold()
    #     for train_i, test_i in gkf.split(df_train[features], df_train[target], hh_id):
    #         train_idx.append(train_i)
    #         test_idx.append(test_i)
    #     pickle.dump(
    #         [train_idx, test_idx],
    #         open(
    #             path + "strat_group_k_fold_mtmc_all.pickle",
    #             "wb",
    #         ),
    #     )
    #     pickle.dump(
    #         df_train,
    #         open(
    #             path + "train_set_switzerland.pkl",
    #             "wb",
    #         ),
    #     )
    #     pickle.dump(
    #         df_test,
    #         open(
    #             path + "test_set_switzerland.pkl",
    #             "wb",
    #         ),
    #     )

    # folds = zip(train_idx, test_idx)

    folds = None

    return df_train, df_test, folds, z_idx


# Sample a dataset grouped by `groups` and stratified by `y`
# Source: https://www.kaggle.com/jakubwasikowski/stratified-group-k-fold-cross-validation
def stratified_group_k_fold(X, y, groups, k, seed=None):
    """
    Stratified Group K-Fold cross-validator
    Provides train/test indices to split data in train/test sets.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        The input samples.
    y : array-like of shape (n_samples,)
        The target values.
    groups : array-like of shape (n_samples,)
        Group labels for the samples used while splitting the dataset into train/test set.
    k : int
        Number of folds. Must be at least 2.
    seed : int, optional
        Random seed for shuffling the data.

    Yields
    ------
    train : ndarray
        The training set indices for that split.
    test : ndarray
        The testing set indices for that split.
    """
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
            label_std = np.std(
                [y_counts_per_fold[i][label] / y_distr[label] for i in range(k)]
            )
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


def prepare_dataset(
    rum_structure,
    df_train,
    num_classes,
    df_test=None,
    target="choice",
    free_raw_data=False,
    save_dataset=None,
    load_dataset=None,
):
    """
    Prepare and save if required the datasets for RUMBoost.

    Parameters
    ----------
    rum_structure : list of dict
        The structure of the RUM model.
    df_train : pandas DataFrame
        The training dataset.
    params : dict
        The parameters of the model.
    num_classes : int
        The number of classes.
    df_test : list of pandas DataFrame, optional
        The list of test datasets.
    target : str, optional
        The target variable.
    free_raw_data : bool, optional
        If the raw data should be freed.
    save_dataset : str, optional
        The path to save the datasets.
    load_dataset : str, optional
        The path to load the datasets.

    Returns
    -------
    train_sets : dict
        The training datasets.
    valid_sets : dict
        The validation datasets.
    """
    valid_sets = {}
    num_datasets = len(rum_structure)

    if load_dataset:
        try:
            with open(f"{load_dataset}_train_sets.pkl", "rb") as f:
                train_sets = pickle.load(f)
            if df_test is not None:
                with open(f"{load_dataset}_valid_sets.pkl", "rb") as f:
                    valid_sets = pickle.load(f)
        except:
            raise FileNotFoundError(
                "Error loading datasets, try running again this function without the load_dataset parameter."
            )
        train_set_J = []
        reduced_valid_sets_J = []
        try:
            for j, _ in enumerate(rum_structure):
                print(
                    "-" * 30
                    + "\n"
                    + f"[{j+1}/{num_datasets}] \t Loading dataset {j+1}..."
                )
                train_set_J.append(Dataset(data=f"{load_dataset}_train_set_{j}.bin", free_raw_data=False))
                if df_test is not None:
                    reduced_valid_sets_j = []
                    for i, _ in enumerate(df_test):
                        reduced_valid_sets_j.append(
                            Dataset(data=f"{load_dataset}_valid_set_{j}_{i}.bin", free_raw_data=False, reference=train_set_J[j])
                        )
                    reduced_valid_sets_J.append(reduced_valid_sets_j)
                print("\t done! \n" + "-" * 30 + "\n")
        except:
            raise FileNotFoundError(
                "Error loading dataset, try running again this function without the load_dataset parameter."
            )

        train_sets["train_sets"] = train_set_J
        if df_test is not None:
            valid_sets["valid_sets"] = np.array(reduced_valid_sets_J).T.tolist()

        return train_sets, valid_sets

    labels = df_train[target].to_numpy().astype(np.int32)
    labels_j = (labels[:, None] == np.array(range(num_classes))[None, :]).astype(
        np.int8
    )

    num_obs = df_train.shape[0]
    if df_test is not None:
        valid_labels = []
        num_obs_test = []
        for df in df_test:
            valid_labels += [df[target].to_numpy().astype(np.int32)]
            num_obs_test += [df.shape[0]]

    val_labels_j = [
        (val_labs[:, None] == np.array(range(num_classes))[None, :]).astype(np.int8)
        for val_labs in valid_labels
    ]

    train_set_J = []
    reduced_valid_sets_J = []
    for j, struct in enumerate(rum_structure):
        print("-" * 30 + "\n" + f"[{j+1}/{num_datasets}] \t Loading dataset {j+1}...")
        if struct:
            if "variables" in struct:
                train_set_j_data = df_train[
                    struct["variables"]
                ]  # only relevant features for the jth booster
                
                if struct['shared']==True:
                    new_label = labels_j[:, struct["utility"][:len(struct['variables'])]].reshape(
                        -1, order="F"
                    )
                else:
                    new_label = labels_j[:, 0].reshape(
                        -1, order="F"
                    )
                train_set_j = Dataset(
                    train_set_j_data.values.reshape(
                        (len(new_label), -1), order="A"
                    ),
                    label=new_label,
                    free_raw_data=free_raw_data,
                )  # create and build dataset
                categorical_feature = struct['boosting_params'].get("categorical_feature", "auto")
                train_set_j._update_params(
                    struct["boosting_params"]
                )._set_predictor(None).set_feature_name(
                    "auto"
                ).set_categorical_feature(
                    categorical_feature
                )

                if df_test is not None:
                    reduced_valid_sets_j = []
                    for i, valid_set in enumerate(df_test):
                        # create and build validation sets
                        valid_set_j_data = valid_set[
                            struct["variables"]
                        ]  # only relevant features for the jth booster

                        if struct['shared']==True:
                            label_valid = val_labels_j[i][:, struct["utility"][:len(struct['variables'])]].reshape(
                                -1, order="F"
                            )
                        else:
                            label_valid = val_labels_j[i][:, 0].reshape(
                                -1, order="F"
                            )
                        valid_set_j = Dataset(
                            valid_set_j_data.values.reshape(
                                (len(label_valid), -1),
                                order="A",
                            ),
                            label=label_valid,
                            free_raw_data=False,
                            reference=train_set_j,
                        )  # create and build dataset
                        valid_set_j._update_params(struct["boosting_params"])

                        reduced_valid_sets_j.append(valid_set_j)

                        if save_dataset:
                            valid_set_j.save_binary(
                                f"{save_dataset}_valid_set_{j}_{i}.bin"
                            )

                train_set_J.append(train_set_j)
                if save_dataset:
                    train_set_j.save_binary(f"{save_dataset}_train_set_{j}.bin")
                if df_test is not None:
                    reduced_valid_sets_J.append(reduced_valid_sets_j)
                del (
                    train_set_j_data,
                    valid_set_j_data,
                    train_set_j,
                    valid_set_j,
                )
                gc.collect()
                print("\t done! \n" + "-" * 30 + "\n")

            else:
                # if no alternative specific datasets
                new_label = np.where(labels == j, 1, 0)
                train_set_j = Dataset(
                    df_train.values, label=new_label, free_raw_data=False
                )
                if df_test is not None:
                    reduced_valid_sets_j = df_test[:]

    train_sets = {"num_data": num_obs, "labels": labels, "labels_j": labels_j}
    if df_test is not None:
        valid_sets = {
            "num_data": num_obs_test,
            "valid_labels": valid_labels,
        }

    if save_dataset:
        with open(f"{save_dataset}_train_sets.pkl", "wb") as f:
            pickle.dump(train_sets, f)
        if df_test is not None:
            with open(f"{save_dataset}_valid_sets.pkl", "wb") as f:
                pickle.dump(valid_sets, f)

    train_sets["train_sets"] = train_set_J
    if df_test is not None:
        reduced_valid_sets_J = np.array(reduced_valid_sets_J).T.tolist()
        valid_sets["valid_sets"] = reduced_valid_sets_J

    return train_sets, valid_sets
