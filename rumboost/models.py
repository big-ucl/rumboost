import biogeme.database as db
import biogeme.biogeme as bio
from biogeme.expressions import Beta
from biogeme.models import loglogit, logit, lognested, nested, logcnl_avail, cnl_avail
import biogeme.logging as blog
import pandas as pd


def SwissMetro(dataset_train: pd.DataFrame, for_prob=False):
    """
    Create a MNL on the swissmetro dataset.

    Parameters
    ----------
    dataset_train : pandas DataFrame
        The training dataset.

    Returns
    -------
    biogeme : bio.BIOGEME
        The BIOGEME object containing the model.
    """
    database_train = db.Database("swissmetro_train", dataset_train)

    globals().update(database_train.variables)

    # parameters to be estimated
    ASC_CAR = Beta("ASC_CAR", 0, None, None, 0)
    ASC_SM = Beta("ASC_SM", 0, None, None, 1)
    ASC_TRAIN = Beta("ASC_TRAIN", 0, None, None, 0)

    B_COST = Beta("B_COST", 0, None, 0, 0)
    B_TIME = Beta("B_TIME", 0, None, 0, 0)
    B_HE = Beta("B_HE", 0, None, 0, 0)

    B_TIME_CAR = Beta("B_TIME_CAR", 0, None, 0, 0)
    B_COST_CAR = Beta("B_COST_CAR", 0, None, 0, 0)
    B_TIME_RAIL = Beta("B_TIME_RAIL", 0, None, 0, 0)
    B_COST_RAIL = Beta("B_COST_RAIL", 0, None, 0, 0)
    B_HE_RAIL = Beta("B_HE_RAIL", 0, None, 0, 0)
    B_TIME_SM = Beta("B_TIME_SM", 0, None, 0, 0)
    B_COST_SM = Beta("B_COST_SM", 0, None, 0, 0)
    B_HE_SM = Beta("B_HE_SM", 0, None, 0, 0)

    B_FIRST = Beta("B_FIRST", 0, None, None, 0)
    B_MALE = Beta("B_MALE", 0, None, None, 0)
    B_SM_SEATS = Beta("B_SM_SEATS", 0, None, None, 0)
    B_SEV_LUGGAGES = Beta("B_SEV_LUGGAGES", 0, None, None, 0)
    B_ORIG_ROM = Beta("B_ORIG_ROM", 0, None, None, 0)
    B_ORIG_TIC = Beta("B_ORIG_TIC", 0, None, None, 0)
    B_DEST_ROM = Beta("B_DEST_ROM", 0, None, None, 0)
    B_DEST_TIC = Beta("B_DEST_TIC", 0, None, None, 0)

    # utilities
    V_TRAIN = (
        ASC_TRAIN
        + B_TIME * TRAIN_TT
        + B_COST * TRAIN_COST
        + B_HE * TRAIN_HE
        + B_FIRST * FIRST
        + B_MALE * MALE
        + B_SEV_LUGGAGES * SEV_LUGGAGES
        + B_ORIG_ROM * ORIG_ROM
        + B_ORIG_TIC * ORIG_TIC
        + B_DEST_ROM * DEST_ROM
        + B_DEST_TIC * DEST_TIC
    )
    V_SM = (
        ASC_SM
        + B_TIME * SM_TT
        + B_COST * SM_COST
        + B_HE * SM_HE
        + B_MALE * MALE
        + B_SEV_LUGGAGES * SEV_LUGGAGES
        + B_ORIG_ROM * ORIG_ROM
        + B_ORIG_TIC * ORIG_TIC
        + B_DEST_ROM * DEST_ROM
        + B_DEST_TIC * DEST_TIC
    )  # + B_FIRST * FIRST
    V_CAR = (
        ASC_CAR
        + B_TIME * CAR_TT
        + B_COST * CAR_CO
        + B_MALE * MALE
        + B_SEV_LUGGAGES * SEV_LUGGAGES
        + B_ORIG_ROM * ORIG_ROM
        + B_ORIG_TIC * ORIG_TIC
        + B_DEST_ROM * DEST_ROM
        + B_DEST_TIC * DEST_TIC
    )

    # simple model
    # V_TRAIN = ASC_TRAIN + B_TIME * TRAIN_TT + B_COST * TRAIN_COST + B_HE * TRAIN_HE
    # V_SM    = ASC_SM    + B_TIME * SM_TT    + B_COST * SM_COST    + B_HE * SM_HE
    # V_CAR   = ASC_CAR   + B_TIME * CAR_TT   + B_COST * CAR_CO

    V = {0: V_TRAIN, 1: V_SM, 2: V_CAR}
    av = {0: 1, 1: 1, 2: 1}

    # choice model
    logprob = loglogit(V, av, choice)
    biogeme = bio.BIOGEME(database_train, logprob)
    biogeme.modelName = "SwissmetroMNL"

    biogeme.generate_html = False
    biogeme.generate_pickle = False

    if for_prob:
        prob_train = logit(V, av, 0)
        prob_SM = logit(V, av, 1)
        prob_car = logit(V, av, 2)

        simulate = {
            "Prob. train": prob_train,
            "Prob. SM": prob_SM,
            "Prob. car": prob_car,
        }
        biosim = bio.BIOGEME(database_train, simulate)
        biosim.modelName = "swissmetro_logit_test"

        biosim.generate_html = False
        biosim.generate_pickle = False

        return biosim

    return biogeme


def SwissMetro_normalised(dataset_train: pd.DataFrame, for_prob=False):
    """
    Create a MNL on the swissmetro dataset, normalised for biogeme estimation.

    Parameters
    ----------
    dataset_train : pandas DataFrame
        The training dataset.

    Returns
    -------
    biogeme : bio.BIOGEME
        The BIOGEME object containing the model.
    """
    database_train = db.Database("swissmetro_train", dataset_train)

    globals().update(database_train.variables)

    # parameters to be estimated
    ASC_CAR = Beta("ASC_CAR", 0, None, None, 0)
    ASC_SM = Beta("ASC_SM", 0, None, None, 1)
    ASC_TRAIN = Beta("ASC_TRAIN", 0, None, None, 0)

    B_COST = Beta("B_COST", 0, None, 0, 0)
    B_TIME = Beta("B_TIME", 0, None, 0, 0)
    B_HE = Beta("B_HE", 0, None, 0, 0)

    B_TIME_CAR = Beta("B_TIME_CAR", 0, None, 0, 0)
    B_COST_CAR = Beta("B_COST_CAR", 0, None, 0, 0)
    B_TIME_RAIL = Beta("B_TIME_RAIL", 0, None, 0, 0)
    B_COST_RAIL = Beta("B_COST_RAIL", 0, None, 0, 0)
    B_HE_RAIL = Beta("B_HE_RAIL", 0, None, 0, 0)
    B_TIME_SM = Beta("B_TIME_SM", 0, None, 0, 0)
    B_COST_SM = Beta("B_COST_SM", 0, None, 0, 0)
    B_HE_SM = Beta("B_HE_SM", 0, None, 0, 0)

    B_FIRST = Beta("B_FIRST", 0, None, None, 0)
    B_MALE = Beta("B_MALE", 0, None, None, 0)
    B_SM_SEATS = Beta("B_SM_SEATS", 0, None, None, 0)
    B_SEV_LUGGAGES = Beta("B_SEV_LUGGAGES", 0, None, None, 0)
    B_ORIG_ROM = Beta("B_ORIG_ROM", 0, None, None, 0)
    B_ORIG_TIC = Beta("B_ORIG_TIC", 0, None, None, 0)
    B_DEST_ROM = Beta("B_DEST_ROM", 0, None, None, 0)
    B_DEST_TIC = Beta("B_DEST_TIC", 0, None, None, 0)

    # utilities
    V_TRAIN = (
        ASC_TRAIN
        + B_TIME * TRAIN_TT
        + B_COST * TRAIN_COST
        + B_HE * TRAIN_HE
        + B_FIRST * FIRST
        + B_MALE * MALE
        + B_SEV_LUGGAGES * SEV_LUGGAGES
        + B_ORIG_ROM * ORIG_ROM
        + B_ORIG_TIC * ORIG_TIC
        + B_DEST_ROM * DEST_ROM
        + B_DEST_TIC * DEST_TIC
    )
    V_SM = (
        ASC_SM
        + B_TIME * SM_TT
        + B_COST * SM_COST
        + B_HE * SM_HE
        + B_MALE * MALE
        + B_SEV_LUGGAGES * SEV_LUGGAGES
        + B_ORIG_ROM * ORIG_ROM
        + B_ORIG_TIC * ORIG_TIC
        + B_DEST_ROM * DEST_ROM
        + B_DEST_TIC * DEST_TIC
    )  # + B_FIRST * FIRST
    V_CAR = (
        ASC_CAR + B_TIME * CAR_TT + B_COST * CAR_CO
    )  # + B_MALE * MALE   + B_SEV_LUGGAGES * SEV_LUGGAGES + B_ORIG_ROM * ORIG_ROM + B_ORIG_TIC * ORIG_TIC + B_DEST_ROM * DEST_ROM + B_DEST_TIC * DEST_TIC

    # simple model
    # V_TRAIN = ASC_TRAIN + B_TIME * TRAIN_TT + B_COST * TRAIN_COST + B_HE * TRAIN_HE
    # V_SM    = ASC_SM    + B_TIME * SM_TT    + B_COST * SM_COST    + B_HE * SM_HE
    # V_CAR   = ASC_CAR   + B_TIME * CAR_TT   + B_COST * CAR_CO

    V = {0: V_TRAIN, 1: V_SM, 2: V_CAR}
    av = {0: 1, 1: 1, 2: 1}

    # choice model
    logprob = loglogit(V, av, choice)
    biogeme = bio.BIOGEME(database_train, logprob)
    biogeme.modelName = "SwissmetroMNL"

    biogeme.generate_html = False
    biogeme.generate_pickle = False

    if for_prob:
        prob_train = logit(V, av, 0)
        prob_SM = logit(V, av, 1)
        prob_car = logit(V, av, 2)

        simulate = {
            "Prob. train": prob_train,
            "Prob. SM": prob_SM,
            "Prob. car": prob_car,
        }
        biosim = bio.BIOGEME(database_train, simulate)
        biosim.modelName = "swissmetro_logit_test"

        biosim.generate_html = False
        biosim.generate_pickle = False

        return biosim

    return biogeme


def SwissMetro_nested(dataset_train: pd.DataFrame, for_prob=False):
    """
    Create a nested logit model on the swissmetro dataset.

    Parameters
    ----------
    dataset_train : pandas DataFrame
        The training dataset.

    Returns
    -------
    biogeme : bio.BIOGEME
        The BIOGEME object containing the model.
    """
    database_train = db.Database("swissmetro_train", dataset_train)

    globals().update(database_train.variables)
    # Parameters to be estimated
    MU = Beta("MU", 1, 1, 10, 0)
    # parameters to be estimated
    ASC_CAR = Beta("ASC_CAR", 0, None, None, 0)
    ASC_SM = Beta("ASC_SM", 0, None, None, 1)
    ASC_TRAIN = Beta("ASC_TRAIN", 0, None, None, 0)

    B_COST = Beta("B_COST", 0, None, 0, 0)
    B_TIME = Beta("B_TIME", 0, None, 0, 0)
    B_HE = Beta("B_HE", 0, None, 0, 0)

    B_TIME_CAR = Beta("B_TIME_CAR", 0, None, 0, 0)
    B_COST_CAR = Beta("B_COST_CAR", 0, None, 0, 0)
    B_TIME_RAIL = Beta("B_TIME_RAIL", 0, None, 0, 0)
    B_COST_RAIL = Beta("B_COST_RAIL", 0, None, 0, 0)
    B_HE_RAIL = Beta("B_HE_RAIL", 0, None, 0, 0)
    B_TIME_SM = Beta("B_TIME_SM", 0, None, 0, 0)
    B_COST_SM = Beta("B_COST_SM", 0, None, 0, 0)
    B_HE_SM = Beta("B_HE_SM", 0, None, 0, 0)

    B_FIRST = Beta("B_FIRST", 0, None, None, 0)
    B_MALE = Beta("B_MALE", 0, None, None, 0)
    B_SM_SEATS = Beta("B_SM_SEATS", 0, None, None, 0)
    B_SEV_LUGGAGES = Beta("B_SEV_LUGGAGES", 0, None, None, 0)
    B_ORIG_ROM = Beta("B_ORIG_ROM", 0, None, None, 0)
    B_ORIG_TIC = Beta("B_ORIG_TIC", 0, None, None, 0)
    B_DEST_ROM = Beta("B_DEST_ROM", 0, None, None, 0)
    B_DEST_TIC = Beta("B_DEST_TIC", 0, None, None, 0)

    # utilities
    V_TRAIN = (
        ASC_TRAIN
        + B_TIME * TRAIN_TT
        + B_COST * TRAIN_COST
        + B_HE * TRAIN_HE
        + B_FIRST * FIRST
        + B_MALE * MALE
        + B_SEV_LUGGAGES * SEV_LUGGAGES
        + B_ORIG_ROM * ORIG_ROM
        + B_ORIG_TIC * ORIG_TIC
        + B_DEST_ROM * DEST_ROM
        + B_DEST_TIC * DEST_TIC
    )
    V_SM = (
        ASC_SM
        + B_TIME * SM_TT
        + B_COST * SM_COST
        + B_HE * SM_HE
        + B_MALE * MALE
        + B_SEV_LUGGAGES * SEV_LUGGAGES
        + B_ORIG_ROM * ORIG_ROM
        + B_ORIG_TIC * ORIG_TIC
        + B_DEST_ROM * DEST_ROM
        + B_DEST_TIC * DEST_TIC
    )  # + B_FIRST * FIRST
    V_CAR = (
        ASC_CAR
        + B_TIME * CAR_TT
        + B_COST * CAR_CO
        + B_MALE * MALE
        + B_SEV_LUGGAGES * SEV_LUGGAGES
        + B_ORIG_ROM * ORIG_ROM
        + B_ORIG_TIC * ORIG_TIC
        + B_DEST_ROM * DEST_ROM
        + B_DEST_TIC * DEST_TIC
    )

    V = {0: V_TRAIN, 1: V_SM, 2: V_CAR}
    av = {0: 1, 1: 1, 2: 1}

    # Definition of nests:
    # 1: nests parameter
    # 2: list of alternatives
    existing = MU, [0, 1]
    future = 1.0, [2]
    nests = existing, future

    # Definition of the model. This is the contribution of each
    # observation to the log likelihood function.
    # The choice model is a nested logit, with availability conditions
    logprob = lognested(V, av, nests, choice)

    # Create the Biogeme object
    the_biogeme = bio.BIOGEME(database_train, logprob)
    the_biogeme.modelName = "b09nested"

    the_biogeme.generate_html = False
    the_biogeme.generate_pickle = False

    if for_prob:
        prob_train = nested(V, av, nests, 0)
        prob_SM = nested(V, av, nests, 1)
        prob_car = nested(V, av, nests, 2)

        simulate = {
            "Prob. SM": prob_SM,
            "Prob. train": prob_train,
            "Prob. car": prob_car,
        }
        biosim = bio.BIOGEME(database_train, simulate)
        biosim.modelName = "swissmetro_logit_test"

        biosim.generate_html = False
        biosim.generate_pickle = False

        return biosim

    return the_biogeme


def SwissMetro_MNL(dataset_train: pd.DataFrame, for_prob=False):
    """
    Create a simple MNL on the swissmetro dataset.

    Parameters
    ----------
    dataset_train : pandas DataFrame
        The training dataset.

    Returns
    -------
    biogeme : bio.BIOGEME
        The BIOGEME object containing the model.
    """
    database_train = db.Database("swissmetro_train", dataset_train)

    globals().update(database_train.variables)
    # Parameters to be estimated
    ASC_CAR = Beta("ASC_CAR", 0, None, None, 0)
    ASC_TRAIN = Beta("ASC_TRAIN", 0, None, None, 0)
    ASC_SM = Beta("ASC_SM", 0, None, None, 1)
    B_TIME = Beta("B_TIME", 0, None, None, 0)
    B_COST = Beta("B_COST", 0, None, None, 0)

    # Definition of the utility functions
    V1 = ASC_TRAIN + B_TIME * TRAIN_TT + B_COST * TRAIN_COST
    V2 = ASC_SM + B_TIME * SM_TT + B_COST * SM_COST
    V3 = ASC_CAR + B_TIME * CAR_TT + B_COST * CAR_CO

    # Associate utility functions with the numbering of alternatives
    V = {0: V1, 1: V2, 2: V3}

    # Associate the availability conditions with the alternatives
    av = {0: 1, 1: 1, 2: 1}

    logprob = loglogit(V, av, choice)
    biogeme = bio.BIOGEME(database_train, logprob)
    biogeme.modelName = "SwissmetroMNL"
    biogeme.generate_html = False
    biogeme.generate_pickle = False

    if for_prob:
        prob_train = logit(V, av, 0)
        prob_SM = logit(V, av, 1)
        prob_car = logit(V, av, 2)

        simulate = {
            "Prob. SM": prob_SM,
            "Prob. train": prob_train,
            "Prob. car": prob_car,
        }
        biosim = bio.BIOGEME(database_train, simulate)
        biosim.modelName = "swissmetro_logit_test"

        biosim.generate_html = False
        biosim.generate_pickle = False

        return biosim

    return biogeme


def LPMC(dataset_train, for_prob=False):
    """
    Create a MNL on the LPMC dataset.
    The model is a slightly modified version from teh code that can be found here: https://github.com/JoseAngelMartinB/prediction-behavioural-analysis-ml-travel-mode-choice.

    Parameters
    ----------
    dataset_train : pandas DataFrame
        The training dataset.

    Returns
    -------
    biogeme : bio.BIOGEME
        The BIOGEME object containing the model.

    """
    database_train = db.Database("LTDS_train", dataset_train)

    globals().update(database_train.variables)

    # several model specifications are available below - the best one is the uncommented one.

    # best model until now, 0.6790 with lr = 0.3
    # MNL_beta_params_positive = ['B_car_ownership_Car', 'B_driving_license_Car']
    # MNL_beta_params_negative = ['B_dur_walking_Walk',  'B_dur_cycling_Bike', 'B_dur_pt_access_Public_Transport', 'B_dur_pt_rail_Public_Transport', 'B_dur_pt_bus_Public_Transport', 'B_dur_pt_int_waiting_Public_Transport', 'B_dur_pt_int_walking_Public_Transport', 'B_pt_n_interchanges_Public_Transport', 'B_cost_transit_Public_Transport', 'B_dur_driving_Car', 'B_cost_driving_total_Car', 'B_distance_Walk', 'B_distance_Bike', 'B_distance_Public_Transport', 'B_distance_Car']#, 'B_traffic_perc_Car']
    # MNL_beta_params_neutral = ['ASC_Bike', 'ASC_Public_Transport', 'ASC_Car', 'B_car_ownership_Walk', 'B_car_ownership_Bike', 'B_car_ownership_Public_Transport', 'B_driving_license_Walk', 'B_driving_license_Bike', 'B_driving_license_Public_Transport', 'B_age_Walk', 'B_age_Bike', 'B_age_Public_Transport', 'B_age_Car', 'B_female_Walk', 'B_female_Bike', 'B_female_Public_Transport', 'B_female_Car', 'B_day_of_week_Walk', 'B_day_of_week_Bike', 'B_day_of_week_Public_Transport', 'B_day_of_week_Car', 'B_start_time_linear_Walk', 'B_start_time_linear_Bike', 'B_start_time_linear_Public_Transport', 'B_start_time_linear_Car', 'B_purpose_B_Walk', 'B_purpose_B_Bike', 'B_purpose_B_Public_Transport', 'B_purpose_B_Car', 'B_purpose_HBE_Walk', 'B_purpose_HBE_Bike', 'B_purpose_HBE_Public_Transport', 'B_purpose_HBE_Car', 'B_purpose_HBO_Walk', 'B_purpose_HBO_Bike', 'B_purpose_HBO_Public_Transport', 'B_purpose_HBO_Car', 'B_purpose_HBW_Walk', 'B_purpose_HBW_Bike', 'B_purpose_HBW_Public_Transport', 'B_purpose_HBW_Car', 'B_purpose_NHBO_Walk', 'B_purpose_NHBO_Bike', 'B_purpose_NHBO_Public_Transport', 'B_purpose_NHBO_Car', 'B_fueltype_Avrg_Walk', 'B_fueltype_Avrg_Bike', 'B_fueltype_Avrg_Public_Transport', 'B_fueltype_Avrg_Car', 'B_fueltype_Diesel_Walk', 'B_fueltype_Diesel_Bike', 'B_fueltype_Diesel_Public_Transport', 'B_fueltype_Diesel_Car', 'B_fueltype_Hybrid_Walk', 'B_fueltype_Hybrid_Bike', 'B_fueltype_Hybrid_Public_Transport', 'B_fueltype_Hybrid_Car', 'B_fueltype_Petrol_Walk', 'B_fueltype_Petrol_Bike', 'B_fueltype_Petrol_Public_Transport', 'B_fueltype_Petrol_Car']

    # MNL_utilities = {0: 'B_age_Walk*age + B_female_Walk*female + B_day_of_week_Walk*day_of_week + B_start_time_linear_Walk*start_time_linear + B_car_ownership_Walk*car_ownership + B_driving_license_Walk*driving_license + B_purpose_B_Walk*purpose_B + B_purpose_HBE_Walk*purpose_HBE + B_purpose_HBO_Walk*purpose_HBO + B_purpose_HBW_Walk*purpose_HBW + B_purpose_NHBO_Walk*purpose_NHBO + B_fueltype_Avrg_Walk*fueltype_Average + B_fueltype_Diesel_Walk*fueltype_Diesel + B_fueltype_Hybrid_Walk*fueltype_Hybrid + B_fueltype_Petrol_Walk*fueltype_Petrol + B_distance_Walk*distance + B_dur_walking_Walk*dur_walking',
    #                     1: 'ASC_Bike + B_age_Bike*age + B_female_Bike*female + B_day_of_week_Bike*day_of_week + B_start_time_linear_Bike*start_time_linear + B_car_ownership_Bike*car_ownership + B_driving_license_Bike*driving_license + B_purpose_B_Bike*purpose_B + B_purpose_HBE_Bike*purpose_HBE + B_purpose_HBO_Bike*purpose_HBO + B_purpose_HBW_Bike*purpose_HBW + B_purpose_NHBO_Bike*purpose_NHBO + B_fueltype_Avrg_Bike*fueltype_Average + B_fueltype_Diesel_Bike*fueltype_Diesel + B_fueltype_Hybrid_Bike*fueltype_Hybrid + B_fueltype_Petrol_Bike*fueltype_Petrol + B_distance_Bike*distance + B_dur_cycling_Bike*dur_cycling',
    #                     2: 'ASC_Public_Transport + B_age_Public_Transport*age + B_female_Public_Transport*female + B_day_of_week_Public_Transport*day_of_week + B_start_time_linear_Public_Transport*start_time_linear + B_car_ownership_Public_Transport*car_ownership + B_driving_license_Public_Transport*driving_license + B_purpose_B_Public_Transport*purpose_B + B_purpose_HBE_Public_Transport*purpose_HBE + B_purpose_HBO_Public_Transport*purpose_HBO + B_purpose_HBW_Public_Transport*purpose_HBW + B_purpose_NHBO_Public_Transport*purpose_NHBO + B_fueltype_Avrg_Public_Transport*fueltype_Average + B_fueltype_Diesel_Public_Transport*fueltype_Diesel + B_fueltype_Hybrid_Public_Transport*fueltype_Hybrid + B_fueltype_Petrol_Public_Transport*fueltype_Petrol + B_distance_Public_Transport*distance + B_dur_pt_access_Public_Transport*dur_pt_access + B_dur_pt_rail_Public_Transport*dur_pt_rail + B_dur_pt_bus_Public_Transport*dur_pt_bus + B_dur_pt_int_waiting_Public_Transport*dur_pt_int_waiting + B_dur_pt_int_walking_Public_Transport*dur_pt_int_walking + B_pt_n_interchanges_Public_Transport*pt_n_interchanges + B_cost_transit_Public_Transport*cost_transit',
    #                     3: 'ASC_Car + B_age_Car*age + B_female_Car*female + B_day_of_week_Car*day_of_week + B_start_time_linear_Car*start_time_linear + B_car_ownership_Car*car_ownership + B_driving_license_Car*driving_license + B_purpose_B_Car*purpose_B + B_purpose_HBE_Car*purpose_HBE + B_purpose_HBO_Car*purpose_HBO + B_purpose_HBW_Car*purpose_HBW + B_purpose_NHBO_Car*purpose_NHBO + B_fueltype_Avrg_Car*fueltype_Average + B_fueltype_Diesel_Car*fueltype_Diesel + B_fueltype_Hybrid_Car*fueltype_Hybrid + B_fueltype_Petrol_Car*fueltype_Petrol + B_distance_Car*distance + B_dur_driving_Car*dur_driving + B_cost_driving_total_Car*cost_driving_total'}

    # driving_percentage, congestion charge as a binary variable NEW PB 0.6730 with lr = 0.1
    MNL_beta_params_positive = ["B_car_ownership_Car", "B_driving_license_Car"]
    MNL_beta_params_negative = [
        "B_dur_walking_Walk",
        "B_dur_cycling_Bike",
        "B_dur_pt_access_Public_Transport",
        "B_dur_pt_rail_Public_Transport",
        "B_dur_pt_bus_Public_Transport",
        "B_dur_pt_int_waiting_Public_Transport",
        "B_dur_pt_int_walking_Public_Transport",
        "B_pt_n_interchanges_Public_Transport",
        "B_cost_transit_Public_Transport",
        "B_dur_driving_Car",
        "B_cost_driving_fuel_Car",
        "B_distance_Walk",
        "B_distance_Bike",
        "B_distance_Public_Transport",
        "B_distance_Car",
        "B_con_charge_Car",
        "B_traffic_perc_Car",
    ]
    MNL_beta_params_neutral = [
        "ASC_Bike",
        "ASC_Public_Transport",
        "ASC_Car",
        "B_car_ownership_Walk",
        "B_car_ownership_Bike",
        "B_car_ownership_Public_Transport",
        "B_driving_license_Walk",
        "B_driving_license_Bike",
        "B_driving_license_Public_Transport",
        "B_age_Walk",
        "B_age_Bike",
        "B_age_Public_Transport",
        "B_age_Car",
        "B_female_Walk",
        "B_female_Bike",
        "B_female_Public_Transport",
        "B_female_Car",
        "B_day_of_week_Walk",
        "B_day_of_week_Bike",
        "B_day_of_week_Public_Transport",
        "B_day_of_week_Car",
        "B_start_time_linear_Walk",
        "B_start_time_linear_Bike",
        "B_start_time_linear_Public_Transport",
        "B_start_time_linear_Car",
        "B_purpose_B_Walk",
        "B_purpose_B_Bike",
        "B_purpose_B_Public_Transport",
        "B_purpose_B_Car",
        "B_purpose_HBE_Walk",
        "B_purpose_HBE_Bike",
        "B_purpose_HBE_Public_Transport",
        "B_purpose_HBE_Car",
        "B_purpose_HBO_Walk",
        "B_purpose_HBO_Bike",
        "B_purpose_HBO_Public_Transport",
        "B_purpose_HBO_Car",
        "B_purpose_HBW_Walk",
        "B_purpose_HBW_Bike",
        "B_purpose_HBW_Public_Transport",
        "B_purpose_HBW_Car",
        "B_purpose_NHBO_Walk",
        "B_purpose_NHBO_Bike",
        "B_purpose_NHBO_Public_Transport",
        "B_purpose_NHBO_Car",
        "B_fueltype_Avrg_Walk",
        "B_fueltype_Avrg_Bike",
        "B_fueltype_Avrg_Public_Transport",
        "B_fueltype_Avrg_Car",
        "B_fueltype_Diesel_Walk",
        "B_fueltype_Diesel_Bike",
        "B_fueltype_Diesel_Public_Transport",
        "B_fueltype_Diesel_Car",
        "B_fueltype_Hybrid_Walk",
        "B_fueltype_Hybrid_Bike",
        "B_fueltype_Hybrid_Public_Transport",
        "B_fueltype_Hybrid_Car",
        "B_fueltype_Petrol_Walk",
        "B_fueltype_Petrol_Bike",
        "B_fueltype_Petrol_Public_Transport",
        "B_fueltype_Petrol_Car",
    ]

    MNL_utilities = {
        0: "B_age_Walk*age + B_female_Walk*female + B_day_of_week_Walk*day_of_week + B_start_time_linear_Walk*start_time_linear + B_car_ownership_Walk*car_ownership + B_driving_license_Walk*driving_license + B_purpose_B_Walk*purpose_B + B_purpose_HBE_Walk*purpose_HBE + B_purpose_HBO_Walk*purpose_HBO + B_purpose_HBW_Walk*purpose_HBW + B_purpose_NHBO_Walk*purpose_NHBO + B_fueltype_Avrg_Walk*fueltype_Average + B_fueltype_Diesel_Walk*fueltype_Diesel + B_fueltype_Hybrid_Walk*fueltype_Hybrid + B_fueltype_Petrol_Walk*fueltype_Petrol + B_distance_Walk*distance + B_dur_walking_Walk*dur_walking",
        1: "ASC_Bike + B_age_Bike*age + B_female_Bike*female + B_day_of_week_Bike*day_of_week + B_start_time_linear_Bike*start_time_linear + B_car_ownership_Bike*car_ownership + B_driving_license_Bike*driving_license + B_purpose_B_Bike*purpose_B + B_purpose_HBE_Bike*purpose_HBE + B_purpose_HBO_Bike*purpose_HBO + B_purpose_HBW_Bike*purpose_HBW + B_purpose_NHBO_Bike*purpose_NHBO + B_fueltype_Avrg_Bike*fueltype_Average + B_fueltype_Diesel_Bike*fueltype_Diesel + B_fueltype_Hybrid_Bike*fueltype_Hybrid + B_fueltype_Petrol_Bike*fueltype_Petrol + B_distance_Bike*distance + B_dur_cycling_Bike*dur_cycling",
        2: "ASC_Public_Transport + B_age_Public_Transport*age + B_female_Public_Transport*female + B_day_of_week_Public_Transport*day_of_week + B_start_time_linear_Public_Transport*start_time_linear + B_car_ownership_Public_Transport*car_ownership + B_driving_license_Public_Transport*driving_license + B_purpose_B_Public_Transport*purpose_B + B_purpose_HBE_Public_Transport*purpose_HBE + B_purpose_HBO_Public_Transport*purpose_HBO + B_purpose_HBW_Public_Transport*purpose_HBW + B_purpose_NHBO_Public_Transport*purpose_NHBO + B_fueltype_Avrg_Public_Transport*fueltype_Average + B_fueltype_Diesel_Public_Transport*fueltype_Diesel + B_fueltype_Hybrid_Public_Transport*fueltype_Hybrid + B_fueltype_Petrol_Public_Transport*fueltype_Petrol + B_distance_Public_Transport*distance + B_dur_pt_access_Public_Transport*dur_pt_access + B_dur_pt_rail_Public_Transport*dur_pt_rail + B_dur_pt_bus_Public_Transport*dur_pt_bus + B_dur_pt_int_waiting_Public_Transport*dur_pt_int_waiting + B_dur_pt_int_walking_Public_Transport*dur_pt_int_walking + B_pt_n_interchanges_Public_Transport*pt_n_interchanges + B_cost_transit_Public_Transport*cost_transit",
        3: "ASC_Car + B_age_Car*age + B_female_Car*female + B_day_of_week_Car*day_of_week + B_start_time_linear_Car*start_time_linear + B_car_ownership_Car*car_ownership + B_driving_license_Car*driving_license + B_purpose_B_Car*purpose_B + B_purpose_HBE_Car*purpose_HBE + B_purpose_HBO_Car*purpose_HBO + B_purpose_HBW_Car*purpose_HBW + B_purpose_NHBO_Car*purpose_NHBO + B_fueltype_Avrg_Car*fueltype_Average + B_fueltype_Diesel_Car*fueltype_Diesel + B_fueltype_Hybrid_Car*fueltype_Hybrid + B_fueltype_Petrol_Car*fueltype_Petrol + B_distance_Car*distance + B_dur_driving_Car*dur_driving + B_cost_driving_fuel_Car*cost_driving_fuel + B_con_charge_Car*congestion_charge + B_traffic_perc_Car*driving_traffic_percent",
    }

    # driving_percentage, congestion charge as a binary variable start time linear and travel month as cyclical variable (also day of travel need to be cyclical) removed car ownership in the cost of driving, worse results somehow
    # MNL_beta_params_positive = ['B_car_ownership_Car', 'B_driving_license_Car']
    # MNL_beta_params_negative = ['B_dur_walking_Walk',  'B_dur_cycling_Bike', 'B_dur_pt_access_Public_Transport', 'B_dur_pt_rail_Public_Transport', 'B_dur_pt_bus_Public_Transport', 'B_dur_pt_int_waiting_Public_Transport', 'B_dur_pt_int_walking_Public_Transport', 'B_pt_n_interchanges_Public_Transport', 'B_cost_transit_Public_Transport', 'B_dur_driving_Car', 'B_cost_driving_fuel_Car', 'B_distance_Walk', 'B_distance_Bike', 'B_distance_Public_Transport', 'B_distance_Car', 'B_con_charge_Car', 'B_traffic_perc_Car']
    # MNL_beta_params_neutral = ['ASC_Bike', 'ASC_Public_Transport', 'ASC_Car', 'B_car_ownership_Walk', 'B_car_ownership_Bike', 'B_car_ownership_Public_Transport', 'B_driving_license_Walk', 'B_driving_license_Bike', 'B_driving_license_Public_Transport', 'B_age_Walk', 'B_age_Bike', 'B_age_Public_Transport', 'B_age_Car', 'B_female_Walk', 'B_female_Bike', 'B_female_Public_Transport', 'B_female_Car', 'B_day_of_week_Walk', 'B_day_of_week_Bike', 'B_day_of_week_Public_Transport', 'B_day_of_week_Car', 'B_start_time_linear_cos_Walk', 'B_start_time_linear_cos_Bike', 'B_start_time_linear_cos_Public_Transport', 'B_start_time_linear_cos_Car', 'B_start_time_linear_sin_Walk', 'B_start_time_linear_sin_Bike', 'B_start_time_linear_sin_Public_Transport', 'B_start_time_linear_sin_Car', 'B_purpose_B_Walk', 'B_purpose_B_Bike', 'B_purpose_B_Public_Transport', 'B_purpose_B_Car', 'B_purpose_HBE_Walk', 'B_purpose_HBE_Bike', 'B_purpose_HBE_Public_Transport', 'B_purpose_HBE_Car', 'B_purpose_HBO_Walk', 'B_purpose_HBO_Bike', 'B_purpose_HBO_Public_Transport', 'B_purpose_HBO_Car', 'B_purpose_HBW_Walk', 'B_purpose_HBW_Bike', 'B_purpose_HBW_Public_Transport', 'B_purpose_HBW_Car', 'B_purpose_NHBO_Walk', 'B_purpose_NHBO_Bike', 'B_purpose_NHBO_Public_Transport', 'B_purpose_NHBO_Car', 'B_fueltype_Avrg_Walk', 'B_fueltype_Avrg_Bike', 'B_fueltype_Avrg_Public_Transport', 'B_fueltype_Avrg_Car', 'B_fueltype_Diesel_Walk', 'B_fueltype_Diesel_Bike', 'B_fueltype_Diesel_Public_Transport', 'B_fueltype_Diesel_Car', 'B_fueltype_Hybrid_Walk', 'B_fueltype_Hybrid_Bike', 'B_fueltype_Hybrid_Public_Transport', 'B_fueltype_Hybrid_Car', 'B_fueltype_Petrol_Walk', 'B_fueltype_Petrol_Bike', 'B_fueltype_Petrol_Public_Transport', 'B_fueltype_Petrol_Car', 'B_travel_month_cos_Walk', 'B_travel_month_cos_Bike', 'B_travel_month_cos_Public_Transport', 'B_travel_month_cos_Car', 'B_travel_month_sin_Walk', 'B_travel_month_sin_Bike', 'B_travel_month_sin_Public_Transport', 'B_travel_month_sin_Car']

    # MNL_utilities = {0: 'B_age_Walk*age + B_female_Walk*female + B_day_of_week_Walk*day_of_week + B_start_time_linear_cos_Walk*start_time_linear_cos + B_start_time_linear_sin_Walk*start_time_linear_sin + B_car_ownership_Walk*car_ownership + B_driving_license_Walk*driving_license + B_purpose_B_Walk*purpose_B + B_purpose_HBE_Walk*purpose_HBE + B_purpose_HBO_Walk*purpose_HBO + B_purpose_HBW_Walk*purpose_HBW + B_purpose_NHBO_Walk*purpose_NHBO + B_fueltype_Avrg_Walk*fueltype_Average + B_fueltype_Diesel_Walk*fueltype_Diesel + B_fueltype_Hybrid_Walk*fueltype_Hybrid + B_fueltype_Petrol_Walk*fueltype_Petrol + B_distance_Walk*distance + B_dur_walking_Walk*dur_walking + B_travel_month_cos_Walk*travel_month_cos + B_travel_month_sin_Walk*travel_month_sin',
    #                     1: 'ASC_Bike + B_age_Bike*age + B_female_Bike*female + B_day_of_week_Bike*day_of_week + B_start_time_linear_cos_Bike*start_time_linear_cos + B_start_time_linear_sin_Bike*start_time_linear_sin + B_car_ownership_Bike*car_ownership + B_driving_license_Bike*driving_license + B_purpose_B_Bike*purpose_B + B_purpose_HBE_Bike*purpose_HBE + B_purpose_HBO_Bike*purpose_HBO + B_purpose_HBW_Bike*purpose_HBW + B_purpose_NHBO_Bike*purpose_NHBO + B_fueltype_Avrg_Bike*fueltype_Average + B_fueltype_Diesel_Bike*fueltype_Diesel + B_fueltype_Hybrid_Bike*fueltype_Hybrid + B_fueltype_Petrol_Bike*fueltype_Petrol + B_distance_Bike*distance + B_dur_cycling_Bike*dur_cycling + B_travel_month_cos_Bike*travel_month_cos + B_travel_month_sin_Bike*travel_month_sin',
    #                     2: 'ASC_Public_Transport + B_age_Public_Transport*age + B_female_Public_Transport*female + B_day_of_week_Public_Transport*day_of_week + B_start_time_linear_cos_Public_Transport*start_time_linear_cos + B_start_time_linear_sin_Public_Transport*start_time_linear_sin + B_car_ownership_Public_Transport*car_ownership + B_driving_license_Public_Transport*driving_license + B_purpose_B_Public_Transport*purpose_B + B_purpose_HBE_Public_Transport*purpose_HBE + B_purpose_HBO_Public_Transport*purpose_HBO + B_purpose_HBW_Public_Transport*purpose_HBW + B_purpose_NHBO_Public_Transport*purpose_NHBO + B_fueltype_Avrg_Public_Transport*fueltype_Average + B_fueltype_Diesel_Public_Transport*fueltype_Diesel + B_fueltype_Hybrid_Public_Transport*fueltype_Hybrid + B_fueltype_Petrol_Public_Transport*fueltype_Petrol + B_distance_Public_Transport*distance + B_dur_pt_access_Public_Transport*dur_pt_access + B_dur_pt_rail_Public_Transport*dur_pt_rail + B_dur_pt_bus_Public_Transport*dur_pt_bus + B_dur_pt_int_waiting_Public_Transport*dur_pt_int_waiting + B_dur_pt_int_walking_Public_Transport*dur_pt_int_walking + B_pt_n_interchanges_Public_Transport*pt_n_interchanges + B_cost_transit_Public_Transport*cost_transit + B_travel_month_cos_Public_Transport*travel_month_cos + B_travel_month_sin_Public_Transport*travel_month_sin',
    #                     3: 'ASC_Car + B_age_Car*age + B_female_Car*female + B_day_of_week_Car*day_of_week + B_start_time_linear_cos_Car*start_time_linear_cos + B_start_time_linear_sin_Car*start_time_linear_sin + B_car_ownership_Car*car_ownership + B_driving_license_Car*driving_license + B_purpose_B_Car*purpose_B + B_purpose_HBE_Car*purpose_HBE + B_purpose_HBO_Car*purpose_HBO + B_purpose_HBW_Car*purpose_HBW + B_purpose_NHBO_Car*purpose_NHBO + B_fueltype_Avrg_Car*fueltype_Average + B_fueltype_Diesel_Car*fueltype_Diesel + B_fueltype_Hybrid_Car*fueltype_Hybrid + B_fueltype_Petrol_Car*fueltype_Petrol + B_distance_Car*distance + B_dur_driving_Car*dur_driving + B_cost_driving_fuel_Car*cost_driving_fuel + B_con_charge_Car*congestion_charge + B_traffic_perc_Car*driving_traffic_percent + B_travel_month_cos_Car*travel_month_cos + B_travel_month_sin_Car*travel_month_sin'}

    # driving_percentage, congestion charge as a binary variable + distance not monotonic 0.6731 no good
    # MNL_beta_params_positive = ['B_car_ownership_Car', 'B_driving_license_Car']
    # MNL_beta_params_negative = ['B_dur_walking_Walk',  'B_dur_cycling_Bike', 'B_dur_pt_access_Public_Transport', 'B_dur_pt_rail_Public_Transport', 'B_dur_pt_bus_Public_Transport', 'B_dur_pt_int_waiting_Public_Transport', 'B_dur_pt_int_walking_Public_Transport', 'B_pt_n_interchanges_Public_Transport', 'B_cost_transit_Public_Transport', 'B_dur_driving_Car', 'B_cost_driving_fuel_Car', 'B_con_charge_Car', 'B_traffic_perc_Car']
    # MNL_beta_params_neutral = ['ASC_Bike', 'ASC_Public_Transport', 'ASC_Car', 'B_car_ownership_Walk', 'B_car_ownership_Bike', 'B_car_ownership_Public_Transport', 'B_driving_license_Walk', 'B_driving_license_Bike', 'B_driving_license_Public_Transport', 'B_age_Walk', 'B_age_Bike', 'B_age_Public_Transport', 'B_age_Car', 'B_female_Walk', 'B_female_Bike', 'B_female_Public_Transport', 'B_female_Car', 'B_day_of_week_Walk', 'B_day_of_week_Bike', 'B_day_of_week_Public_Transport', 'B_day_of_week_Car', 'B_start_time_linear_Walk', 'B_start_time_linear_Bike', 'B_start_time_linear_Public_Transport', 'B_start_time_linear_Car', 'B_purpose_B_Walk', 'B_purpose_B_Bike', 'B_purpose_B_Public_Transport', 'B_purpose_B_Car', 'B_purpose_HBE_Walk', 'B_purpose_HBE_Bike', 'B_purpose_HBE_Public_Transport', 'B_purpose_HBE_Car', 'B_purpose_HBO_Walk', 'B_purpose_HBO_Bike', 'B_purpose_HBO_Public_Transport', 'B_purpose_HBO_Car', 'B_purpose_HBW_Walk', 'B_purpose_HBW_Bike', 'B_purpose_HBW_Public_Transport', 'B_purpose_HBW_Car', 'B_purpose_NHBO_Walk', 'B_purpose_NHBO_Bike', 'B_purpose_NHBO_Public_Transport', 'B_purpose_NHBO_Car', 'B_fueltype_Avrg_Walk', 'B_fueltype_Avrg_Bike', 'B_fueltype_Avrg_Public_Transport', 'B_fueltype_Avrg_Car', 'B_fueltype_Diesel_Walk', 'B_fueltype_Diesel_Bike', 'B_fueltype_Diesel_Public_Transport', 'B_fueltype_Diesel_Car', 'B_fueltype_Hybrid_Walk', 'B_fueltype_Hybrid_Bike', 'B_fueltype_Hybrid_Public_Transport', 'B_fueltype_Hybrid_Car', 'B_fueltype_Petrol_Walk', 'B_fueltype_Petrol_Bike', 'B_fueltype_Petrol_Public_Transport', 'B_fueltype_Petrol_Car','B_distance_Walk', 'B_distance_Bike', 'B_distance_Public_Transport', 'B_distance_Car']

    # MNL_utilities = {0: 'B_age_Walk*age + B_female_Walk*female + B_day_of_week_Walk*day_of_week + B_start_time_linear_Walk*start_time_linear + B_car_ownership_Walk*car_ownership + B_driving_license_Walk*driving_license + B_purpose_B_Walk*purpose_B + B_purpose_HBE_Walk*purpose_HBE + B_purpose_HBO_Walk*purpose_HBO + B_purpose_HBW_Walk*purpose_HBW + B_purpose_NHBO_Walk*purpose_NHBO + B_fueltype_Avrg_Walk*fueltype_Average + B_fueltype_Diesel_Walk*fueltype_Diesel + B_fueltype_Hybrid_Walk*fueltype_Hybrid + B_fueltype_Petrol_Walk*fueltype_Petrol + B_distance_Walk*distance + B_dur_walking_Walk*dur_walking',
    #                     1: 'ASC_Bike + B_age_Bike*age + B_female_Bike*female + B_day_of_week_Bike*day_of_week + B_start_time_linear_Bike*start_time_linear + B_car_ownership_Bike*car_ownership + B_driving_license_Bike*driving_license + B_purpose_B_Bike*purpose_B + B_purpose_HBE_Bike*purpose_HBE + B_purpose_HBO_Bike*purpose_HBO + B_purpose_HBW_Bike*purpose_HBW + B_purpose_NHBO_Bike*purpose_NHBO + B_fueltype_Avrg_Bike*fueltype_Average + B_fueltype_Diesel_Bike*fueltype_Diesel + B_fueltype_Hybrid_Bike*fueltype_Hybrid + B_fueltype_Petrol_Bike*fueltype_Petrol + B_distance_Bike*distance + B_dur_cycling_Bike*dur_cycling',
    #                     2: 'ASC_Public_Transport + B_age_Public_Transport*age + B_female_Public_Transport*female + B_day_of_week_Public_Transport*day_of_week + B_start_time_linear_Public_Transport*start_time_linear + B_car_ownership_Public_Transport*car_ownership + B_driving_license_Public_Transport*driving_license + B_purpose_B_Public_Transport*purpose_B + B_purpose_HBE_Public_Transport*purpose_HBE + B_purpose_HBO_Public_Transport*purpose_HBO + B_purpose_HBW_Public_Transport*purpose_HBW + B_purpose_NHBO_Public_Transport*purpose_NHBO + B_fueltype_Avrg_Public_Transport*fueltype_Average + B_fueltype_Diesel_Public_Transport*fueltype_Diesel + B_fueltype_Hybrid_Public_Transport*fueltype_Hybrid + B_fueltype_Petrol_Public_Transport*fueltype_Petrol + B_distance_Public_Transport*distance + B_dur_pt_access_Public_Transport*dur_pt_access + B_dur_pt_rail_Public_Transport*dur_pt_rail + B_dur_pt_bus_Public_Transport*dur_pt_bus + B_dur_pt_int_waiting_Public_Transport*dur_pt_int_waiting + B_dur_pt_int_walking_Public_Transport*dur_pt_int_walking + B_pt_n_interchanges_Public_Transport*pt_n_interchanges + B_cost_transit_Public_Transport*cost_transit',
    #                     3: 'ASC_Car + B_age_Car*age + B_female_Car*female + B_day_of_week_Car*day_of_week + B_start_time_linear_Car*start_time_linear + B_car_ownership_Car*car_ownership + B_driving_license_Car*driving_license + B_purpose_B_Car*purpose_B + B_purpose_HBE_Car*purpose_HBE + B_purpose_HBO_Car*purpose_HBO + B_purpose_HBW_Car*purpose_HBW + B_purpose_NHBO_Car*purpose_NHBO + B_fueltype_Avrg_Car*fueltype_Average + B_fueltype_Diesel_Car*fueltype_Diesel + B_fueltype_Hybrid_Car*fueltype_Hybrid + B_fueltype_Petrol_Car*fueltype_Petrol + B_distance_Car*distance + B_dur_driving_Car*dur_driving + B_cost_driving_fuel_Car*cost_driving_fuel + B_con_charge_Car*congestion_charge + B_traffic_perc_Car*driving_traffic_percent'}

    # 0.6805
    # MNL_beta_params_positive = ['B_car_ownership_Car', 'B_driving_license_Car']
    # MNL_beta_params_negative = ['B_age_Bike', 'B_distance_Bike', 'B_distance_Public_Transport', 'B_distance_Car','B_driving_license_Public_Transport', 'B_car_ownership_Bike', 'B_car_ownership_Public_Transport', 'B_driving_license_Bike', 'B_dur_walking_Walk', 'B_dur_cycling_Bike', 'B_dur_pt_access_Public_Transport', 'B_dur_pt_rail_Public_Transport', 'B_dur_pt_bus_Public_Transport', 'B_dur_pt_int_waiting_Public_Transport', 'B_dur_pt_int_walking_Public_Transport', 'B_pt_n_interchanges_Public_Transport', 'B_cost_transit_Public_Transport', 'B_dur_driving_Car', 'B_cost_driving_total_Car']
    # MNL_beta_params_neutral = ['ASC_Bike', 'ASC_Public_Transport', 'ASC_Car', 'B_age_Public_Transport', 'B_age_Car', 'B_female_Bike', 'B_female_Public_Transport', 'B_female_Car', 'B_day_of_week_Bike', 'B_day_of_week_Public_Transport', 'B_day_of_week_Car', 'B_start_time_linear_Bike', 'B_start_time_linear_Public_Transport', 'B_start_time_linear_Car', 'B_purpose_B_Bike', 'B_purpose_B_Public_Transport', 'B_purpose_B_Car', 'B_purpose_HBE_Bike', 'B_purpose_HBE_Public_Transport', 'B_purpose_HBE_Car', 'B_purpose_HBO_Bike', 'B_purpose_HBO_Public_Transport', 'B_purpose_HBO_Car', 'B_purpose_HBW_Bike', 'B_purpose_HBW_Public_Transport', 'B_purpose_HBW_Car', 'B_purpose_NHBO_Bike', 'B_purpose_NHBO_Public_Transport', 'B_purpose_NHBO_Car','B_fueltype_Avrg_Bike', 'B_fueltype_Avrg_Public_Transport', 'B_fueltype_Avrg_Car', 'B_fueltype_Diesel_Bike', 'B_fueltype_Diesel_Public_Transport', 'B_fueltype_Diesel_Car', 'B_fueltype_Hybrid_Bike', 'B_fueltype_Hybrid_Public_Transport', 'B_fueltype_Hybrid_Car', 'B_fueltype_Petrol_Bike', 'B_fueltype_Petrol_Public_Transport', 'B_fueltype_Petrol_Car']

    # MNL_utilities = {0: 'B_dur_walking_Walk*dur_walking',
    #                  1: 'ASC_Bike + B_age_Bike*age + B_female_Bike*female + B_day_of_week_Bike*day_of_week + B_start_time_linear_Bike*start_time_linear + B_car_ownership_Bike*car_ownership + B_driving_license_Bike*driving_license + B_purpose_B_Bike*purpose_B + B_purpose_HBE_Bike*purpose_HBE + B_purpose_HBO_Bike*purpose_HBO + B_purpose_HBW_Bike*purpose_HBW + B_purpose_NHBO_Bike*purpose_NHBO + B_fueltype_Avrg_Bike*fueltype_Average + B_fueltype_Diesel_Bike*fueltype_Diesel + B_fueltype_Hybrid_Bike*fueltype_Hybrid + B_fueltype_Petrol_Bike*fueltype_Petrol + B_distance_Bike*distance + B_dur_cycling_Bike*dur_cycling',
    #                  2: 'ASC_Public_Transport + B_age_Public_Transport*age + B_female_Public_Transport*female + B_day_of_week_Public_Transport*day_of_week + B_start_time_linear_Public_Transport*start_time_linear + B_car_ownership_Public_Transport*car_ownership + B_driving_license_Public_Transport*driving_license + B_purpose_B_Public_Transport*purpose_B + B_purpose_HBE_Public_Transport*purpose_HBE + B_purpose_HBO_Public_Transport*purpose_HBO + B_purpose_HBW_Public_Transport*purpose_HBW + B_purpose_NHBO_Public_Transport*purpose_NHBO + B_fueltype_Avrg_Public_Transport*fueltype_Average + B_fueltype_Diesel_Public_Transport*fueltype_Diesel + B_fueltype_Hybrid_Public_Transport*fueltype_Hybrid + B_fueltype_Petrol_Public_Transport*fueltype_Petrol + B_dur_pt_access_Public_Transport*dur_pt_access + B_dur_pt_rail_Public_Transport*dur_pt_rail + B_dur_pt_bus_Public_Transport*dur_pt_bus + B_dur_pt_int_waiting_Public_Transport*dur_pt_int_waiting + B_dur_pt_int_walking_Public_Transport*dur_pt_int_walking + B_pt_n_interchanges_Public_Transport*pt_n_interchanges + B_cost_transit_Public_Transport*cost_transit + B_distance_Public_Transport*distance',
    #                  3: 'ASC_Car + B_age_Car*age + B_female_Car*female + B_day_of_week_Car*day_of_week + B_start_time_linear_Car*start_time_linear + B_car_ownership_Car*car_ownership + B_driving_license_Car*driving_license + B_purpose_B_Car*purpose_B + B_purpose_HBE_Car*purpose_HBE + B_purpose_HBO_Car*purpose_HBO + B_purpose_HBW_Car*purpose_HBW + B_purpose_NHBO_Car*purpose_NHBO + B_fueltype_Avrg_Car*fueltype_Average + B_fueltype_Diesel_Car*fueltype_Diesel + B_fueltype_Hybrid_Car*fueltype_Hybrid + B_fueltype_Petrol_Car*fueltype_Petrol + B_dur_driving_Car*dur_driving + B_cost_driving_total_Car*cost_driving_total + B_distance_Car*distance'}

    # EXTREME-alternative specific model 0.683
    # MNL_beta_params_positive = ['B_car_ownership_Car', 'B_driving_license_Car']
    # MNL_beta_params_negative = ['B_dur_walking_Walk',  'B_dur_cycling_Bike', 'B_dur_pt_access_Public_Transport', 'B_dur_pt_rail_Public_Transport', 'B_dur_pt_bus_Public_Transport', 'B_dur_pt_int_waiting_Public_Transport', 'B_dur_pt_int_walking_Public_Transport', 'B_pt_n_interchanges_Public_Transport', 'B_cost_transit_Public_Transport', 'B_dur_driving_Car', 'B_cost_driving_total_Car']#, 'B_traffic_perc_Car']
    # MNL_beta_params_neutral = ['ASC_Bike', 'ASC_Public_Transport', 'ASC_Car', 'B_car_ownership_Walk', 'B_car_ownership_Bike', 'B_car_ownership_Public_Transport', 'B_driving_license_Walk', 'B_driving_license_Bike', 'B_driving_license_Public_Transport', 'B_age_Walk', 'B_age_Bike', 'B_age_Public_Transport', 'B_age_Car', 'B_female_Walk', 'B_female_Bike', 'B_female_Public_Transport', 'B_female_Car', 'B_distance_Walk', 'B_distance_Bike', 'B_day_of_week_Walk', 'B_day_of_week_Bike', 'B_day_of_week_Public_Transport', 'B_day_of_week_Car', 'B_start_time_linear_Walk', 'B_start_time_linear_Bike', 'B_start_time_linear_Public_Transport', 'B_start_time_linear_Car', 'B_purpose_B_Walk', 'B_purpose_B_Bike', 'B_purpose_B_Public_Transport', 'B_purpose_B_Car', 'B_purpose_HBE_Walk', 'B_purpose_HBE_Bike', 'B_purpose_HBE_Public_Transport', 'B_purpose_HBE_Car', 'B_purpose_HBO_Walk', 'B_purpose_HBO_Bike', 'B_purpose_HBO_Public_Transport', 'B_purpose_HBO_Car', 'B_purpose_HBW_Walk', 'B_purpose_HBW_Bike', 'B_purpose_HBW_Public_Transport', 'B_purpose_HBW_Car', 'B_purpose_NHBO_Walk', 'B_purpose_NHBO_Bike', 'B_purpose_NHBO_Public_Transport', 'B_purpose_NHBO_Car', 'B_fueltype_Avrg_Walk', 'B_fueltype_Avrg_Bike', 'B_fueltype_Avrg_Public_Transport', 'B_fueltype_Avrg_Car', 'B_fueltype_Diesel_Walk', 'B_fueltype_Diesel_Bike', 'B_fueltype_Diesel_Public_Transport', 'B_fueltype_Diesel_Car', 'B_fueltype_Hybrid_Walk', 'B_fueltype_Hybrid_Bike', 'B_fueltype_Hybrid_Public_Transport', 'B_fueltype_Hybrid_Car', 'B_fueltype_Petrol_Walk', 'B_fueltype_Petrol_Bike', 'B_fueltype_Petrol_Public_Transport', 'B_fueltype_Petrol_Car']

    # MNL_utilities = {0: 'B_age_Walk*age + B_female_Walk*female + B_day_of_week_Walk*day_of_week + B_start_time_linear_Walk*start_time_linear + B_purpose_B_Walk*purpose_B + B_purpose_HBE_Walk*purpose_HBE + B_purpose_HBO_Walk*purpose_HBO + B_purpose_HBW_Walk*purpose_HBW + B_purpose_NHBO_Walk*purpose_NHBO + B_distance_Walk*distance + B_dur_walking_Walk*dur_walking',
    #                  1: 'ASC_Bike + B_age_Bike*age + B_female_Bike*female + B_day_of_week_Bike*day_of_week + B_start_time_linear_Bike*start_time_linear + B_purpose_B_Bike*purpose_B + B_purpose_HBE_Bike*purpose_HBE + B_purpose_HBO_Bike*purpose_HBO + B_purpose_HBW_Bike*purpose_HBW + B_purpose_NHBO_Bike*purpose_NHBO + B_distance_Bike*distance + B_dur_cycling_Bike*dur_cycling',
    #                  2: 'ASC_Public_Transport + B_age_Public_Transport*age + B_female_Public_Transport*female + B_day_of_week_Public_Transport*day_of_week + B_start_time_linear_Public_Transport*start_time_linear + B_purpose_B_Public_Transport*purpose_B + B_purpose_HBE_Public_Transport*purpose_HBE + B_purpose_HBO_Public_Transport*purpose_HBO + B_purpose_HBW_Public_Transport*purpose_HBW + B_purpose_NHBO_Public_Transport*purpose_NHBO + B_dur_pt_access_Public_Transport*dur_pt_access + B_dur_pt_rail_Public_Transport*dur_pt_rail + B_dur_pt_bus_Public_Transport*dur_pt_bus + B_dur_pt_int_waiting_Public_Transport*dur_pt_int_waiting + B_dur_pt_int_walking_Public_Transport*dur_pt_int_walking + B_pt_n_interchanges_Public_Transport*pt_n_interchanges + B_cost_transit_Public_Transport*cost_transit',
    #                  3: 'ASC_Car + B_age_Car*age + B_female_Car*female + B_day_of_week_Car*day_of_week + B_start_time_linear_Car*start_time_linear + B_car_ownership_Car*car_ownership + B_driving_license_Car*driving_license + B_purpose_B_Car*purpose_B + B_purpose_HBE_Car*purpose_HBE + B_purpose_HBO_Car*purpose_HBO + B_purpose_HBW_Car*purpose_HBW + B_purpose_NHBO_Car*purpose_NHBO + B_fueltype_Avrg_Car*fueltype_Average + B_fueltype_Diesel_Car*fueltype_Diesel + B_fueltype_Hybrid_Car*fueltype_Hybrid + B_fueltype_Petrol_Car*fueltype_Petrol + B_dur_driving_Car*dur_driving + B_cost_driving_total_Car*cost_driving_total'}

    # 0.6796
    # MNL_beta_params_positive = ['B_car_ownership_Car', 'B_driving_license_Car']
    # MNL_beta_params_negative = ['B_dur_walking_Walk',  'B_dur_cycling_Bike', 'B_dur_pt_access_Public_Transport', 'B_dur_pt_rail_Public_Transport', 'B_dur_pt_bus_Public_Transport', 'B_dur_pt_int_waiting_Public_Transport', 'B_dur_pt_int_walking_Public_Transport', 'B_pt_n_interchanges_Public_Transport', 'B_cost_transit_Public_Transport', 'B_dur_driving_Car', 'B_cost_driving_total_Car']#, 'B_traffic_perc_Car']
    # MNL_beta_params_neutral = ['ASC_Bike', 'ASC_Public_Transport', 'ASC_Car', 'B_car_ownership_Walk', 'B_car_ownership_Bike', 'B_car_ownership_Public_Transport', 'B_driving_license_Walk', 'B_driving_license_Bike', 'B_driving_license_Public_Transport', 'B_age_Walk', 'B_age_Bike', 'B_age_Public_Transport', 'B_age_Car', 'B_female_Walk', 'B_female_Bike', 'B_female_Public_Transport', 'B_female_Car', 'B_distance_Walk', 'B_distance_Bike', 'B_distance_Public_Transport', 'B_distance_Car', 'B_day_of_week_Walk', 'B_day_of_week_Bike', 'B_day_of_week_Public_Transport', 'B_day_of_week_Car', 'B_start_time_linear_Walk', 'B_start_time_linear_Bike', 'B_start_time_linear_Public_Transport', 'B_start_time_linear_Car', 'B_purpose_B_Walk', 'B_purpose_B_Bike', 'B_purpose_B_Public_Transport', 'B_purpose_B_Car', 'B_purpose_HBE_Walk', 'B_purpose_HBE_Bike', 'B_purpose_HBE_Public_Transport', 'B_purpose_HBE_Car', 'B_purpose_HBO_Walk', 'B_purpose_HBO_Bike', 'B_purpose_HBO_Public_Transport', 'B_purpose_HBO_Car', 'B_purpose_HBW_Walk', 'B_purpose_HBW_Bike', 'B_purpose_HBW_Public_Transport', 'B_purpose_HBW_Car', 'B_purpose_NHBO_Walk', 'B_purpose_NHBO_Bike', 'B_purpose_NHBO_Public_Transport', 'B_purpose_NHBO_Car', 'B_fueltype_Avrg_Walk', 'B_fueltype_Avrg_Bike', 'B_fueltype_Avrg_Public_Transport', 'B_fueltype_Avrg_Car', 'B_fueltype_Diesel_Walk', 'B_fueltype_Diesel_Bike', 'B_fueltype_Diesel_Public_Transport', 'B_fueltype_Diesel_Car', 'B_fueltype_Hybrid_Walk', 'B_fueltype_Hybrid_Bike', 'B_fueltype_Hybrid_Public_Transport', 'B_fueltype_Hybrid_Car', 'B_fueltype_Petrol_Walk', 'B_fueltype_Petrol_Bike', 'B_fueltype_Petrol_Public_Transport', 'B_fueltype_Petrol_Car']

    # MNL_utilities = {0: 'B_age_Walk*age + B_female_Walk*female + B_day_of_week_Walk*day_of_week + B_start_time_linear_Walk*start_time_linear + B_car_ownership_Walk*car_ownership + B_driving_license_Walk*driving_license + B_purpose_B_Walk*purpose_B + B_purpose_HBE_Walk*purpose_HBE + B_purpose_HBO_Walk*purpose_HBO + B_purpose_HBW_Walk*purpose_HBW + B_purpose_NHBO_Walk*purpose_NHBO + B_fueltype_Avrg_Walk*fueltype_Average + B_fueltype_Diesel_Walk*fueltype_Diesel + B_fueltype_Hybrid_Walk*fueltype_Hybrid + B_fueltype_Petrol_Walk*fueltype_Petrol + B_distance_Walk*distance + B_dur_walking_Walk*dur_walking',
    #                  1: 'ASC_Bike + B_age_Bike*age + B_female_Bike*female + B_day_of_week_Bike*day_of_week + B_start_time_linear_Bike*start_time_linear + B_car_ownership_Bike*car_ownership + B_driving_license_Bike*driving_license + B_purpose_B_Bike*purpose_B + B_purpose_HBE_Bike*purpose_HBE + B_purpose_HBO_Bike*purpose_HBO + B_purpose_HBW_Bike*purpose_HBW + B_purpose_NHBO_Bike*purpose_NHBO + B_fueltype_Avrg_Bike*fueltype_Average + B_fueltype_Diesel_Bike*fueltype_Diesel + B_fueltype_Hybrid_Bike*fueltype_Hybrid + B_fueltype_Petrol_Bike*fueltype_Petrol + B_distance_Bike*distance + B_dur_cycling_Bike*dur_cycling',
    #                  2: 'ASC_Public_Transport + B_age_Public_Transport*age + B_female_Public_Transport*female + B_day_of_week_Public_Transport*day_of_week + B_start_time_linear_Public_Transport*start_time_linear + B_car_ownership_Public_Transport*car_ownership + B_driving_license_Public_Transport*driving_license + B_purpose_B_Public_Transport*purpose_B + B_purpose_HBE_Public_Transport*purpose_HBE + B_purpose_HBO_Public_Transport*purpose_HBO + B_purpose_HBW_Public_Transport*purpose_HBW + B_purpose_NHBO_Public_Transport*purpose_NHBO + B_fueltype_Avrg_Public_Transport*fueltype_Average + B_fueltype_Diesel_Public_Transport*fueltype_Diesel + B_fueltype_Hybrid_Public_Transport*fueltype_Hybrid + B_fueltype_Petrol_Public_Transport*fueltype_Petrol + B_distance_Public_Transport*distance + B_dur_pt_access_Public_Transport*dur_pt_access + B_dur_pt_rail_Public_Transport*dur_pt_rail + B_dur_pt_bus_Public_Transport*dur_pt_bus + B_dur_pt_int_waiting_Public_Transport*dur_pt_int_waiting + B_dur_pt_int_walking_Public_Transport*dur_pt_int_walking + B_pt_n_interchanges_Public_Transport*pt_n_interchanges + B_cost_transit_Public_Transport*cost_transit',
    #                  3: 'ASC_Car + B_age_Car*age + B_female_Car*female + B_day_of_week_Car*day_of_week + B_start_time_linear_Car*start_time_linear + B_car_ownership_Car*car_ownership + B_driving_license_Car*driving_license + B_purpose_B_Car*purpose_B + B_purpose_HBE_Car*purpose_HBE + B_purpose_HBO_Car*purpose_HBO + B_purpose_HBW_Car*purpose_HBW + B_purpose_NHBO_Car*purpose_NHBO + B_fueltype_Avrg_Car*fueltype_Average + B_fueltype_Diesel_Car*fueltype_Diesel + B_fueltype_Hybrid_Car*fueltype_Hybrid + B_fueltype_Petrol_Car*fueltype_Petrol + B_distance_Car*distance + B_dur_driving_Car*dur_driving + B_cost_driving_total_Car*cost_driving_total'}

    # 0.6789 with lr 0.3 --- change with above model; B_age_Bike and B_age_Walk in negative mono, got rid of driving license in cycling, got rid of fueltype in other features than car, got rid of car ownership in walk
    # MNL_beta_params_positive = ['B_car_ownership_Car', 'B_driving_license_Car']
    # MNL_beta_params_negative = ['B_age_Walk', 'B_age_Bike', 'B_dur_walking_Walk',  'B_dur_cycling_Bike', 'B_dur_pt_access_Public_Transport', 'B_dur_pt_rail_Public_Transport', 'B_dur_pt_bus_Public_Transport', 'B_dur_pt_int_waiting_Public_Transport', 'B_dur_pt_int_walking_Public_Transport', 'B_pt_n_interchanges_Public_Transport', 'B_cost_transit_Public_Transport', 'B_dur_driving_Car', 'B_cost_driving_total_Car', 'B_distance_Walk', 'B_distance_Bike', 'B_distance_Public_Transport', 'B_distance_Car']#, 'B_traffic_perc_Car']
    # MNL_beta_params_neutral = ['ASC_Bike', 'ASC_Public_Transport', 'ASC_Car', 'B_car_ownership_Bike', 'B_car_ownership_Public_Transport', 'B_driving_license_Walk', 'B_driving_license_Public_Transport', 'B_age_Public_Transport', 'B_age_Car', 'B_female_Walk', 'B_female_Bike', 'B_female_Public_Transport', 'B_female_Car', 'B_day_of_week_Walk', 'B_day_of_week_Bike', 'B_day_of_week_Public_Transport', 'B_day_of_week_Car', 'B_start_time_linear_Walk', 'B_start_time_linear_Bike', 'B_start_time_linear_Public_Transport', 'B_start_time_linear_Car', 'B_purpose_B_Walk', 'B_purpose_B_Bike', 'B_purpose_B_Public_Transport', 'B_purpose_B_Car', 'B_purpose_HBE_Walk', 'B_purpose_HBE_Bike', 'B_purpose_HBE_Public_Transport', 'B_purpose_HBE_Car', 'B_purpose_HBO_Walk', 'B_purpose_HBO_Bike', 'B_purpose_HBO_Public_Transport', 'B_purpose_HBO_Car', 'B_purpose_HBW_Walk', 'B_purpose_HBW_Bike', 'B_purpose_HBW_Public_Transport', 'B_purpose_HBW_Car', 'B_purpose_NHBO_Walk', 'B_purpose_NHBO_Bike', 'B_purpose_NHBO_Public_Transport', 'B_purpose_NHBO_Car', 'B_fueltype_Avrg_Car', 'B_fueltype_Diesel_Car', 'B_fueltype_Hybrid_Car', 'B_fueltype_Petrol_Car']

    # MNL_utilities = {0: 'B_age_Walk*age + B_female_Walk*female + B_day_of_week_Walk*day_of_week + B_start_time_linear_Walk*start_time_linear + B_driving_license_Walk*driving_license + B_purpose_B_Walk*purpose_B + B_purpose_HBE_Walk*purpose_HBE + B_purpose_HBO_Walk*purpose_HBO + B_purpose_HBW_Walk*purpose_HBW + B_purpose_NHBO_Walk*purpose_NHBO + B_distance_Walk*distance + B_dur_walking_Walk*dur_walking',
    #                  1: 'ASC_Bike + B_age_Bike*age + B_female_Bike*female + B_day_of_week_Bike*day_of_week + B_start_time_linear_Bike*start_time_linear + B_car_ownership_Bike*car_ownership + B_purpose_B_Bike*purpose_B + B_purpose_HBE_Bike*purpose_HBE + B_purpose_HBO_Bike*purpose_HBO + B_purpose_HBW_Bike*purpose_HBW + B_purpose_NHBO_Bike*purpose_NHBO + B_distance_Bike*distance + B_dur_cycling_Bike*dur_cycling',
    #                  2: 'ASC_Public_Transport + B_age_Public_Transport*age + B_female_Public_Transport*female + B_day_of_week_Public_Transport*day_of_week + B_start_time_linear_Public_Transport*start_time_linear + B_car_ownership_Public_Transport*car_ownership + B_driving_license_Public_Transport*driving_license + B_purpose_B_Public_Transport*purpose_B + B_purpose_HBE_Public_Transport*purpose_HBE + B_purpose_HBO_Public_Transport*purpose_HBO + B_purpose_HBW_Public_Transport*purpose_HBW + B_purpose_NHBO_Public_Transport*purpose_NHBO + B_distance_Public_Transport*distance + B_dur_pt_access_Public_Transport*dur_pt_access + B_dur_pt_rail_Public_Transport*dur_pt_rail + B_dur_pt_bus_Public_Transport*dur_pt_bus + B_dur_pt_int_waiting_Public_Transport*dur_pt_int_waiting + B_dur_pt_int_walking_Public_Transport*dur_pt_int_walking + B_pt_n_interchanges_Public_Transport*pt_n_interchanges + B_cost_transit_Public_Transport*cost_transit',
    #                  3: 'ASC_Car + B_age_Car*age + B_female_Car*female + B_day_of_week_Car*day_of_week + B_start_time_linear_Car*start_time_linear + B_car_ownership_Car*car_ownership + B_driving_license_Car*driving_license + B_purpose_B_Car*purpose_B + B_purpose_HBE_Car*purpose_HBE + B_purpose_HBO_Car*purpose_HBO + B_purpose_HBW_Car*purpose_HBW + B_purpose_NHBO_Car*purpose_NHBO + B_fueltype_Avrg_Car*fueltype_Average + B_fueltype_Diesel_Car*fueltype_Diesel + B_fueltype_Hybrid_Car*fueltype_Hybrid + B_fueltype_Petrol_Car*fueltype_Petrol + B_distance_Car*distance + B_dur_driving_Car*dur_driving + B_cost_driving_total_Car*cost_driving_total'}

    # 0.6789 with lr 0.3 --- change with above model; got rid of start time linear
    # MNL_beta_params_positive = ['B_car_ownership_Car', 'B_driving_license_Car']
    # MNL_beta_params_negative = ['B_age_Walk', 'B_age_Bike', 'B_dur_walking_Walk',  'B_dur_cycling_Bike', 'B_dur_pt_access_Public_Transport', 'B_dur_pt_rail_Public_Transport', 'B_dur_pt_bus_Public_Transport', 'B_dur_pt_int_waiting_Public_Transport', 'B_dur_pt_int_walking_Public_Transport', 'B_pt_n_interchanges_Public_Transport', 'B_cost_transit_Public_Transport', 'B_dur_driving_Car', 'B_cost_driving_total_Car', 'B_distance_Walk', 'B_distance_Bike', 'B_distance_Public_Transport', 'B_distance_Car']#, 'B_traffic_perc_Car']
    # MNL_beta_params_neutral = ['ASC_Bike', 'ASC_Public_Transport', 'ASC_Car', 'B_car_ownership_Bike', 'B_car_ownership_Public_Transport', 'B_driving_license_Walk', 'B_driving_license_Public_Transport', 'B_age_Public_Transport', 'B_age_Car', 'B_female_Walk', 'B_female_Bike', 'B_female_Public_Transport', 'B_female_Car', 'B_day_of_week_Walk', 'B_day_of_week_Bike', 'B_day_of_week_Public_Transport', 'B_day_of_week_Car', 'B_purpose_B_Walk', 'B_purpose_B_Bike', 'B_purpose_B_Public_Transport', 'B_purpose_B_Car', 'B_purpose_HBE_Walk', 'B_purpose_HBE_Bike', 'B_purpose_HBE_Public_Transport', 'B_purpose_HBE_Car', 'B_purpose_HBO_Walk', 'B_purpose_HBO_Bike', 'B_purpose_HBO_Public_Transport', 'B_purpose_HBO_Car', 'B_purpose_HBW_Walk', 'B_purpose_HBW_Bike', 'B_purpose_HBW_Public_Transport', 'B_purpose_HBW_Car', 'B_purpose_NHBO_Walk', 'B_purpose_NHBO_Bike', 'B_purpose_NHBO_Public_Transport', 'B_purpose_NHBO_Car', 'B_fueltype_Avrg_Car', 'B_fueltype_Diesel_Car', 'B_fueltype_Hybrid_Car', 'B_fueltype_Petrol_Car']

    # MNL_utilities = {0: 'B_age_Walk*age + B_female_Walk*female + B_day_of_week_Walk*day_of_week + B_driving_license_Walk*driving_license + B_purpose_B_Walk*purpose_B + B_purpose_HBE_Walk*purpose_HBE + B_purpose_HBO_Walk*purpose_HBO + B_purpose_HBW_Walk*purpose_HBW + B_purpose_NHBO_Walk*purpose_NHBO + B_distance_Walk*distance + B_dur_walking_Walk*dur_walking',
    #                  1: 'ASC_Bike + B_age_Bike*age + B_female_Bike*female + B_day_of_week_Bike*day_of_week + B_car_ownership_Bike*car_ownership + B_purpose_B_Bike*purpose_B + B_purpose_HBE_Bike*purpose_HBE + B_purpose_HBO_Bike*purpose_HBO + B_purpose_HBW_Bike*purpose_HBW + B_purpose_NHBO_Bike*purpose_NHBO + B_distance_Bike*distance + B_dur_cycling_Bike*dur_cycling',
    #                  2: 'ASC_Public_Transport + B_age_Public_Transport*age + B_female_Public_Transport*female + B_day_of_week_Public_Transport*day_of_week + B_car_ownership_Public_Transport*car_ownership + B_driving_license_Public_Transport*driving_license + B_purpose_B_Public_Transport*purpose_B + B_purpose_HBE_Public_Transport*purpose_HBE + B_purpose_HBO_Public_Transport*purpose_HBO + B_purpose_HBW_Public_Transport*purpose_HBW + B_purpose_NHBO_Public_Transport*purpose_NHBO + B_distance_Public_Transport*distance + B_dur_pt_access_Public_Transport*dur_pt_access + B_dur_pt_rail_Public_Transport*dur_pt_rail + B_dur_pt_bus_Public_Transport*dur_pt_bus + B_dur_pt_int_waiting_Public_Transport*dur_pt_int_waiting + B_dur_pt_int_walking_Public_Transport*dur_pt_int_walking + B_pt_n_interchanges_Public_Transport*pt_n_interchanges + B_cost_transit_Public_Transport*cost_transit',
    #                  3: 'ASC_Car + B_age_Car*age + B_female_Car*female + B_day_of_week_Car*day_of_week + B_car_ownership_Car*car_ownership + B_driving_license_Car*driving_license + B_purpose_B_Car*purpose_B + B_purpose_HBE_Car*purpose_HBE + B_purpose_HBO_Car*purpose_HBO + B_purpose_HBW_Car*purpose_HBW + B_purpose_NHBO_Car*purpose_NHBO + B_fueltype_Avrg_Car*fueltype_Average + B_fueltype_Diesel_Car*fueltype_Diesel + B_fueltype_Hybrid_Car*fueltype_Hybrid + B_fueltype_Petrol_Car*fueltype_Petrol + B_distance_Car*distance + B_dur_driving_Car*dur_driving + B_cost_driving_total_Car*cost_driving_total'}

    # with travel month and traffic percentage
    # MNL_beta_params_positive = ['B_car_ownership_Car', 'B_driving_license_Car']
    # MNL_beta_params_negative = ['B_dur_walking_Walk',  'B_dur_cycling_Bike', 'B_dur_pt_access_Public_Transport', 'B_dur_pt_rail_Public_Transport', 'B_dur_pt_bus_Public_Transport', 'B_dur_pt_int_waiting_Public_Transport', 'B_dur_pt_int_walking_Public_Transport', 'B_pt_n_interchanges_Public_Transport', 'B_cost_transit_Public_Transport', 'B_dur_driving_Car', 'B_cost_driving_total_Car', 'B_traffic_perc_Car']
    # MNL_beta_params_neutral = ['ASC_Bike', 'ASC_Public_Transport', 'ASC_Car', 'B_car_ownership_Walk', 'B_car_ownership_Bike', 'B_car_ownership_Public_Transport', 'B_driving_license_Walk', 'B_driving_license_Bike', 'B_driving_license_Public_Transport', 'B_age_Walk', 'B_age_Bike', 'B_age_Public_Transport', 'B_age_Car', 'B_female_Walk', 'B_female_Bike', 'B_female_Public_Transport', 'B_female_Car', 'B_distance_Walk', 'B_distance_Bike', 'B_distance_Public_Transport', 'B_distance_Car', 'B_day_of_week_Walk', 'B_day_of_week_Bike', 'B_day_of_week_Public_Transport', 'B_day_of_week_Car', 'B_start_time_linear_Walk', 'B_start_time_linear_Bike', 'B_start_time_linear_Public_Transport', 'B_start_time_linear_Car', 'B_purpose_B_Walk', 'B_purpose_B_Bike', 'B_purpose_B_Public_Transport', 'B_purpose_B_Car', 'B_purpose_HBE_Walk', 'B_purpose_HBE_Bike', 'B_purpose_HBE_Public_Transport', 'B_purpose_HBE_Car', 'B_purpose_HBO_Walk', 'B_purpose_HBO_Bike', 'B_purpose_HBO_Public_Transport', 'B_purpose_HBO_Car', 'B_purpose_HBW_Walk', 'B_purpose_HBW_Bike', 'B_purpose_HBW_Public_Transport', 'B_purpose_HBW_Car', 'B_purpose_NHBO_Walk', 'B_purpose_NHBO_Bike', 'B_purpose_NHBO_Public_Transport', 'B_purpose_NHBO_Car', 'B_fueltype_Avrg_Walk', 'B_fueltype_Avrg_Bike', 'B_fueltype_Avrg_Public_Transport', 'B_fueltype_Avrg_Car', 'B_fueltype_Diesel_Walk', 'B_fueltype_Diesel_Bike', 'B_fueltype_Diesel_Public_Transport', 'B_fueltype_Diesel_Car', 'B_fueltype_Hybrid_Walk', 'B_fueltype_Hybrid_Bike', 'B_fueltype_Hybrid_Public_Transport', 'B_fueltype_Hybrid_Car', 'B_fueltype_Petrol_Walk', 'B_fueltype_Petrol_Bike', 'B_fueltype_Petrol_Public_Transport', 'B_fueltype_Petrol_Car', 'B_travel_month_Walk', 'B_travel_month_Bike', 'B_travel_month_Public_Transport', 'B_travel_month_Car']

    # MNL_utilities = {0: 'B_travel_month_Walk*travel_month + B_age_Walk*age + B_female_Walk*female + B_day_of_week_Walk*day_of_week + B_start_time_linear_Walk*start_time_linear + B_car_ownership_Walk*car_ownership + B_driving_license_Walk*driving_license + B_purpose_B_Walk*purpose_B + B_purpose_HBE_Walk*purpose_HBE + B_purpose_HBO_Walk*purpose_HBO + B_purpose_HBW_Walk*purpose_HBW + B_purpose_NHBO_Walk*purpose_NHBO + B_fueltype_Avrg_Walk*fueltype_Average + B_fueltype_Diesel_Walk*fueltype_Diesel + B_fueltype_Hybrid_Walk*fueltype_Hybrid + B_fueltype_Petrol_Walk*fueltype_Petrol + B_distance_Walk*distance + B_dur_walking_Walk*dur_walking',
    #                  1: 'ASC_Bike + B_travel_month_Bike*travel_month + B_age_Bike*age + B_female_Bike*female + B_day_of_week_Bike*day_of_week + B_start_time_linear_Bike*start_time_linear + B_car_ownership_Bike*car_ownership + B_driving_license_Bike*driving_license + B_purpose_B_Bike*purpose_B + B_purpose_HBE_Bike*purpose_HBE + B_purpose_HBO_Bike*purpose_HBO + B_purpose_HBW_Bike*purpose_HBW + B_purpose_NHBO_Bike*purpose_NHBO + B_fueltype_Avrg_Bike*fueltype_Average + B_fueltype_Diesel_Bike*fueltype_Diesel + B_fueltype_Hybrid_Bike*fueltype_Hybrid + B_fueltype_Petrol_Bike*fueltype_Petrol + B_distance_Bike*distance + B_dur_cycling_Bike*dur_cycling',
    #                  2: 'ASC_Public_Transport + B_travel_month_Public_Transport*travel_month + B_age_Public_Transport*age + B_female_Public_Transport*female + B_day_of_week_Public_Transport*day_of_week + B_start_time_linear_Public_Transport*start_time_linear + B_car_ownership_Public_Transport*car_ownership + B_driving_license_Public_Transport*driving_license + B_purpose_B_Public_Transport*purpose_B + B_purpose_HBE_Public_Transport*purpose_HBE + B_purpose_HBO_Public_Transport*purpose_HBO + B_purpose_HBW_Public_Transport*purpose_HBW + B_purpose_NHBO_Public_Transport*purpose_NHBO + B_fueltype_Avrg_Public_Transport*fueltype_Average + B_fueltype_Diesel_Public_Transport*fueltype_Diesel + B_fueltype_Hybrid_Public_Transport*fueltype_Hybrid + B_fueltype_Petrol_Public_Transport*fueltype_Petrol + B_distance_Public_Transport*distance + B_dur_pt_access_Public_Transport*dur_pt_access + B_dur_pt_rail_Public_Transport*dur_pt_rail + B_dur_pt_bus_Public_Transport*dur_pt_bus + B_dur_pt_int_waiting_Public_Transport*dur_pt_int_waiting + B_dur_pt_int_walking_Public_Transport*dur_pt_int_walking + B_pt_n_interchanges_Public_Transport*pt_n_interchanges + B_cost_transit_Public_Transport*cost_transit',
    #                  3: 'ASC_Car + B_travel_month_Car*travel_month + B_traffic_perc_Car*driving_traffic_percent + B_age_Car*age + B_female_Car*female + B_day_of_week_Car*day_of_week + B_start_time_linear_Car*start_time_linear + B_car_ownership_Car*car_ownership + B_driving_license_Car*driving_license + B_purpose_B_Car*purpose_B + B_purpose_HBE_Car*purpose_HBE + B_purpose_HBO_Car*purpose_HBO + B_purpose_HBW_Car*purpose_HBW + B_purpose_NHBO_Car*purpose_NHBO + B_fueltype_Avrg_Car*fueltype_Average + B_fueltype_Diesel_Car*fueltype_Diesel + B_fueltype_Hybrid_Car*fueltype_Hybrid + B_fueltype_Petrol_Car*fueltype_Petrol + B_distance_Car*distance + B_dur_driving_Car*dur_driving + B_cost_driving_total_Car*cost_driving_total'}

    # market segmentation with weekend
    # MNL_beta_params_positive = ['B_car_ownership_Car', 'B_driving_license_Car']
    # MNL_beta_params_negative = ['B_dur_walking_Walk',  'B_dur_cycling_Bike', 'B_dur_pt_access_Public_Transport', 'B_dur_pt_rail_Public_Transport', 'B_dur_pt_bus_Public_Transport', 'B_dur_pt_int_waiting_Public_Transport', 'B_dur_pt_int_walking_Public_Transport', 'B_pt_n_interchanges_Public_Transport', 'B_cost_transit_Public_Transport', 'B_dur_driving_Car', 'B_cost_driving_total_Car', 'B_distance_Walk', 'B_distance_Bike', 'B_distance_Public_Transport', 'B_distance_Car']#, 'B_traffic_perc_Car']
    # MNL_beta_params_neutral = ['ASC_Bike', 'ASC_Public_Transport', 'ASC_Car', 'B_car_ownership_Walk', 'B_car_ownership_Bike', 'B_car_ownership_Public_Transport', 'B_driving_license_Walk', 'B_driving_license_Bike', 'B_driving_license_Public_Transport', 'B_age_Walk', 'B_age_Bike', 'B_age_Public_Transport', 'B_age_Car', 'B_female_Walk', 'B_female_Bike', 'B_female_Public_Transport', 'B_female_Car', 'B_weekend_Walk', 'B_weekend_Bike', 'B_weekend_Public_Transport', 'B_weekend_Car', 'B_start_time_linear_Walk', 'B_start_time_linear_Bike', 'B_start_time_linear_Public_Transport', 'B_start_time_linear_Car', 'B_purpose_B_Walk', 'B_purpose_B_Bike', 'B_purpose_B_Public_Transport', 'B_purpose_B_Car', 'B_purpose_HBE_Walk', 'B_purpose_HBE_Bike', 'B_purpose_HBE_Public_Transport', 'B_purpose_HBE_Car', 'B_purpose_HBO_Walk', 'B_purpose_HBO_Bike', 'B_purpose_HBO_Public_Transport', 'B_purpose_HBO_Car', 'B_purpose_HBW_Walk', 'B_purpose_HBW_Bike', 'B_purpose_HBW_Public_Transport', 'B_purpose_HBW_Car', 'B_purpose_NHBO_Walk', 'B_purpose_NHBO_Bike', 'B_purpose_NHBO_Public_Transport', 'B_purpose_NHBO_Car', 'B_fueltype_Avrg_Walk', 'B_fueltype_Avrg_Bike', 'B_fueltype_Avrg_Public_Transport', 'B_fueltype_Avrg_Car', 'B_fueltype_Diesel_Walk', 'B_fueltype_Diesel_Bike', 'B_fueltype_Diesel_Public_Transport', 'B_fueltype_Diesel_Car', 'B_fueltype_Hybrid_Walk', 'B_fueltype_Hybrid_Bike', 'B_fueltype_Hybrid_Public_Transport', 'B_fueltype_Hybrid_Car', 'B_fueltype_Petrol_Walk', 'B_fueltype_Petrol_Bike', 'B_fueltype_Petrol_Public_Transport', 'B_fueltype_Petrol_Car']

    # MNL_utilities = {0: 'B_age_Walk*age + B_female_Walk*female + B_weekend_Walk*weekend + B_start_time_linear_Walk*start_time_linear + B_car_ownership_Walk*car_ownership + B_driving_license_Walk*driving_license + B_purpose_B_Walk*purpose_B + B_purpose_HBE_Walk*purpose_HBE + B_purpose_HBO_Walk*purpose_HBO + B_purpose_HBW_Walk*purpose_HBW + B_purpose_NHBO_Walk*purpose_NHBO + B_fueltype_Avrg_Walk*fueltype_Average + B_fueltype_Diesel_Walk*fueltype_Diesel + B_fueltype_Hybrid_Walk*fueltype_Hybrid + B_fueltype_Petrol_Walk*fueltype_Petrol + B_distance_Walk*distance + B_dur_walking_Walk*dur_walking',
    #                  1: 'ASC_Bike + B_age_Bike*age + B_female_Bike*female + B_weekend_Bike*weekend + B_start_time_linear_Bike*start_time_linear + B_car_ownership_Bike*car_ownership + B_driving_license_Bike*driving_license + B_purpose_B_Bike*purpose_B + B_purpose_HBE_Bike*purpose_HBE + B_purpose_HBO_Bike*purpose_HBO + B_purpose_HBW_Bike*purpose_HBW + B_purpose_NHBO_Bike*purpose_NHBO + B_fueltype_Avrg_Bike*fueltype_Average + B_fueltype_Diesel_Bike*fueltype_Diesel + B_fueltype_Hybrid_Bike*fueltype_Hybrid + B_fueltype_Petrol_Bike*fueltype_Petrol + B_distance_Bike*distance + B_dur_cycling_Bike*dur_cycling',
    #                  2: 'ASC_Public_Transport + B_age_Public_Transport*age + B_female_Public_Transport*female + B_weekend_Public_Transport*weekend + B_start_time_linear_Public_Transport*start_time_linear + B_car_ownership_Public_Transport*car_ownership + B_driving_license_Public_Transport*driving_license + B_purpose_B_Public_Transport*purpose_B + B_purpose_HBE_Public_Transport*purpose_HBE + B_purpose_HBO_Public_Transport*purpose_HBO + B_purpose_HBW_Public_Transport*purpose_HBW + B_purpose_NHBO_Public_Transport*purpose_NHBO + B_fueltype_Avrg_Public_Transport*fueltype_Average + B_fueltype_Diesel_Public_Transport*fueltype_Diesel + B_fueltype_Hybrid_Public_Transport*fueltype_Hybrid + B_fueltype_Petrol_Public_Transport*fueltype_Petrol + B_distance_Public_Transport*distance + B_dur_pt_access_Public_Transport*dur_pt_access + B_dur_pt_rail_Public_Transport*dur_pt_rail + B_dur_pt_bus_Public_Transport*dur_pt_bus + B_dur_pt_int_waiting_Public_Transport*dur_pt_int_waiting + B_dur_pt_int_walking_Public_Transport*dur_pt_int_walking + B_pt_n_interchanges_Public_Transport*pt_n_interchanges + B_cost_transit_Public_Transport*cost_transit',
    #                  3: 'ASC_Car + B_age_Car*age + B_female_Car*female + B_weekend_Car*weekend + B_start_time_linear_Car*start_time_linear + B_car_ownership_Car*car_ownership + B_driving_license_Car*driving_license + B_purpose_B_Car*purpose_B + B_purpose_HBE_Car*purpose_HBE + B_purpose_HBO_Car*purpose_HBO + B_purpose_HBW_Car*purpose_HBW + B_purpose_NHBO_Car*purpose_NHBO + B_fueltype_Avrg_Car*fueltype_Average + B_fueltype_Diesel_Car*fueltype_Diesel + B_fueltype_Hybrid_Car*fueltype_Hybrid + B_fueltype_Petrol_Car*fueltype_Petrol + B_distance_Car*distance + B_dur_driving_Car*dur_driving + B_cost_driving_total_Car*cost_driving_total'}

    # construct the model parameters
    for beta in MNL_beta_params_positive:
        exec("{} = Beta('{}', 0, 0, None, 0)".format(beta, beta), globals())
    for beta in MNL_beta_params_negative:
        exec("{} = Beta('{}', 0, None, 0, 0)".format(beta, beta), globals())
    for beta in MNL_beta_params_neutral:
        exec("{} = Beta('{}', 0, None, None, 0)".format(beta, beta), globals())

    # define utility functions
    for utility_idx in MNL_utilities.keys():
        exec("V_{} = {}".format(utility_idx, MNL_utilities[utility_idx]), globals())

    # assign utility functions to utility indices
    exec("V_dict = {}", globals())
    for utility_idx in MNL_utilities.keys():
        exec("V_dict[{}] = V_{}".format(utility_idx, utility_idx), globals())

    # associate the availability conditions with the alternatives
    exec("av = {}", globals())
    for utility_idx in MNL_utilities.keys():
        exec("av[{}] = 1".format(utility_idx), globals())

    # definition of the model
    logprob = loglogit(V_dict, av, choice)

    # create the Biogeme object
    biogeme = bio.BIOGEME(database_train, logprob)
    biogeme.modelName = "LPMC"

    biogeme.generate_html = False
    biogeme.generate_pickle = False

    if for_prob:
        prob_0 = logit(V_dict, av, 0)
        prob_1 = logit(V_dict, av, 1)
        prob_2 = logit(V_dict, av, 2)
        prob_3 = logit(V_dict, av, 3)

        simulate = {
            "Prob. 0": prob_0,
            "Prob. 1": prob_1,
            "Prob. 2": prob_2,
            "Prob. 3": prob_3,
        }
        biosim = bio.BIOGEME(database_train, simulate)
        biosim.modelName = "LPMC_logit_test"

        biosim.generate_html = False
        biosim.generate_pickle = False

        return biosim

    return biogeme


def LPMC_normalised(dataset_train, for_prob=False):
    """
    Create a MNL on the LPMC dataset, normalised for biogeme estimation.
    The model is a slightly modified version from teh code that can be found here: https://github.com/JoseAngelMartinB/prediction-behavioural-analysis-ml-travel-mode-choice.

    Parameters
    ----------
    dataset_train : pandas DataFrame
        The training dataset.

    Returns
    -------
    biogeme : bio.BIOGEME
        The BIOGEME object containing the model.

    """
    database_train = db.Database("LTDS_train", dataset_train)

    globals().update(database_train.variables)

    # several model specifications are available below - the best one is the uncommented one.

    # driving_percentage, congestion charge as a binary variable NEW PB 0.6730 with lr = 0.1
    MNL_beta_params_positive = ["B_car_ownership_Car", "B_driving_license_Car"]
    MNL_beta_params_negative = [
        "B_dur_walking_Walk",
        "B_dur_cycling_Bike",
        "B_dur_pt_access_Public_Transport",
        "B_dur_pt_rail_Public_Transport",
        "B_dur_pt_bus_Public_Transport",
        "B_dur_pt_int_waiting_Public_Transport",
        "B_dur_pt_int_walking_Public_Transport",
        "B_pt_n_interchanges_Public_Transport",
        "B_cost_transit_Public_Transport",
        "B_dur_driving_Car",
        "B_cost_driving_fuel_Car",
        "B_distance_Walk",
        "B_distance_Bike",
        "B_distance_Public_Transport",
        "B_distance_Car",
        "B_con_charge_Car",
        "B_traffic_perc_Car",
    ]
    MNL_beta_params_neutral = [
        "ASC_Bike",
        "ASC_Public_Transport",
        "ASC_Car",
        "B_car_ownership_Walk",
        "B_car_ownership_Bike",
        "B_car_ownership_Public_Transport",
        "B_driving_license_Walk",
        "B_driving_license_Bike",
        "B_driving_license_Public_Transport",
        "B_age_Walk",
        "B_age_Bike",
        "B_age_Public_Transport",
        "B_age_Car",
        "B_female_Walk",
        "B_female_Bike",
        "B_female_Public_Transport",
        "B_female_Car",
        "B_day_of_week_Walk",
        "B_day_of_week_Bike",
        "B_day_of_week_Public_Transport",
        "B_day_of_week_Car",
        "B_start_time_linear_Walk",
        "B_start_time_linear_Bike",
        "B_start_time_linear_Public_Transport",
        "B_start_time_linear_Car",
        "B_purpose_B_Walk",
        "B_purpose_B_Bike",
        "B_purpose_B_Public_Transport",
        "B_purpose_B_Car",
        "B_purpose_HBE_Walk",
        "B_purpose_HBE_Bike",
        "B_purpose_HBE_Public_Transport",
        "B_purpose_HBE_Car",
        "B_purpose_HBO_Walk",
        "B_purpose_HBO_Bike",
        "B_purpose_HBO_Public_Transport",
        "B_purpose_HBO_Car",
        "B_purpose_HBW_Walk",
        "B_purpose_HBW_Bike",
        "B_purpose_HBW_Public_Transport",
        "B_purpose_HBW_Car",
        "B_purpose_NHBO_Walk",
        "B_purpose_NHBO_Bike",
        "B_purpose_NHBO_Public_Transport",
        "B_purpose_NHBO_Car",
        "B_fueltype_Avrg_Walk",
        "B_fueltype_Avrg_Bike",
        "B_fueltype_Avrg_Public_Transport",
        "B_fueltype_Avrg_Car",
        "B_fueltype_Diesel_Walk",
        "B_fueltype_Diesel_Bike",
        "B_fueltype_Diesel_Public_Transport",
        "B_fueltype_Diesel_Car",
        "B_fueltype_Hybrid_Walk",
        "B_fueltype_Hybrid_Bike",
        "B_fueltype_Hybrid_Public_Transport",
        "B_fueltype_Hybrid_Car",
        "B_fueltype_Petrol_Walk",
        "B_fueltype_Petrol_Bike",
        "B_fueltype_Petrol_Public_Transport",
        "B_fueltype_Petrol_Car",
    ]

    MNL_utilities = {
        0: "B_dur_walking_Walk*dur_walking",  # + B_age_Walk*age + B_female_Walk*female + B_day_of_week_Walk*day_of_week + B_start_time_linear_Walk*start_time_linear + B_car_ownership_Walk*car_ownership + B_driving_license_Walk*driving_license + B_purpose_B_Walk*purpose_B + B_purpose_HBE_Walk*purpose_HBE + B_purpose_HBO_Walk*purpose_HBO + B_purpose_HBW_Walk*purpose_HBW + B_purpose_NHBO_Walk*purpose_NHBO + B_fueltype_Avrg_Walk*fueltype_Average + B_fueltype_Diesel_Walk*fueltype_Diesel + B_fueltype_Hybrid_Walk*fueltype_Hybrid + B_fueltype_Petrol_Walk*fueltype_Petrol + B_distance_Walk*distance',
        1: "ASC_Bike + B_age_Bike*age + B_female_Bike*female + B_day_of_week_Bike*day_of_week + B_start_time_linear_Bike*start_time_linear + B_car_ownership_Bike*car_ownership + B_driving_license_Bike*driving_license + B_purpose_B_Bike*purpose_B + B_purpose_HBE_Bike*purpose_HBE + B_purpose_HBO_Bike*purpose_HBO + B_purpose_HBW_Bike*purpose_HBW + B_purpose_NHBO_Bike*purpose_NHBO + B_fueltype_Avrg_Bike*fueltype_Average + B_fueltype_Diesel_Bike*fueltype_Diesel + B_fueltype_Hybrid_Bike*fueltype_Hybrid + B_fueltype_Petrol_Bike*fueltype_Petrol + B_distance_Bike*distance + B_dur_cycling_Bike*dur_cycling",
        2: "ASC_Public_Transport + B_age_Public_Transport*age + B_female_Public_Transport*female + B_day_of_week_Public_Transport*day_of_week + B_start_time_linear_Public_Transport*start_time_linear + B_car_ownership_Public_Transport*car_ownership + B_driving_license_Public_Transport*driving_license + B_purpose_B_Public_Transport*purpose_B + B_purpose_HBE_Public_Transport*purpose_HBE + B_purpose_HBO_Public_Transport*purpose_HBO + B_purpose_HBW_Public_Transport*purpose_HBW + B_purpose_NHBO_Public_Transport*purpose_NHBO + B_fueltype_Avrg_Public_Transport*fueltype_Average + B_fueltype_Diesel_Public_Transport*fueltype_Diesel + B_fueltype_Hybrid_Public_Transport*fueltype_Hybrid + B_fueltype_Petrol_Public_Transport*fueltype_Petrol + B_distance_Public_Transport*distance + B_dur_pt_access_Public_Transport*dur_pt_access + B_dur_pt_rail_Public_Transport*dur_pt_rail + B_dur_pt_bus_Public_Transport*dur_pt_bus + B_dur_pt_int_waiting_Public_Transport*dur_pt_int_waiting + B_dur_pt_int_walking_Public_Transport*dur_pt_int_walking + B_pt_n_interchanges_Public_Transport*pt_n_interchanges + B_cost_transit_Public_Transport*cost_transit",
        3: "ASC_Car + B_age_Car*age + B_female_Car*female + B_day_of_week_Car*day_of_week + B_start_time_linear_Car*start_time_linear + B_car_ownership_Car*car_ownership + B_driving_license_Car*driving_license + B_purpose_B_Car*purpose_B + B_purpose_HBE_Car*purpose_HBE + B_purpose_HBO_Car*purpose_HBO + B_purpose_HBW_Car*purpose_HBW + B_purpose_NHBO_Car*purpose_NHBO + B_fueltype_Avrg_Car*fueltype_Average + B_fueltype_Diesel_Car*fueltype_Diesel + B_fueltype_Hybrid_Car*fueltype_Hybrid + B_fueltype_Petrol_Car*fueltype_Petrol + B_distance_Car*distance + B_dur_driving_Car*dur_driving + B_cost_driving_fuel_Car*cost_driving_fuel + B_con_charge_Car*congestion_charge + B_traffic_perc_Car*driving_traffic_percent",
    }

    # construct the model parameters
    for beta in MNL_beta_params_positive:
        exec("{} = Beta('{}', 0, 0, None, 0)".format(beta, beta), globals())
    for beta in MNL_beta_params_negative:
        exec("{} = Beta('{}', 0, None, 0, 0)".format(beta, beta), globals())
    for beta in MNL_beta_params_neutral:
        exec("{} = Beta('{}', 0, None, None, 0)".format(beta, beta), globals())

    # define utility functions
    for utility_idx in MNL_utilities.keys():
        exec("V_{} = {}".format(utility_idx, MNL_utilities[utility_idx]), globals())

    # assign utility functions to utility indices
    exec("V_dict = {}", globals())
    for utility_idx in MNL_utilities.keys():
        exec("V_dict[{}] = V_{}".format(utility_idx, utility_idx), globals())

    # associate the availability conditions with the alternatives
    exec("av = {}", globals())
    for utility_idx in MNL_utilities.keys():
        exec("av[{}] = 1".format(utility_idx), globals())

    # definition of the model
    logprob = loglogit(V_dict, av, choice)

    # create the Biogeme object
    biogeme = bio.BIOGEME(database_train, logprob)
    biogeme.modelName = "LPMC"

    biogeme.generate_html = False
    biogeme.generate_pickle = False

    if for_prob:
        prob_0 = logit(V_dict, av, 0)
        prob_1 = logit(V_dict, av, 1)
        prob_2 = logit(V_dict, av, 2)
        prob_3 = logit(V_dict, av, 3)

        simulate = {
            "Prob. 0": prob_0,
            "Prob. 1": prob_1,
            "Prob. 2": prob_2,
            "Prob. 3": prob_3,
        }
        biosim = bio.BIOGEME(database_train, simulate)
        biosim.modelName = "LPMC_logit_test"

        biosim.generate_html = False
        biosim.generate_pickle = False

        return biosim

    return biogeme


def LPMC_nested(dataset_train, for_prob=False):
    """
    Create a nested logit model on the LPMC dataset.
    The model is a slightly modified version from teh code that can be found here: https://github.com/JoseAngelMartinB/prediction-behavioural-analysis-ml-travel-mode-choice.

    Parameters
    ----------
    dataset_train : pandas DataFrame
        The training dataset.

    Returns
    -------
    biogeme : bio.BIOGEME
        The BIOGEME object containing the model.

    """
    database_train = db.Database("LTDS_train", dataset_train)

    globals().update(database_train.variables)

    logger = blog.get_screen_logger(level=blog.DEBUG)
    logger.info("Nested LPMC")

    # several model specifications are available below - the best one is the uncommented one.
    # driving_percentage, congestion charge as a binary variable NEW PB 0.6730 with lr = 0.1
    MNL_beta_params_positive = ["B_car_ownership_Car", "B_driving_license_Car"]
    MNL_beta_params_negative = [
        "B_dur_walking_Walk",
        "B_dur_cycling_Bike",
        "B_dur_pt_access_Public_Transport",
        "B_dur_pt_rail_Public_Transport",
        "B_dur_pt_bus_Public_Transport",
        "B_dur_pt_int_waiting_Public_Transport",
        "B_dur_pt_int_walking_Public_Transport",
        "B_pt_n_interchanges_Public_Transport",
        "B_cost_transit_Public_Transport",
        "B_dur_driving_Car",
        "B_cost_driving_fuel_Car",
        "B_distance_Walk",
        "B_distance_Bike",
        "B_distance_Public_Transport",
        "B_distance_Car",
        "B_con_charge_Car",
        "B_traffic_perc_Car",
    ]
    MNL_beta_params_neutral = [
        "ASC_Bike",
        "ASC_Public_Transport",
        "ASC_Car",
        "B_car_ownership_Walk",
        "B_car_ownership_Bike",
        "B_car_ownership_Public_Transport",
        "B_driving_license_Walk",
        "B_driving_license_Bike",
        "B_driving_license_Public_Transport",
        "B_age_Walk",
        "B_age_Bike",
        "B_age_Public_Transport",
        "B_age_Car",
        "B_female_Walk",
        "B_female_Bike",
        "B_female_Public_Transport",
        "B_female_Car",
        "B_day_of_week_Walk",
        "B_day_of_week_Bike",
        "B_day_of_week_Public_Transport",
        "B_day_of_week_Car",
        "B_start_time_linear_Walk",
        "B_start_time_linear_Bike",
        "B_start_time_linear_Public_Transport",
        "B_start_time_linear_Car",
        "B_purpose_B_Walk",
        "B_purpose_B_Bike",
        "B_purpose_B_Public_Transport",
        "B_purpose_B_Car",
        "B_purpose_HBE_Walk",
        "B_purpose_HBE_Bike",
        "B_purpose_HBE_Public_Transport",
        "B_purpose_HBE_Car",
        "B_purpose_HBO_Walk",
        "B_purpose_HBO_Bike",
        "B_purpose_HBO_Public_Transport",
        "B_purpose_HBO_Car",
        "B_purpose_HBW_Walk",
        "B_purpose_HBW_Bike",
        "B_purpose_HBW_Public_Transport",
        "B_purpose_HBW_Car",
        "B_purpose_NHBO_Walk",
        "B_purpose_NHBO_Bike",
        "B_purpose_NHBO_Public_Transport",
        "B_purpose_NHBO_Car",
        "B_fueltype_Avrg_Walk",
        "B_fueltype_Avrg_Bike",
        "B_fueltype_Avrg_Public_Transport",
        "B_fueltype_Avrg_Car",
        "B_fueltype_Diesel_Walk",
        "B_fueltype_Diesel_Bike",
        "B_fueltype_Diesel_Public_Transport",
        "B_fueltype_Diesel_Car",
        "B_fueltype_Hybrid_Walk",
        "B_fueltype_Hybrid_Bike",
        "B_fueltype_Hybrid_Public_Transport",
        "B_fueltype_Hybrid_Car",
        "B_fueltype_Petrol_Walk",
        "B_fueltype_Petrol_Bike",
        "B_fueltype_Petrol_Public_Transport",
        "B_fueltype_Petrol_Car",
    ]

    MNL_utilities = {
        0: "B_age_Walk*age + B_female_Walk*female + B_day_of_week_Walk*day_of_week + B_start_time_linear_Walk*start_time_linear + B_car_ownership_Walk*car_ownership + B_driving_license_Walk*driving_license + B_purpose_B_Walk*purpose_B + B_purpose_HBE_Walk*purpose_HBE + B_purpose_HBO_Walk*purpose_HBO + B_purpose_HBW_Walk*purpose_HBW + B_purpose_NHBO_Walk*purpose_NHBO + B_fueltype_Avrg_Walk*fueltype_Average + B_fueltype_Diesel_Walk*fueltype_Diesel + B_fueltype_Hybrid_Walk*fueltype_Hybrid + B_fueltype_Petrol_Walk*fueltype_Petrol + B_distance_Walk*distance + B_dur_walking_Walk*dur_walking",
        1: "ASC_Bike + B_age_Bike*age + B_female_Bike*female + B_day_of_week_Bike*day_of_week + B_start_time_linear_Bike*start_time_linear + B_car_ownership_Bike*car_ownership + B_driving_license_Bike*driving_license + B_purpose_B_Bike*purpose_B + B_purpose_HBE_Bike*purpose_HBE + B_purpose_HBO_Bike*purpose_HBO + B_purpose_HBW_Bike*purpose_HBW + B_purpose_NHBO_Bike*purpose_NHBO + B_fueltype_Avrg_Bike*fueltype_Average + B_fueltype_Diesel_Bike*fueltype_Diesel + B_fueltype_Hybrid_Bike*fueltype_Hybrid + B_fueltype_Petrol_Bike*fueltype_Petrol + B_distance_Bike*distance + B_dur_cycling_Bike*dur_cycling",
        2: "ASC_Public_Transport + B_age_Public_Transport*age + B_female_Public_Transport*female + B_day_of_week_Public_Transport*day_of_week + B_start_time_linear_Public_Transport*start_time_linear + B_car_ownership_Public_Transport*car_ownership + B_driving_license_Public_Transport*driving_license + B_purpose_B_Public_Transport*purpose_B + B_purpose_HBE_Public_Transport*purpose_HBE + B_purpose_HBO_Public_Transport*purpose_HBO + B_purpose_HBW_Public_Transport*purpose_HBW + B_purpose_NHBO_Public_Transport*purpose_NHBO + B_fueltype_Avrg_Public_Transport*fueltype_Average + B_fueltype_Diesel_Public_Transport*fueltype_Diesel + B_fueltype_Hybrid_Public_Transport*fueltype_Hybrid + B_fueltype_Petrol_Public_Transport*fueltype_Petrol + B_distance_Public_Transport*distance + B_dur_pt_access_Public_Transport*dur_pt_access + B_dur_pt_rail_Public_Transport*dur_pt_rail + B_dur_pt_bus_Public_Transport*dur_pt_bus + B_dur_pt_int_waiting_Public_Transport*dur_pt_int_waiting + B_dur_pt_int_walking_Public_Transport*dur_pt_int_walking + B_pt_n_interchanges_Public_Transport*pt_n_interchanges + B_cost_transit_Public_Transport*cost_transit",
        3: "ASC_Car + B_age_Car*age + B_female_Car*female + B_day_of_week_Car*day_of_week + B_start_time_linear_Car*start_time_linear + B_car_ownership_Car*car_ownership + B_driving_license_Car*driving_license + B_purpose_B_Car*purpose_B + B_purpose_HBE_Car*purpose_HBE + B_purpose_HBO_Car*purpose_HBO + B_purpose_HBW_Car*purpose_HBW + B_purpose_NHBO_Car*purpose_NHBO + B_fueltype_Avrg_Car*fueltype_Average + B_fueltype_Diesel_Car*fueltype_Diesel + B_fueltype_Hybrid_Car*fueltype_Hybrid + B_fueltype_Petrol_Car*fueltype_Petrol + B_distance_Car*distance + B_dur_driving_Car*dur_driving + B_cost_driving_fuel_Car*cost_driving_fuel + B_con_charge_Car*congestion_charge + B_traffic_perc_Car*driving_traffic_percent",
    }

    # best model until now, 0.6790 with lr = 0.3
    # MNL_beta_params_positive = ['B_car_ownership_Car', 'B_driving_license_Car']
    # MNL_beta_params_negative = ['B_dur_walking_Walk',  'B_dur_cycling_Bike', 'B_dur_pt_access_Public_Transport', 'B_dur_pt_rail_Public_Transport', 'B_dur_pt_bus_Public_Transport', 'B_dur_pt_int_waiting_Public_Transport', 'B_dur_pt_int_walking_Public_Transport', 'B_pt_n_interchanges_Public_Transport', 'B_cost_transit_Public_Transport', 'B_dur_driving_Car', 'B_cost_driving_total_Car', 'B_distance_Walk', 'B_distance_Bike', 'B_distance_Public_Transport', 'B_distance_Car']#, 'B_traffic_perc_Car']
    # MNL_beta_params_neutral = ['ASC_Bike', 'ASC_Public_Transport', 'ASC_Car', 'B_car_ownership_Walk', 'B_car_ownership_Bike', 'B_car_ownership_Public_Transport', 'B_driving_license_Walk', 'B_driving_license_Bike', 'B_driving_license_Public_Transport', 'B_age_Walk', 'B_age_Bike', 'B_age_Public_Transport', 'B_age_Car', 'B_female_Walk', 'B_female_Bike', 'B_female_Public_Transport', 'B_female_Car', 'B_day_of_week_Walk', 'B_day_of_week_Bike', 'B_day_of_week_Public_Transport', 'B_day_of_week_Car', 'B_start_time_linear_Walk', 'B_start_time_linear_Bike', 'B_start_time_linear_Public_Transport', 'B_start_time_linear_Car', 'B_purpose_B_Walk', 'B_purpose_B_Bike', 'B_purpose_B_Public_Transport', 'B_purpose_B_Car', 'B_purpose_HBE_Walk', 'B_purpose_HBE_Bike', 'B_purpose_HBE_Public_Transport', 'B_purpose_HBE_Car', 'B_purpose_HBO_Walk', 'B_purpose_HBO_Bike', 'B_purpose_HBO_Public_Transport', 'B_purpose_HBO_Car', 'B_purpose_HBW_Walk', 'B_purpose_HBW_Bike', 'B_purpose_HBW_Public_Transport', 'B_purpose_HBW_Car', 'B_purpose_NHBO_Walk', 'B_purpose_NHBO_Bike', 'B_purpose_NHBO_Public_Transport', 'B_purpose_NHBO_Car', 'B_fueltype_Avrg_Walk', 'B_fueltype_Avrg_Bike', 'B_fueltype_Avrg_Public_Transport', 'B_fueltype_Avrg_Car', 'B_fueltype_Diesel_Walk', 'B_fueltype_Diesel_Bike', 'B_fueltype_Diesel_Public_Transport', 'B_fueltype_Diesel_Car', 'B_fueltype_Hybrid_Walk', 'B_fueltype_Hybrid_Bike', 'B_fueltype_Hybrid_Public_Transport', 'B_fueltype_Hybrid_Car', 'B_fueltype_Petrol_Walk', 'B_fueltype_Petrol_Bike', 'B_fueltype_Petrol_Public_Transport', 'B_fueltype_Petrol_Car']

    # MNL_utilities = {0: 'B_dur_walking_Walk*dur_walking', #B_age_Walk*age + B_female_Walk*female + B_day_of_week_Walk*day_of_week + B_start_time_linear_Walk*start_time_linear + B_car_ownership_Walk*car_ownership + B_driving_license_Walk*driving_license + B_purpose_B_Walk*purpose_B + B_purpose_HBE_Walk*purpose_HBE + B_purpose_HBO_Walk*purpose_HBO + B_purpose_HBW_Walk*purpose_HBW + B_purpose_NHBO_Walk*purpose_NHBO + B_fueltype_Avrg_Walk*fueltype_Average + B_fueltype_Diesel_Walk*fueltype_Diesel + B_fueltype_Hybrid_Walk*fueltype_Hybrid + B_fueltype_Petrol_Walk*fueltype_Petrol + B_distance_Walk*distance + B_dur_walking_Walk*dur_walking',
    #                     1: 'ASC_Bike + B_age_Bike*age + B_female_Bike*female + B_day_of_week_Bike*day_of_week + B_start_time_linear_Bike*start_time_linear + B_car_ownership_Bike*car_ownership + B_driving_license_Bike*driving_license + B_purpose_B_Bike*purpose_B + B_purpose_HBE_Bike*purpose_HBE + B_purpose_HBO_Bike*purpose_HBO + B_purpose_HBW_Bike*purpose_HBW + B_purpose_NHBO_Bike*purpose_NHBO + B_fueltype_Avrg_Bike*fueltype_Average + B_fueltype_Diesel_Bike*fueltype_Diesel + B_fueltype_Hybrid_Bike*fueltype_Hybrid + B_fueltype_Petrol_Bike*fueltype_Petrol + B_distance_Bike*distance + B_dur_cycling_Bike*dur_cycling',
    #                     2: 'ASC_Public_Transport + B_age_Public_Transport*age + B_female_Public_Transport*female + B_day_of_week_Public_Transport*day_of_week + B_start_time_linear_Public_Transport*start_time_linear + B_car_ownership_Public_Transport*car_ownership + B_driving_license_Public_Transport*driving_license + B_purpose_B_Public_Transport*purpose_B + B_purpose_HBE_Public_Transport*purpose_HBE + B_purpose_HBO_Public_Transport*purpose_HBO + B_purpose_HBW_Public_Transport*purpose_HBW + B_purpose_NHBO_Public_Transport*purpose_NHBO + B_fueltype_Avrg_Public_Transport*fueltype_Average + B_fueltype_Diesel_Public_Transport*fueltype_Diesel + B_fueltype_Hybrid_Public_Transport*fueltype_Hybrid + B_fueltype_Petrol_Public_Transport*fueltype_Petrol + B_distance_Public_Transport*distance + B_dur_pt_access_Public_Transport*dur_pt_access + B_dur_pt_rail_Public_Transport*dur_pt_rail + B_dur_pt_bus_Public_Transport*dur_pt_bus + B_dur_pt_int_waiting_Public_Transport*dur_pt_int_waiting + B_dur_pt_int_walking_Public_Transport*dur_pt_int_walking + B_pt_n_interchanges_Public_Transport*pt_n_interchanges + B_cost_transit_Public_Transport*cost_transit',
    #                     3: 'ASC_Car + B_age_Car*age + B_female_Car*female + B_day_of_week_Car*day_of_week + B_start_time_linear_Car*start_time_linear + B_car_ownership_Car*car_ownership + B_driving_license_Car*driving_license + B_purpose_B_Car*purpose_B + B_purpose_HBE_Car*purpose_HBE + B_purpose_HBO_Car*purpose_HBO + B_purpose_HBW_Car*purpose_HBW + B_purpose_NHBO_Car*purpose_NHBO + B_fueltype_Avrg_Car*fueltype_Average + B_fueltype_Diesel_Car*fueltype_Diesel + B_fueltype_Hybrid_Car*fueltype_Hybrid + B_fueltype_Petrol_Car*fueltype_Petrol + B_distance_Car*distance + B_dur_driving_Car*dur_driving + B_cost_driving_total_Car*cost_driving_total'}

    # 0.6805
    # MNL_beta_params_positive = ['B_car_ownership_Car', 'B_driving_license_Car']
    # MNL_beta_params_negative = ['B_age_Bike', 'B_distance_Bike', 'B_distance_Public_Transport', 'B_distance_Car','B_driving_license_Public_Transport', 'B_car_ownership_Bike', 'B_car_ownership_Public_Transport', 'B_driving_license_Bike', 'B_dur_walking_Walk', 'B_dur_cycling_Bike', 'B_dur_pt_access_Public_Transport', 'B_dur_pt_rail_Public_Transport', 'B_dur_pt_bus_Public_Transport', 'B_dur_pt_int_waiting_Public_Transport', 'B_dur_pt_int_walking_Public_Transport', 'B_pt_n_interchanges_Public_Transport', 'B_cost_transit_Public_Transport', 'B_dur_driving_Car', 'B_cost_driving_total_Car']
    # MNL_beta_params_neutral = ['ASC_Bike', 'ASC_Public_Transport', 'ASC_Car', 'B_age_Public_Transport', 'B_age_Car', 'B_female_Bike', 'B_female_Public_Transport', 'B_female_Car', 'B_day_of_week_Bike', 'B_day_of_week_Public_Transport', 'B_day_of_week_Car', 'B_start_time_linear_Bike', 'B_start_time_linear_Public_Transport', 'B_start_time_linear_Car', 'B_purpose_B_Bike', 'B_purpose_B_Public_Transport', 'B_purpose_B_Car', 'B_purpose_HBE_Bike', 'B_purpose_HBE_Public_Transport', 'B_purpose_HBE_Car', 'B_purpose_HBO_Bike', 'B_purpose_HBO_Public_Transport', 'B_purpose_HBO_Car', 'B_purpose_HBW_Bike', 'B_purpose_HBW_Public_Transport', 'B_purpose_HBW_Car', 'B_purpose_NHBO_Bike', 'B_purpose_NHBO_Public_Transport', 'B_purpose_NHBO_Car','B_fueltype_Avrg_Bike', 'B_fueltype_Avrg_Public_Transport', 'B_fueltype_Avrg_Car', 'B_fueltype_Diesel_Bike', 'B_fueltype_Diesel_Public_Transport', 'B_fueltype_Diesel_Car', 'B_fueltype_Hybrid_Bike', 'B_fueltype_Hybrid_Public_Transport', 'B_fueltype_Hybrid_Car', 'B_fueltype_Petrol_Bike', 'B_fueltype_Petrol_Public_Transport', 'B_fueltype_Petrol_Car']

    # MNL_utilities = {0: 'B_dur_walking_Walk*dur_walking',
    #                  1: 'ASC_Bike + B_age_Bike*age + B_female_Bike*female + B_day_of_week_Bike*day_of_week + B_start_time_linear_Bike*start_time_linear + B_car_ownership_Bike*car_ownership + B_driving_license_Bike*driving_license + B_purpose_B_Bike*purpose_B + B_purpose_HBE_Bike*purpose_HBE + B_purpose_HBO_Bike*purpose_HBO + B_purpose_HBW_Bike*purpose_HBW + B_purpose_NHBO_Bike*purpose_NHBO + B_fueltype_Avrg_Bike*fueltype_Average + B_fueltype_Diesel_Bike*fueltype_Diesel + B_fueltype_Hybrid_Bike*fueltype_Hybrid + B_fueltype_Petrol_Bike*fueltype_Petrol + B_distance_Bike*distance + B_dur_cycling_Bike*dur_cycling',
    #                  2: 'ASC_Public_Transport + B_age_Public_Transport*age + B_female_Public_Transport*female + B_day_of_week_Public_Transport*day_of_week + B_start_time_linear_Public_Transport*start_time_linear + B_car_ownership_Public_Transport*car_ownership + B_driving_license_Public_Transport*driving_license + B_purpose_B_Public_Transport*purpose_B + B_purpose_HBE_Public_Transport*purpose_HBE + B_purpose_HBO_Public_Transport*purpose_HBO + B_purpose_HBW_Public_Transport*purpose_HBW + B_purpose_NHBO_Public_Transport*purpose_NHBO + B_fueltype_Avrg_Public_Transport*fueltype_Average + B_fueltype_Diesel_Public_Transport*fueltype_Diesel + B_fueltype_Hybrid_Public_Transport*fueltype_Hybrid + B_fueltype_Petrol_Public_Transport*fueltype_Petrol + B_dur_pt_access_Public_Transport*dur_pt_access + B_dur_pt_rail_Public_Transport*dur_pt_rail + B_dur_pt_bus_Public_Transport*dur_pt_bus + B_dur_pt_int_waiting_Public_Transport*dur_pt_int_waiting + B_dur_pt_int_walking_Public_Transport*dur_pt_int_walking + B_pt_n_interchanges_Public_Transport*pt_n_interchanges + B_cost_transit_Public_Transport*cost_transit + B_distance_Public_Transport*distance',
    #                  3: 'ASC_Car + B_age_Car*age + B_female_Car*female + B_day_of_week_Car*day_of_week + B_start_time_linear_Car*start_time_linear + B_car_ownership_Car*car_ownership + B_driving_license_Car*driving_license + B_purpose_B_Car*purpose_B + B_purpose_HBE_Car*purpose_HBE + B_purpose_HBO_Car*purpose_HBO + B_purpose_HBW_Car*purpose_HBW + B_purpose_NHBO_Car*purpose_NHBO + B_fueltype_Avrg_Car*fueltype_Average + B_fueltype_Diesel_Car*fueltype_Diesel + B_fueltype_Hybrid_Car*fueltype_Hybrid + B_fueltype_Petrol_Car*fueltype_Petrol + B_dur_driving_Car*dur_driving + B_cost_driving_total_Car*cost_driving_total + B_distance_Car*distance'}

    # 0.6791 but misspecification
    # MNL_beta_params_positive = ['B_car_ownership_Car', 'B_driving_license_Car']
    # MNL_beta_params_negative = ['B_dur_walking_Walk',  'B_dur_cycling_Bike', 'B_dur_pt_access_Public_Transport', 'B_dur_pt_rail_Public_Transport', 'B_dur_pt_bus_Public_Transport', 'B_dur_pt_int_waiting_Public_Transport', 'B_dur_pt_int_walking_Public_Transport', 'B_pt_n_interchanges_Public_Transport', 'B_cost_transit_Public_Transport', 'B_dur_driving_Car', 'B_cost_driving_total_Car']#, 'B_traffic_perc_Car']
    # MNL_beta_params_neutral = ['ASC_Bike', 'ASC_Public_Transport', 'ASC_Car', 'B_car_ownership_Walk', 'B_car_ownership_Bike', 'B_car_ownership_Public_Transport', 'B_driving_license_Walk', 'B_driving_license_Bike', 'B_driving_license_Public_Transport', 'B_age_Walk', 'B_age_Bike', 'B_age_Public_Transport', 'B_age_Car', 'B_female_Walk', 'B_female_Bike', 'B_female_Public_Transport', 'B_female_Car', 'B_distance_Walk', 'B_distance_Bike', 'B_day_of_week_Walk', 'B_day_of_week_Bike', 'B_day_of_week_Public_Transport', 'B_day_of_week_Car', 'B_start_time_linear_Walk', 'B_start_time_linear_Bike', 'B_start_time_linear_Public_Transport', 'B_start_time_linear_Car', 'B_purpose_B_Walk', 'B_purpose_B_Bike', 'B_purpose_B_Public_Transport', 'B_purpose_B_Car', 'B_purpose_HBE_Walk', 'B_purpose_HBE_Bike', 'B_purpose_HBE_Public_Transport', 'B_purpose_HBE_Car', 'B_purpose_HBO_Walk', 'B_purpose_HBO_Bike', 'B_purpose_HBO_Public_Transport', 'B_purpose_HBO_Car', 'B_purpose_HBW_Walk', 'B_purpose_HBW_Bike', 'B_purpose_HBW_Public_Transport', 'B_purpose_HBW_Car', 'B_purpose_NHBO_Walk', 'B_purpose_NHBO_Bike', 'B_purpose_NHBO_Public_Transport', 'B_purpose_NHBO_Car', 'B_fueltype_Avrg_Walk', 'B_fueltype_Avrg_Bike', 'B_fueltype_Avrg_Public_Transport', 'B_fueltype_Avrg_Car', 'B_fueltype_Diesel_Walk', 'B_fueltype_Diesel_Bike', 'B_fueltype_Diesel_Public_Transport', 'B_fueltype_Diesel_Car', 'B_fueltype_Hybrid_Walk', 'B_fueltype_Hybrid_Bike', 'B_fueltype_Hybrid_Public_Transport', 'B_fueltype_Hybrid_Car', 'B_fueltype_Petrol_Walk', 'B_fueltype_Petrol_Bike', 'B_fueltype_Petrol_Public_Transport', 'B_fueltype_Petrol_Car']

    # MNL_utilities = {0: 'B_age_Walk*age + B_female_Walk*female + B_day_of_week_Walk*day_of_week + B_start_time_linear_Walk*start_time_linear + B_car_ownership_Walk*car_ownership + B_driving_license_Walk*driving_license + B_purpose_B_Walk*purpose_B + B_purpose_HBE_Walk*purpose_HBE + B_purpose_HBO_Walk*purpose_HBO + B_purpose_HBW_Walk*purpose_HBW + B_purpose_NHBO_Walk*purpose_NHBO + B_fueltype_Avrg_Walk*fueltype_Average + B_fueltype_Diesel_Walk*fueltype_Diesel + B_fueltype_Hybrid_Walk*fueltype_Hybrid + B_fueltype_Petrol_Walk*fueltype_Petrol + B_distance_Walk*distance + B_dur_walking_Walk*dur_walking',
    #                  1: 'ASC_Bike + B_age_Bike*age + B_female_Bike*female + B_day_of_week_Bike*day_of_week + B_start_time_linear_Bike*start_time_linear + B_car_ownership_Bike*car_ownership + B_driving_license_Bike*driving_license + B_purpose_B_Bike*purpose_B + B_purpose_HBE_Bike*purpose_HBE + B_purpose_HBO_Bike*purpose_HBO + B_purpose_HBW_Bike*purpose_HBW + B_purpose_NHBO_Bike*purpose_NHBO + B_fueltype_Avrg_Bike*fueltype_Average + B_fueltype_Diesel_Bike*fueltype_Diesel + B_fueltype_Hybrid_Bike*fueltype_Hybrid + B_fueltype_Petrol_Bike*fueltype_Petrol + B_distance_Bike*distance + B_dur_cycling_Bike*dur_cycling',
    #                  2: 'ASC_Public_Transport + B_age_Public_Transport*age + B_female_Public_Transport*female + B_day_of_week_Public_Transport*day_of_week + B_start_time_linear_Public_Transport*start_time_linear + B_car_ownership_Public_Transport*car_ownership + B_driving_license_Public_Transport*driving_license + B_purpose_B_Public_Transport*purpose_B + B_purpose_HBE_Public_Transport*purpose_HBE + B_purpose_HBO_Public_Transport*purpose_HBO + B_purpose_HBW_Public_Transport*purpose_HBW + B_purpose_NHBO_Public_Transport*purpose_NHBO + B_fueltype_Avrg_Public_Transport*fueltype_Average + B_fueltype_Diesel_Public_Transport*fueltype_Diesel + B_fueltype_Hybrid_Public_Transport*fueltype_Hybrid + B_fueltype_Petrol_Public_Transport*fueltype_Petrol + B_dur_pt_access_Public_Transport*dur_pt_access + B_dur_pt_rail_Public_Transport*dur_pt_rail + B_dur_pt_bus_Public_Transport*dur_pt_bus + B_dur_pt_int_waiting_Public_Transport*dur_pt_int_waiting + B_dur_pt_int_walking_Public_Transport*dur_pt_int_walking + B_pt_n_interchanges_Public_Transport*pt_n_interchanges + B_cost_transit_Public_Transport*cost_transit',
    #                  3: 'ASC_Car + B_age_Car*age + B_female_Car*female + B_day_of_week_Car*day_of_week + B_start_time_linear_Car*start_time_linear + B_car_ownership_Car*car_ownership + B_driving_license_Car*driving_license + B_purpose_B_Car*purpose_B + B_purpose_HBE_Car*purpose_HBE + B_purpose_HBO_Car*purpose_HBO + B_purpose_HBW_Car*purpose_HBW + B_purpose_NHBO_Car*purpose_NHBO + B_fueltype_Avrg_Car*fueltype_Average + B_fueltype_Diesel_Car*fueltype_Diesel + B_fueltype_Hybrid_Car*fueltype_Hybrid + B_fueltype_Petrol_Car*fueltype_Petrol + B_dur_driving_Car*dur_driving + B_cost_driving_total_Car*cost_driving_total'}

    # 0.6796
    # MNL_beta_params_positive = ['B_car_ownership_Car', 'B_driving_license_Car']
    # MNL_beta_params_negative = ['B_dur_walking_Walk',  'B_dur_cycling_Bike', 'B_dur_pt_access_Public_Transport', 'B_dur_pt_rail_Public_Transport', 'B_dur_pt_bus_Public_Transport', 'B_dur_pt_int_waiting_Public_Transport', 'B_dur_pt_int_walking_Public_Transport', 'B_pt_n_interchanges_Public_Transport', 'B_cost_transit_Public_Transport', 'B_dur_driving_Car', 'B_cost_driving_total_Car']#, 'B_traffic_perc_Car']
    # MNL_beta_params_neutral = ['ASC_Bike', 'ASC_Public_Transport', 'ASC_Car', 'B_car_ownership_Walk', 'B_car_ownership_Bike', 'B_car_ownership_Public_Transport', 'B_driving_license_Walk', 'B_driving_license_Bike', 'B_driving_license_Public_Transport', 'B_age_Walk', 'B_age_Bike', 'B_age_Public_Transport', 'B_age_Car', 'B_female_Walk', 'B_female_Bike', 'B_female_Public_Transport', 'B_female_Car', 'B_distance_Walk', 'B_distance_Bike', 'B_distance_Public_Transport', 'B_distance_Car', 'B_day_of_week_Walk', 'B_day_of_week_Bike', 'B_day_of_week_Public_Transport', 'B_day_of_week_Car', 'B_start_time_linear_Walk', 'B_start_time_linear_Bike', 'B_start_time_linear_Public_Transport', 'B_start_time_linear_Car', 'B_purpose_B_Walk', 'B_purpose_B_Bike', 'B_purpose_B_Public_Transport', 'B_purpose_B_Car', 'B_purpose_HBE_Walk', 'B_purpose_HBE_Bike', 'B_purpose_HBE_Public_Transport', 'B_purpose_HBE_Car', 'B_purpose_HBO_Walk', 'B_purpose_HBO_Bike', 'B_purpose_HBO_Public_Transport', 'B_purpose_HBO_Car', 'B_purpose_HBW_Walk', 'B_purpose_HBW_Bike', 'B_purpose_HBW_Public_Transport', 'B_purpose_HBW_Car', 'B_purpose_NHBO_Walk', 'B_purpose_NHBO_Bike', 'B_purpose_NHBO_Public_Transport', 'B_purpose_NHBO_Car', 'B_fueltype_Avrg_Walk', 'B_fueltype_Avrg_Bike', 'B_fueltype_Avrg_Public_Transport', 'B_fueltype_Avrg_Car', 'B_fueltype_Diesel_Walk', 'B_fueltype_Diesel_Bike', 'B_fueltype_Diesel_Public_Transport', 'B_fueltype_Diesel_Car', 'B_fueltype_Hybrid_Walk', 'B_fueltype_Hybrid_Bike', 'B_fueltype_Hybrid_Public_Transport', 'B_fueltype_Hybrid_Car', 'B_fueltype_Petrol_Walk', 'B_fueltype_Petrol_Bike', 'B_fueltype_Petrol_Public_Transport', 'B_fueltype_Petrol_Car']

    # MNL_utilities = {0: 'B_age_Walk*age + B_female_Walk*female + B_day_of_week_Walk*day_of_week + B_start_time_linear_Walk*start_time_linear + B_car_ownership_Walk*car_ownership + B_driving_license_Walk*driving_license + B_purpose_B_Walk*purpose_B + B_purpose_HBE_Walk*purpose_HBE + B_purpose_HBO_Walk*purpose_HBO + B_purpose_HBW_Walk*purpose_HBW + B_purpose_NHBO_Walk*purpose_NHBO + B_fueltype_Avrg_Walk*fueltype_Average + B_fueltype_Diesel_Walk*fueltype_Diesel + B_fueltype_Hybrid_Walk*fueltype_Hybrid + B_fueltype_Petrol_Walk*fueltype_Petrol + B_distance_Walk*distance + B_dur_walking_Walk*dur_walking',
    #                  1: 'ASC_Bike + B_age_Bike*age + B_female_Bike*female + B_day_of_week_Bike*day_of_week + B_start_time_linear_Bike*start_time_linear + B_car_ownership_Bike*car_ownership + B_driving_license_Bike*driving_license + B_purpose_B_Bike*purpose_B + B_purpose_HBE_Bike*purpose_HBE + B_purpose_HBO_Bike*purpose_HBO + B_purpose_HBW_Bike*purpose_HBW + B_purpose_NHBO_Bike*purpose_NHBO + B_fueltype_Avrg_Bike*fueltype_Average + B_fueltype_Diesel_Bike*fueltype_Diesel + B_fueltype_Hybrid_Bike*fueltype_Hybrid + B_fueltype_Petrol_Bike*fueltype_Petrol + B_distance_Bike*distance + B_dur_cycling_Bike*dur_cycling',
    #                  2: 'ASC_Public_Transport + B_age_Public_Transport*age + B_female_Public_Transport*female + B_day_of_week_Public_Transport*day_of_week + B_start_time_linear_Public_Transport*start_time_linear + B_car_ownership_Public_Transport*car_ownership + B_driving_license_Public_Transport*driving_license + B_purpose_B_Public_Transport*purpose_B + B_purpose_HBE_Public_Transport*purpose_HBE + B_purpose_HBO_Public_Transport*purpose_HBO + B_purpose_HBW_Public_Transport*purpose_HBW + B_purpose_NHBO_Public_Transport*purpose_NHBO + B_fueltype_Avrg_Public_Transport*fueltype_Average + B_fueltype_Diesel_Public_Transport*fueltype_Diesel + B_fueltype_Hybrid_Public_Transport*fueltype_Hybrid + B_fueltype_Petrol_Public_Transport*fueltype_Petrol + B_distance_Public_Transport*distance + B_dur_pt_access_Public_Transport*dur_pt_access + B_dur_pt_rail_Public_Transport*dur_pt_rail + B_dur_pt_bus_Public_Transport*dur_pt_bus + B_dur_pt_int_waiting_Public_Transport*dur_pt_int_waiting + B_dur_pt_int_walking_Public_Transport*dur_pt_int_walking + B_pt_n_interchanges_Public_Transport*pt_n_interchanges + B_cost_transit_Public_Transport*cost_transit',
    #                  3: 'ASC_Car + B_age_Car*age + B_female_Car*female + B_day_of_week_Car*day_of_week + B_start_time_linear_Car*start_time_linear + B_car_ownership_Car*car_ownership + B_driving_license_Car*driving_license + B_purpose_B_Car*purpose_B + B_purpose_HBE_Car*purpose_HBE + B_purpose_HBO_Car*purpose_HBO + B_purpose_HBW_Car*purpose_HBW + B_purpose_NHBO_Car*purpose_NHBO + B_fueltype_Avrg_Car*fueltype_Average + B_fueltype_Diesel_Car*fueltype_Diesel + B_fueltype_Hybrid_Car*fueltype_Hybrid + B_fueltype_Petrol_Car*fueltype_Petrol + B_distance_Car*distance + B_dur_driving_Car*dur_driving + B_cost_driving_total_Car*cost_driving_total'}

    # 0.6789 with lr 0.3 --- change with above model; B_age_Bike and B_age_Walk in negative mono, got rid of driving license in cycling, got rid of fueltype in other features than car, got rid of car ownership in walk
    # MNL_beta_params_positive = ['B_car_ownership_Car', 'B_driving_license_Car']
    # MNL_beta_params_negative = ['B_age_Walk', 'B_age_Bike', 'B_dur_walking_Walk',  'B_dur_cycling_Bike', 'B_dur_pt_access_Public_Transport', 'B_dur_pt_rail_Public_Transport', 'B_dur_pt_bus_Public_Transport', 'B_dur_pt_int_waiting_Public_Transport', 'B_dur_pt_int_walking_Public_Transport', 'B_pt_n_interchanges_Public_Transport', 'B_cost_transit_Public_Transport', 'B_dur_driving_Car', 'B_cost_driving_total_Car', 'B_distance_Walk', 'B_distance_Bike', 'B_distance_Public_Transport', 'B_distance_Car']#, 'B_traffic_perc_Car']
    # MNL_beta_params_neutral = ['ASC_Bike', 'ASC_Public_Transport', 'ASC_Car', 'B_car_ownership_Bike', 'B_car_ownership_Public_Transport', 'B_driving_license_Walk', 'B_driving_license_Public_Transport', 'B_age_Public_Transport', 'B_age_Car', 'B_female_Walk', 'B_female_Bike', 'B_female_Public_Transport', 'B_female_Car', 'B_day_of_week_Walk', 'B_day_of_week_Bike', 'B_day_of_week_Public_Transport', 'B_day_of_week_Car', 'B_start_time_linear_Walk', 'B_start_time_linear_Bike', 'B_start_time_linear_Public_Transport', 'B_start_time_linear_Car', 'B_purpose_B_Walk', 'B_purpose_B_Bike', 'B_purpose_B_Public_Transport', 'B_purpose_B_Car', 'B_purpose_HBE_Walk', 'B_purpose_HBE_Bike', 'B_purpose_HBE_Public_Transport', 'B_purpose_HBE_Car', 'B_purpose_HBO_Walk', 'B_purpose_HBO_Bike', 'B_purpose_HBO_Public_Transport', 'B_purpose_HBO_Car', 'B_purpose_HBW_Walk', 'B_purpose_HBW_Bike', 'B_purpose_HBW_Public_Transport', 'B_purpose_HBW_Car', 'B_purpose_NHBO_Walk', 'B_purpose_NHBO_Bike', 'B_purpose_NHBO_Public_Transport', 'B_purpose_NHBO_Car', 'B_fueltype_Avrg_Car', 'B_fueltype_Diesel_Car', 'B_fueltype_Hybrid_Car', 'B_fueltype_Petrol_Car']

    # MNL_utilities = {0: 'B_age_Walk*age + B_female_Walk*female + B_day_of_week_Walk*day_of_week + B_start_time_linear_Walk*start_time_linear + B_driving_license_Walk*driving_license + B_purpose_B_Walk*purpose_B + B_purpose_HBE_Walk*purpose_HBE + B_purpose_HBO_Walk*purpose_HBO + B_purpose_HBW_Walk*purpose_HBW + B_purpose_NHBO_Walk*purpose_NHBO + B_distance_Walk*distance + B_dur_walking_Walk*dur_walking',
    #                  1: 'ASC_Bike + B_age_Bike*age + B_female_Bike*female + B_day_of_week_Bike*day_of_week + B_start_time_linear_Bike*start_time_linear + B_car_ownership_Bike*car_ownership + B_purpose_B_Bike*purpose_B + B_purpose_HBE_Bike*purpose_HBE + B_purpose_HBO_Bike*purpose_HBO + B_purpose_HBW_Bike*purpose_HBW + B_purpose_NHBO_Bike*purpose_NHBO + B_distance_Bike*distance + B_dur_cycling_Bike*dur_cycling',
    #                  2: 'ASC_Public_Transport + B_age_Public_Transport*age + B_female_Public_Transport*female + B_day_of_week_Public_Transport*day_of_week + B_start_time_linear_Public_Transport*start_time_linear + B_car_ownership_Public_Transport*car_ownership + B_driving_license_Public_Transport*driving_license + B_purpose_B_Public_Transport*purpose_B + B_purpose_HBE_Public_Transport*purpose_HBE + B_purpose_HBO_Public_Transport*purpose_HBO + B_purpose_HBW_Public_Transport*purpose_HBW + B_purpose_NHBO_Public_Transport*purpose_NHBO + B_distance_Public_Transport*distance + B_dur_pt_access_Public_Transport*dur_pt_access + B_dur_pt_rail_Public_Transport*dur_pt_rail + B_dur_pt_bus_Public_Transport*dur_pt_bus + B_dur_pt_int_waiting_Public_Transport*dur_pt_int_waiting + B_dur_pt_int_walking_Public_Transport*dur_pt_int_walking + B_pt_n_interchanges_Public_Transport*pt_n_interchanges + B_cost_transit_Public_Transport*cost_transit',
    #                  3: 'ASC_Car + B_age_Car*age + B_female_Car*female + B_day_of_week_Car*day_of_week + B_start_time_linear_Car*start_time_linear + B_car_ownership_Car*car_ownership + B_driving_license_Car*driving_license + B_purpose_B_Car*purpose_B + B_purpose_HBE_Car*purpose_HBE + B_purpose_HBO_Car*purpose_HBO + B_purpose_HBW_Car*purpose_HBW + B_purpose_NHBO_Car*purpose_NHBO + B_fueltype_Avrg_Car*fueltype_Average + B_fueltype_Diesel_Car*fueltype_Diesel + B_fueltype_Hybrid_Car*fueltype_Hybrid + B_fueltype_Petrol_Car*fueltype_Petrol + B_distance_Car*distance + B_dur_driving_Car*dur_driving + B_cost_driving_total_Car*cost_driving_total'}

    # 0.6789 with lr 0.3 --- change with above model; got rid of start time linear
    # MNL_beta_params_positive = ['B_car_ownership_Car', 'B_driving_license_Car']
    # MNL_beta_params_negative = ['B_age_Walk', 'B_age_Bike', 'B_dur_walking_Walk',  'B_dur_cycling_Bike', 'B_dur_pt_access_Public_Transport', 'B_dur_pt_rail_Public_Transport', 'B_dur_pt_bus_Public_Transport', 'B_dur_pt_int_waiting_Public_Transport', 'B_dur_pt_int_walking_Public_Transport', 'B_pt_n_interchanges_Public_Transport', 'B_cost_transit_Public_Transport', 'B_dur_driving_Car', 'B_cost_driving_total_Car', 'B_distance_Walk', 'B_distance_Bike', 'B_distance_Public_Transport', 'B_distance_Car']#, 'B_traffic_perc_Car']
    # MNL_beta_params_neutral = ['ASC_Bike', 'ASC_Public_Transport', 'ASC_Car', 'B_car_ownership_Bike', 'B_car_ownership_Public_Transport', 'B_driving_license_Walk', 'B_driving_license_Public_Transport', 'B_age_Public_Transport', 'B_age_Car', 'B_female_Walk', 'B_female_Bike', 'B_female_Public_Transport', 'B_female_Car', 'B_day_of_week_Walk', 'B_day_of_week_Bike', 'B_day_of_week_Public_Transport', 'B_day_of_week_Car', 'B_purpose_B_Walk', 'B_purpose_B_Bike', 'B_purpose_B_Public_Transport', 'B_purpose_B_Car', 'B_purpose_HBE_Walk', 'B_purpose_HBE_Bike', 'B_purpose_HBE_Public_Transport', 'B_purpose_HBE_Car', 'B_purpose_HBO_Walk', 'B_purpose_HBO_Bike', 'B_purpose_HBO_Public_Transport', 'B_purpose_HBO_Car', 'B_purpose_HBW_Walk', 'B_purpose_HBW_Bike', 'B_purpose_HBW_Public_Transport', 'B_purpose_HBW_Car', 'B_purpose_NHBO_Walk', 'B_purpose_NHBO_Bike', 'B_purpose_NHBO_Public_Transport', 'B_purpose_NHBO_Car', 'B_fueltype_Avrg_Car', 'B_fueltype_Diesel_Car', 'B_fueltype_Hybrid_Car', 'B_fueltype_Petrol_Car']

    # MNL_utilities = {0: 'B_age_Walk*age + B_female_Walk*female + B_day_of_week_Walk*day_of_week + B_driving_license_Walk*driving_license + B_purpose_B_Walk*purpose_B + B_purpose_HBE_Walk*purpose_HBE + B_purpose_HBO_Walk*purpose_HBO + B_purpose_HBW_Walk*purpose_HBW + B_purpose_NHBO_Walk*purpose_NHBO + B_distance_Walk*distance + B_dur_walking_Walk*dur_walking',
    #                  1: 'ASC_Bike + B_age_Bike*age + B_female_Bike*female + B_day_of_week_Bike*day_of_week + B_car_ownership_Bike*car_ownership + B_purpose_B_Bike*purpose_B + B_purpose_HBE_Bike*purpose_HBE + B_purpose_HBO_Bike*purpose_HBO + B_purpose_HBW_Bike*purpose_HBW + B_purpose_NHBO_Bike*purpose_NHBO + B_distance_Bike*distance + B_dur_cycling_Bike*dur_cycling',
    #                  2: 'ASC_Public_Transport + B_age_Public_Transport*age + B_female_Public_Transport*female + B_day_of_week_Public_Transport*day_of_week + B_car_ownership_Public_Transport*car_ownership + B_driving_license_Public_Transport*driving_license + B_purpose_B_Public_Transport*purpose_B + B_purpose_HBE_Public_Transport*purpose_HBE + B_purpose_HBO_Public_Transport*purpose_HBO + B_purpose_HBW_Public_Transport*purpose_HBW + B_purpose_NHBO_Public_Transport*purpose_NHBO + B_distance_Public_Transport*distance + B_dur_pt_access_Public_Transport*dur_pt_access + B_dur_pt_rail_Public_Transport*dur_pt_rail + B_dur_pt_bus_Public_Transport*dur_pt_bus + B_dur_pt_int_waiting_Public_Transport*dur_pt_int_waiting + B_dur_pt_int_walking_Public_Transport*dur_pt_int_walking + B_pt_n_interchanges_Public_Transport*pt_n_interchanges + B_cost_transit_Public_Transport*cost_transit',
    #                  3: 'ASC_Car + B_age_Car*age + B_female_Car*female + B_day_of_week_Car*day_of_week + B_car_ownership_Car*car_ownership + B_driving_license_Car*driving_license + B_purpose_B_Car*purpose_B + B_purpose_HBE_Car*purpose_HBE + B_purpose_HBO_Car*purpose_HBO + B_purpose_HBW_Car*purpose_HBW + B_purpose_NHBO_Car*purpose_NHBO + B_fueltype_Avrg_Car*fueltype_Average + B_fueltype_Diesel_Car*fueltype_Diesel + B_fueltype_Hybrid_Car*fueltype_Hybrid + B_fueltype_Petrol_Car*fueltype_Petrol + B_distance_Car*distance + B_dur_driving_Car*dur_driving + B_cost_driving_total_Car*cost_driving_total'}

    # with travel month and traffic percentage
    # MNL_beta_params_positive = ['B_car_ownership_Car', 'B_driving_license_Car']
    # MNL_beta_params_negative = ['B_dur_walking_Walk',  'B_dur_cycling_Bike', 'B_dur_pt_access_Public_Transport', 'B_dur_pt_rail_Public_Transport', 'B_dur_pt_bus_Public_Transport', 'B_dur_pt_int_waiting_Public_Transport', 'B_dur_pt_int_walking_Public_Transport', 'B_pt_n_interchanges_Public_Transport', 'B_cost_transit_Public_Transport', 'B_dur_driving_Car', 'B_cost_driving_total_Car', 'B_traffic_perc_Car']
    # MNL_beta_params_neutral = ['ASC_Bike', 'ASC_Public_Transport', 'ASC_Car', 'B_car_ownership_Walk', 'B_car_ownership_Bike', 'B_car_ownership_Public_Transport', 'B_driving_license_Walk', 'B_driving_license_Bike', 'B_driving_license_Public_Transport', 'B_age_Walk', 'B_age_Bike', 'B_age_Public_Transport', 'B_age_Car', 'B_female_Walk', 'B_female_Bike', 'B_female_Public_Transport', 'B_female_Car', 'B_distance_Walk', 'B_distance_Bike', 'B_distance_Public_Transport', 'B_distance_Car', 'B_day_of_week_Walk', 'B_day_of_week_Bike', 'B_day_of_week_Public_Transport', 'B_day_of_week_Car', 'B_start_time_linear_Walk', 'B_start_time_linear_Bike', 'B_start_time_linear_Public_Transport', 'B_start_time_linear_Car', 'B_purpose_B_Walk', 'B_purpose_B_Bike', 'B_purpose_B_Public_Transport', 'B_purpose_B_Car', 'B_purpose_HBE_Walk', 'B_purpose_HBE_Bike', 'B_purpose_HBE_Public_Transport', 'B_purpose_HBE_Car', 'B_purpose_HBO_Walk', 'B_purpose_HBO_Bike', 'B_purpose_HBO_Public_Transport', 'B_purpose_HBO_Car', 'B_purpose_HBW_Walk', 'B_purpose_HBW_Bike', 'B_purpose_HBW_Public_Transport', 'B_purpose_HBW_Car', 'B_purpose_NHBO_Walk', 'B_purpose_NHBO_Bike', 'B_purpose_NHBO_Public_Transport', 'B_purpose_NHBO_Car', 'B_fueltype_Avrg_Walk', 'B_fueltype_Avrg_Bike', 'B_fueltype_Avrg_Public_Transport', 'B_fueltype_Avrg_Car', 'B_fueltype_Diesel_Walk', 'B_fueltype_Diesel_Bike', 'B_fueltype_Diesel_Public_Transport', 'B_fueltype_Diesel_Car', 'B_fueltype_Hybrid_Walk', 'B_fueltype_Hybrid_Bike', 'B_fueltype_Hybrid_Public_Transport', 'B_fueltype_Hybrid_Car', 'B_fueltype_Petrol_Walk', 'B_fueltype_Petrol_Bike', 'B_fueltype_Petrol_Public_Transport', 'B_fueltype_Petrol_Car', 'B_travel_month_Walk', 'B_travel_month_Bike', 'B_travel_month_Public_Transport', 'B_travel_month_Car']

    # MNL_utilities = {0: 'B_travel_month_Walk*travel_month + B_age_Walk*age + B_female_Walk*female + B_day_of_week_Walk*day_of_week + B_start_time_linear_Walk*start_time_linear + B_car_ownership_Walk*car_ownership + B_driving_license_Walk*driving_license + B_purpose_B_Walk*purpose_B + B_purpose_HBE_Walk*purpose_HBE + B_purpose_HBO_Walk*purpose_HBO + B_purpose_HBW_Walk*purpose_HBW + B_purpose_NHBO_Walk*purpose_NHBO + B_fueltype_Avrg_Walk*fueltype_Average + B_fueltype_Diesel_Walk*fueltype_Diesel + B_fueltype_Hybrid_Walk*fueltype_Hybrid + B_fueltype_Petrol_Walk*fueltype_Petrol + B_distance_Walk*distance + B_dur_walking_Walk*dur_walking',
    #                  1: 'ASC_Bike + B_travel_month_Bike*travel_month + B_age_Bike*age + B_female_Bike*female + B_day_of_week_Bike*day_of_week + B_start_time_linear_Bike*start_time_linear + B_car_ownership_Bike*car_ownership + B_driving_license_Bike*driving_license + B_purpose_B_Bike*purpose_B + B_purpose_HBE_Bike*purpose_HBE + B_purpose_HBO_Bike*purpose_HBO + B_purpose_HBW_Bike*purpose_HBW + B_purpose_NHBO_Bike*purpose_NHBO + B_fueltype_Avrg_Bike*fueltype_Average + B_fueltype_Diesel_Bike*fueltype_Diesel + B_fueltype_Hybrid_Bike*fueltype_Hybrid + B_fueltype_Petrol_Bike*fueltype_Petrol + B_distance_Bike*distance + B_dur_cycling_Bike*dur_cycling',
    #                  2: 'ASC_Public_Transport + B_travel_month_Public_Transport*travel_month + B_age_Public_Transport*age + B_female_Public_Transport*female + B_day_of_week_Public_Transport*day_of_week + B_start_time_linear_Public_Transport*start_time_linear + B_car_ownership_Public_Transport*car_ownership + B_driving_license_Public_Transport*driving_license + B_purpose_B_Public_Transport*purpose_B + B_purpose_HBE_Public_Transport*purpose_HBE + B_purpose_HBO_Public_Transport*purpose_HBO + B_purpose_HBW_Public_Transport*purpose_HBW + B_purpose_NHBO_Public_Transport*purpose_NHBO + B_fueltype_Avrg_Public_Transport*fueltype_Average + B_fueltype_Diesel_Public_Transport*fueltype_Diesel + B_fueltype_Hybrid_Public_Transport*fueltype_Hybrid + B_fueltype_Petrol_Public_Transport*fueltype_Petrol + B_distance_Public_Transport*distance + B_dur_pt_access_Public_Transport*dur_pt_access + B_dur_pt_rail_Public_Transport*dur_pt_rail + B_dur_pt_bus_Public_Transport*dur_pt_bus + B_dur_pt_int_waiting_Public_Transport*dur_pt_int_waiting + B_dur_pt_int_walking_Public_Transport*dur_pt_int_walking + B_pt_n_interchanges_Public_Transport*pt_n_interchanges + B_cost_transit_Public_Transport*cost_transit',
    #                  3: 'ASC_Car + B_travel_month_Car*travel_month + B_traffic_perc_Car*driving_traffic_percent + B_age_Car*age + B_female_Car*female + B_day_of_week_Car*day_of_week + B_start_time_linear_Car*start_time_linear + B_car_ownership_Car*car_ownership + B_driving_license_Car*driving_license + B_purpose_B_Car*purpose_B + B_purpose_HBE_Car*purpose_HBE + B_purpose_HBO_Car*purpose_HBO + B_purpose_HBW_Car*purpose_HBW + B_purpose_NHBO_Car*purpose_NHBO + B_fueltype_Avrg_Car*fueltype_Average + B_fueltype_Diesel_Car*fueltype_Diesel + B_fueltype_Hybrid_Car*fueltype_Hybrid + B_fueltype_Petrol_Car*fueltype_Petrol + B_distance_Car*distance + B_dur_driving_Car*dur_driving + B_cost_driving_total_Car*cost_driving_total'}

    # market segmentation with weekend
    # MNL_beta_params_positive = ['B_car_ownership_Car', 'B_driving_license_Car']
    # MNL_beta_params_negative = ['B_dur_walking_Walk',  'B_dur_cycling_Bike', 'B_dur_pt_access_Public_Transport', 'B_dur_pt_rail_Public_Transport', 'B_dur_pt_bus_Public_Transport', 'B_dur_pt_int_waiting_Public_Transport', 'B_dur_pt_int_walking_Public_Transport', 'B_pt_n_interchanges_Public_Transport', 'B_cost_transit_Public_Transport', 'B_dur_driving_Car', 'B_cost_driving_total_Car', 'B_distance_Walk', 'B_distance_Bike', 'B_distance_Public_Transport', 'B_distance_Car']#, 'B_traffic_perc_Car']
    # MNL_beta_params_neutral = ['ASC_Bike', 'ASC_Public_Transport', 'ASC_Car', 'B_car_ownership_Walk', 'B_car_ownership_Bike', 'B_car_ownership_Public_Transport', 'B_driving_license_Walk', 'B_driving_license_Bike', 'B_driving_license_Public_Transport', 'B_age_Walk', 'B_age_Bike', 'B_age_Public_Transport', 'B_age_Car', 'B_female_Walk', 'B_female_Bike', 'B_female_Public_Transport', 'B_female_Car', 'B_weekend_Walk', 'B_weekend_Bike', 'B_weekend_Public_Transport', 'B_weekend_Car', 'B_start_time_linear_Walk', 'B_start_time_linear_Bike', 'B_start_time_linear_Public_Transport', 'B_start_time_linear_Car', 'B_purpose_B_Walk', 'B_purpose_B_Bike', 'B_purpose_B_Public_Transport', 'B_purpose_B_Car', 'B_purpose_HBE_Walk', 'B_purpose_HBE_Bike', 'B_purpose_HBE_Public_Transport', 'B_purpose_HBE_Car', 'B_purpose_HBO_Walk', 'B_purpose_HBO_Bike', 'B_purpose_HBO_Public_Transport', 'B_purpose_HBO_Car', 'B_purpose_HBW_Walk', 'B_purpose_HBW_Bike', 'B_purpose_HBW_Public_Transport', 'B_purpose_HBW_Car', 'B_purpose_NHBO_Walk', 'B_purpose_NHBO_Bike', 'B_purpose_NHBO_Public_Transport', 'B_purpose_NHBO_Car', 'B_fueltype_Avrg_Walk', 'B_fueltype_Avrg_Bike', 'B_fueltype_Avrg_Public_Transport', 'B_fueltype_Avrg_Car', 'B_fueltype_Diesel_Walk', 'B_fueltype_Diesel_Bike', 'B_fueltype_Diesel_Public_Transport', 'B_fueltype_Diesel_Car', 'B_fueltype_Hybrid_Walk', 'B_fueltype_Hybrid_Bike', 'B_fueltype_Hybrid_Public_Transport', 'B_fueltype_Hybrid_Car', 'B_fueltype_Petrol_Walk', 'B_fueltype_Petrol_Bike', 'B_fueltype_Petrol_Public_Transport', 'B_fueltype_Petrol_Car']

    # MNL_utilities = {0: 'B_age_Walk*age + B_female_Walk*female + B_weekend_Walk*weekend + B_start_time_linear_Walk*start_time_linear + B_car_ownership_Walk*car_ownership + B_driving_license_Walk*driving_license + B_purpose_B_Walk*purpose_B + B_purpose_HBE_Walk*purpose_HBE + B_purpose_HBO_Walk*purpose_HBO + B_purpose_HBW_Walk*purpose_HBW + B_purpose_NHBO_Walk*purpose_NHBO + B_fueltype_Avrg_Walk*fueltype_Average + B_fueltype_Diesel_Walk*fueltype_Diesel + B_fueltype_Hybrid_Walk*fueltype_Hybrid + B_fueltype_Petrol_Walk*fueltype_Petrol + B_distance_Walk*distance + B_dur_walking_Walk*dur_walking',
    #                  1: 'ASC_Bike + B_age_Bike*age + B_female_Bike*female + B_weekend_Bike*weekend + B_start_time_linear_Bike*start_time_linear + B_car_ownership_Bike*car_ownership + B_driving_license_Bike*driving_license + B_purpose_B_Bike*purpose_B + B_purpose_HBE_Bike*purpose_HBE + B_purpose_HBO_Bike*purpose_HBO + B_purpose_HBW_Bike*purpose_HBW + B_purpose_NHBO_Bike*purpose_NHBO + B_fueltype_Avrg_Bike*fueltype_Average + B_fueltype_Diesel_Bike*fueltype_Diesel + B_fueltype_Hybrid_Bike*fueltype_Hybrid + B_fueltype_Petrol_Bike*fueltype_Petrol + B_distance_Bike*distance + B_dur_cycling_Bike*dur_cycling',
    #                  2: 'ASC_Public_Transport + B_age_Public_Transport*age + B_female_Public_Transport*female + B_weekend_Public_Transport*weekend + B_start_time_linear_Public_Transport*start_time_linear + B_car_ownership_Public_Transport*car_ownership + B_driving_license_Public_Transport*driving_license + B_purpose_B_Public_Transport*purpose_B + B_purpose_HBE_Public_Transport*purpose_HBE + B_purpose_HBO_Public_Transport*purpose_HBO + B_purpose_HBW_Public_Transport*purpose_HBW + B_purpose_NHBO_Public_Transport*purpose_NHBO + B_fueltype_Avrg_Public_Transport*fueltype_Average + B_fueltype_Diesel_Public_Transport*fueltype_Diesel + B_fueltype_Hybrid_Public_Transport*fueltype_Hybrid + B_fueltype_Petrol_Public_Transport*fueltype_Petrol + B_distance_Public_Transport*distance + B_dur_pt_access_Public_Transport*dur_pt_access + B_dur_pt_rail_Public_Transport*dur_pt_rail + B_dur_pt_bus_Public_Transport*dur_pt_bus + B_dur_pt_int_waiting_Public_Transport*dur_pt_int_waiting + B_dur_pt_int_walking_Public_Transport*dur_pt_int_walking + B_pt_n_interchanges_Public_Transport*pt_n_interchanges + B_cost_transit_Public_Transport*cost_transit',
    #                  3: 'ASC_Car + B_age_Car*age + B_female_Car*female + B_weekend_Car*weekend + B_start_time_linear_Car*start_time_linear + B_car_ownership_Car*car_ownership + B_driving_license_Car*driving_license + B_purpose_B_Car*purpose_B + B_purpose_HBE_Car*purpose_HBE + B_purpose_HBO_Car*purpose_HBO + B_purpose_HBW_Car*purpose_HBW + B_purpose_NHBO_Car*purpose_NHBO + B_fueltype_Avrg_Car*fueltype_Average + B_fueltype_Diesel_Car*fueltype_Diesel + B_fueltype_Hybrid_Car*fueltype_Hybrid + B_fueltype_Petrol_Car*fueltype_Petrol + B_distance_Car*distance + B_dur_driving_Car*dur_driving + B_cost_driving_total_Car*cost_driving_total'}

    # construct the model parameters
    for beta in MNL_beta_params_positive:
        exec("{} = Beta('{}', 0, 0, None, 0)".format(beta, beta), globals())
    for beta in MNL_beta_params_negative:
        exec("{} = Beta('{}', 0, None, 0, 0)".format(beta, beta), globals())
    for beta in MNL_beta_params_neutral:
        exec("{} = Beta('{}', 0, None, None, 0)".format(beta, beta), globals())

    # define utility functions
    for utility_idx in MNL_utilities.keys():
        exec("V_{} = {}".format(utility_idx, MNL_utilities[utility_idx]), globals())

    # assign utility functions to utility indices
    exec("V_dict = {}", globals())
    for utility_idx in MNL_utilities.keys():
        exec("V_dict[{}] = V_{}".format(utility_idx, utility_idx), globals())

    # associate the availability conditions with the alternatives
    exec("av = {}", globals())
    for utility_idx in MNL_utilities.keys():
        exec("av[{}] = 1".format(utility_idx), globals())

    MU_nm = Beta("MU_nm", 1, 1, 10, 0)
    MU_m = Beta("MU_m", 1, 1, 10, 0)

    non_motorised = MU_nm, [0, 1]
    motorised = MU_m, [2, 3]
    nests = non_motorised, motorised

    # definition of the model
    logprob = lognested(V_dict, av, nests, choice)

    # create the Biogeme object
    biogeme = bio.BIOGEME(database_train, logprob)
    biogeme.modelName = "LPMC"

    biogeme.generate_html = False
    biogeme.generate_pickle = False

    if for_prob:
        prob_0 = logit(V_dict, av, 0)
        prob_1 = logit(V_dict, av, 1)
        prob_2 = logit(V_dict, av, 2)
        prob_3 = logit(V_dict, av, 3)

        simulate = {
            "Prob. 0": prob_0,
            "Prob. 1": prob_1,
            "Prob. 2": prob_2,
            "Prob. 3": prob_3,
        }
        biosim = bio.BIOGEME(database_train, simulate)
        biosim.modelName = "LPMC_logit_test"

        biosim.generate_html = False
        biosim.generate_pickle = False

        return biosim

    return biogeme


def LPMC_nested_normalised(dataset_train, for_prob=False):
    """
    Create a nested logit model on the LPMC dataset, normalised for biogeme estimation.
    The model is a slightly modified version from teh code that can be found here: https://github.com/JoseAngelMartinB/prediction-behavioural-analysis-ml-travel-mode-choice.

    Parameters
    ----------
    dataset_train : pandas DataFrame
        The training dataset.

    Returns
    -------
    biogeme : bio.BIOGEME
        The BIOGEME object containing the model.

    """
    database_train = db.Database("LTDS_train", dataset_train)

    globals().update(database_train.variables)

    logger = blog.get_screen_logger(level=blog.DEBUG)
    logger.info("Nested LPMC")

    # several model specifications are available below - the best one is the uncommented one.
    # driving_percentage, congestion charge as a binary variable NEW PB 0.6730 with lr = 0.1
    MNL_beta_params_positive = ["B_car_ownership_Car", "B_driving_license_Car"]
    MNL_beta_params_negative = [
        "B_dur_walking_Walk",
        "B_dur_cycling_Bike",
        "B_dur_pt_access_Public_Transport",
        "B_dur_pt_rail_Public_Transport",
        "B_dur_pt_bus_Public_Transport",
        "B_dur_pt_int_waiting_Public_Transport",
        "B_dur_pt_int_walking_Public_Transport",
        "B_pt_n_interchanges_Public_Transport",
        "B_cost_transit_Public_Transport",
        "B_dur_driving_Car",
        "B_cost_driving_fuel_Car",
        "B_distance_Walk",
        "B_distance_Bike",
        "B_distance_Public_Transport",
        "B_distance_Car",
        "B_con_charge_Car",
        "B_traffic_perc_Car",
    ]
    MNL_beta_params_neutral = [
        "ASC_Bike",
        "ASC_Public_Transport",
        "ASC_Car",
        "B_car_ownership_Walk",
        "B_car_ownership_Bike",
        "B_car_ownership_Public_Transport",
        "B_driving_license_Walk",
        "B_driving_license_Bike",
        "B_driving_license_Public_Transport",
        "B_age_Walk",
        "B_age_Bike",
        "B_age_Public_Transport",
        "B_age_Car",
        "B_female_Walk",
        "B_female_Bike",
        "B_female_Public_Transport",
        "B_female_Car",
        "B_day_of_week_Walk",
        "B_day_of_week_Bike",
        "B_day_of_week_Public_Transport",
        "B_day_of_week_Car",
        "B_start_time_linear_Walk",
        "B_start_time_linear_Bike",
        "B_start_time_linear_Public_Transport",
        "B_start_time_linear_Car",
        "B_purpose_B_Walk",
        "B_purpose_B_Bike",
        "B_purpose_B_Public_Transport",
        "B_purpose_B_Car",
        "B_purpose_HBE_Walk",
        "B_purpose_HBE_Bike",
        "B_purpose_HBE_Public_Transport",
        "B_purpose_HBE_Car",
        "B_purpose_HBO_Walk",
        "B_purpose_HBO_Bike",
        "B_purpose_HBO_Public_Transport",
        "B_purpose_HBO_Car",
        "B_purpose_HBW_Walk",
        "B_purpose_HBW_Bike",
        "B_purpose_HBW_Public_Transport",
        "B_purpose_HBW_Car",
        "B_purpose_NHBO_Walk",
        "B_purpose_NHBO_Bike",
        "B_purpose_NHBO_Public_Transport",
        "B_purpose_NHBO_Car",
        "B_fueltype_Avrg_Walk",
        "B_fueltype_Avrg_Bike",
        "B_fueltype_Avrg_Public_Transport",
        "B_fueltype_Avrg_Car",
        "B_fueltype_Diesel_Walk",
        "B_fueltype_Diesel_Bike",
        "B_fueltype_Diesel_Public_Transport",
        "B_fueltype_Diesel_Car",
        "B_fueltype_Hybrid_Walk",
        "B_fueltype_Hybrid_Bike",
        "B_fueltype_Hybrid_Public_Transport",
        "B_fueltype_Hybrid_Car",
        "B_fueltype_Petrol_Walk",
        "B_fueltype_Petrol_Bike",
        "B_fueltype_Petrol_Public_Transport",
        "B_fueltype_Petrol_Car",
    ]

    MNL_utilities = {
        0: "B_dur_walking_Walk*dur_walking",  # B_age_Walk*age + B_female_Walk*female + B_day_of_week_Walk*day_of_week + B_start_time_linear_Walk*start_time_linear + B_car_ownership_Walk*car_ownership + B_driving_license_Walk*driving_license + B_purpose_B_Walk*purpose_B + B_purpose_HBE_Walk*purpose_HBE + B_purpose_HBO_Walk*purpose_HBO + B_purpose_HBW_Walk*purpose_HBW + B_purpose_NHBO_Walk*purpose_NHBO + B_fueltype_Avrg_Walk*fueltype_Average + B_fueltype_Diesel_Walk*fueltype_Diesel + B_fueltype_Hybrid_Walk*fueltype_Hybrid + B_fueltype_Petrol_Walk*fueltype_Petrol + B_distance_Walk*distance + B_dur_walking_Walk*dur_walking',
        1: "ASC_Bike + B_age_Bike*age + B_female_Bike*female + B_day_of_week_Bike*day_of_week + B_start_time_linear_Bike*start_time_linear + B_car_ownership_Bike*car_ownership + B_driving_license_Bike*driving_license + B_purpose_B_Bike*purpose_B + B_purpose_HBE_Bike*purpose_HBE + B_purpose_HBO_Bike*purpose_HBO + B_purpose_HBW_Bike*purpose_HBW + B_purpose_NHBO_Bike*purpose_NHBO + B_fueltype_Avrg_Bike*fueltype_Average + B_fueltype_Diesel_Bike*fueltype_Diesel + B_fueltype_Hybrid_Bike*fueltype_Hybrid + B_fueltype_Petrol_Bike*fueltype_Petrol + B_distance_Bike*distance + B_dur_cycling_Bike*dur_cycling",
        2: "ASC_Public_Transport + B_age_Public_Transport*age + B_female_Public_Transport*female + B_day_of_week_Public_Transport*day_of_week + B_start_time_linear_Public_Transport*start_time_linear + B_car_ownership_Public_Transport*car_ownership + B_driving_license_Public_Transport*driving_license + B_purpose_B_Public_Transport*purpose_B + B_purpose_HBE_Public_Transport*purpose_HBE + B_purpose_HBO_Public_Transport*purpose_HBO + B_purpose_HBW_Public_Transport*purpose_HBW + B_purpose_NHBO_Public_Transport*purpose_NHBO + B_fueltype_Avrg_Public_Transport*fueltype_Average + B_fueltype_Diesel_Public_Transport*fueltype_Diesel + B_fueltype_Hybrid_Public_Transport*fueltype_Hybrid + B_fueltype_Petrol_Public_Transport*fueltype_Petrol + B_distance_Public_Transport*distance + B_dur_pt_access_Public_Transport*dur_pt_access + B_dur_pt_rail_Public_Transport*dur_pt_rail + B_dur_pt_bus_Public_Transport*dur_pt_bus + B_dur_pt_int_waiting_Public_Transport*dur_pt_int_waiting + B_dur_pt_int_walking_Public_Transport*dur_pt_int_walking + B_pt_n_interchanges_Public_Transport*pt_n_interchanges + B_cost_transit_Public_Transport*cost_transit",
        3: "ASC_Car + B_age_Car*age + B_female_Car*female + B_day_of_week_Car*day_of_week + B_start_time_linear_Car*start_time_linear + B_car_ownership_Car*car_ownership + B_driving_license_Car*driving_license + B_purpose_B_Car*purpose_B + B_purpose_HBE_Car*purpose_HBE + B_purpose_HBO_Car*purpose_HBO + B_purpose_HBW_Car*purpose_HBW + B_purpose_NHBO_Car*purpose_NHBO + B_fueltype_Avrg_Car*fueltype_Average + B_fueltype_Diesel_Car*fueltype_Diesel + B_fueltype_Hybrid_Car*fueltype_Hybrid + B_fueltype_Petrol_Car*fueltype_Petrol + B_distance_Car*distance + B_dur_driving_Car*dur_driving + B_cost_driving_fuel_Car*cost_driving_fuel + B_con_charge_Car*congestion_charge + B_traffic_perc_Car*driving_traffic_percent",
    }

    # construct the model parameters
    for beta in MNL_beta_params_positive:
        exec("{} = Beta('{}', 0, 0, None, 0)".format(beta, beta), globals())
    for beta in MNL_beta_params_negative:
        exec("{} = Beta('{}', 0, None, 0, 0)".format(beta, beta), globals())
    for beta in MNL_beta_params_neutral:
        exec("{} = Beta('{}', 0, None, None, 0)".format(beta, beta), globals())

    # define utility functions
    for utility_idx in MNL_utilities.keys():
        exec("V_{} = {}".format(utility_idx, MNL_utilities[utility_idx]), globals())

    # assign utility functions to utility indices
    exec("V_dict = {}", globals())
    for utility_idx in MNL_utilities.keys():
        exec("V_dict[{}] = V_{}".format(utility_idx, utility_idx), globals())

    # associate the availability conditions with the alternatives
    exec("av = {}", globals())
    for utility_idx in MNL_utilities.keys():
        exec("av[{}] = 1".format(utility_idx), globals())

    # MU_nm = Beta('MU_nm', 1, 1, 10, 0)
    MU_m = Beta("MU_m", 1, 1, 10, 0)

    walk = 1, [0]
    cycle = 1, [1]
    motorised = MU_m, [2, 3]
    nests = walk, cycle, motorised

    # definition of the model
    logprob = lognested(V_dict, av, nests, choice)

    # create the Biogeme object
    biogeme = bio.BIOGEME(database_train, logprob)
    biogeme.modelName = "LPMC"

    biogeme.generate_html = False
    biogeme.generate_pickle = False

    if for_prob:
        prob_0 = nested(V_dict, av, nests, 0)
        prob_1 = nested(V_dict, av, nests, 1)
        prob_2 = nested(V_dict, av, nests, 2)
        prob_3 = nested(V_dict, av, nests, 3)

        simulate = {
            "Prob. 0": prob_0,
            "Prob. 1": prob_1,
            "Prob. 2": prob_2,
            "Prob. 3": prob_3,
        }
        biosim = bio.BIOGEME(database_train, simulate)
        biosim.modelName = "LPMC_logit_test"

        biosim.generate_html = False
        biosim.generate_pickle = False

        return biosim

    return biogeme


def Optima(dataset_train, for_prob=False):
    """
    Create a MNL on the OPTIMA dataset.
    The model is a slightly modified version from the code that can be found here: https://github.com/JoseAngelMartinB/prediction-behavioural-analysis-ml-travel-mode-choice.

    Parameters
    ----------
    dataset_train : pandas DataFrame
        The training dataset.

    Returns
    -------
    biogeme : bio.BIOGEME
        The BIOGEME object containing the model.

    """
    database_train = db.Database("OP", dataset_train)

    globals().update(database_train.variables)

    # model
    MNL_beta_params_negative = [
        "B_TimePT_PT",
        "B_MarginalCostPT_PT",
        "B_distance_km_PT",
        "B_TimeCar_PM",
        "B_CostCarCHF_PM",
        "B_distance_km_PM",
        "B_distance_km_SM",
    ]
    MNL_beta_params_neutral = [
        "ASC_PM",
        "ASC_SM",
        "B_age_PT",
        "B_age_PM",
        "B_age_SM",
        "B_NbChild_PT",
        "B_NbChild_PM",
        "B_NbChild_SM",
        "B_NbCar_PT",
        "B_NbCar_PM",
        "B_NbCar_SM",
        "B_NbMoto_PT",
        "B_NbMoto_PM",
        "B_NbMoto_SM",
        "B_NbBicy_PT",
        "B_NbBicy_PM",
        "B_NbBicy_SM",
        "B_OccupStat_fulltime_PT",
        "B_OccupStat_fulltime_PM",
        "B_OccupStat_fulltime_SM",
        "B_Gender_man_PT",
        "B_Gender_man_PM",
        "B_Gender_man_SM",
        "B_Gender_woman_PT",
        "B_Gender_woman_PM",
        "B_Gender_woman_SM",
        "B_Gender_unreported_PT",
        "B_Gender_unreported_PM",
        "B_Gender_unreported_SM",
    ]
    MNL_utilities = {
        0: "B_age_PT*age + B_NbChild_PT*NbChild + B_NbCar_PT*NbCar + B_NbMoto_PT*NbMoto + B_NbBicy_PT*NbBicy + B_OccupStat_fulltime_PT*OccupStat_fulltime + B_Gender_man_PT*Gender_man + B_Gender_woman_PT*Gender_woman + B_Gender_unreported_PT*Gender_unreported + B_TimePT_PT*TimePT + B_MarginalCostPT_PT*MarginalCostPT + B_distance_km_PT*distance_km",
        1: "ASC_PM + B_age_PM*age + B_NbChild_PM*NbChild + B_NbCar_PM*NbCar + B_NbMoto_PM*NbMoto + B_NbBicy_PM*NbBicy + B_OccupStat_fulltime_PM*OccupStat_fulltime + B_Gender_man_PM*Gender_man + B_Gender_woman_PM*Gender_woman + B_Gender_unreported_PM*Gender_unreported + B_TimeCar_PM*TimeCar + B_CostCarCHF_PM*CostCarCHF + B_distance_km_PM*distance_km",
        2: "ASC_SM + B_age_SM*age + B_NbChild_SM*NbChild + B_NbCar_SM*NbCar + B_NbMoto_SM*NbMoto + B_NbBicy_SM*NbBicy + B_OccupStat_fulltime_SM*OccupStat_fulltime + B_Gender_man_SM*Gender_man + B_Gender_woman_SM*Gender_woman + B_Gender_unreported_SM*Gender_unreported + B_distance_km_SM*distance_km",
    }

    # construct the model parameters
    for beta in MNL_beta_params_negative:
        exec("{} = Beta('{}', 0, None, 0, 0)".format(beta, beta), globals())
    for beta in MNL_beta_params_neutral:
        exec("{} = Beta('{}', 0, None, None, 0)".format(beta, beta), globals())

    # define utility functions
    for utility_idx in MNL_utilities.keys():
        exec("V_{} = {}".format(utility_idx, MNL_utilities[utility_idx]), globals())

    # assign utility functions to utility indices
    exec("V_dict = {}", globals())
    for utility_idx in MNL_utilities.keys():
        exec("V_dict[{}] = V_{}".format(utility_idx, utility_idx), globals())

    # associate the availability conditions with the alternatives
    exec("av = {}", globals())
    for utility_idx in MNL_utilities.keys():
        exec("av[{}] = 1".format(utility_idx), globals())

    # definition of the model
    logprob = loglogit(V_dict, av, choice)

    # create the Biogeme object
    biogeme = bio.BIOGEME(database_train, logprob)
    biogeme.modelName = "Optima"

    biogeme.generate_html = False
    biogeme.generate_pickle = False

    if for_prob:
        prob_0 = logit(V_dict, av, 0)
        prob_1 = logit(V_dict, av, 1)
        prob_2 = logit(V_dict, av, 2)

        simulate = {"Prob. 0": prob_0, "Prob. 1": prob_1, "Prob. 2": prob_2}
        biosim = bio.BIOGEME(database_train, simulate)
        biosim.modelName = "optima_logit_test"

        biosim.generate_html = False
        biosim.generate_pickle = False

        return biosim

    return biogeme


def Netherlands(df_train, for_prob=False):

    database_train = db.Database("netherlands_train", df_train)
    pd.options.display.float_format = "{:.3g}".format

    globals().update(database_train.variables)

    # Parameters to be estimated
    # Arguments:
    #   1  Name for report. Typically, the same as the variable
    #   2  Starting value
    #   3  Lower bound
    #   4  Upper bound
    #   5  0: estimate the parameter, 1: keep it fixed
    ASC_CAR = Beta("ASC_CAR", 0, None, None, 0)
    ASC_RAIL = Beta("ASC_RAIL", 0, None, None, 1)
    BETA_COST_CAR = Beta("BETA_COST_CAR", 0, None, 0, 0)
    BETA_COST_RAIL = Beta("BETA_COST_RAIL", 0, None, 0, 0)
    # BETA_TT    = Beta('BETA_TT',0,None,None,0)
    BETA_TT_CAR = Beta("BETA_TT_CAR", 0, None, 0, 0)
    BETA_TT_RAIL = Beta("BETA_TT_RAIL", 0, None, 0, 0)

    BETA_PURPOSE = Beta("BETA_PURPOSE", 0, None, None, 1)
    BETA_NPERSONS = Beta("BETA_NPERSONS", 0, None, None, 1)
    BETA_AGE = Beta("BETA_AGE", 0, None, None, 1)
    BETA_EMPLOY_STATUS = Beta("BETA_EMPLOY_STATUS", 0, None, None, 1)
    BETA_MAINEARN = Beta("BETA_MAINEARN", 0, None, None, 1)
    BETA_ARRIVAL_TIME = Beta("BETA_ARRIVAL_TIME", 0, None, None, 1)
    BETA_GENDER = Beta("BETA_GENDER", 0, None, None, 1)
    BETA_TRANSFERS = Beta("BETA_TRANSFERS", 0, None, None, 1)
    BETA_SEAT_STATUS = Beta("BETA_SEAT_STATUS", 0, None, None, 1)

    # Utilities
    __Car = (
        ASC_CAR + BETA_COST_CAR * car_cost_euro + BETA_TT_CAR * car_time
    )  # + BETA_GENDER * gender + BETA_PURPOSE * purpose + BETA_NPERSONS * npersons + BETA_AGE * age + BETA_EMPLOY_STATUS * employ_status + BETA_MAINEARN * mainearn + BETA_ARRIVAL_TIME * arrival_time
    __Rail = (
        ASC_RAIL + BETA_COST_RAIL * rail_cost_euro + BETA_TT_RAIL * rail_time
    )  # + BETA_TRANSFERS * rp_transfer + BETA_SEAT_STATUS * seat_status

    __V = {0: __Car, 1: __Rail}
    __av = {0: 1, 1: 1}

    # The choice model is a logit, with availability conditions
    logprob = loglogit(__V, __av, choice)
    biogeme = bio.BIOGEME(database_train, logprob)
    biogeme.modelName = "binary_specific_netherlands"
    biogeme.generate_html = False
    biogeme.generate_pickle = False

    if for_prob:
        prob_0 = logit(__V, __av, 0)
        prob_1 = logit(__V, __av, 1)

        simulate = {"Prob. 0": prob_0, "Prob. 1": prob_1}
        biosim = bio.BIOGEME(database_train, simulate)
        biosim.modelName = "netherlands_logit_test"

        biosim.generate_html = False
        biosim.generate_pickle = False

        return biosim

    return biogeme


def Airplane(df_train, for_prob=False):

    database_train = db.Database("airlane_train", df_train)
    pd.options.display.float_format = "{:.3g}".format

    globals().update(database_train.variables)

    # Parameters to be estimated
    # Arguments:
    #   1  Name for report. Typically, the same as the variable
    #   2  Starting value
    #   3  Lower bound
    #   4  Upper bound
    #   5  0: estimate the parameter, 1: keep it fixed
    ASC_NOSTOP = Beta("ASC_NOSTOP", 0, None, None, 1)
    ASC_TRANSFER = Beta("ASC_TRANSFER", 0, None, None, 0)
    ASC_TRANSFER_TWOAIRLINES = Beta("ASC_TRANSFER_TWOAIRLINES", 0, None, None, 0)
    BETA_FARE_NOSTOP = Beta("BETA_FARE_NOSTOP", 0, None, 0, 0)
    BETA_FARE_TRANSFER = Beta("BETA_FARE_TRANSFER", 0, None, 0, 0)
    BETA_FARE_TRANSFER_TWOAIRLINES = Beta(
        "BETA_FARE_TRANSFER_TWOAIRLINES", 0, None, 0, 0
    )
    # BETA_TTDIFF_NOSTOP = Beta('BETA_TTDIFF_NOSTOP',0,None,None,1)
    BETA_TTDIFF_TRANSFER = Beta("BETA_TTDIFF_TRANSFER", 0, None, 0, 0)
    BETA_TTDIFF_TRANSFER_TWOAIRLINES = Beta(
        "BETA_TTDIFF_TRANSFER_TWOAIRLINES", 0, None, 0, 0
    )
    BETA_DEP_NOSTOP = Beta("BETA_DEP_NOSTOP", 0, None, None, 1)
    BETA_DEP_TRANSFER = Beta("BETA_DEP_TRANSFER", 0, None, None, 1)
    BETA_DEP_TRANSFER_TWOAIRLINES = Beta(
        "BETA_DEP_TRANSFER_TWOAIRLINES", 0, None, None, 1
    )
    BETA_ARR_NOSTOP = Beta("BETA_ARR_NOSTOP", 0, None, None, 1)
    BETA_ARR_TRANSFER = Beta("BETA_ARR_TRANSFER", 0, None, None, 1)
    BETA_ARR_TRANSFER_TWOAIRLINES = Beta(
        "BETA_ARR_TRANSFER_TWOAIRLINES", 0, None, None, 1
    )
    BETA_LEG_NOSTOP = Beta("BETA_LEG_NOSTOP", 0, None, None, 0)
    BETA_LEG_TRANSFER = Beta("BETA_LEG_TRANSFER", 0, None, None, 0)
    BETA_LEG_TRANSFER_TWOAIRLINES = Beta(
        "BETA_LEG_TRANSFER_TWOAIRLINES", 0, None, None, 0
    )

    # Utilities
    V_NOSTOP = (
        ASC_NOSTOP
        + BETA_FARE_NOSTOP * Fare_1_scaled
        + BETA_LEG_NOSTOP * Legroom_1
        + BETA_DEP_NOSTOP * DepartureTimeHours_1
    )  # + BETA_ARR_NOSTOP * ArrivalTimeHours_1
    V_TRANSFER = (
        ASC_TRANSFER
        + BETA_FARE_TRANSFER * Fare_2_scaled
        + BETA_TTDIFF_TRANSFER * TTDIFF_TRANSFER
        + BETA_LEG_TRANSFER * Legroom_2
        + BETA_DEP_TRANSFER * DepartureTimeHours_2
    )  # + BETA_ARR_TRANSFER * ArrivalTimeHours_2
    V_TRANSFER_TWOAIRLINES = (
        ASC_TRANSFER_TWOAIRLINES
        + BETA_FARE_TRANSFER_TWOAIRLINES * Fare_3_scaled
        + BETA_TTDIFF_TRANSFER_TWOAIRLINES * TTDIFF_TRANSFER_TWOAIRLINES
        + BETA_LEG_TRANSFER_TWOAIRLINES * Legroom_3
        + BETA_DEP_TRANSFER_TWOAIRLINES * DepartureTimeHours_3
    )  # + BETA_ARR_TRANSFER_TWOAIRLINES * ArrivalTimeHours_3

    __V = {0: V_NOSTOP, 1: V_TRANSFER, 2: V_TRANSFER_TWOAIRLINES}
    __av = {0: 1, 1: 1, 2: 1}

    # The choice model is a logit, with availability conditions
    logprob = loglogit(__V, __av, choice)
    biogeme = bio.BIOGEME(database_train, logprob)
    biogeme.modelName = "MNL_Airplane"
    biogeme.generate_html = False
    biogeme.generate_pickle = False

    if for_prob:
        prob_0 = logit(__V, __av, 0)
        prob_1 = logit(__V, __av, 1)
        prob_2 = logit(__V, __av, 2)

        simulate = {"Prob. 0": prob_0, "Prob. 1": prob_1, "Prob. 2": prob_2}
        biosim = bio.BIOGEME(database_train, simulate)
        biosim.modelName = "airplane_logit_test"

        biosim.generate_html = False
        biosim.generate_pickle = False

        return biosim

    return biogeme


def Telephone(df_train, for_prob=False):

    database_train = db.Database("telephone_train", df_train)
    pd.options.display.float_format = "{:.3g}".format

    globals().update(database_train.variables)

    # Parameters to be estimated
    # Arguments:
    #   1  Name for report. Typically, the same as the variable
    #   2  Starting value
    #   3  Lower bound
    #   4  Upper bound
    #   5  0: estimate the parameter, 1: keep it fixed
    ASC_budg_meas = Beta("ASC_budg_meas", 0, None, None, 0)
    ASC_stand_meas = Beta("ASC_stand_meas", 0, None, None, 0)
    ASC_loc_flat = Beta("ASC_loc_flat", 0, None, None, 0)
    ASC_ext_flat = Beta("ASC_ext_flat", 0, None, None, 0)
    ASC_metro_flat = Beta("ASC_metro_flat", 0, None, None, 0)
    BETA_employ_budg_meas = Beta("BETA_employ_budg_meas", 0, None, None, 0)
    BETA_employ_stand_meas = Beta("BETA_employ_stand_meas", 0, None, None, 0)
    BETA_employ_loc_flat = Beta("BETA_employ_loc_flat", 0, None, None, 0)
    BETA_employ_ext_flat = Beta("BETA_employ_ext_flat", 0, None, None, 0)
    BETA_employ_metro_flat = Beta("BETA_employ_metro_flat", 0, None, None, 0)
    BETA_users_budg_meas = Beta("BETA_users_budg_meas", 0, None, None, 0)
    BETA_users_stand_meas = Beta("BETA_users_stand_meas", 0, None, None, 0)
    BETA_users_loc_flat = Beta("BETA_users_loc_flat", 0, None, None, 0)
    BETA_users_ext_flat = Beta("BETA_users_ext_flat", 0, None, None, 0)
    BETA_users_metro_flat = Beta("BETA_users_metro_flat", 0, None, None, 0)

    # BETA_cost_budg_meas = Beta('BETA_cost_budg_meas',0,None,0,0)
    # BETA_cost_stand_meas = Beta('BETA_cost_stand_meas',0,None,0,0)
    # BETA_cost_loc_flat = Beta('BETA_cost_loc_flat',0,None,0,0)
    # BETA_cost_ext_flat = Beta('BETA_cost_ext_flat',0,None,0,0)
    # BETA_cost_metro_flat = Beta('BETA_cost_metro_flat',0,None,0,0)
    BETA_cost = Beta("BETA_cost", 0, None, None, 0)

    # Utilities
    V_budg_meas = (
        ASC_budg_meas + BETA_cost * cost1_scaled
    )  # + BETA_users_budg_meas * users + BETA_employ_budg_meas * employ #+ BETA_users_budg_meas * users + BETA_cost_budg_meas * cost1_scaled
    V_stand_meas = (
        ASC_stand_meas + BETA_cost * cost2_scaled
    )  # + BETA_users_stand_meas * users + BETA_employ_stand_meas * employ #+ BETA_users_stand_meas * users + BETA_cost_stand_meas * cost2_scaled
    V_loc_flat = (
        ASC_loc_flat + BETA_cost * cost3_scaled + BETA_users_loc_flat * users
    )  # + BETA_employ_loc_flat * employ #+ BETA_users_loc_flat * users + BETA_cost_loc_flat * cost3_scaled
    V_ext_flat = (
        ASC_ext_flat + BETA_cost * cost4_scaled + BETA_users_ext_flat * users
    )  # + BETA_employ_ext_flat * employ #+ BETA_users_ext_flat * users + BETA_cost_ext_flat * cost4_scaled
    V_metro_flat = (
        ASC_metro_flat + BETA_cost * cost5_scaled
    )  # + BETA_users_metro_flat * users + BETA_employ_metro_flat * employ #+ BETA_users_metro_flat * users + BETA_cost_metro_flat * cost5_scaled

    __V = {
        0: V_budg_meas,
        1: V_stand_meas,
        2: V_loc_flat,
        3: V_ext_flat,
        4: V_metro_flat,
    }
    __av = {0: 1, 1: 1, 2: 1, 3: 1, 4: 1}

    # The choice model is a logit, with availability conditions
    logprob = loglogit(__V, __av, choice)
    biogeme = bio.BIOGEME(database_train, logprob)
    biogeme.modelName = "MNL_Telephone"
    biogeme.generate_html = False
    biogeme.generate_pickle = False

    if for_prob:
        prob_0 = logit(__V, __av, 0)
        prob_1 = logit(__V, __av, 1)
        prob_2 = logit(__V, __av, 2)
        prob_3 = logit(__V, __av, 3)
        prob_4 = logit(__V, __av, 4)

        simulate = {
            "Prob. 0": prob_0,
            "Prob. 1": prob_1,
            "Prob. 2": prob_2,
            "Prob. 3": prob_3,
            "Prob. 4": prob_4,
        }
        biosim = bio.BIOGEME(database_train, simulate)
        biosim.modelName = "telephone_logit_test"

        biosim.generate_html = False
        biosim.generate_pickle = False

        return biosim

    return biogeme


def Parking(df_train, for_prob=False):

    database_train = db.Database("parking_train", df_train)
    pd.options.display.float_format = "{:.3g}".format

    globals().update(database_train.variables)

    # Parameters to be estimated
    # Arguments:
    #   1  Name for report. Typically, the same as the variable
    #   2  Starting value
    #   3  Lower bound
    #   4  Upper bound
    #   5  0: estimate the parameter, 1: keep it fixed
    ASC_FSP = Beta("ASC_FSP", 0, None, None, 1)
    ASC_PSP = Beta("ASC_PSP", 0, None, None, 0)
    ASC_PUP = Beta("ASC_PUP", 0, None, None, 0)
    BETA_AT_FSP = Beta("BETA_AT_FSP", 0, None, 0, 0)
    BETA_AT_PSP = Beta("BETA_AT_PSP", 0, None, 0, 0)
    BETA_AT_PUP = Beta("BETA_AT_PUP", 0, None, 0, 0)
    BETA_TD = Beta("BETA_TD", 0, None, 0, 0)
    # BETA_TD_FSP = Beta('BETA_TD_FSP',0,None,None,0)
    # BETA_TD_PSP = Beta('BETA_TD_PSP',0,None,None,0)
    # BETA_TD_PUP = Beta('BETA_TD_PUP',0,None,None,0)
    BETA_FEE_FSP = Beta("BETA_FEE_FSP", 0, None, 0, 1)
    BETA_FEE_PSP = Beta("BETA_FEE_PSP", 0, None, 0, 0)
    BETA_FEE_PUP = Beta("BETA_FEE_PUP", 0, None, 0, 0)

    BETA_GENDER_FSP = Beta("BETA_GENDER_FSP", 0, None, None, 1)
    BETA_GENDER_PSP = Beta("BETA_GENDER_PSP", 0, None, None, 1)
    BETA_GENDER_PUP = Beta("BETA_GENDER_PUP", 0, None, None, 1)
    BETA_INCH_FSP = Beta("BETA_INCH_FSP", 0, None, None, 1)
    BETA_INCH_PSP = Beta("BETA_INCH_PSP", 0, None, None, 0)
    BETA_INCH_PUP = Beta("BETA_INCH_PUP", 0, None, None, 0)

    # Utilities
    V_FSP = (
        ASC_FSP
        + BETA_AT_FSP * AT1
        + BETA_TD * TD1
        + BETA_FEE_FSP * FEE1
        + BETA_INCH_FSP * INCH
        + BETA_GENDER_FSP * GENDER
    )
    V_PSP = (
        ASC_PSP
        + BETA_AT_PSP * AT2
        + BETA_TD * TD2
        + BETA_FEE_PSP * FEE2
        + BETA_INCH_PSP * INCH
        + BETA_GENDER_PSP * GENDER
    )
    V_PUP = (
        ASC_PUP
        + BETA_AT_PUP * AT3
        + BETA_TD * TD3
        + BETA_FEE_PUP * FEE3
        + BETA_INCH_PUP * INCH
        + BETA_GENDER_PUP * GENDER
    )

    __V = {0: V_FSP, 1: V_PSP, 2: V_PUP}
    __av = {0: 1, 1: 1, 2: 1}

    # The choice model is a logit, with availability conditions
    logprob = loglogit(__V, __av, choice)
    biogeme = bio.BIOGEME(database_train, logprob)
    biogeme.modelName = "MNL_Parking"
    biogeme.generate_html = False
    biogeme.generate_pickle = False

    if for_prob:
        prob_0 = logit(__V, __av, 0)
        prob_1 = logit(__V, __av, 1)
        prob_2 = logit(__V, __av, 2)

        simulate = {"Prob. 0": prob_0, "Prob. 1": prob_1, "Prob. 2": prob_2}
        biosim = bio.BIOGEME(database_train, simulate)
        biosim.modelName = "parking_logit_test"

        biosim.generate_html = False
        biosim.generate_pickle = False

        return biosim

    return biogeme


def Vaccines(dataset_train: pd.DataFrame, for_prob=False):
    """
    Create a MNL on the Vaccine dataset.

    Parameters
    ----------
    dataset_train : pandas DataFrame
        The training dataset.

    Returns
    -------
    biogeme : bio.BIOGEME
        The BIOGEME object containing the model.
    """
    database_train = db.Database("vaccine_train", dataset_train)

    globals().update(database_train.variables)

    # parameters to be estimated
    ASC_V1 = Beta("ASC_V1", 0, None, None, 1)
    ASC_V2 = Beta("ASC_V2", 0, None, None, 0)

    # optout alternative
    ASC_OPTOUT = Beta("ASC_OPTOUT", 0, None, None, 0)
    ASC_MALE = Beta("ASC_MALE", 0, None, None, 0)
    ASC_BBOLD = Beta("ASC_BBOLD", 0, None, None, 0)
    ASC_MIL = Beta("ASC_MIL", 0, None, None, 0)
    ASC_GENX = Beta("ASC_GENX", 0, None, None, 0)
    ASC_BMI = Beta("ASC_BMI", 0, None, None, 0)
    ASC_FUL = Beta("ASC_FUL", 0, None, None, 0)
    ASC_OHH = Beta("ASC_OHH", 0, None, None, 0)
    ASC_CHH = Beta("ASC_CHH", 0, None, None, 0)
    ASC_NIN = Beta("ASC_NIN", 0, None, None, 0)
    ASC_TES = Beta("ASC_TES", 0, None, None, 0)
    ASC_VCO = Beta("ASC_VCO", 0, None, None, 0)
    ASC_REP = Beta("ASC_REP", 0, None, None, 0)
    ASC_ASI = Beta("ASC_ASI", 0, None, None, 0)
    ASC_HIS = Beta("ASC_HIS", 0, None, None, 0)
    ASC_AGE = Beta("ASC_AGE", 0, None, None, 0)
    ASC_BSC = Beta("ASC_BSC", 0, None, None, 0)
    ASC_PGDT = Beta("ASC_PGDT", 0, None, None, 0)
    ASC_HHINC = Beta("ASC_HHINC", 0, None, None, 0)
    ASC_BLACKAFRAMERICAN = Beta("ASC_BLACKAFRAMERICAN", 0, None, None, 0)
    ASC_DEMOCRAT = Beta("ASC_DEMOCRAT", 0, None, None, 0)
    ASC_COVID = Beta("ASC_COVID", 0, None, None, 0)
    ASC_FLUSHOT = Beta("ASC_FLUSHOT", 0, None, None, 0)
    ASC_CONDITIONS = Beta("ASC_CONDITIONS", 0, None, None, 0)

    # vaccines parameters
    B_COST = Beta("B_COST", 0, None, 0, 0)
    B_EFFECTIVE = Beta("B_EFFECTIVE", 0, 0, None, 0)
    B_PROTECT = Beta("B_PROTECT", 0, 0, None, 0)
    B_INCUB = Beta("B_INCUB", 0, None, 0, 0)
    B_SEVERESIDE = Beta("B_SEVERESIDE", 0, None, 0, 0)
    B_MILDSIDE_1 = Beta("B_MILDSIDE_1", 0, None, 0, 0)
    B_MILDSIDE_3 = Beta("B_MILDSIDE_3", 0, None, 0, 0)
    B_DOSE_1 = Beta("B_DOSE_1", 0, None, 0, 0)
    B_DOSE_3 = Beta("B_DOSE_3", 0, None, 0, 0)
    B_USA = Beta("B_USA", 0, None, None, 0)
    B_BOOST = Beta("B_BOOST", 0, None, 0, 0)
    B_CHI = Beta("B_CHI", 0, None, None, 0)

    # utilities
    # V_V1 = ASC_V1 + B_COST * cost1 + B_EFFECTIVE * effectiveness1 + B_PROTECT * protection1 + B_INCUB * incubation1 + B_SEVERESIDE * severe1 + B_MILDSIDE * mild1 + B_DOSE * doses1 + B_USA * USA1
    # V_OPTOUT = ASC_OPTOUT + ASC_MALE * Male + ASC_BLACKAFRAMERICAN * Black +  ASC_BSC * BSc +  ASC_PGDT * PostGrad + ASC_HHINC * HHInc10K + ASC_BBOLD * babyboomolder +  ASC_DEMOCRAT * Democrat + ASC_COVID * covidpos + ASC_FLUSHOT * FluShot + ASC_CONDITIONS * Underlying
    # V_V2 = ASC_V2 + B_COST * cost3 + B_EFFECTIVE * effectiveness3 + B_PROTECT * protection3 + B_INCUB * incubation3 + B_SEVERESIDE * severe3 + B_MILDSIDE * mild3 + B_DOSE * doses3 + B_USA * USA3

    # utilities
    V_V1 = (
        ASC_V1
        + B_COST * cost1
        + B_EFFECTIVE * effectiveness1
        + B_PROTECT * protection1
        + B_INCUB * incubation1
        + B_SEVERESIDE * severe1
        + B_USA * USA1
    )  # + B_MILDSIDE_1 * mild1 #+ B_DOSE_1 * doses1#+ ASC_AGE * Age + ASC_BMI * BMI #+ ASC_MALE * Male + ASC_BLACKAFRAMERICAN * Black +  ASC_BSC * BSc +  ASC_PGDT * PostGrad + ASC_HHINC * HHInc10K + ASC_BBOLD * babyboomolder +  ASC_DEMOCRAT * Democrat + ASC_COVID * covidpos + ASC_FLUSHOT * FluShot + ASC_CONDITIONS * Underlying + ASC_MIL * millenial + ASC_GENX * genX + ASC_BMI * BMI + ASC_FUL * FullTime + ASC_OHH * olderinHH + ASC_CHH * chilinHH + ASC_TES * Tested + ASC_HIS * Hispanic + ASC_AGE * Age + ASC_ASI * Asian + ASC_NIN * noIns + ASC_VCO * VeryConservative + ASC_REP * Republican + B_BOOST * booster1 + B_CHI * China1 #+ B_UK * UK1 + B_GER * Germany1 + B_RUS * Russia1
    V_OPTOUT = (
        ASC_OPTOUT
        + ASC_MALE * Male
        + ASC_BLACKAFRAMERICAN * Black
        + ASC_BSC * BSc
        + ASC_PGDT * PostGrad
        + ASC_HHINC * HHInc10K
        + ASC_BBOLD * babyboomolder
        + ASC_DEMOCRAT * Democrat
        + ASC_FLUSHOT * FluShot
        + ASC_CONDITIONS * Underlying
    )  # + ASC_MIL * millenial + ASC_GENX * genX + ASC_OHH * olderinHH + ASC_CHH * chilinHH + ASC_TES * Tested + ASC_AGE * Age + ASC_VCO * VeryConservative #+ ASC_REP * Republican + ASC_NIN * noIns + ASC_HIS * Hispanic + ASC_ASI * Asian + ASC_FUL * FullTime + ASC_BMI * BMI + ASC_COVID * covidpos
    V_V2 = (
        ASC_V2
        + B_COST * cost3
        + B_EFFECTIVE * effectiveness3
        + B_PROTECT * protection3
        + B_INCUB * incubation3
        + B_SEVERESIDE * severe3
        + B_USA * USA3
    )  # + B_MILDSIDE_3 * mild3 #+ B_DOSE_3 * doses3#+ ASC_AGE * Age + ASC_BMI * BMI #+ ASC_MALE * Male + ASC_BLACKAFRAMERICAN * Black +  ASC_BSC * BSc +  ASC_PGDT * PostGrad + ASC_HHINC * HHInc10K + ASC_BBOLD * babyboomolder +  ASC_DEMOCRAT * Democrat + ASC_COVID * covidpos + ASC_FLUSHOT * FluShot + ASC_CONDITIONS * Underlying + ASC_MIL * millenial + ASC_GENX * genX + ASC_BMI * BMI + ASC_FUL * FullTime + ASC_OHH * olderinHH + ASC_CHH * chilinHH + ASC_TES * Tested + ASC_HIS * Hispanic + ASC_AGE * Age + ASC_ASI * Asian + ASC_NIN * noIns + ASC_VCO * VeryConservative + ASC_REP * Republican + B_BOOST * booster3 + B_CHI * China3 #+ B_UK * UK3 + B_GER * Germany3 + B_RUS * Russia3

    V = {0: V_V1, 1: V_OPTOUT, 2: V_V2}
    av = {0: 1, 1: 1, 2: 1}

    # choice model
    logprob = loglogit(V, av, choice)
    biogeme = bio.BIOGEME(database_train, logprob)
    biogeme.modelName = "VaccineMNL"

    biogeme.generate_html = False
    biogeme.generate_pickle = False

    if for_prob:
        prob_V1 = logit(V, av, 0)
        prob_OPTOUT = logit(V, av, 1)
        prob_V2 = logit(V, av, 2)

        simulate = {
            "Prob. V1": prob_V1,
            "Prob. OPTOUT": prob_OPTOUT,
            "Prob. V2": prob_V2,
        }
        biosim = bio.BIOGEME(database_train, simulate)
        biosim.modelName = "vaccine_logit_test"

        biosim.generate_html = False
        biosim.generate_pickle = False

        return biosim

    return biogeme


def MTMC_lausanne_MNL(dataset_train: pd.DataFrame, for_prob=False, results=None):
    """
    Estimation of a MNL model.

    Parameters
    ----------
    dataset_train : pandas DataFrame
        The training dataset.
    for_prob : bool, optional
        If True, the function returns a BIOGEME object for probability calculation.
    results : bio.BIOGEME, optional (default=None)
        The biogeme model estimated.

    Returns
    -------
    biogeme : bio.BIOGEME
        The BIOGEME object containing the model.
    """
    database = db.Database("mtmc_train", dataset_train)
    globals().update(database.variables)

    # define level of verbosity
    logger = blog.get_screen_logger(level=blog.DEBUG)
    logger.info("LPMC MNL")

    alt = 88

    ASC_c = Beta("ASC_c", 0, None, None, 0)
    ASC_pt = Beta("ASC_pt", 0, None, None, 0)
    ASC_act = Beta("ASC_act", 0, None, None, 1)

    # car specific
    B_TIME_c = Beta("B_TIME_c", 0, None, None, 0)
    # pt specific
    B_TIME_pt = Beta("B_TIME_pt", 0, None, None, 0)
    # soft modes specific
    B_TIME_act = Beta("B_TIME_act", 0, None, None, 0)

    # activity specific
    B_JobD_WORK = Beta("B_JobD_WORK", 0, None, None, 0)
    B_PopD_EDUC = Beta("B_PopD_EDUC", 0, None, None, 0)
    B_PopD_LEIS = Beta("B_PopD_LEIS", 0, None, None, 0)
    B_JobD_PopD_SHOP = Beta("B_JobD_PopD_SHOP", 0, None, None, 0)
    B_PopD_ESC = Beta("B_PopD_ESC", 0, None, None, 0)
    B_PopD_ERR = Beta("B_PopD_ERR", 0, None, None, 0)

    # utilities
    V_CAR = [
        ASC_c
        + B_TIME_c * globals()["CAR_TT_" + str(i + 6334)]
        + work * B_JobD_WORK * globals()["job_density_" + str(i)]
        + education * B_PopD_EDUC * globals()["pop_density_" + str(i)]
        + leisure * B_PopD_LEIS * (globals()["pop_density_" + str(i)])
        + shopping
        * B_JobD_PopD_SHOP
        * (globals()["pop_density_" + str(i)] + globals()["job_density_" + str(i)])
        + escort * B_PopD_ESC * globals()["pop_density_" + str(i)]
        + errands * B_PopD_ERR * globals()["pop_density_" + str(i)]
        for i in range(0, alt)
    ]
    V_PT = [
        ASC_pt
        + B_TIME_pt * globals()["PT_TT_" + str(i + 6334)]
        + work * B_JobD_WORK * globals()["job_density_" + str(i)]
        + education * B_PopD_EDUC * globals()["pop_density_" + str(i)]
        + leisure * B_PopD_LEIS * (globals()["pop_density_" + str(i)])
        + shopping
        * B_JobD_PopD_SHOP
        * (globals()["pop_density_" + str(i)] + globals()["job_density_" + str(i)])
        + escort * B_PopD_ESC * globals()["pop_density_" + str(i)]
        + errands * B_PopD_ERR * globals()["pop_density_" + str(i)]
        for i in range(0, alt)
    ]
    V_ACT = [
        ASC_act
        + B_TIME_act * globals()["ACT_TT_" + str(i + 6334)]
        + work * B_JobD_WORK * globals()["job_density_" + str(i)]
        + education * B_PopD_EDUC * globals()["pop_density_" + str(i)]
        + leisure * B_PopD_LEIS * (globals()["pop_density_" + str(i)])
        + shopping
        * B_JobD_PopD_SHOP
        * (globals()["pop_density_" + str(i)] + globals()["job_density_" + str(i)])
        + escort * B_PopD_ESC * globals()["pop_density_" + str(i)]
        + errands * B_PopD_ERR * globals()["pop_density_" + str(i)]
        for i in range(0, alt)
    ]

    # assignment of utility function to alternatives
    V_car = {i: V_CAR[i] for i in range(0, alt)}
    V_pt = {i + alt: V_PT[i] for i in range(0, alt)}
    V_act = {i + 2 * alt: V_ACT[i] for i in range(0, alt)}
    # combining different modes together
    V = {**V_car, **V_pt, **V_act}

    # define alternatives availability -> here everything is available
    avail = {i: 1 for i in range(0, alt * 3)}

    # for predicting
    # if for_prob:
    #     simulate ={'Prob. '+str(i):logit(V, None, i) for i in range(0, 3*alt)}
    #     biosim = bio.BIOGEME(database, simulate)
    #     biosim.modelName = "MTMC_Lausanne_MNL_test"

    #     biosim.generate_html = False
    #     biosim.generate_pickle = False

    #     return biosim

    if for_prob:
        logprob = logit(V, avail, choice)
        simulated_choices = logprob.getValue_c(
            betas=results.getBetaValues(), database=database, prepareIds=True
        )
        return simulated_choices

    # define model
    logprob = loglogit(V, avail, choice)
    # create the Biogeme object
    biogeme = bio.BIOGEME(database, logprob)
    biogeme.modelName = "MTMC_Lausanne_MNL"

    return biogeme


def MTMC_lausanne_CNL(dataset_train: pd.DataFrame, for_prob=False, results=None):
    """
    Estimation of a CNL model.

    Parameters
    ----------
    dataset_train : pandas DataFrame
        The training dataset.
    for_prob : bool, optional
        If True, the function returns a BIOGEME object for probability calculation.
    results : bio.BIOGEME, optional (default=None)
        The biogeme model estimated.

    Returns
    -------
    biogeme : bio.BIOGEME
        The BIOGEME object containing the model.
    """
    database = db.Database("mtmc_train", dataset_train)
    globals().update(database.variables)

    alt = 88
    # defining groups of destinations for the cross-nested logit model
    west_loz = list(range(0, 25))
    east_loz = [27, 30, 46, 50, 54] + list(range(68, 88))
    city_center = [i for i in range(0, 88) if i not in west_loz if i not in east_loz]

    ASC_c = Beta("ASC_c", 0, None, None, 0)
    ASC_pt = Beta("ASC_pt", 0, None, None, 0)
    ASC_act = Beta("ASC_act", 0, None, None, 1)

    # nest parameter
    MU_CAR = Beta("MU_CAR", 1, 1, 10, 0)
    MU_PT = Beta("MU_PT", 1, 1, 10, 0)
    MU_ACT = Beta("MU_ACT", 1, 1, 10, 1)
    MU_WEST = Beta("MU_WEST", 1, 1, 10, 0)
    MU_CENTER = Beta("MU_CENTER", 1, 1, 10, 0)
    MU_EAST = Beta("MU_EAST", 1, 1, 10, 0)

    # car specific
    B_TIME_c = Beta("B_TIME_c", 0, None, None, 0)
    # pt specific
    B_TIME_pt = Beta("B_TIME_pt", 0, None, None, 0)
    # soft modes specific
    B_TIME_act = Beta("B_TIME_act", 0, None, None, 0)

    # activity specific
    B_JobD_WORK = Beta("B_JobD_WORK", 0, None, None, 0)
    B_PopD_EDUC = Beta("B_PopD_EDUC", 0, None, None, 0)
    B_PopD_LEIS = Beta("B_PopD_LEIS", 0, None, None, 0)
    B_JobD_PopD_SHOP = Beta("B_JobD_PopD_SHOP", 0, None, None, 0)
    B_PopD_ESC = Beta("B_PopD_ESC", 0, None, None, 0)
    B_PopD_ERR = Beta("B_PopD_ERR", 0, None, None, 0)

    # utilities
    V_CAR = [
        ASC_c
        + B_TIME_c * globals()["CAR_TT_" + str(i + 6334)]
        + work * B_JobD_WORK * globals()["job_density_" + str(i)]
        + education * B_PopD_EDUC * globals()["pop_density_" + str(i)]
        + leisure * B_PopD_LEIS * (globals()["pop_density_" + str(i)])
        + shopping
        * B_JobD_PopD_SHOP
        * (globals()["pop_density_" + str(i)] + globals()["job_density_" + str(i)])
        + escort * B_PopD_ESC * globals()["pop_density_" + str(i)]
        + errands * B_PopD_ERR * globals()["pop_density_" + str(i)]
        for i in range(0, alt)
    ]
    V_PT = [
        ASC_pt
        + B_TIME_pt * globals()["PT_TT_" + str(i + 6334)]
        + work * B_JobD_WORK * globals()["job_density_" + str(i)]
        + education * B_PopD_EDUC * globals()["pop_density_" + str(i)]
        + leisure * B_PopD_LEIS * (globals()["pop_density_" + str(i)])
        + shopping
        * B_JobD_PopD_SHOP
        * (globals()["pop_density_" + str(i)] + globals()["job_density_" + str(i)])
        + escort * B_PopD_ESC * globals()["pop_density_" + str(i)]
        + errands * B_PopD_ERR * globals()["pop_density_" + str(i)]
        for i in range(0, alt)
    ]
    V_ACT = [
        ASC_act
        + B_TIME_act * globals()["ACT_TT_" + str(i + 6334)]
        + work * B_JobD_WORK * globals()["job_density_" + str(i)]
        + education * B_PopD_EDUC * globals()["pop_density_" + str(i)]
        + leisure * B_PopD_LEIS * (globals()["pop_density_" + str(i)])
        + shopping
        * B_JobD_PopD_SHOP
        * (globals()["pop_density_" + str(i)] + globals()["job_density_" + str(i)])
        + escort * B_PopD_ESC * globals()["pop_density_" + str(i)]
        + errands * B_PopD_ERR * globals()["pop_density_" + str(i)]
        for i in range(0, alt)
    ]

    # assignment of utility function to alternatives
    V_car = {i: V_CAR[i] for i in range(0, alt)}
    V_pt = {i + alt: V_PT[i] for i in range(0, alt)}
    V_act = {i + 2 * alt: V_ACT[i] for i in range(0, alt)}
    # combining different modes together
    V = {**V_car, **V_pt, **V_act}

    # define alternatives availability -> here everything is available
    avail = {i: 1 for i in range(0, alt * 3)}

    # define which alternatives belong to which nest
    ALPHA_CAR = {i: 0.5 if i < alt else 0.0 for i in range(0, 3 * alt)}
    ALPHA_PT = {i: 0.5 if i >= alt and i < 2 * alt else 0.0 for i in range(0, 3 * alt)}
    ALPHA_ACT = {i: 0.5 if i >= 2 * alt else 0.0 for i in range(0, 3 * alt)}
    # using modulo to get zones for all 3 modes belonging to each groups
    ALPHA_WEST = {i: 0.5 if i % alt in west_loz else 0.0 for i in range(0, 3 * alt)}
    ALPHA_CENTER = {
        i: 0.5 if i % alt in city_center else 0.0 for i in range(0, 3 * alt)
    }
    ALPHA_EAST = {i: 0.5 if i % alt in east_loz else 0.0 for i in range(0, 3 * alt)}

    # nest definition
    nest_car = MU_CAR, ALPHA_CAR
    nest_pt = MU_PT, ALPHA_PT
    nest_act = MU_ACT, ALPHA_ACT
    nest_west = MU_WEST, ALPHA_WEST
    nest_center = MU_CENTER, ALPHA_CENTER
    nest_east = MU_EAST, ALPHA_EAST

    nests = nest_car, nest_pt, nest_act, nest_west, nest_center, nest_east

    # define level of verbosity
    logger = blog.get_screen_logger(level=blog.DEBUG)
    logger.info("LPMC CNL")

    # for predicting
    # if for_prob:
    #     simulate ={'Prob. '+str(i):cnl_avail(V, avail, i) for i in range(0, 3*alt)}
    #     biosim = bio.BIOGEME(database, simulate)
    #     biosim.modelName = "MTMC_Lausanne_CNL_test"

    #     biosim.generate_html = False
    #     biosim.generate_pickle = False

    #     return biosim

    if for_prob:
        logprob = cnl_avail(V, None, nests, choice)
        simulated_choices = logprob.getValue_c(
            betas=results.getBetaValues(), database=database, prepareIds=True
        )
        return simulated_choices

    # define model
    logprob = logcnl_avail(V, avail, nests, choice)

    # create the Biogeme object
    biogeme = bio.BIOGEME(database, logprob)
    biogeme.modelName = "MTMC_Lausanne_CNL"

    return biogeme
