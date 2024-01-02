import sys
sys.path
sys.path.append('/rumboost')
from utils import *
from utility_smoothing import *
from rumboost import *
from dataset import *
from models import *
from utility_plotting import *
import lightgbm
from sklearn.model_selection import train_test_split, KFold

# swissmetro_train, swissmetro_test = load_preprocess_SwissMetro(test_size=0.2,random_state = 35)
# swissmetro_model = SwissMetro(swissmetro_train)

# params = {'learning_rate':0.3,
#           'max_depth':1,
#           'num_classes':3,
#           'num_iterations':500,
#           'objective':'multiclass',
#           'verbosity': 1,
#           'early_stopping_round':50}

# rum_structure = bio_to_rumboost(swissmetro_model)

# features = [f for f in swissmetro_train.columns if f != "CHOICE"]
# label = "CHOICE"


# kf = KFold(n_splits=5)
# train_idx = []
# test_idx = []
# for (train_i, test_i) in kf.split(swissmetro_train):
#     train_idx.append(train_i)
#     test_idx.append(test_i)
# folds = zip(train_idx, test_idx)


# dataset_train = lightgbm.Dataset(swissmetro_train[features], label=swissmetro_train[label], free_raw_data=False)
# dataset_test = lightgbm.Dataset(swissmetro_test[features], label=swissmetro_test[label], free_raw_data=False)
# trained_model = rum_train(params,dataset_train,rum_structure, valid_sets=[dataset_test])
#trained_model = rum_cv(params,dataset_train,folds =folds,rum_structure=rum_structure, verbose_eval=2)

LPMC_train, LPMC_test, folds = load_preprocess_LPMC()
LPMC_model = LPMC(LPMC_train)

params = {'learning_rate':0.1,
          'max_depth':3,
          'num_classes':4,
          'num_iterations':2000,
          'objective':'multiclass',
          'verbosity': 1,
          'early_stopping_round':50,
          'min_data_in_leaf':10,
          'min_sum_hessian_in_leaf':3,
          'lambda_l1': 0.7,
          'lambda_l2': 0.1,
          'feature_fraction': 0.7
          }

rum_structure = bio_to_rumboost(LPMC_model)

features = [f for f in LPMC_train.columns if f != "choice"]

label = "choice"
train_idx = []
test_idx = []
for (train_i, test_i) in stratified_group_k_fold(LPMC_train[features], LPMC_train['choice'], LPMC_train['household_id'], k=5):
    train_idx.append(train_i)
    test_idx.append(test_i)
folds = zip(train_idx, test_idx)

# LPMC_train_train, LPMC_train_test = train_test_split(LPMC_train, test_size=0.2)

dataset_train = lightgbm.Dataset(LPMC_train[features], label=LPMC_train[label], free_raw_data=False)
dataset_test = lightgbm.Dataset(LPMC_test[features], label=LPMC_test[label], free_raw_data=False)
#trained_model = rum_cv(params,dataset_train,folds =folds,rum_structure=rum_structure, verbose_eval=2)
trained_model = rum_train(params,dataset_train,rum_structure, valid_sets=[dataset_test])


# Optima_train, Optima_test, folds = load_preprocess_Optima()
# Optima_model = Optima(Optima_train)

# params = {'learning_rate':0.1,
#           'max_depth':2,
#           'num_classes':3,
#           'num_iterations':2000,
#           'objective':'multiclass',
#           'verbosity': 1,
#           'early_stopping_round':50,
#           'min_data_in_leaf':10,
#           'min_sum_hessian_in_leaf':3,
#           'lambda_l1': 0.7,
#           'lambda_l2': 0.1,
#           'feature_fraction': 0.7
#           }

# rum_structure = bio_to_rumboost(Optima_model)

# features = [f for f in Optima_train.columns if f != 'choice']
# label = 'choice'

# dataset_train = lightgbm.Dataset(Optima_train[features], label=Optima_train[label], free_raw_data=False)
# dataset_test = lightgbm.Dataset(Optima_test[features], label=Optima_test[label], free_raw_data=False)
# trained_model = rum_cv(params,dataset_train,folds =folds,rum_structure=rum_structure, verbose_eval=2)
# trained_model = rum_train(params,dataset_train,rum_structure, valid_sets=[dataset_test])
