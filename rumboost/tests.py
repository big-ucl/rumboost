from utils import *
from basic_functions import *
from utility_smoothing import *
from rumboost import *
from dataset import *
from models import *
from utility_plotting import *
import lightgbm
from sklearn.model_selection import train_test_split

# swissmetro_train, swissmetro_test = load_preprocess_SwissMetro(test_size=0.2,random_state = 2)
# swissmetro_model = SwissMetro(swissmetro_train)

# params = {'learning_rate':0.2,
#           'max_depth':1,
#           'num_classes':3,
#           'num_iterations':500,
#           'objective':'multiclass',
#           'verbosity': 1,
#           'early_stopping_round':50}

# rum_structure = bio_to_rumboost(swissmetro_model)

# features = [f for f in swissmetro_train.columns if f != "CHOICE"]
# label = "CHOICE"

# train_idx = []
# test_idx = []
# for (train_i, test_i) in stratified_group_k_fold(swissmetro_train[features], swissmetro_train['CHOICE'], swissmetro_train.index, k=5):
#     train_idx.append(train_i)
#     test_idx.append(test_i)
# folds = zip(train_idx, test_idx)


# dataset_train = lightgbm.Dataset(swissmetro_train[features], label=swissmetro_train[label], free_raw_data=False)
# dataset_test = lightgbm.Dataset(swissmetro_test[features], label=swissmetro_test[label], free_raw_data=False)
#trained_model = rum_train(params,dataset_train,rum_structure, valid_sets=[dataset_test])

LPMC_train, LPMC_test, folds = load_preprocess_LPMC()
LPMC_model = LPMC(LPMC_train)

params = {'learning_rate':0.2,
          'max_depth':1,
          'num_classes':4,
          'num_iterations':600,
          'objective':'multiclass',
          'verbosity': 1,
          'early_stopping_round':20
          }

rum_structure = bio_to_rumboost(LPMC_model)

features = [f for f in LPMC_train.columns if f != "choice"]

label = "choice"
train_idx = []
test_idx = []
for (train_i, test_i) in stratified_group_k_fold(LPMC_train[features], LPMC_train['choice'], LPMC_train.index, k=5):
    train_idx.append(train_i)
    test_idx.append(test_i)
folds = zip(train_idx, test_idx)

dataset_train = lightgbm.Dataset(LPMC_train[features], label=LPMC_train[label], free_raw_data=False)
ataset_test = lightgbm.Dataset(LPMC_test[features], label=LPMC_test[label], free_raw_data=False)
trained_model = rum_cv(params,dataset_train,folds =folds,rum_structure=rum_structure, verbose_eval=2)
#trained_model = rum_train(params,dataset_train,rum_structure, valid_sets=[dataset_test])
