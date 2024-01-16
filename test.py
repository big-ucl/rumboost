
import sys
sys.path.append('rumboost/')
from utils import *
from utility_smoothing import *
from rumboost import *
from dataset import *
from models import *
from utility_plotting import *

import lightgbm

#load dataset
LPMC_train, LPMC_test, folds = load_preprocess_LPMC()

#load model
LPMC_model = LPMC(LPMC_train)

#parameters
params = {'n_jobs': -1,
          'num_classes':4,
          'objective':'multiclass',
          'boosting': 'gbdt',
          'monotone_constraints_method': 'advanced',
          'verbosity': 1,
          'num_iterations':3000,
          'early_stopping_round':100,
          'learning_rate':0.1,
          'max_depth':1
          }

#features and label column names

features = [f for f in LPMC_train.columns if f != "choice"]

label = "choice"
#create lightgbm dataset
dataset_train = lightgbm.Dataset(LPMC_train[features], label=LPMC_train[label], free_raw_data=False)
dataset_test = lightgbm.Dataset(LPMC_test[features], label=LPMC_test[label], free_raw_data=False)

mu = [1, 1, 1.16]
# mu = [1, 1, 1.16]

alphas  =np.array([[1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [0, 0, 1]])

nests = {0:0, 1:1, 2:2, 3:2}

# ce_loss = 0
# num_trees = 0
# for i, (train_idx, test_idx) in enumerate(folds):
#     train_set = dataset_train.subset(sorted(train_idx))
#     test_set = dataset_train.subset(sorted(test_idx))

#     # Create the classifier
#     param = copy.deepcopy(params)
#     rum_structure = bio_to_rumboost(LPMC_model)


#     print('-'*50 + '\n')
#     print(f'Iteration {i+1}')
#     LPMC_model_trained = rum_train(param,train_set,rum_structure=rum_structure, valid_sets = [test_set], mu=mu, alphas=alphas)
#     # LPMC_model_trained = rum_train(param,train_set,rum_structure=rum_structure, valid_sets = [test_set], mu=mu, nests=nests)
#     ce_loss += LPMC_model_trained.best_score
#     num_trees += LPMC_model_trained.best_iteration
#     print('-'*50 + '\n')
#     print(f'Best cross entropy loss: {LPMC_model_trained.best_score}')
#     print(f'Best number of trees: {LPMC_model_trained.best_iteration}')

# ce_loss = ce_loss/5
# num_trees = num_trees/5
# print('-'*50 + '\n')
# print(f'Cross validation negative cross entropy loss: {ce_loss}')
# print(f'With a number of trees on average of {num_trees}')

rum_structure = bio_to_rumboost(LPMC_model)
params = {'n_jobs': -1,
          'num_classes':4,
          'objective':'multiclass',
          'boosting': 'gbdt',
          'monotone_constraints_method': 'advanced',
          'verbosity': 1,
          'num_iterations':1256,
          'early_stopping_round':100,
          'learning_rate':0.1,
          'max_depth':1
          }

#LPMC_model_fully_trained = rum_train(params, dataset_train, rum_structure, valid_sets=[dataset_test], mu=mu, nests=nests)
LPMC_model_fully_trained = rum_train(params, dataset_train, rum_structure, valid_sets=[dataset_test], mu=mu, alphas=alphas)

preds, _, _ = LPMC_model_fully_trained.predict(dataset_test, mu=mu, alphas=alphas)

ce_test = cross_entropy(preds, dataset_test.get_label().astype(int))

print('-'*50)
print(f'Final negative cross-entropy on the test set: {ce_test}')