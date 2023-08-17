from utils import *
from basic_functions import *
from utility_smoothing import *
from rumboost import *
from dataset import *
from models import *
from utility_plotting import *
import lightgbm

swissmetro_train, swissmetro_test = load_preprocess_SwissMetro()
swissmetro_model = SwissMetro(swissmetro_train)

params = {'learning_rate':0.3,
          'max_depth':1,
          'num_classes':3,
          'num_boost_rounds':50}

rum_structure = bio_to_rumboost(swissmetro_model)

features = [f for f in swissmetro_train.columns if f != "CHOICE"]

label = swissmetro_train["CHOICE"]

dataset_train = lightgbm.Dataset(swissmetro_train[features], label=label)

trained_model = rum_train(params,dataset_train,rum_structure)