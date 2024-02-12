import pytest
import rumboost as rumb
import numpy as np
np.set_printoptions(precision=10, floatmode='maxprec')
from lightgbm import Dataset

def test_rumboost_object():
    #create a rumboost object
    model = rumb.RUMBoost()

    #check if the object is created with correct attributes
    assert model is not None
    assert isinstance(model.boosters, list)
    assert model.valid_sets is None
    assert model.num_classes is None
    assert model.mu is None
    assert model.nests is None
    assert model.alphas is None
    assert model.functional_effects is None


def test_f_obj():
    #create a RUMBoost object
    rumboost = rumb.RUMBoost()

    #assign some values to the object
    rumboost._preds = np.array([[0.5], [0.5], [0.5]])
    rumboost._current_j = 0
    rumboost.num_classes = 2

    #create a dummy dataset
    train_data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    train_labels = np.array([0, 1, 0])
    train_set = Dataset(train_data, label=train_labels)

    #call the f_obj method
    grad, hess = rumboost.f_obj(None, train_set)

    #perform assertions
    assert np.array_equal(grad, np.array([0.5, -0.5, 0.5]))
    assert np.array_equal(hess, np.array([0.5, 0.5, 0.5]))

    #assign some values to the object
    rumboost._preds = np.array([[0.1, 0.2, 0.3], [0.5, 0.2, 0.3], [0.7, 0.1, 0.2]])
    rumboost._current_j = 2
    rumboost.num_classes = 3

    #call the f_obj method
    grad, hess = rumboost.f_obj(None, train_set)

    #perform assertions
    assert np.array_equal(grad, np.array([0.3, -0.7, 0.2]))
    assert np.allclose(hess, np.array([0.315, 0.315, 0.24 ])) #allclose is used because of floating point precision

    rumboost._current_j = 1

    #call the f_obj method
    grad, hess = rumboost.f_obj(None, train_set)

    #perform assertions
    assert np.array_equal(grad, np.array([0.2, -0.8, 0.1]))
    assert np.allclose(hess, np.array([0.24 , 0.24 , 0.135])) #allclose is used because of floating point precision
    
def test_f_obj_nest():
    #create a RUMBoost object
    rumboost = rumb.RUMBoost()

    #assign some values to the object
    rumboost._current_j = 2
    rumboost.preds_i_m = np.array([[0.5, 1, 0.5], [0.3, 1, 0.7], [0.2, 1, 0.8]])
    rumboost.preds_m = np.array([[0.4, 0.6], [0.3, 0.7], [0.5, 0.5]])
    rumboost.nests = {0:0, 1:1, 2:0}
    rumboost.num_classes = 3
    rumboost.mu = np.array([1.5, 1])
    rumboost.labels = np.array([0, 0, 1])
    rumboost.labels_nest = np.array([1, 0, 1])
    
    #create a dummy dataset
    train_data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    train_labels = np.array([0, 1, 0])
    train_set = Dataset(train_data, label=train_labels)

    #call the f_obj_nest method
    grad, hess = rumboost.f_obj_nest(None, train_set)
    print(grad, hess)
    #perform assertions
    assert np.allclose(grad, np.array([0.45, 0.21, -0.7])) #allclose is used because of floating point precision
    assert np.allclose(hess, np.array([0.3975, 0.3549, 0.6]))  #allclose is used because of floating point precision