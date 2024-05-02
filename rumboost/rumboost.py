# coding: utf-8
"""Library with training routines of LightGBM."""
import collections
import copy
import json
import numpy as np

from scipy.special import softmax
from operator import attrgetter
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from lightgbm import callback
from lightgbm.basic import Booster, Dataset, LightGBMError, _ConfigAliases, _InnerPredictor, _choose_param_value, _log_warning
from lightgbm.compat import SKLEARN_INSTALLED, _LGBMGroupKFold, _LGBMStratifiedKFold

from rumboost.utils import bio_to_rumboost, cross_entropy, nest_probs, cross_nested_probs

_LGBM_CustomObjectiveFunction = Callable[
    [Union[List, np.ndarray], Dataset],
    Tuple[Union[List, np.ndarray], Union[List, np.ndarray]]
]
_LGBM_CustomMetricFunction = Callable[
    [Union[List, np.ndarray], Dataset],
    Tuple[str, float, bool]
]

class RUMBoost:
    """RUMBoost for doing Random Utility Modelling in LightGBM.
    
    Auxiliary data structure to implement boosters of ``rum_train()`` function for multiclass classification.
    This class has the same methods as Booster class.
    All method calls, except for the following methods, are actually performed for underlying Boosters.

    - ``model_from_string()``
    - ``model_to_string()``
    - ``save_model()``

    Attributes
    ----------
    boosters : list of Booster
        The list of fitted models.
    valid_sets : None
        Validation sets of the RUMBoost. By default None, to avoid computing cross entropy if there are no 
        validation sets.
    """
    def __init__(self, model_file = None):
        """Initialize the RUMBoost.

        Parameters
        ----------
        model_file : str, pathlib.Path or None, optional (default=None)
            Path to the RUMBoost model file.
        """
        self.boosters = []
        self.valid_sets = None
        self.num_classes = None #need to be specify by user

        #for nested and cross-nested rumboost
        self.mu = None
        self.nests = None
        self.alphas = None

        #for functional effect rumboost
        self.functional_effects = None

        if model_file is not None:
            with open(model_file, "r") as file:
                self._from_dict(json.load(file))
    
    def f_obj(
            self,
            _,
            train_set: Dataset
        ):
            """
            Objective function of the binary classification boosters, but based on softmax predictions.

            Parameters
            ----------
            train_set : Dataset
                Training set used to train the jth booster. It means that it is not the full training set but rather another dataset containing the relevant features for that utility. It is the jth dataset in the RUMBoost object.

            Returns
            -------
            grad : numpy array
                The gradient with the cross-entropy loss function. It is the predictions minus the binary labels (if it is used for the jth booster, labels will be 1 if the chosen class is j, 0 if it is any other classes).
            hess : numpy array
                The hessian with the cross-entropy loss function (second derivative approximation rather than the hessian). Calculated as factor * preds * (1 - preds).
            """
            j = self._current_j #jth booster
            if self.shared_ensembles and j >= self.shared_start_idx:
                preds = self._preds[:,self.shared_ensembles[j]].reshape(-1, order='A') #corresponding predictions
            else:
                preds = self._preds[:,j] #corresponding predictions
            factor = self.num_classes/(self.num_classes-1) #factor to correct redundancy (see Friedmann, Greedy Function Approximation)
            eps = 1e-6
            labels = self.labels_j[j]
            grad = preds - labels
            hess = np.maximum(factor * preds * (1 - preds), eps) #truncate low values to avoid numerical errors
            return grad, hess
    
    def f_obj_nest(
            self,
            _,
            train_set: Dataset
        ):
            """
            Objective function of the binary classification boosters, for a nested rumboost.

            Parameters
            ----------
            train_set : Dataset
                Training set used to train the jth booster. It means that it is not the full training set but rather another dataset containing the relevant features for that utility. It is the jth dataset in the RUMBoost object.

            Returns
            -------
            grad : numpy array
                The gradient with the cross-entropy loss function and nested probabilities.
            hess : numpy array
                The hessian with the cross-entropy loss function and nested probabilities (second derivative approximation rather than the hessian).
            """
            if self.shared_ensembles and self._current_j >= self.shared_start_idx:
                grad = np.array([])
                hess = np.array([])
                for j in self.shared_ensembles[self._current_j]:
                    labels = self.labels
                    labels_nest = self.labels_nest
                    pred_i_m = self.preds_i_m[:,j] #prediction of choice i knowing nest m
                    pred_m = self.preds_m[:, self.nests[j]] #prediction of choosing nest m
                    factor = self.num_classes/(self.num_classes-1) #factor to correct redundancy (see Friedmann, Greedy Function Approximation)

                    #three cases: 1. choice i = j, 2. j is in the same nest than choice i, 3. j is in another nest.
                    grad = np.concatenate([grad, (labels == j) * (-self.mu[self.nests[j]] * (1 - pred_i_m) - pred_i_m * (1 - pred_m)) + \
                                                 (labels_nest == self.nests[j]) * (1 - (labels == j)) * (self.mu[self.nests[j]] * pred_i_m - pred_i_m * (1 - pred_m)) + \
                                                 (1 - (labels_nest == self.nests[j])) * (pred_i_m * pred_m)])
                    hess = np.concatenate([hess, (labels == j) * (-self.mu[self.nests[j]] * pred_i_m * (1 - pred_i_m) * (1 - self.mu[self.nests[j]] - pred_m) + pred_i_m**2 * pred_m * (1 - pred_m)) + \
                                                 (labels_nest == self.nests[j]) * (1 - (labels == j)) * (-self.mu[self.nests[j]] * pred_i_m * (1 - pred_i_m) * (1 - self.mu[self.nests[j]] - pred_m) + pred_i_m**2 * pred_m * (1 - pred_m)) + \
                                                 (1 - (labels_nest == self.nests[j])) * (-pred_i_m * pred_m * (-self.mu[self.nests[j]] * (1 - pred_i_m) - pred_i_m * (1 - pred_m)))])
                    
                    hess = factor * hess

                return grad, hess
            else:
                j = self._current_j
                labels = self.labels
                labels_nest = self.labels_nest
                pred_i_m = self.preds_i_m[:,j] #prediction of choice i knowing nest m
                pred_m = self.preds_m[:, self.nests[j]] #prediction of choosing nest m
                factor = self.num_classes/(self.num_classes-1) #factor to correct redundancy (see Friedmann, Greedy Function Approximation)

                #three cases: 1. choice i = j, 2. j is in the same nest than choice i, 3. j is in another nest.
                grad = (labels == j) * (-self.mu[self.nests[j]] * (1 - pred_i_m) - pred_i_m * (1 - pred_m)) + \
                    (labels_nest == self.nests[j]) * (1 - (labels == j)) * (self.mu[self.nests[j]] * pred_i_m - pred_i_m * (1 - pred_m)) + \
                    (1 - (labels_nest == self.nests[j])) * (pred_i_m * pred_m)
                hess = (labels == j) * (-self.mu[self.nests[j]] * pred_i_m * (1 - pred_i_m) * (1 - self.mu[self.nests[j]] - pred_m) + pred_i_m**2 * pred_m * (1 - pred_m)) + \
                    (labels_nest == self.nests[j]) * (1 - (labels == j)) * (-self.mu[self.nests[j]] * pred_i_m * (1 - pred_i_m) * (1 - self.mu[self.nests[j]] - pred_m) + pred_i_m**2 * pred_m * (1 - pred_m)) + \
                    (1 - (labels_nest == self.nests[j])) * (-pred_i_m * pred_m * (-self.mu[self.nests[j]] * (1 - pred_i_m) - pred_i_m * (1 - pred_m)))
                
                hess = factor * hess

                return grad, hess
    
    def f_obj_cross_nested(self, _, train_set: Dataset):
        """
        Objective function of the binary classification boosters, for a cross-nested rumboost.

        Parameters
        ----------
        train_set : Dataset
            Training set used to train the jth booster. It means that it is not the full training set but rather another dataset containing the relevant features for that utility. It is the jth dataset in the RUMBoost object.

        Returns
        -------
        grad : numpy array
            The gradient with the cross-entropy loss function and cross-nested probabilities.
        hess : numpy array
            The hessian with the cross-entropy loss function and cross-nested probabilities (second derivative approximation rather than the hessian).
        """
        if self.shared_ensembles and self._current_j >= self.shared_start_idx:
            grad = np.array([]).reshape(0, 1)
            hess = np.array([]).reshape(0, 1)
            for j in self.shared_ensembles[self._current_j]:
                labels = self.labels
                mu = np.array(self.mu).reshape(1, len(self.mu))
                data_idx = np.arange(self.preds_i_m.shape[0])
                factor = self.num_classes / (self.num_classes - 1)  #factor to correct redundancy (see Friedmann, Greedy Function Approximation)

                pred_j_m = self.preds_i_m[:, j, :]  #pred of alternative j knowing nest m
                pred_i_m = self.preds_i_m[data_idx, labels, :]  #prediction of choice i knowing nest m
                pred_m = self.preds_m[:, j, :]  #prediction of choosing nest m
                pred_i = self._preds[data_idx, labels].reshape(-1,1)  #pred of choice i
                pred_j = self._preds[:, j].reshape(-1,1)  #pred of alt j

                d_pred_i_Vi = np.sum((pred_i_m * pred_m * (pred_i_m * (1 - mu) + mu - pred_i)), axis=1, keepdims=True)  #first derivative of pred i with respect to Vi
                d_pred_i_Vj = np.sum((pred_i_m * pred_m * (pred_j_m * (1 - mu) - pred_j)), axis=1, keepdims=True)  #first derivative of pred i with respect to Vj
                d_pred_j_Vj = np.sum((pred_j_m * pred_m * (pred_j_m * (1 - mu) + mu - pred_j)), axis=1, keepdims=True)  #first derivative of pred j with respect to Vj
                d2_pred_i_Vi = np.sum((pred_i_m * pred_m * (mu ** 2 * (2 * pred_i_m ** 2 - 3 * pred_i_m + 1) + mu * (-3 * pred_i_m ** 2 + 3 * pred_i_m + 2 * pred_i * (pred_i_m - 1)) + (pred_i_m ** 2 - 2 * pred_i_m * pred_i + pred_i ** 2 - d_pred_i_Vi))), axis=1, keepdims=True)
                d2_pred_i_Vj = np.sum((pred_i_m * pred_m * (mu ** 2 * (-pred_j_m) + mu * (-pred_j_m ** 2 + pred_j_m) + (pred_j_m - pred_j) ** 2 - d_pred_j_Vj)), axis=1, keepdims=True)

                #two cases: 1. alt j is choice i, 2. alt j is not choice i
                grad = np.concatenate([grad,np.where(labels.reshape(-1,1) == j, (-1 / pred_i) * d_pred_i_Vi, (-1 / pred_i) * d_pred_i_Vj)])
                hess = np.concatenate([hess,np.where(labels.reshape(-1,1) == j, (-1 / pred_i ** 2) * (d2_pred_i_Vi * pred_i - d_pred_i_Vi ** 2), (-1 / pred_i ** 2) * (d2_pred_i_Vj * pred_i - d_pred_i_Vj ** 2))])
                hess = factor * hess

            return grad.reshape(-1), hess.reshape(-1)
        else:
            j = self._current_j
            labels = self.labels
            mu = np.array(self.mu).reshape(1, len(self.mu))
            data_idx = np.arange(self.preds_i_m.shape[0])
            factor = self.num_classes / (self.num_classes - 1)  #factor to correct redundancy (see Friedmann, Greedy Function Approximation)

            pred_j_m = self.preds_i_m[:, j, :]  #pred of alternative j knowing nest m
            pred_i_m = self.preds_i_m[data_idx, labels, :]  #prediction of choice i knowing nest m
            pred_m = self.preds_m[:, j, :]  #prediction of choosing nest m
            pred_i = self._preds[data_idx, labels]  #pred of choice i
            pred_j = self._preds[:, j]  #pred of alt j

            d_pred_i_Vi = np.sum((pred_i_m * pred_m * (pred_i_m * (1 - mu) + mu - pred_i)), axis=1, keepdims=True)  #first derivative of pred i with respect to Vi
            d_pred_i_Vj = np.sum((pred_i_m * pred_m * (pred_j_m * (1 - mu) - pred_j)), axis=1, keepdims=True)  #first derivative of pred i with respect to Vj
            d_pred_j_Vj = np.sum((pred_j_m * pred_m * (pred_j_m * (1 - mu) + mu - pred_j)), axis=1, keepdims=True)  #first derivative of pred j with respect to Vj
            d2_pred_i_Vi = np.sum((pred_i_m * pred_m * (mu ** 2 * (2 * pred_i_m ** 2 - 3 * pred_i_m + 1) + mu * (-3 * pred_i_m ** 2 + 3 * pred_i_m + 2 * pred_i * (pred_i_m - 1)) + (pred_i_m ** 2 - 2 * pred_i_m * pred_i + pred_i ** 2 - d_pred_i_Vi))), axis=1, keepdims=True)
            d2_pred_i_Vj = np.sum((pred_i_m * pred_m * (mu ** 2 * (-pred_j_m) + mu * (-pred_j_m ** 2 + pred_j_m) + (pred_j_m - pred_j) ** 2 - d_pred_j_Vj)), axis=1, keepdims=True)

            #two cases: 1. alt j is choice i, 2. alt j is not choice i
            grad = np.where(labels == j, (-1 / pred_i) * d_pred_i_Vi, (-1 / pred_i) * d_pred_i_Vj)
            hess = np.where(labels == j, (-1 / pred_i ** 2) * (d2_pred_i_Vi * pred_i - d_pred_i_Vi ** 2), (-1 / pred_i ** 2) * (d2_pred_i_Vj * pred_i - d_pred_i_Vj ** 2))
            hess = factor * hess

            return grad.reshape(-1), hess.reshape(-1)
            
    def predict(
        self,
        data,
        start_iteration: int = 0,
        num_iteration: int = -1,
        raw_score: bool = True,
        pred_leaf: bool = False,
        pred_contrib: bool = False,
        data_has_header: bool = False,
        validate_features: bool = False,
        utilities: bool = False,
        nests: dict = None,
        mu: list[float] = None,
        alphas: np.array = None
    ):
        """Predict logic.

        Parameters
        ----------
        data : str, pathlib.Path, numpy array, pandas DataFrame, H2O DataTable's Frame or scipy.sparse
            Data source for prediction.
            If str or pathlib.Path, it represents the path to a text file (CSV, TSV, or LibSVM).
        start_iteration : int, optional (default=0)
            Start index of the iteration to predict.
        num_iteration : int, optional (default=-1)
            Iteration used for prediction.
        raw_score : bool, optional (default=False)
            Whether to predict raw scores.
        pred_leaf : bool, optional (default=False)
            Whether to predict leaf index.
        pred_contrib : bool, optional (default=False)
            Whether to predict feature contributions.
        data_has_header : bool, optional (default=False)
            Whether data has header.
            Used only for txt data.
        validate_features : bool, optional (default=False)
            If True, ensure that the features used to predict match the ones used to train.
            Used only if data is pandas DataFrame.
        utilities : bool, optional (default=True)
            If True, return raw utilities for each class, without generating probabilities.
        nests : dict, optional (default=None)
            If not none, compute predictions with the nested probability function. The dictionary keys are alternatives number and their values are
            their nest number. By example {0:0, 1:1, 2:0} means that alt 0 and 2 are in nest 0 and alt 1 is in nest 1.
        mu : list, optional (default=None)
            Only used, and required, if nests is True. It is the list of mu values for each nest.
            The first value correspond to the first nest and so on.
        alphas : ndarray, optional (default=None)
            An array of J (alternatives) by M (nests).
            alpha_jn represents the degree of membership of alternative j to nest n
            By example, alpha_12 = 0.5 means that alternative one belongs 50% to nest 2.

        Returns
        -------
        result : numpy array, scipy.sparse or list of scipy.sparse
            Prediction result.
            Can be sparse or a list of sparse objects (each element represents predictions for one class) for feature contributions (when ``pred_contrib=True``).
        """    
        #compute utilities with corresponding features
        #split data
        new_data, _ = self._preprocess_data(data, return_data=True)
        #compute U
        raw_preds = [booster.predict(new_data[k].get_data(), 
                                start_iteration, 
                                num_iteration, 
                                raw_score, 
                                pred_leaf, 
                                pred_contrib,
                                data_has_header,
                                validate_features) for k, booster in enumerate(self.boosters)]

        #if shared ensembles, get the shared predictions out and reorder them for easier addition later
        if self.shared_ensembles:
            raw_shared_preds = np.concatenate([arr.reshape((data.num_data(), -1)) for arr in raw_preds[self.shared_start_idx:]], axis=1)
            if self.shared_start_idx == 0:
                raw_preds = [np.zeros(data.num_data())]*self.num_classes
            else:
                raw_preds = raw_preds[:self.shared_start_idx]

        raw_preds = np.array(raw_preds).T

        #if functional effect, sum the two ensembles (of attributes and socio-economic characteristics) of each alternative
        if self.functional_effects:
            raw_preds = raw_preds.reshape((-1, self.num_classes, 2)).sum(axis=2)

        #if shared ensembles, add the shared ensembles to the individual specific ensembles
        if self.shared_ensembles:
            for shared_ensembles in self.shared_ensembles.values():
                raw_preds[:, shared_ensembles] += raw_shared_preds[:,:len(shared_ensembles)]
                raw_shared_preds = raw_shared_preds[:,len(shared_ensembles):]

        #compute nested probabilities. pred_i_m is predictions of choosing i knowing m, pred_m is prediction of choosing nest m and preds is pred_i_m * pred_m
        if nests:
            preds, pred_i_m, pred_m = nest_probs(raw_preds, mu=mu, nests=nests)
            return preds, pred_i_m, pred_m
        
        #compute cross-nested probabilities. pred_i_m is predictions of choosing i knowing m, pred_m is prediction of choosing nest m and preds is pred_i_m * pred_m
        if alphas is not None:
            preds, pred_i_m, pred_m = cross_nested_probs(raw_preds, mu=mu, alphas=alphas)
            return preds, pred_i_m, pred_m

        #softmax
        if not utilities:
            preds = softmax(raw_preds, axis=1)
            return preds
   
        return raw_preds
    
    def _inner_predict(
        self,
        data_idx: int = 0,
        utilities: bool = False,
        nests: bool = False,
        mu = None,
        alphas: np.array = None
    ):
        """
        Predict logic for training RUMBoost object. This _inner_predict function is much faster than the predict function.
        But the function takes advantage of the inner_prediction function of lightGBM boosters, and shouldn't be used
        when predicting outside of training, as datasets might not be stored inside boosters.

        Parameters
        ----------
        data_idx: int (default=0)
            The index of the dataset. 0 means training set, and following numbers are validation sets, in the specified order.
        utilities : bool, optional (default=True)
            If True, return raw utilities for each class, without generating probabilities.
        nests : bool, optional (default=False)
            If True, compute predictions with the nested probability function.
        mu : list, optional (default=None)
            Only used, and required, if nests is True. It is the list of mu values for each nest.
            The first value correspond to the first nest and so on.
        alphas : ndarray, optional (default=None)
            An array of J (alternatives) by M (nests).
            alpha_jn represents the degree of membership of alternative j to nest n
            By example, alpha_12 = 0.5 means that alternative one belongs 50% to nest 2.

        Returns
        -------
        result : numpy array, scipy.sparse or list of scipy.sparse
            Prediction result.
            Can be sparse or a list of sparse objects (each element represents predictions for one class) for feature contributions (when ``pred_contrib=True``).
        """
        #getting raw prediction from lightGBM booster's inner predict
        raw_preds = [booster._Booster__inner_predict(data_idx) for _, booster in enumerate(self.boosters)]

        #if shared ensembles, get the shared predictions out and reorder them for easier addition later
        if self.shared_ensembles:
            raw_shared_preds = np.concatenate([arr.reshape((-1, self.num_obs[data_idx])) for arr in raw_preds[self.shared_start_idx:]]).T
            if self.shared_start_idx == 0:
                raw_preds = [np.zeros(self.num_obs[data_idx])]*self.num_classes
            else:
                raw_preds = raw_preds[:self.shared_start_idx]

        raw_preds = np.array(raw_preds).T

        #if functional effect, sum the two ensembles (of attributes and socio-economic characteristics) of each alternative
        if self.functional_effects:
            raw_preds = raw_preds.reshape((-1, self.num_classes, 2)).sum(axis=2)

        #if shared ensembles, add the shared ensembles to the individual specific ensembles
        if self.shared_ensembles:
            for shared_ensembles in self.shared_ensembles.values():
                raw_preds[:, shared_ensembles] += raw_shared_preds[:,:len(shared_ensembles)]
                raw_shared_preds = raw_shared_preds[:,len(shared_ensembles):]
            
        #compute nested probabilities. pred_i_m is predictions of choosing i knowing m, pred_m is prediction of choosing nest m and preds is pred_i_m * pred_m
        if nests:
            if self.mu:
                mu = self.mu
            nest = self.nests
            preds, pred_i_m, pred_m = nest_probs(raw_preds, mu=mu, nests=nest)
            return preds, pred_i_m, pred_m
        
        #compute cross-nested probabilities. pred_i_m is predictions of choosing i knowing m, pred_m is prediction of choosing nest m and preds is pred_i_m * pred_m
        if alphas is not None:
            if self.mu:
                mu = self.mu
            preds, pred_i_m, pred_m = cross_nested_probs(raw_preds, mu=mu, alphas=alphas)
            return preds, pred_i_m, pred_m

        #softmax
        if not utilities:
            preds = softmax(raw_preds, axis=1)
            return preds

        return raw_preds
    
    def _preprocess_data(self, data: Dataset, reduced_valid_set = None, return_data: bool = False):
        """Set up J training (and, if specified, validation) datasets.
        
        Parameters
        ----------
        data : Dataset
            The full training dataset (i.e. the union of the socio-economic features with the alternative-specific features).
            Note, the argument free_raw_data shall be set to False when creating the dataset.
        reduced_valid_set : Dataset or list of Dataset, optional (default = None)
            The full dataset used for validation. There can be several datasets.
        return_data : bool, optional (default = False)
            If True, returns the J preprocessed datasets (and potential validation sets)

        Returns
        -------
        train_set_J : list[Dataset]
            If return_data is True, return a list with J preprocessed datasets corresponding to the J boosters.
        reduced_valid_sets_J : list[Dataset] or list[list[Dataset]], optional
            If return_data is True, and reduced_valid_set is not None, return one or several list(s) with J preprocessed validation sets corresponding to the J boosters.
        """
        train_set_J = []
        reduced_valid_sets_J = []

        #to access data
        data.construct()
        self.num_obs = [data.num_data()] #saving number of observations
        if reduced_valid_set:
            for valid_set in reduced_valid_set:
                valid_set.construct()
                self.num_obs.append(valid_set.num_data())

        self.labels = data.get_label().astype('int16') #saving labels
        self.valid_labels = []
        self.labels_j = []

        if self.shared_ensembles:
            shared_labels = {}
            shared_valids = {}

        #loop over all J utilities
        for j, struct in enumerate(self.rum_structure):
            if struct:
                if 'columns' in struct:
                    train_set_j_data = data.get_data()[struct['columns']] #only relevant features for the jth booster

                    #transforming labels for functional effects
                    if self.functional_effects and j < 2*self.num_classes:
                        l = int(j/2)
                    else:
                        l = j

                    if self.shared_ensembles:
                        if l >= self.shared_start_idx:
                            if not shared_labels:
                                shared_labels = {a: np.where(data.get_label() == a, 1, 0) for a in range(self.num_classes)}
                            new_label = np.hstack([shared_labels[s] for s in self.shared_ensembles[l]])
                            self.labels_j.append(new_label.astype('int8'))
                            train_set_j = Dataset(train_set_j_data.values.reshape((-1, 1), order='A'), label=new_label, free_raw_data=False, params={'verbosity':-1}) #create and build dataset
                            train_set_j.construct()
                        else:
                            new_label = np.where(data.get_label() == l, 1, 0) #new binary label, used for multiclassification
                            shared_labels[l] = new_label
                            self.labels_j.append(new_label.astype('int8'))
                            train_set_j = Dataset(train_set_j_data, label=new_label, free_raw_data=False, params={'verbosity':-1}) #create and build dataset
                            train_set_j.construct()
                    else:
                        new_label = np.where(data.get_label() == l, 1, 0) #new binary label, used for multiclassification
                        self.labels_j.append(new_label.astype('int8'))
                        train_set_j = Dataset(train_set_j_data, label=new_label, free_raw_data=False, params={'verbosity':-1}) #create and build dataset
                        train_set_j.construct()
                     
                    if reduced_valid_set is not None:
                        reduced_valid_sets_j = []
                        for valid_set in reduced_valid_set:
                            #create and build validation sets
                            valid_set.construct()
                            valid_set_j_data = valid_set.get_data()[struct['columns']] #only relevant features for the jth booster
                            self.valid_labels.append(valid_set.get_label().astype('int16')) #saving labels
                            
                            if self.shared_ensembles:
                                if l >= self.shared_start_idx:
                                    if not shared_valids:
                                        shared_valids = {a: np.where(valid_set.get_label() == a, 1, 0) for a in range(self.num_classes)}
                                    label_valid = np.hstack([shared_valids[s] for s in self.shared_ensembles[l]])
                                    valid_set_j = Dataset(valid_set_j_data.values.reshape((-1, 1), order='A'), label=label_valid, free_raw_data=False, reference=train_set_j, params={'verbosity':-1}) #create and build dataset
                                    valid_set_j.construct()
                                else:
                                    label_valid = np.where(valid_set.get_label() == l, 1, 0) #new binary label, used for multiclassification
                                    shared_valids[l] = label_valid
                                    valid_set_j = Dataset(valid_set_j_data, label=label_valid, free_raw_data=False, reference=train_set_j, params={'verbosity':-1}) #create and build dataset
                                    valid_set_j.construct()
                            else:
                                label_valid = np.where(valid_set.get_label() == l, 1, 0) #new binary label, used for multiclassification
                                valid_set_j = Dataset(valid_set_j_data, label=label_valid, reference= train_set_j, free_raw_data=False)
                                valid_set_j.construct()
                            
                            reduced_valid_sets_j.append(valid_set_j)

                else:
                    #if no alternative specific datasets
                    new_label = np.where(data.get_label() == j, 1, 0)
                    train_set_j = Dataset(data.get_data(), label=new_label, free_raw_data=False)
                    if reduced_valid_set is not None:
                        reduced_valid_sets_j = reduced_valid_set[:]

            #store all training and potential validation sets in lists
            train_set_J.append(train_set_j)
            if reduced_valid_set is not None:
                reduced_valid_sets_J.append(reduced_valid_sets_j)

        #store them in the RUMBoost object
        self.train_set = train_set_J
        self.valid_sets = np.array(reduced_valid_sets_J).T.tolist()
        if return_data:
            return train_set_J, reduced_valid_sets_J
    
    def _preprocess_params(self, params: dict, return_params: bool = False, params_fe: dict = None):
        """Set up J set of parameters.
        
        Parameters
        ----------
        params : dict
            Dictionary containing parameters. The syntax must follow the one from LightGBM.
        return_params : bool, optional (default = False)
            If True, returns the J sets of parameters (and potential validation sets)
        params_fe: dict, optional (default=None)
            Second set of parameters, for the functional effect model. These parameters are applied to the socio-economic characteristics ensembles

        Returns
        -------
        params_J : list[dict]
            A list of dictionary containing J (or 2*J if functional effect model) sets of parameters.
        """

        #create the J parameters dictionaries
        if params_fe is not None: #for functional effect, two sets of parameters
            params_J = [{**copy.deepcopy(params),
                        'verbosity': -1,
                        'objective': 'binary',
                        'num_classes': 1,
                        'monotone_constraints': struct.get('monotone_constraints', []) if struct else [],
                        'interaction_constraints': struct.get('interaction_constraints', []) if struct else [],
                        'categorical_feature': struct.get('categorical_feature', []) if struct else []
                        } if i%2 == 0 else 
                        {**copy.deepcopy(params_fe),
                        'verbosity': -1,
                        'objective': 'binary',
                        'num_classes': 1,
                        'monotone_constraints': struct.get('monotone_constraints', []) if struct else [],
                        'interaction_constraints': struct.get('interaction_constraints', []) if struct else [],
                        'categorical_feature': struct.get('categorical_feature', []) if struct else []
                        } for i, struct in enumerate(self.rum_structure)]
        else:
            params_J = [{**copy.deepcopy(params),
                        'verbosity': -1,
                        'objective': 'binary',
                        'num_classes': 1,
                        'monotone_constraints': struct.get('monotone_constraints', []) if struct else [],
                        'interaction_constraints': struct.get('interaction_constraints', []) if struct else [],
                        'categorical_feature': struct.get('categorical_feature', []) if struct else []
                        } for struct in self.rum_structure]

        #store the set of parameters in RUMBoost
        self.params = params_J
        if return_params:
            return params_J
        
    def _preprocess_valids(self, train_set: Dataset, params: dict, valid_sets = None, valid_names = None):
        """Prepare validation sets.
        
        Parameters
        ----------
        train_set : Dataset
            The full training dataset (i.e. the union of the socio-economic features with the alternative-specific features).
        params : dict
            Dictionary containing parameters. The syntax must follow the one from LightGBM.
        valid_sets : Dataset or list[Dataset], optional (default = None)
            The full dataset used for validation. There can be several datasets.
        valid_names : str or list[str], optional (default = None)
            The names of the validation sets.

        Returns
        -------
        reduced_valid_sets : list[Dataset]
            List of prepared validation sets.
        name_valid_sets : list[str]
            List of names of validation sets.
        is_valid_contain_train: bool
            True if the training set is in the validation sets.
        train_data_name: str
            Name of training dataset : 'training'.
        """
        #initialise variables
        is_valid_contain_train = False
        train_data_name = "training"
        reduced_valid_sets = []
        name_valid_sets = []

        #finalise validation sets for training
        if valid_sets is not None:
            if isinstance(valid_sets, Dataset):
                valid_sets = [valid_sets]
            if isinstance(valid_names, str):
                valid_names = [valid_names]
            for i, valid_data in enumerate(valid_sets):
                if valid_data is train_set:
                    is_valid_contain_train = True #store if train set is in validation set
                    if valid_names is not None:
                        train_data_name = valid_names[i]
                    continue
                if not isinstance(valid_data, Dataset):
                    raise TypeError("Training only accepts Dataset object")
                reduced_valid_sets.append(valid_data._update_params(params).set_reference(train_set))
                if valid_names is not None and len(valid_names) > i:
                    name_valid_sets.append(valid_names[i])
                else:
                    name_valid_sets.append(f'valid_{i}')

        return reduced_valid_sets, name_valid_sets, is_valid_contain_train, train_data_name
        
    
    def _construct_boosters(self, train_data_name = "Training", is_valid_contain_train = False,
                            name_valid_sets = ["Valid_0"]):
        """Construct boosters of the RUMBoost model with corresponding set of parameters, training datasets, and validation sets and store them in the RUMBoost object.
        
        Parameters
        ----------
        train_data_name: str, optional (default = 'Training')
            Name of training dataset.
        is_valid_contain_train: bool
            True if the training set is in the validation sets.
        name_valid_sets : list[str]
            List of names of validation sets.
        """
        #getting parameters, training, and validation sets
        params_J = self.params
        train_set_J = self.train_set
        reduced_valid_sets_J = self.valid_sets

        for j, (param_j, train_set_j) in enumerate(zip(params_J, train_set_J)):
            train_set_j._update_params(param_j) #update parameters of the jth training set
            #construct booster and perform basic preparations
            try:
                booster = Booster(params=param_j, train_set=train_set_j)
                if is_valid_contain_train:
                    booster.set_train_data_name(train_data_name)
                for valid_set, name_valid_set in zip(reduced_valid_sets_J, name_valid_sets):
                    valid_set[j]._update_params(param_j).set_reference(train_set_j)
                    booster.add_valid(valid_set[j], name_valid_set)
            finally:
                train_set_j._reverse_update_params()
                for valid_set in reduced_valid_sets_J:
                    valid_set[j]._reverse_update_params()
            
            #initialise and store boosters in a list
            booster.best_iteration = 0
            self._append(booster)

        #initialise RUMBoost score information
        self.best_iteration = 0
        self.best_score = 1e6
        self.best_score_train = 1e6

    def _append(self, booster: Booster) -> None:
        """Add a booster to RUMBoost."""
        self.boosters.append(booster)

    def _from_dict(self, models: Dict[str, Any]) -> None:
        """Load RUMBoost from dict."""
        self.best_iteration = models["best_iteration"]
        self.best_score = models["best_score"]
        self.boosters = []
        for model_str in models["boosters"]:
            self._append(Booster(model_str=model_str))

    def _to_dict(self, num_iteration: Optional[int], start_iteration: int, importance_type: str) -> Dict[str, Any]:
        """Serialize RUMBoost to dict."""
        models_str = []
        for booster in self.boosters:
            models_str.append(booster.model_to_string(num_iteration=num_iteration, start_iteration=start_iteration,
                                                      importance_type=importance_type))
        return {"boosters": models_str, "best_iteration": self.best_iteration, "best_score": self.best_score}

    def __getattr__(self, name: str) -> Callable[[Any, Any], List[Any]]:
        """Redirect methods call of RUMBoost."""
        def handler_function(*args: Any, **kwargs: Any) -> List[Any]:
            """Call methods with each booster, and concatenate their results."""
            ret = []
            for booster in self.boosters:
                ret.append(getattr(booster, name)(*args, **kwargs))
            return ret
        return handler_function

    def __getstate__(self) -> Dict[str, Any]:
        return vars(self)

    def __setstate__(self, state: Dict[str, Any]) -> None:
        vars(self).update(state)

    def model_from_string(self, model_str: str):
        """Load RUMBoost from a string.

        Parameters
        ----------
        model_str : str
            Model will be loaded from this string.

        Returns
        -------
        self : RUMBoost
            Loaded RUMBoost object.
        """
        self._from_dict(json.loads(model_str))
        return self

    def model_to_string(
        self,
        num_iteration: Optional[int] = None,
        start_iteration: int = 0,
        importance_type: str = 'split'
    ) -> str:
        """Save RUMBoost to JSON string.

        Parameters
        ----------
        num_iteration : int or None, optional (default=None)
            Index of the iteration that should be saved.
            If None, if the best iteration exists, it is saved; otherwise, all iterations are saved.
            If <= 0, all iterations are saved.
        start_iteration : int, optional (default=0)
            Start index of the iteration that should be saved.
        importance_type : str, optional (default="split")
            What type of feature importance should be saved.
            If "split", result contains numbers of times the feature is used in a model.
            If "gain", result contains total gains of splits which use the feature.

        Returns
        -------
        str_repr : str
            JSON string representation of RUMBoost.
        """
        return json.dumps(self._to_dict(num_iteration, start_iteration, importance_type))

    def save_model(
        self,
        filename: Union[str, Path],
        num_iteration: Optional[int] = None,
        start_iteration: int = 0,
        importance_type: str = 'split'
    ) -> "RUMBoost":
        """Save RUMBoost to a file as JSON text.

        Parameters
        ----------
        filename : str or pathlib.Path
            Filename to save RUMBoost.
        num_iteration : int or None, optional (default=None)
            Index of the iteration that should be saved.
            If None, if the best iteration exists, it is saved; otherwise, all iterations are saved.
            If <= 0, all iterations are saved.
        start_iteration : int, optional (default=0)
            Start index of the iteration that should be saved.
        importance_type : str, optional (default="split")
            What type of feature importance should be saved.
            If "split", result contains numbers of times the feature is used in a model.
            If "gain", result contains total gains of splits which use the feature.

        Returns
        -------
        self : RUMBoost
            Returns self.
        """
        with open(filename, "w") as file:
            json.dump(self._to_dict(num_iteration, start_iteration, importance_type), file)

        return self

def rum_train(
    params: dict[str, Any],
    train_set: Dataset | dict[str, Any],
    rum_structure: list[dict[str, Any]] = None,
    num_boost_round: int = 100,
    valid_sets: Optional[list[Dataset]] | Optional[dict] = None,
    valid_names: Optional[list[str]] = None,
    feval: Optional[Union[_LGBM_CustomMetricFunction, list[_LGBM_CustomMetricFunction]]] = None,
    init_model: Optional[Union[str, Path, Booster]] = None,
    feature_name: Union[list[str], str] = 'auto',
    categorical_feature: Union[list[str], list[int], str] = 'auto',
    keep_training_booster: bool = False,
    callbacks: Optional[list[Callable]] = None,
    nests: dict = None,
    mu: list = None,
    params_fe: dict = None,
    alphas: np.array = None,
    shared_ensembles: dict = None
) -> RUMBoost:
    """Perform the RUM training with given parameters.

    Parameters
    ----------
    params : dict
        Parameters for training. Values passed through ``params`` take precedence over those
        supplied via arguments. If num_classes > 2, please specify params['objective'] = 'multiclass'.
    train_set : Dataset or dict[int, Any]
        Data to be trained on. Set free_raw_data=False when creating the dataset. If it is 
        a dictionary, the key-value pairs should be:
            - "train_set":  the corresponding preprocessed Dataset. 
            - "num_data": the number of observations in the dataset.
            - "labels": the labels of the full dataset.
            - "labels_j": the labels of the dataset for each class (binary).
    rum_structure : list[dict[str, Any]], optional (default = None)
        List of dictionaries specifying the RUM structure. 
        The list must contain one dictionary for each class, which describes the 
        utility structure for that class. 
        Each dictionary has three allowed keys. 
        'cols': list of columns included in that class
        'monotone_constraints': list of monotonic constraints on parameters
        'interaction_constraints': list of interaction constraints on features
        if None, a biogeme_model must be specified
    biogeme_model : BIOGEME, optional (default = None)
        A BIOGEME object representing a biogeme model, used to create the rum_structure.
        A biogeme model is required if rum_structure is None, otherwise should be None.
    num_boost_round : int, optional (default = 100)
        Number of boosting iterations.
    valid_sets : list of Dataset, dict, or None, optional (default = None)
        List of data to be evaluated on during training. If the train_set is passed as
        already preprocessed, it is assumed that valid_sets are also preprocessed. Therefore it
        should be a dictionary following this structure:
            - "valid_sets":  a list of list of corresponding preprocessed validation Datasets. 
            - "valid_labels": a list of the valid dataset labels.
            - "num_data": a list of the number of data in validation datasets. 
    valid_names : list of str, or None, optional (default = None)
        Names of ``valid_sets``.
    feval : callable, list of callable, or None, optional (default = None)
        Customized evaluation function.
        Each evaluation function should accept two parameters: preds, eval_data,
        and return (eval_name, eval_result, is_higher_better) or list of such tuples.

            preds : numpy 1-D array or numpy 2-D array (for multi-class task)
                The predicted values.
                For multi-class task, preds are numpy 2-D array of shape = [n_samples, n_classes].
                If custom objective function is used, predicted values are returned before any transformation,
                e.g. they are raw margin instead of probability of positive class for binary task in this case.
            eval_data : Dataset
                A ``Dataset`` to evaluate.
            eval_name : str
                The name of evaluation function (without whitespaces).
            eval_result : float
                The eval result.
            is_higher_better : bool
                Is eval result higher better, e.g. AUC is ``is_higher_better``.

        To ignore the default metric corresponding to the used objective,
        set the ``metric`` parameter to the string ``"None"`` in ``params``.
    init_model : str, pathlib.Path, Booster or None, optional (default = None)
        Filename of LightGBM model or Booster instance used for continue training.
    feature_name : list of str, or 'auto', optional (default = "auto")
        Feature names.
        If 'auto' and data is pandas DataFrame, data columns names are used.
    categorical_feature : list of str or int, or 'auto', optional (default = "auto")
        Categorical features.
        If list of int, interpreted as indices.
        If list of str, interpreted as feature names (need to specify ``feature_name`` as well).
        If 'auto' and data is pandas DataFrame, pandas unordered categorical columns are used.
        All values in categorical features will be cast to int32 and thus should be less than int32 max value (2147483647).
        Large values could be memory consuming. Consider using consecutive integers starting from zero.
        All negative values in categorical features will be treated as missing values.
        The output cannot be monotonically constrained with respect to a categorical feature.
        Floating point numbers in categorical features will be rounded towards 0.
    keep_training_booster : bool, optional (default = False)
        Whether the returned Booster will be used to keep training.
        If False, the returned value will be converted into _InnerPredictor before returning.
        This means you won't be able to use ``eval``, ``eval_train`` or ``eval_valid`` methods of the returned Booster.
        When your model is very large and cause the memory error,
        you can try to set this param to ``True`` to avoid the model conversion performed during the internal call of ``model_to_string``.
        You can still use _InnerPredictor as ``init_model`` for future continue training.
    callbacks : list of callable, or None, optional (default = None)
        List of callback functions that are applied at each iteration.
        See Callbacks in Python API for more information.
    mu : list, optional (default=None)
        List of mu values, the scaling parameter, for each nest. The first value of the list correspond to nest 0, and so on.
    nest : dict, optional (default=None)
        Dictionary representing the nesting structure. Keys are alternatives, and values are the nest they belong to. By example,
        {0:0, 1:1, 2:0} means alt 0 and 2 belong to nest 0 and alt 1 belongs to nest 1. 
    params_fe : dict, optional (default=None)
        Parameters for training the socio-economic part of a functional effect model.
    alphas : ndarray, optional (default=None)
        An array of J (alternatives) by M (nests).
        alpha_jn represents the degree of membership of alternative j to nest n
        By example, alpha_12 = 0.5 means that alternative one belongs 50% to nest 2.
    shared_ensembles : dict, optional (default=None)
        Dictionary of shared ensembles. Keys are the index of position in the rum_structure list of the shared ensembles, 
        and values are the list of alternatives that share the parameter.
        

    Note
    ----
    A custom objective function can be provided for the ``objective`` parameter.
    It should accept two parameters: preds, train_data and return (grad, hess).

        preds : numpy 1-D array or numpy 2-D array (for multi-class task)
            The predicted values.
            Predicted values are returned before any transformation,
            e.g. they are raw margin instead of probability of positive class for binary task.
        train_data : Dataset
            The training dataset.
        grad : numpy 1-D array or numpy 2-D array (for multi-class task)
            The value of the first order derivative (gradient) of the loss
            with respect to the elements of preds for each sample point.
        hess : numpy 1-D array or numpy 2-D array (for multi-class task)
            The value of the second order derivative (Hessian) of the loss
            with respect to the elements of preds for each sample point.

    For multi-class task, preds are numpy 2-D array of shape = [n_samples, n_classes],
    and grad and hess should be returned in the same format.

    Returns
    -------
    rum_booster : RUMBoost
        The trained RUMBoost model.
    """
    for alias in _ConfigAliases.get("verbosity"): 
        if alias in params:
            verbosity = params[alias]
    #create predictor first
    params = copy.deepcopy(params)
    params = _choose_param_value(
        main_param_name='objective',
        params=params,
        default_value=None
    )
    fobj: Optional[_LGBM_CustomObjectiveFunction] = None
    if callable(params["objective"]):
        fobj = params["objective"]
        params["objective"] = 'none'
    for alias in _ConfigAliases.get("num_iterations"):
        if alias in params:
            num_boost_round = params.pop(alias)
            _log_warning(f"Found `{alias}` in params. Will use it instead of argument")
    params["num_iterations"] = num_boost_round
    #setting early stopping via global params should be possible
    params = _choose_param_value(
        main_param_name="early_stopping_round",
        params=params,
        default_value=None
    )
    if params["early_stopping_round"] is None:
        params["early_stopping_round"] = 10000
    first_metric_only = params.get('first_metric_only', False)

    if num_boost_round <= 0:
        raise ValueError("num_boost_round should be greater than zero.")
    predictor: Optional[_InnerPredictor] = None
    if isinstance(init_model, (str, Path)):
        predictor = _InnerPredictor(model_file=init_model, pred_parameter=params)
    elif isinstance(init_model, Booster):
        predictor = init_model._to_predictor(dict(init_model.params, **params))
    init_iteration = predictor.num_total_iteration if predictor is not None else 0

    #process callbacks
    if callbacks is None:
        callbacks_set = set()
    else:
        for i, cb in enumerate(callbacks):
            cb.__dict__.setdefault('order', i - len(callbacks))
        callbacks_set = set(callbacks)

    # if "early_stopping_round" in params:
    #    callbacks_set.add(
    #        callback.early_stopping(
    #            stopping_rounds=params["early_stopping_round"],
    #            first_metric_only=first_metric_only,
    #            verbose=_choose_param_value(
    #                main_param_name="verbosity",
    #                params=params,
    #                default_value=1
    #            ).pop("verbosity") > 0
    #        )
    #    )

    callbacks_before_iter_set = {cb for cb in callbacks_set if getattr(cb, 'before_iteration', False)}
    callbacks_after_iter_set = callbacks_set - callbacks_before_iter_set
    callbacks_before_iter = sorted(callbacks_before_iter_set, key=attrgetter('order'))
    callbacks_after_iter = sorted(callbacks_after_iter_set, key=attrgetter('order'))

    #construct rumboost object
    rumb = RUMBoost()

    #check number of classes
    if 'num_classes' not in params:
        raise ValueError('Specify the number of classes in the dictionary of parameters with the key num_classes')
    rumb.num_classes = params.get('num_classes') #saving number of classes

    #initialise shared ensembles if they are specified
    if shared_ensembles is not None:
        rumb.shared_ensembles = shared_ensembles
        rumb.shared_start_idx = [*shared_ensembles][0]
    else:
        rumb.shared_ensembles = None
        rumb.shared_start_idx = 0

    #initialise RUMBoost for functional effects if param_fe is passed
    if params_fe is not None:
        if (len(rum_structure) - rumb.shared_start_idx) == 2 * params['num_classes']:
            rumb.functional_effects = True
        else:
            raise ValueError('Functional effects model requires a rum_structure of length 2 * num_classes (without \
                             shared ensembles) or 2 * num_classes + number of shared ensembles (with shared ensembles)')    
    else:
        rumb.functional_effects = False

    #store usefull information in RUMBoost object
    rumb.rum_structure = rum_structure #saving utility structure

    #check dataset and preprocess it
    if isinstance(train_set, dict):
        if 'num_data' not in train_set:
            raise ValueError('The dictionary must contain the number of observations with the key num_data')
        if 'labels' not in train_set:
            raise ValueError('The dictionary must contain the labels with the key labels')
        rumb.train_set = train_set['train_sets'] #assign the J previously preprocessed datasets
        rumb.labels = train_set['labels']
        rumb.labels_j = train_set['labels_j']
        rumb.num_obs = [train_set['num_data']]
        if isinstance(valid_sets, dict):
            rumb.valid_sets = valid_sets['valid_sets'] #assign the J previously preprocessed validation sets
            rumb.valid_labels = valid_sets['valid_labels']
            rumb.num_obs.extend(valid_sets['num_data'])
        is_valid_contain_train = False
        train_data_name = "training"
        name_valid_sets = "valid_0"
    elif not isinstance(train_set, Dataset):
        raise TypeError("Training only accepts Dataset object")
    else:
        train_set._update_params(params) \
                ._set_predictor(predictor) \
                .set_feature_name(feature_name) \
                .set_categorical_feature(categorical_feature)
        reduced_valid_sets, \
        name_valid_sets, \
        is_valid_contain_train, \
        train_data_name = rumb._preprocess_valids(train_set, params, valid_sets) #prepare validation sets
        rumb._preprocess_data(train_set, reduced_valid_sets) #prepare J datasets with relevant features

    #preprocess parameters
    rumb._preprocess_params(params, params_fe = params_fe) #prepare J set of parameters
    
    #create J boosters with corresponding params and datasets
    rumb._construct_boosters(train_data_name, is_valid_contain_train, name_valid_sets) #build boosters with corresponding params and dataset

    #initialise nested probabilities if they are specified
    if nests is not None:
        rumb.mu = mu
        rumb.nests = nests
        f_obj = rumb.f_obj_nest
        rumb.labels_nest = np.array([nests[l] for l in rumb.labels])
        rumb._preds, rumb.preds_i_m, rumb.preds_m = rumb._inner_predict(nests=True)
    elif alphas is not None:
        rumb.mu = mu
        f_obj = rumb.f_obj_cross_nested
        rumb._preds, rumb.preds_i_m, rumb.preds_m = rumb._inner_predict(alphas=alphas)
    else:
        f_obj = rumb.f_obj
        rumb._preds = rumb._inner_predict()

    #start training
    for i in range(init_iteration, init_iteration + num_boost_round):
        #update all binary boosters of the rumb
        for j, booster in enumerate(rumb.boosters):
            for cb in callbacks_before_iter:
                cb(callback.CallbackEnv(model=booster,
                                        params=rumb.params[j],
                                        iteration=i,
                                        begin_iteration=init_iteration,
                                        end_iteration=init_iteration + num_boost_round,
                                        evaluation_result_list=None))       
    
            #update booster with custom binary objective function, and relevant features
            if rumb.functional_effects and j < 2*rumb.num_classes:
                rumb._current_j = int(j/2) #if functional effect keep same j for the two ensembles of each alternative
            else:
                rumb._current_j = j

            booster.update(train_set=rumb.train_set[j], fobj=f_obj)
            
            #check evaluation result. (from lightGBM initial code, check on all J binary boosters)
            evaluation_result_list = []
            if valid_sets is not None:
                if is_valid_contain_train:
                    evaluation_result_list.extend(booster.eval_train(feval))
                evaluation_result_list.extend(booster.eval_valid(feval))
            try:
                for cb in callbacks_after_iter:
                    cb(callback.CallbackEnv(model=booster,
                                            params=rumb.params[j],
                                            iteration=i,
                                            begin_iteration=init_iteration,
                                            end_iteration=init_iteration + num_boost_round,
                                            evaluation_result_list=evaluation_result_list))
            except callback.EarlyStopException as earlyStopException:
                booster.best_iteration = earlyStopException.best_iteration + 1
                evaluation_result_list = earlyStopException.best_score

        #make predictions after boosting round to compute new cross entropy and for next iteration grad and hess
        if nests is not None:
            rumb._preds, rumb.preds_i_m, rumb.preds_m = rumb._inner_predict(nests=True)
        elif alphas is not None:
            rumb._preds, rumb.preds_i_m, rumb.preds_m = rumb._inner_predict(alphas=alphas)
        else:
            rumb._preds = rumb._inner_predict()

        #compute cross validation on training or validation test
        if valid_sets is not None:
            if is_valid_contain_train:
                cross_entropy_test = cross_entropy(rumb._preds, rumb.labels)
            else:
                for k, _ in enumerate(valid_sets):
                    if nests is not None:
                        preds_valid, _, _ = rumb._inner_predict(k+1, nests=True)
                    elif alphas is not None:
                        preds_valid,  _, _ = rumb._inner_predict(k+1, alphas=alphas)
                    else:
                        preds_valid = rumb._inner_predict(k+1)
                    cross_entropy_train = cross_entropy(rumb._preds, rumb.labels)
                    cross_entropy_test = cross_entropy(preds_valid, rumb.valid_labels[k])

            #update best score and best iteration
            if cross_entropy_test < rumb.best_score:
                rumb.best_score = cross_entropy_test
                if is_valid_contain_train:
                    rumb.best_score_train = cross_entropy_test
                else:
                    rumb.best_score_train = cross_entropy_train
                rumb.best_iteration = i+1

            #verbosity
            if (verbosity >= 1) and (i % 10 == 0):
                if is_valid_contain_train:
                    print('[{}] -- NCE value on train set: {}'.format(i + 1, cross_entropy_test))
                else:
                    print('[{}] -- NCE value on train set: {} \n     --  NCE value on test set: {}'.format(i + 1, cross_entropy_train, cross_entropy_test))      
    
        #early stopping
        if (params["early_stopping_round"] != 0) and (rumb.best_iteration + params["early_stopping_round"] < i + 1):
            if is_valid_contain_train:
                print('Early stopping at iteration {}, with a best score of {}'.format(rumb.best_iteration, rumb.best_score))
            else:
                print('Early stopping at iteration {}, with a best score on test set of {}, and on train set of {}'.format(rumb.best_iteration, rumb.best_score, rumb.best_score_train))
            break

    for booster in rumb.boosters:
        booster.best_score_lgb = collections.defaultdict(collections.OrderedDict)
        for dataset_name, eval_name, score, _ in evaluation_result_list:
            booster.best_score_lgb[dataset_name][eval_name] = score
        if not keep_training_booster:
            booster.model_from_string(booster.model_to_string()).free_dataset()
    return rumb

class CVRUMBoost:
    """CVRUMBoost in LightGBM.

    Auxiliary data structure to hold and redirect all boosters of ``cv`` function.
    This class has the same methods as Booster class.
    All method calls are actually performed for underlying Boosters and then all returned results are returned in a list.

    Attributes
    ----------
    rum_boosters : list of RUMBoost
        The list of underlying fitted models.
    best_iteration : int
        The best iteration of fitted model.
    """

    def __init__(self):
        """Initialize the CVBooster.

        Generally, no need to instantiate manually.
        """
        self.RUMBoosts = []
        self.best_iteration = -1
        self.best_score = 100000

    def _append(self, rum_booster):
        """Add a booster to CVBooster."""
        self.RUMBoosts.append(rum_booster)

    def __getattr__(self, name):
        """Redirect methods call of CVBooster."""
        def handler_function(*args, **kwargs):
            """Call methods with each booster, and concatenate their results."""
            ret = []
            for rum_booster in self.RUMBoosts:
                for booster in rum_booster:
                    ret.append(getattr(booster, name)(*args, **kwargs))
                return ret
        return handler_function


def _make_n_folds(full_data, folds, nfold, params, seed, fpreproc=None, stratified=True,
                  shuffle=True, eval_train_metric=False, rum_structure=None, biogeme_model=None):
    """Make a n-fold list of Booster from random indices."""
    full_data = full_data.construct()
    num_data = full_data.num_data()
    if folds is not None:
        if not hasattr(folds, '__iter__') and not hasattr(folds, 'split'):
            raise AttributeError("folds should be a generator or iterator of (train_idx, test_idx) tuples "
                                 "or scikit-learn splitter object with split method")
        if hasattr(folds, 'split'):
            group_info = full_data.get_group()
            if group_info is not None:
                group_info = np.array(group_info, dtype=np.int32, copy=False)
                flatted_group = np.repeat(range(len(group_info)), repeats=group_info)
            else:
                flatted_group = np.zeros(num_data, dtype=np.int32)
            folds = folds.split(X=np.empty(num_data), y=full_data.get_label(), groups=flatted_group)
    else:
        if any(params.get(obj_alias, "") in {"lambdarank", "rank_xendcg", "xendcg",
                                             "xe_ndcg", "xe_ndcg_mart", "xendcg_mart"}
               for obj_alias in _ConfigAliases.get("objective")):
            if not SKLEARN_INSTALLED:
                raise LightGBMError('scikit-learn is required for ranking cv')
            # ranking task, split according to groups
            group_info = np.array(full_data.get_group(), dtype=np.int32, copy=False)
            flatted_group = np.repeat(range(len(group_info)), repeats=group_info)
            group_kfold = _LGBMGroupKFold(n_splits=nfold)
            folds = group_kfold.split(X=np.empty(num_data), groups=flatted_group)
        elif stratified:
            if not SKLEARN_INSTALLED:
                raise LightGBMError('scikit-learn is required for stratified cv')
            skf = _LGBMStratifiedKFold(n_splits=nfold, shuffle=shuffle, random_state=seed)
            folds = skf.split(X=np.empty(num_data), y=full_data.get_label())
        else:
            if shuffle:
                randidx = np.random.RandomState(seed).permutation(num_data)
            else:
                randidx = np.arange(num_data)
            kstep = int(num_data / nfold)
            test_id = [randidx[i: i + kstep] for i in range(0, num_data, kstep)]
            train_id = [np.concatenate([test_id[i] for i in range(nfold) if k != i]) for k in range(nfold)]
            folds = zip(train_id, test_id)

    ret = CVRUMBoost()

    for train_idx, test_idx in folds:
        train_set = full_data.subset(sorted(train_idx))
        valid_set = full_data.subset(sorted(test_idx))
        # run preprocessing on the data set if needed
        if fpreproc is not None:
            train_set, valid_set, tparam = fpreproc(train_set, valid_set, params.copy())
        else:
            tparam = params
        #create RUMBoosts with corresponding training, validation, and parameters sets
        cvbooster = RUMBoost()
        if rum_structure is not None:
            cvbooster.rum_structure = rum_structure #save utility structure
        elif biogeme_model is not None:
            cvbooster.rum_structure = bio_to_rumboost(biogeme_model, max_depth=params['max_depth'])
        else:
            raise ValueError("Either one of rum_structure or biogeme_model arguments must be passed")
        reduced_valid_sets, name_valid_sets, is_valid_contain_train, train_data_name = cvbooster._preprocess_valids(train_set, params, valid_set)
        cvbooster._preprocess_data(train_set, reduced_valid_sets)
        cvbooster._preprocess_params(tparam)
        cvbooster._construct_boosters(train_data_name, is_valid_contain_train,
                                      name_valid_sets=name_valid_sets)

        ret._append(cvbooster)

        ret.best_iteration = 0
        ret.best_score = 100000

    return ret

def _agg_cv_result(raw_results, eval_train_metric=False):
    """Aggregate cross-validation results."""
    cvmap = collections.OrderedDict()
    metric_type = {}
    for one_result in raw_results:
        for one_line in one_result:
            if eval_train_metric:
                key = f"{one_line[0]} {one_line[1]}"
            else:
                key = one_line[1]
            metric_type[key] = one_line[3]
            cvmap.setdefault(key, [])
            cvmap[key].append(one_line[2])
    return [('cv_agg', k, np.mean(v), metric_type[k], np.std(v)) for k, v in cvmap.items()]


def rum_cv(params, train_set, num_boost_round=100,
        folds=None, nfold=5, stratified=True, shuffle=True,
        metrics=None, fobj=None, feval=None, init_model=None,
        feature_name='auto', categorical_feature='auto',
        early_stopping_rounds=None, fpreproc=None,
        verbose_eval=None, show_stdv=True, seed=0,
        callbacks=None, eval_train_metric=False,
        return_cvbooster=False, rum_structure=None,
        biogeme_model=None):
    """Perform the cross-validation with given parameters.

    Parameters
    ----------
    params : dict
        Parameters for Booster.
    train_set : Dataset
        Data to be trained on.
    num_boost_round : int, optional (default=100)
        Number of boosting iterations.
    folds : generator or iterator of (train_idx, test_idx) tuples, scikit-learn splitter object or None, optional (default=None)
        If generator or iterator, it should yield the train and test indices for each fold.
        If object, it should be one of the scikit-learn splitter classes
        (https://scikit-learn.org/stable/modules/classes.html#splitter-classes)
        and have ``split`` method.
        This argument has highest priority over other data split arguments.
    nfold : int, optional (default=5)
        Number of folds in CV.
    stratified : bool, optional (default=True)
        Whether to perform stratified sampling.
    shuffle : bool, optional (default=True)
        Whether to shuffle before splitting data.
    metrics : str, list of str, or None, optional (default=None)
        Evaluation metrics to be monitored while CV.
        If not None, the metric in ``params`` will be overridden.
    fobj : callable or None, optional (default=None)
        Customized objective function.
        Should accept two parameters: preds, train_data,
        and return (grad, hess).

            preds : list or numpy 1-D array
                The predicted values.
                Predicted values are returned before any transformation,
                e.g. they are raw margin instead of probability of positive class for binary task.
            train_data : Dataset
                The training dataset.
            grad : list or numpy 1-D array
                The value of the first order derivative (gradient) of the loss
                with respect to the elements of preds for each sample point.
            hess : list or numpy 1-D array
                The value of the second order derivative (Hessian) of the loss
                with respect to the elements of preds for each sample point.

        For multi-class task, the preds is group by class_id first, then group by row_id.
        If you want to get i-th row preds in j-th class, the access way is score[j * num_data + i]
        and you should group grad and hess in this way as well.

    feval : callable, list of callable, or None, optional (default=None)
        Customized evaluation function.
        Each evaluation function should accept two parameters: preds, train_data,
        and return (eval_name, eval_result, is_higher_better) or list of such tuples.

            preds : list or numpy 1-D array
                The predicted values.
                If ``fobj`` is specified, predicted values are returned before any transformation,
                e.g. they are raw margin instead of probability of positive class for binary task in this case.
            train_data : Dataset
                The training dataset.
            eval_name : str
                The name of evaluation function (without whitespace).
            eval_result : float
                The eval result.
            is_higher_better : bool
                Is eval result higher better, e.g. AUC is ``is_higher_better``.

        For multi-class task, the preds is group by class_id first, then group by row_id.
        If you want to get i-th row preds in j-th class, the access way is preds[j * num_data + i].
        To ignore the default metric corresponding to the used objective,
        set ``metrics`` to the string ``"None"``.
    init_model : str, pathlib.Path, Booster or None, optional (default=None)
        Filename of LightGBM model or Booster instance used for continue training.
    feature_name : list of str, or 'auto', optional (default="auto")
        Feature names.
        If 'auto' and data is pandas DataFrame, data columns names are used.
    categorical_feature : list of str or int, or 'auto', optional (default="auto")
        Categorical features.
        If list of int, interpreted as indices.
        If list of str, interpreted as feature names (need to specify ``feature_name`` as well).
        If 'auto' and data is pandas DataFrame, pandas unordered categorical columns are used.
        All values in categorical features should be less than int32 max value (2147483647).
        Large values could be memory consuming. Consider using consecutive integers starting from zero.
        All negative values in categorical features will be treated as missing values.
        The output cannot be monotonically constrained with respect to a categorical feature.
    early_stopping_rounds : int or None, optional (default=None)
        Activates early stopping.
        CV score needs to improve at least every ``early_stopping_rounds`` round(s)
        to continue.
        Requires at least one metric. If there's more than one, will check all of them.
        To check only the first metric, set the ``first_metric_only`` parameter to ``True`` in ``params``.
        Last entry in evaluation history is the one from the best iteration.
    fpreproc : callable or None, optional (default=None)
        Preprocessing function that takes (dtrain, dtest, params)
        and returns transformed versions of those.
    verbose_eval : bool, int, or None, optional (default=None)
        Whether to display the progress.
        If True, progress will be displayed at every boosting stage.
        If int, progress will be displayed at every given ``verbose_eval`` boosting stage.
    show_stdv : bool, optional (default=True)
        Whether to display the standard deviation in progress.
        Results are not affected by this parameter, and always contain std.
    seed : int, optional (default=0)
        Seed used to generate the folds (passed to numpy.random.seed).
    callbacks : list of callable, or None, optional (default=None)
        List of callback functions that are applied at each iteration.
        See Callbacks in Python API for more information.
    eval_train_metric : bool, optional (default=False)
        Whether to display the train metric in progress.
        The score of the metric is calculated again after each training step, so there is some impact on performance.
    return_cvbooster : bool, optional (default=False)
        Whether to return Booster models trained on each fold through ``CVBooster``.
    rum_structure : dict, optional (default=None)
        List of dictionaries specifying the RUM structure. 
        The list must contain one dictionary for each class, which describes the 
        utility structure for that class. 
        Each dictionary has three allowed keys. 

            cols : list of columns included in that class
            monotone_constraints : list of monotonic constraints on parameters
            interaction_constraints : list of interaction constraints on features
            
        if None, a biogeme_model must be specified
    biogeme_model: biogeme.biogeme.BIOGEME, optional (default=None)
        A biogeme.biogeme.BIOGEME object representing a biogeme model, used to create the rum_structure.
        A biogeme model is required if rum_structure is None, otherwise should be None.

    Returns
    -------
    eval_hist : dict
        Evaluation history.
        The dictionary has the following format:
        {'metric1-mean': [values], 'metric1-stdv': [values],
        'metric2-mean': [values], 'metric2-stdv': [values],
        ...}.
        If ``return_cvbooster=True``, also returns trained boosters via ``cvbooster`` key.
    """
    raise NotImplementedError("This function is not implemented yet")

    if not isinstance(train_set, Dataset):
        raise TypeError("Training only accepts Dataset object")

    params = copy.deepcopy(params)
    if fobj is not None:
        for obj_alias in _ConfigAliases.get("objective"):
            params.pop(obj_alias, None)
        params['objective'] = 'none'
    for alias in _ConfigAliases.get("num_iterations"):
        if alias in params:
            _log_warning(f"Found `{alias}` in params. Will use it instead of argument")
            num_boost_round = params.pop(alias)
    params["num_iterations"] = num_boost_round
    if early_stopping_rounds is not None and early_stopping_rounds > 0:
        _log_warning("'early_stopping_rounds' argument is deprecated and will be removed in a future release of LightGBM. "
                     "Pass 'early_stopping()' callback via 'callbacks' argument instead.")
    for alias in _ConfigAliases.get("early_stopping_round"):
        if alias in params:
            early_stopping_rounds = params.pop(alias)
    params["early_stopping_round"] = early_stopping_rounds
    first_metric_only = params.get('first_metric_only', False)

    if num_boost_round <= 0:
        raise ValueError("num_boost_round should be greater than zero.")
    if isinstance(init_model, (str, Path)):
        predictor = _InnerPredictor(model_file=init_model, pred_parameter=params)
    elif isinstance(init_model, Booster):
        predictor = init_model._to_predictor(dict(init_model.params, **params))
    else:
        predictor = None

    if metrics is not None:
        for metric_alias in _ConfigAliases.get("metric"):
            params.pop(metric_alias, None)
        params['metric'] = metrics

    train_set._update_params(params) \
             ._set_predictor(predictor) \
             .set_feature_name(feature_name) \
             .set_categorical_feature(categorical_feature)

    results = collections.defaultdict(list)
    cvfolds = _make_n_folds(train_set, folds=folds, nfold=nfold,
                            params=params, seed=seed, fpreproc=fpreproc,
                            stratified=stratified, shuffle=shuffle,
                            eval_train_metric=eval_train_metric, rum_structure=rum_structure,
                            biogeme_model=biogeme_model)

    # setup callbacks
    if callbacks is None:
        callbacks = set()
    else:
        for i, cb in enumerate(callbacks):
            cb.__dict__.setdefault('order', i - len(callbacks))
        callbacks = set(callbacks)
    if early_stopping_rounds is not None and early_stopping_rounds > 0:
        callbacks.add(callback.early_stopping(early_stopping_rounds, first_metric_only, verbose=False))
    if verbose_eval is not None:
        _log_warning("'verbose_eval' argument is deprecated and will be removed in a future release of LightGBM. "
                     "Pass 'log_evaluation()' callback via 'callbacks' argument instead.")
    if verbose_eval is True:
        callbacks.add(callback.log_evaluation(show_stdv=show_stdv))
    elif isinstance(verbose_eval, int):
        callbacks.add(callback.log_evaluation(verbose_eval, show_stdv=show_stdv))

    callbacks_before_iter = {cb for cb in callbacks if getattr(cb, 'before_iteration', False)}
    callbacks_after_iter = callbacks - callbacks_before_iter
    callbacks_before_iter = sorted(callbacks_before_iter, key=attrgetter('order'))
    callbacks_after_iter = sorted(callbacks_after_iter, key=attrgetter('order'))

    for i in range(num_boost_round):
        cross_ent = []
        raw_results = []
        #train all RUMBoosts
        for RUMBoost in cvfolds.RUMBoosts:
            RUMBoost._preds = RUMBoost._inner_predict()
            for j, booster in enumerate(RUMBoost.boosters):
                for cb in callbacks_before_iter:
                    cb(callback.CallbackEnv(model=booster,
                                            params=RUMBoost.params[j],
                                            iteration=i,
                                            begin_iteration=0,
                                            end_iteration=num_boost_round,
                                            evaluation_result_list=None))
                RUMBoost._current_j = j
                booster.update(train_set = RUMBoost.train_set[j], fobj=RUMBoost.f_obj)

            valid_sets = RUMBoost.valid_sets
            for valid_set in valid_sets:
                preds_valid = RUMBoost._inner_predict(data = valid_set)
                raw_results.append(preds_valid)
                cross_ent.append(cross_entropy(preds_valid, valid_set[0].get_label().astype(int)))

        results[f'Cross entropy --- mean'].append(np.mean(cross_ent))
        results[f'Cross entropy --- stdv'].append(np.std(cross_ent))
        if verbose_eval:
            print('[{}] -- Cross entropy mean: {}, with std: {}'.format(i + 1, np.mean(cross_ent), np.std(cross_ent)))
        
        if np.mean(cross_ent) < cvfolds.best_score:
            cvfolds.best_score = np.mean(cross_ent)
            cvfolds.best_iteration = i + 1 

        if (int(params.get("early_stopping_round", 0) or 0) > 0) and (cvfolds.best_iteration + params.get("early_stopping_round", 0) < i + 1):
            print('Early stopping at iteration {} with a cross entropy best score of {}'.format(cvfolds.best_iteration,cvfolds.best_score))
            for k in results:
                results[k] = results[k][:cvfolds.best_iteration]
            break
        #res = _agg_cv_result(raw_results, eval_train_metric)
        #try:
        #    for cb in callbacks_after_iter:
        #        cb(callback.CallbackEnv(model=cvfolds,
        #                                params=params,
        #                                iteration=i,
        #                                begin_iteration=0,
        #                                end_iteration=num_boost_round,
        #                                evaluation_result_list=res))
        #except callback.EarlyStopException as earlyStopException:
        #    cvfolds.best_iteration = earlyStopException.best_iteration + 1
        #    for k in results:
        #        results[k] = results[k][:cvfolds.best_iteration]
        #    break

    if return_cvbooster:
        results['cvbooster'] = cvfolds

    return dict(results)