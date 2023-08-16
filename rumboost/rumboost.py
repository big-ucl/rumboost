# coding: utf-8
"""Library with training routines of LightGBM."""
import collections
import copy
import json
import numpy as np
import pandas as pd
from scipy.special import softmax

from operator import attrgetter
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

from lightgbm import callback
from lightgbm.basic import Booster, Dataset, LightGBMError, _ConfigAliases, _InnerPredictor, _choose_param_value, _log_warning
from lightgbm.compat import SKLEARN_INSTALLED, _LGBMGroupKFold, _LGBMStratifiedKFold

from utils import bio_to_rumboost
from utility_smoothing import stairs_to_pw

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
        The list of underlying fitted models.
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
            preds = self._preds[:,j] #corresponding predictions
            factor = self.num_classes/(self.num_classes-1) #factor to correct redundancy (see Friedmann, Greedy Function Approximation)
            eps = 1e-6
            labels = train_set.get_label()
            grad = preds - labels
            hess = np.maximum(factor * preds * (1 - preds), eps) #truncate low values to avoid numerical errors
            return grad, hess
            
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
        piece_wise: bool = False
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
        piece_wise : bool, optional (default=False)
            If True, use piece-wise utility instead of stairs utility.

        Returns
        -------
        result : numpy array, scipy.sparse or list of scipy.sparse
            Prediction result.
            Can be sparse or a list of sparse objects (each element represents predictions for one class) for feature contributions (when ``pred_contrib=True``).
        """    
        #compute utilities with corresponding features
        if piece_wise: #piece-wise utility
            U = self.pw_predict(data, utility= True)
        else:
            #split data
            new_data, _ = self._preprocess_data(data, return_data=True)
            #compute U
            U = [booster.predict(new_data[k].get_data(), 
                                 start_iteration, 
                                 num_iteration, 
                                 raw_score, 
                                 pred_leaf, 
                                 pred_contrib,
                                 data_has_header,
                                 validate_features) for k, booster in enumerate(self.boosters)]

        preds = np.array(U).T

        #softmax
        if not utilities:
            preds = softmax(preds, axis=1)
   
        return preds
    
    def _inner_predict(
        self,
        data = None,
        start_iteration: int = 0,
        num_iteration: int = -1,
        raw_score: bool = True,
        pred_leaf: bool = False,
        pred_contrib: bool = False,
        data_has_header: bool = False,
        validate_features: bool = False,
        utilities: bool = False,
        piece_wise: bool = False
    ):
        """Inner predict logic, the dataset is not assumed to be already build. Should not be used in public.

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
        utilities : bool, optional (default=False)
            If True, return raw utilities for each class, without generating probabilities.     
        piece_wise : bool, optional (default=False)
            If True, use piece-wise utility instead of stairs utility.

        Returns
        -------
        result : numpy array, scipy.sparse or list of scipy.sparse
            Prediction result.
            Can be sparse or a list of sparse objects (each element represents predictions for one class) for feature contributions (when ``pred_contrib=True``).
        """
        #getting dataset
        if data is None:
            data = self.train_set

        #compute utilities with corresponding features
        if piece_wise:
            U = self.pw_predict(data, utility= True)
        else:
            U = [booster.predict(data[k].get_data(), 
                                 start_iteration, 
                                 num_iteration, 
                                 raw_score, 
                                 pred_leaf, 
                                 pred_contrib,
                                 data_has_header,
                                 validate_features) for k, booster in enumerate(self.boosters)]

        U = np.array(U).T

        #softmax
        if not utilities:
            U = softmax(U, axis=1)

        return U

    def pw_predict(self, data: Dataset, data_to_transform: pd.DataFrame = None, utilities: bool = False):
        '''
        Compute predictions with piece-wise utility (WARNING: Not stable).

        Parameters
        ----------
        data : Dataset
            The full training dataset.
        data_to_transform : pandas DataFrame, optional
            The stairs data to be transformed into a piece-wise linear approximation.
        utilities : bool, optional (default=False)
            If True, return raw utilities for each class, without generating probabilities.

        Returns
        -------
        U : list or numpy array
            The piece-wise linear predictions

        '''
        #compute the piece-wise utilities
        U = stairs_to_pw(self, data, data_to_transform)

        #softmax
        if not utilities:
            U = softmax(U.T, axis=1)
        
        return U
    
    def _preprocess_data(self, data: Dataset, reduced_valid_set = None, return_data: bool = False):
        """Set up J training (and, if specified, validation) datasets.
        
        Parameters
        ----------
        data : Dataset
            The full training dataset (i.e. the union of the socio-economic features with the alternative-specific features).
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

        #loop over all J utilities
        for j, struct in enumerate(self.rum_structure):
            if struct:
                if 'columns' in struct:
                    train_set_j_data = data.get_data()[struct['columns']] #only relevant features for the jth booster
                    new_label = np.where(data.get_label() == j, 1, 0) #new binary label, used for multiclassification
                    #create and build dataset
                    if self.with_linear_trees:
                        train_set_j = Dataset(train_set_j_data, label=new_label, free_raw_data=False, params={'linear_trees':True}) #WARNING not stable
                    else:
                        train_set_j = Dataset(train_set_j_data, label=new_label, free_raw_data=False)
                    train_set_j.construct()
                    if reduced_valid_set is not None:
                        reduced_valid_sets_j = []
                        for valid_set in reduced_valid_set:
                            #create and build validation sets
                            valid_set.construct()
                            valid_set_j_data = valid_set.get_data()[struct['columns']] #only relevant features for the j booster
                            label_valid = valid_set.get_label()#new binary label
                            valid_set_j = Dataset(valid_set_j_data, label=label_valid, reference= train_set_j, free_raw_data=False)
                            valid_set_j.construct()
                            reduced_valid_sets_j.append(valid_set_j)

                else:
                    #if no alternative specific datasets
                    train_set_j = data
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
    
    def _preprocess_params(self, params: dict, return_params: bool = False):
        """Set up J set of parameters.
        
        Parameters
        ----------
        params : dict
            Dictionary containing parameters. The syntax must follow the one from LightGBM.
        return_params : bool, optional (default = False)
            If True, returns the J sets of parameters (and potential validation sets)

        Returns
        -------
        params_J : list[dict]
            A list of dictionary containing J sets of parameters.
        """
        #store the number of classes in RUMBoost
        self.num_classes = params['num_classes']

        #create the J parameters dictionaries
        params_J = [{**copy.deepcopy(params),
                    'objective': 'binary',
                    'num_classes':1,
                    'monotone_constraints': struct.get('monotone_constraints', []) if struct else [],
                    'interaction_constraints': struct.get('monotone_constraints', []) if struct else [],
                    'categorical_feature': struct.get('categorical_feature', []) if struct else []
                    } for struct in self.rum_structure]
        
        #update if with linear trees or not
        self.with_linear_trees = params.get('linear_tree', False)

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
        #construct training set to access data
        train_set.construct()

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
                            name_valid_sets = None):
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
            #construct booster and perform basic preparations
            try: 
                booster = Booster(params=param_j, train_set=train_set_j)
                if is_valid_contain_train:
                    booster.set_train_data_name(train_data_name)
                for valid_set, name_valid_set in zip(reduced_valid_sets_J, name_valid_sets):
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
        self.best_score = 1000000

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
    params: Dict[str, Any],
    train_set: Dataset,
    rum_structure: List[Dict[str, Any]],
    biogeme_model: biogeme.biogeme.BIOGEME = None,
    num_boost_round: int = 100,
    valid_sets: Optional[List[Dataset]] = None,
    valid_names: Optional[List[str]] = None,
    feval: Optional[Union[_LGBM_CustomMetricFunction, List[_LGBM_CustomMetricFunction]]] = None,
    init_model: Optional[Union[str, Path, Booster]] = None,
    feature_name: Union[List[str], str] = 'auto',
    categorical_feature: Union[List[str], List[int], str] = 'auto',
    keep_training_booster: bool = False,
    callbacks: Optional[List[Callable]] = None,
    pw_utility: bool = False
) -> RUMBoost:
    """Perform the RUM training with given parameters.

    Parameters
    ----------
    params : dict
        Parameters for training. Values passed through ``params`` take precedence over those
        supplied via arguments.
    train_set : Dataset
        Data to be trained on.
    rum_structure : dict, optional (default=None)
        List of dictionaries specifying the RUM structure. 
        The list must contain one dictionary for each class, which describes the 
        utility structure for that class. 
        Each dictionary has three allowed keys. 
        'cols': list of columns included in that class
        'monotone_constraints': list of monotonic constraints on parameters
        'interaction_constraints': list of interaction constraints on features
        if None, a biogeme_model must be specified
    biogeme_model: biogeme.biogeme.BIOGEME, optional (default=None)
        A biogeme.biogeme.BIOGEME object representing a biogeme model, used to create the rum_structure.
        A biogeme model is required if rum_structure is None, otherwise should be None.
    num_boost_round : int, optional (default=100)
        Number of boosting iterations.
    valid_sets : list of Dataset, or None, optional (default=None)
        List of data to be evaluated on during training.
    valid_names : list of str, or None, optional (default=None)
        Names of ``valid_sets``.
    feval : callable, list of callable, or None, optional (default=None)
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
        All values in categorical features will be cast to int32 and thus should be less than int32 max value (2147483647).
        Large values could be memory consuming. Consider using consecutive integers starting from zero.
        All negative values in categorical features will be treated as missing values.
        The output cannot be monotonically constrained with respect to a categorical feature.
        Floating point numbers in categorical features will be rounded towards 0.
    keep_training_booster : bool, optional (default=False)
        Whether the returned Booster will be used to keep training.
        If False, the returned value will be converted into _InnerPredictor before returning.
        This means you won't be able to use ``eval``, ``eval_train`` or ``eval_valid`` methods of the returned Booster.
        When your model is very large and cause the memory error,
        you can try to set this param to ``True`` to avoid the model conversion performed during the internal call of ``model_to_string``.
        You can still use _InnerPredictor as ``init_model`` for future continue training.
    callbacks : list of callable, or None, optional (default=None)
        List of callback functions that are applied at each iteration.
        See Callbacks in Python API for more information.
    pw_utility: bool, optional (default=False)
        If true, compute continuous feature utility in a piece-wise linear way.
        

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
        params.pop("early_stopping_round")
    first_metric_only = params.get('first_metric_only', False)

    if num_boost_round <= 0:
        raise ValueError("num_boost_round should be greater than zero.")
    predictor: Optional[_InnerPredictor] = None
    if isinstance(init_model, (str, Path)):
        predictor = _InnerPredictor(model_file=init_model, pred_parameter=params)
    elif isinstance(init_model, Booster):
        predictor = init_model._to_predictor(dict(init_model.params, **params))
    init_iteration = predictor.num_total_iteration if predictor is not None else 0
    #check dataset
    if not isinstance(train_set, Dataset):
        raise TypeError("Training only accepts Dataset object")

    train_set._update_params(params) \
             ._set_predictor(predictor) \
             .set_feature_name(feature_name) \
             .set_categorical_feature(categorical_feature)

    #process callbacks
    if callbacks is None:
        callbacks_set = set()
    else:
        for i, cb in enumerate(callbacks):
            cb.__dict__.setdefault('order', i - len(callbacks))
        callbacks_set = set(callbacks)

    if "early_stopping_round" in params:
       callbacks_set.add(
           callback.early_stopping(
               stopping_rounds=params["early_stopping_round"],
               first_metric_only=first_metric_only,
               verbose=_choose_param_value(
                   main_param_name="verbosity",
                   params=params,
                   default_value=1
               ).pop("verbosity") > 0
           )
       )

    callbacks_before_iter_set = {cb for cb in callbacks_set if getattr(cb, 'before_iteration', False)}
    callbacks_after_iter_set = callbacks_set - callbacks_before_iter_set
    callbacks_before_iter = sorted(callbacks_before_iter_set, key=attrgetter('order'))
    callbacks_after_iter = sorted(callbacks_after_iter_set, key=attrgetter('order'))

    #construct boosters
    rum_booster = RUMBoost()
    reduced_valid_sets, \
    name_valid_sets, \
    is_valid_contain_train, \
    train_data_name = rum_booster._preprocess_valids(train_set, params, valid_sets) #prepare validation sets
    if rum_structure is not None:
        rum_booster.rum_structure = rum_structure #save utility structure
    elif biogeme_model is not None:
        rum_booster.rum_structure = bio_to_rumboost(biogeme_model, max_depth=params['max_depth'])
    else:
        raise ValueError("Either one of rum_structure or biogeme_model arguments must be passed")
    rum_booster._preprocess_params(params) #prepare J set of parameters
    rum_booster._preprocess_data(train_set, reduced_valid_sets, return_data=True) #prepare J datasets with relevant features
    rum_booster._construct_boosters(train_data_name, is_valid_contain_train, name_valid_sets) #build boosters with corresponding params and dataset

    #initial prediction for first iteration
    rum_booster._preds = rum_booster._inner_predict()

    #start training
    for i in range(init_iteration, init_iteration + num_boost_round):
        #update all binary boosters of the rum_booster
        for j, booster in enumerate(rum_booster.boosters):
            for cb in callbacks_before_iter:
                cb(callback.CallbackEnv(model=booster,
                                        params=rum_booster.params[j],
                                        iteration=i,
                                        begin_iteration=init_iteration,
                                        end_iteration=init_iteration + num_boost_round,
                                        evaluation_result_list=None))       
    
            #update booster with custom binary objective function, and relevant features
            rum_booster._current_j = j
            booster.update(train_set=rum_booster.train_set[j], fobj=rum_booster.f_obj)
            
            #check evaluation result. (from lightGBM initial code, check on all J binary boosters)
            evaluation_result_list = []
            if valid_sets is not None:
                if is_valid_contain_train:
                    evaluation_result_list.extend(booster.eval_train(feval))
                evaluation_result_list.extend(booster.eval_valid(feval))
            try:
                for cb in callbacks_after_iter:
                    cb(callback.CallbackEnv(model=booster,
                                            params=rum_booster.params[j],
                                            iteration=i,
                                            begin_iteration=init_iteration,
                                            end_iteration=init_iteration + num_boost_round,
                                            evaluation_result_list=evaluation_result_list))
            except callback.EarlyStopException as earlyStopException:
                booster.best_iteration = earlyStopException.best_iteration + 1
                evaluation_result_list = earlyStopException.best_score

        #make predictions after boosting round to compute new cross entropy and for next iteration grad and hess
        rum_booster._preds = rum_booster._inner_predict(piece_wise=pw_utility)
        #compute cross validation on training or validation test
        if valid_sets is not None:
            if is_valid_contain_train:
                cross_entropy = rum_booster.cross_entropy(rum_booster._preds, train_set.get_label().astype(int))
            else:
                for valid_set_J in rum_booster.valid_sets:
                    preds_valid = rum_booster._inner_predict(valid_set_J, piece_wise=pw_utility)
                    cross_entropy_train = rum_booster.cross_entropy(rum_booster._preds, train_set.get_label().astype(int))
                    cross_entropy = rum_booster.cross_entropy(preds_valid, valid_set_J[0].get_label().astype(int))
        
            if cross_entropy < rum_booster.best_score:
                rum_booster.best_score = cross_entropy
                rum_booster.best_iteration = i+1
        
            if (params['verbosity'] >= 1) and (i % 10 == 0):
                if is_valid_contain_train:
                    print('[{}] -- NCE value on train set: {}'.format(i + 1, cross_entropy))
                else:
                    print('[{}] -- NCE value on train set: {} \n     --  NCE value on test set: {}'.format(i + 1, cross_entropy_train, cross_entropy))
        
        #early stopping if early stopping criterion in all boosters
        if (params["early_stopping_round"] != 0) and (rum_booster.best_iteration + params["early_stopping_round"] < i + 1):
            print('Early stopping at iteration {}, with a best score of {}'.format(rum_booster.best_iteration, rum_booster.best_score))
            break

    for booster in rum_booster.boosters:
        booster.best_score = collections.defaultdict(collections.OrderedDict)
        for dataset_name, eval_name, score, _ in evaluation_result_list:
            booster.best_score[dataset_name][eval_name] = score
        if not keep_training_booster:
            booster.model_from_string(booster.model_to_string(), verbose='_silent_false').free_dataset()
    return rum_booster

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
            'cols': list of columns included in that class
            'monotone_constraints': list of monotonic constraints on parameters
            'interaction_constraints': list of interaction constraints on features
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
                cross_ent.append(RUMBoost.cross_entropy(preds_valid, valid_set[0].get_label().astype(int)))

        results[f'Cross entropy --- mean'].append(np.mean(cross_ent))
        results[f'Cross entropy --- stdv'].append(np.std(cross_ent))
        if verbose_eval and i % 10 == 0:
            print('[{}] -- Cross entropy mean: {}, with std: {}'.format(i + 1, np.mean(cross_ent), np.std(cross_ent)))
        
        if np.mean(cross_ent) < cvfolds.best_score:
            cvfolds.best_score = np.mean(cross_ent)
            cvfolds.best_iteration = i + 1 

        if early_stopping_rounds is not None and cvfolds.best_iteration + early_stopping_rounds < i+1:
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