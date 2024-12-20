# coding: utf-8
"""Library with training routines of LightGBM."""
import collections
import copy
import json
import numpy as np

from scipy.special import softmax, expit
from scipy.optimize import minimize
from operator import attrgetter
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from lightgbm import callback
from lightgbm.basic import (
    Booster,
    Dataset,
    LightGBMError,
    _ConfigAliases,
    _InnerPredictor,
    _choose_param_value,
    _log_warning,
)
from lightgbm.compat import SKLEARN_INSTALLED, _LGBMGroupKFold, _LGBMStratifiedKFold

from rumboost.metrics import cross_entropy, binary_cross_entropy, mse, safe_softplus
from rumboost.nested_cross_nested import (
    nest_probs,
    cross_nested_probs,
    optimise_mu_or_alpha,
)
from rumboost.ordinal import (
    diff_to_threshold,
    threshold_to_diff,
    threshold_preds,
    corn_preds,
    optimise_thresholds_coral,
    optimise_thresholds_proportional_odds,
)

from rumboost.utils import optimise_asc

try:
    import torch
    from rumboost.torch_functions import (
        _inner_predict_torch,
        _inner_predict_torch_compiled,
        _f_obj_torch,
        _f_obj_torch_compiled,
        _f_obj_binary_torch,
        _f_obj_binary_torch_compiled,
        _f_obj_mse_torch,
        _f_obj_mse_torch_compiled,
        _f_obj_nested_torch,
        _f_obj_nested_torch_compiled,
        _f_obj_cross_nested_torch,
        _f_obj_cross_nested_torch_compiled,
        _f_obj_proportional_odds_torch,
        _f_obj_proportional_odds_torch_compiled,
        _f_obj_coral_torch,
        _f_obj_coral_torch_compiled,
        _f_obj_corn_torch,
        _f_obj_corn_torch_compiled,
        cross_entropy_torch,
        cross_entropy_torch_compiled,
        binary_cross_entropy_torch,
        binary_cross_entropy_torch_compiled,
        mse_torch,
        mse_torch_compiled,
    )

    torch_installed = True
except ImportError:
    torch_installed = False
try:
    import matplotlib.pyplot as plt

    matplotlib_installed = True
except ImportError:
    matplotlib_installed = False

_LGBM_CustomObjectiveFunction = Callable[
    [Union[List, np.ndarray], Dataset],
    Tuple[Union[List, np.ndarray], Union[List, np.ndarray]],
]
_LGBM_CustomMetricFunction = Callable[
    [Union[List, np.ndarray], Dataset], Tuple[str, float, bool]
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

    def __init__(self, model_file=None, **kwargs):
        """Initialize the RUMBoost.

        Parameters
        ----------
        model_file : str, pathlib.Path or None, optional (default=None)
            Path to the RUMBoost model file.
        **kwargs : dict
            Other attributes for the RUMBoost.
            It could be the following attributes:

            Core model attributes:
            - boosters : list of Booster
                List of LightGBM boosters.
            - rum_structure : list of dict
                RUM structure.
            - num_classes : int
                Number of classes.
            - num_obs : list of int
                Number of observations.
            - params : list of dict
                List of parameters for each booster.
            - labels : numpy array
                Labels of the dataset.
            - labels_j : list of numpy array
                Labels for each booster.
            - valid_labels : list of numpy array
                Validation labels.
            - boost_from_parameter_space : list[bool]
                Whether to boost from the parameter space.
                If so the output of the boosters will be betas,
                instead of piece-wise utility constants. If specified,
                the length of the list must be the same
                as the number of boosters.

            Nested and cross-nested attributes:
            - nests : dict
                Dictionary of nests.
            - mu : ndarray
                array of mu values.
            - alphas : ndarray
                Array of alpha values.
            - nest_alt : dict
                Dictionary of alternative nests.

            Ordinal attributes:
            - ord_model : str
                Type of ordinal model.
            - thresholds : list of float
                Thresholds for ordinal model.

            Torch tensors attributes:
            - device : str
                Device to use for computations.
            - torch_compile : bool
                Whether to use compiled torch functions.

            Early stopping attributes:
            - best_score : float
                Best score.
            - best_score_train : float
                Best score on training set.
            - best_iteration : int
                Best iteration.
        """
        self.boosters = []
        self.__dict__.update(kwargs)

        if model_file is not None:
            with open(model_file, "r") as file:
                self._from_dict(json.load(file))

        if "device" in self.__dict__ and self.device is not None:
            self.device = torch.device(self.device)

        if self.alphas is not None:  # numpy.ndarray so need to specify not None
            self.alphas = np.array(self.alphas)
        if self.mu is not None:  # numpy.ndarray so need to specify not None
            self.mu = np.array(self.mu)
        if self.thresholds is not None:  # numpy.ndarray so need to specify not None
            self.thresholds = np.array(self.thresholds)

    def multiply_grad_hess_by_data(func):
        """
        Decorator to multiply the gradient and hessian by the number of observations for the jth booster.
        This is used to scale the gradient and hessian when
        boosting from the parameter space, according to the chain rule.
        """

        def wrapper(self, preds, data):
            j = self._current_j

            grad, hess = func(self, preds, data)
            if self.boost_from_parameter_space[j]:
                # multiplying gradients and hessians by observations
                # we start with hess because it needs the original values
                # (second derivative of the chain rule)
                # hess = hess * self._monotonise_grad(
                #     self.train_set[j].data.reshape(-1), j
                # ) ** 2 + grad * self._monotonise_hess(
                #     self.train_set[j].data.reshape(-1), j
                # )
                grad *= self._monotonise_grad(self.train_set[j].data.reshape(-1), j)
                hess *= self._monotonise_hess(self.train_set[j].data.reshape(-1), j)
            return grad, hess

        return wrapper

    def _monotonise_with_softplus(self, x, j, force_cpu=False):
        """
        Monotonise the beta values with the softplus function.

        Parameters
        ----------
        beta : numpy array
            Beta values.
        j : int
            Booster index.
        force_cpu : bool, optional (default=False)
            Whether to force the computation on the CPU.

        Returns
        -------
        beta : numpy array
            Monotonised beta values.
        """
        if self.rum_structure[j]["boosting_params"].get("monotone_constraints", []):
            for monotone_constraint in self.rum_structure[j]["boosting_params"][
                "monotone_constraints"
            ]:
                if monotone_constraint != 0:
                    if (self.device is not None) and (not force_cpu):
                        monotone_constraint = (
                            torch.tensor(monotone_constraint).to(self.device).float()
                        )
                        x = monotone_constraint * torch.nn.functional.softplus(
                            x * monotone_constraint, beta=self.softplus_strength
                        )
                    else:
                        x = monotone_constraint * safe_softplus(
                            x * monotone_constraint, beta=self.softplus_strength
                        )
        return x

    def _monotonise_grad_softplus(self, x, j):
        """
        Monotonise the gradient of the beta values
        with the gradient of the softplus function.
        The gradient of the softplus function is the sigmoid function.

        Parameters
        ----------
        x : numpy array
            Data points.
        j : int
            Booster index.

        Returns
        -------
        beta : numpy array
            Monotonised beta values.
        """
        grad_x = x.astype(float)
        if self.rum_structure[j]["boosting_params"].get("monotone_constraints", []):
            for monotone_constraint in self.rum_structure[j]["boosting_params"][
                "monotone_constraints"
            ]:
                if monotone_constraint != 0:
                    if self.device is not None:
                        raw_preds = torch.tensor(
                            self.boosters[j]._Booster__inner_predict_buffer[0]
                        )
                        grad_x *= torch.nn.functional.sigmoid(
                            self.softplus_strength * raw_preds * monotone_constraint
                        ).numpy()
                    else:
                        raw_preds = self.boosters[j]._Booster__inner_predict_buffer[0]
                        grad_x *= expit(
                            self.softplus_strength * raw_preds * monotone_constraint
                        )
        return grad_x

    def _monotonise_hess_softplus(self, x, j):
        """
        Monotonise the hessian of the beta values
        with the hessian of the softplus function.
        The hessian of the softplus function is the sigmoid function times the sigmoid function minus 1.

        Parameters
        ----------
        x : numpy array
            Data points.
        j : int
            Booster index.

        Returns
        -------
        beta : numpy array
            Monotonised beta values.
        """
        hess_xx = x.astype(float) ** 2
        if self.rum_structure[j]["boosting_params"].get("monotone_constraints", []):
            for monotone_constraint in self.rum_structure[j]["boosting_params"][
                "monotone_constraints"
            ]:
                if monotone_constraint != 0:
                    if self.device is not None:
                        raw_preds = torch.tensor(
                            self.boosters[j]._Booster__inner_predict_buffer[0]
                        )
                        hess_xx *= (
                            torch.nn.functional.sigmoid(
                                self.softplus_strength * raw_preds * monotone_constraint
                            ).numpy()
                            * (
                                1
                                - torch.nn.functional.sigmoid(
                                    self.softplus_strength
                                    * raw_preds
                                    * monotone_constraint
                                )
                            ).numpy()
                            * self.softplus_strength
                            * monotone_constraint
                        )
                    else:
                        raw_preds = self.boosters[j]._Booster__inner_predict_buffer[0]
                        hess_xx *= (
                            expit(
                                self.softplus_strength * raw_preds * monotone_constraint
                            )
                            * (
                                1
                                - expit(
                                    self.softplus_strength
                                    * raw_preds
                                    * monotone_constraint
                                )
                            )
                            * self.softplus_strength
                            * monotone_constraint
                        )
        return hess_xx

    def _monotonise_with_relu(self, beta, j, force_cpu=False):
        """
        Monotonise the beta values with the ReLU function.

        Parameters
        ----------
        beta : numpy array
            Beta values.
        j : int
            Booster index.
        force_cpu : bool, optional (default=False)
            Whether to force the computation on the CPU.

        Returns
        -------
        beta : numpy array
            Monotonised beta values.
        """
        if self.rum_structure[j]["boosting_params"].get("monotone_constraints", []):
            for monotone_constraint in self.rum_structure[j]["boosting_params"][
                "monotone_constraints"
            ]:
                if monotone_constraint != 0:
                    if (self.device is not None) and (not force_cpu):
                        monotone_constraint = (
                            torch.tensor(monotone_constraint).to(self.device).float()
                        )
                        beta = monotone_constraint * torch.nn.functional.relu(
                            beta * monotone_constraint
                        )
                    else:
                        beta = monotone_constraint * np.maximum(
                            0, beta * monotone_constraint
                        )
        return beta

    def _monotonise_grad_relu(self, x, j):
        """
        Monotonise the gradient of the beta values
        with the gradient of the ReLU function.

        Parameters
        ----------
        x : numpy array
            Data points.
        j : int
            Booster index.

        Returns
        -------
        beta : numpy array
            Monotonised beta values.
        """
        grad_x = x.astype(float)
        if self.rum_structure[j]["boosting_params"].get("monotone_constraints", []):
            for monotone_constraint in self.rum_structure[j]["boosting_params"][
                "monotone_constraints"
            ]:
                if monotone_constraint != 0:
                    if self.device is not None:
                        raw_preds = torch.tensor(
                            self.boosters[j]._Booster__inner_predict_buffer[0]
                        )
                        grad_x *= (
                            raw_preds * monotone_constraint > 0
                        ).numpy() * monotone_constraint
                    else:
                        raw_preds = self.boosters[j]._Booster__inner_predict_buffer[0]
                        grad_x *= (
                            raw_preds * monotone_constraint > 0
                        ) * monotone_constraint
        return grad_x

    def _monotonise_hess_relu(self, x, j):
        """
        Monotonise the hessian of the beta values
        with the hessian of the ReLU function.

        Parameters
        ----------
        x : numpy array
            Data points.
        j : int
            Booster index.

        Returns
        -------
        beta : numpy array
            Monotonised beta values.
        """
        hess_xx = x.astype(float) ** 2
        return hess_xx

    @multiply_grad_hess_by_data
    def f_obj(self, _, __):
        """
        Objective function of the binary classification boosters, but based on softmax predictions.

        Returns
        -------
        grad : numpy array
            The gradient with the cross-entropy loss function. It is the predictions minus the binary labels (if it is used for the jth booster, labels will be 1 if the chosen class is j, 0 if it is any other classes).
        hess : numpy array
            The hessian with the cross-entropy loss function (second derivative approximation rather than the hessian). Calculated as factor * preds * (1 - preds).
        """
        j = self._current_j  # jth booster
        # call torch functions if required
        if self.device is not None:
            if self.torch_compile:
                grad, hess = _f_obj_torch_compiled(
                    self._preds,
                    self.num_classes,
                    self.rum_structure[j]["utility"],
                    self.labels_j[self.subsample_idx, :],
                )
            else:
                grad, hess = _f_obj_torch(
                    self._preds,
                    self.num_classes,
                    self.rum_structure[j]["utility"],
                    self.labels_j[self.subsample_idx, :],
                )

            grad = grad.cpu().type(torch.float32).numpy()
            hess = hess.cpu().type(torch.float32).numpy()

            if self.subsample_idx.shape[0] < self.num_obs[0]:
                grad_rescaled = np.zeros(
                    (self.num_obs[0], len(self.rum_structure[j]["utility"]))
                )
                hess_rescaled = np.zeros(
                    (self.num_obs[0], len(self.rum_structure[j]["utility"]))
                )
                grad_rescaled[self.subsample_idx.cpu().numpy(), :] = grad.reshape(
                    -1, len(self.rum_structure[j]["utility"]), order="F"
                )
                hess_rescaled[self.subsample_idx.cpu().numpy(), :] = hess.reshape(
                    -1, len(self.rum_structure[j]["utility"]), order="F"
                )

                grad = grad_rescaled
                hess = hess_rescaled

            if (
                not self.rum_structure[j]["shared"]
                and len(self.rum_structure[j]["utility"]) > 1
            ):
                grad = grad.sum(axis=1)
                hess = hess.sum(axis=1)
            elif len(self.rum_structure[j]["variables"]) < len(
                self.rum_structure[j]["utility"]
            ):
                grad = grad.T.reshape(
                    int(
                        len(self.rum_structure[j]["utility"])
                        / len(self.rum_structure[j]["variables"])
                    ),
                    -1,
                ).sum(axis=0)
                hess = hess.T.reshape(
                    int(
                        len(self.rum_structure[j]["utility"])
                        / len(self.rum_structure[j]["variables"])
                    ),
                    -1,
                ).sum(axis=0)

            return grad.reshape(-1, order="F"), hess.reshape(-1, order="F")

        preds = self._preds[:, self.rum_structure[j]["utility"]]
        factor = self.num_classes / (
            self.num_classes - 1
        )  # factor to correct redundancy (see Friedmann, Greedy Function Approximation)
        eps = 1e-6
        labels = self.labels_j[:, self.rum_structure[j]["utility"]][
            self.subsample_idx, :
        ]
        grad = preds - labels
        hess = np.maximum(
            factor * preds * (1 - preds), eps
        )  # truncate low values to avoid numerical errors

        if self.subsample_idx.size < self.num_obs[0]:
            grad_rescaled = np.zeros(
                (self.num_obs[0], len(self.rum_structure[j]["utility"]))
            )
            hess_rescaled = np.zeros(
                (self.num_obs[0], len(self.rum_structure[j]["utility"]))
            )
            grad_rescaled[self.subsample_idx, :] = grad
            hess_rescaled[self.subsample_idx, :] = hess

            grad = grad_rescaled
            hess = hess_rescaled

        if (
            not self.rum_structure[j]["shared"]
            and len(self.rum_structure[j]["utility"]) > 1
        ):
            grad = grad.sum(axis=1)
            hess = hess.sum(axis=1)
        elif len(self.rum_structure[j]["variables"]) < len(
            self.rum_structure[j]["utility"]
        ):
            grad = grad.T.reshape(
                int(
                    len(self.rum_structure[j]["utility"])
                    / len(self.rum_structure[j]["variables"])
                ),
                -1,
            ).sum(axis=0)
            hess = hess.T.reshape(
                int(
                    len(self.rum_structure[j]["utility"])
                    / len(self.rum_structure[j]["variables"])
                ),
                -1,
            ).sum(axis=0)

        return grad.reshape(-1, order="F"), hess.reshape(-1, order="F")

    @multiply_grad_hess_by_data
    def f_obj_binary(self, _, __):
        """
        Objective function of the binary classification boosters, for binary classification.

        Returns
        -------
        grad : numpy array
            The gradient with the cross-entropy loss function. It is the predictions minus the binary labels.
        hess : numpy array
            The hessian with the cross-entropy loss function (second derivative approximation rather than the hessian).
        """
        j = self._current_j  # jth booster
        if self.device is not None:
            if self.torch_compile:
                grad, hess = _f_obj_binary_torch_compiled(
                    self._preds,
                    self.labels[self.subsample_idx],
                )
            else:
                grad, hess = _f_obj_binary_torch(
                    self._preds,
                    self.labels[self.subsample_idx],
                )

            grad = grad.cpu().type(torch.float32).numpy()
            hess = hess.cpu().type(torch.float32).numpy()

            if self.subsample_idx.shape[0] < self.num_obs[0]:
                grad_rescaled = np.zeros(
                    (self.num_obs[0], len(self.rum_structure[j]["utility"]))
                )
                hess_rescaled = np.zeros(
                    (self.num_obs[0], len(self.rum_structure[j]["utility"]))
                )
                grad_rescaled[self.subsample_idx.cpu().numpy(), :] = grad.reshape(
                    -1, len(self.rum_structure[j]["utility"]), order="F"
                )
                hess_rescaled[self.subsample_idx.cpu().numpy(), :] = hess.reshape(
                    -1, len(self.rum_structure[j]["utility"]), order="F"
                )

                grad = grad_rescaled
                hess = hess_rescaled

            return grad.reshape(-1, order="F"), hess.reshape(-1, order="F")

        preds = self._preds.reshape(-1)
        labels = self.labels[self.subsample_idx]
        grad = preds - labels
        hess = preds * (1 - preds)

        if self.subsample_idx.size < self.num_obs[0]:

            grad_rescaled = np.zeros(
                (self.num_obs[0], len(self.rum_structure[j]["utility"]))
            )
            hess_rescaled = np.zeros(
                (self.num_obs[0], len(self.rum_structure[j]["utility"]))
            )
            grad_rescaled[self.subsample_idx, :] = grad
            hess_rescaled[self.subsample_idx, :] = hess

            grad = grad_rescaled
            hess = hess_rescaled

        return grad.reshape(-1, order="F"), hess.reshape(-1, order="F")

    @multiply_grad_hess_by_data
    def f_obj_mse(self, _, __):
        """
        Objective function of the boosters, for regression with mean squared error.

        Returns
        -------
        grad : numpy array
            The gradient with the mean squared error loss function. It is the predictions minus the binary labels.
        hess : numpy array
            The hessian with the mean squared error loss function (second derivative approximation rather than the hessian).
        """
        j = self._current_j
        if self.device is not None:
            if self.torch_compile:
                grad, hess = _f_obj_mse_torch_compiled(
                    self._preds,
                    self.labels[self.subsample_idx],
                )
            else:
                grad, hess = _f_obj_mse_torch(
                    self._preds,
                    self.labels[self.subsample_idx],
                )

            grad = grad.cpu().type(torch.float32).numpy()
            hess = hess.cpu().type(torch.float32).numpy()

            if self.subsample_idx.shape[0] < self.num_obs[0]:
                grad_rescaled = np.zeros(
                    (self.num_obs[0], len(self.rum_structure[j]["utility"]))
                )
                hess_rescaled = np.zeros(
                    (self.num_obs[0], len(self.rum_structure[j]["utility"]))
                )
                grad_rescaled[self.subsample_idx.cpu().numpy(), :] = grad.reshape(
                    -1, len(self.rum_structure[j]["utility"]), order="F"
                )
                hess_rescaled[self.subsample_idx.cpu().numpy(), :] = hess.reshape(
                    -1, len(self.rum_structure[j]["utility"]), order="F"
                )

                grad = grad_rescaled
                hess = hess_rescaled

            return grad.reshape(-1, order="F"), hess.reshape(-1, order="F")

        preds = self._preds
        targets = self.labels[self.subsample_idx]
        grad = 2 * (preds - targets)
        hess = 2 * np.ones_like(preds)

        if self.subsample_idx.size < self.num_obs[0]:
            grad_rescaled = np.zeros(
                (self.num_obs[0], len(self.rum_structure[j]["utility"]))
            )
            hess_rescaled = np.zeros(
                (self.num_obs[0], len(self.rum_structure[j]["utility"]))
            )
            grad_rescaled[self.subsample_idx, :] = grad
            hess_rescaled[self.subsample_idx, :] = hess

            grad = grad_rescaled
            hess = hess_rescaled

        return grad.reshape(-1, order="F"), hess.reshape(-1, order="F")

    @multiply_grad_hess_by_data
    def f_obj_nest(self, _, __):
        """
        Objective function of the binary classification boosters, for a nested rumboost.

        Returns
        -------
        grad : numpy array
            The gradient with the cross-entropy loss function and nested probabilities.
        hess : numpy array
            The hessian with the cross-entropy loss function and nested probabilities (second derivative approximation rather than the hessian).
        """
        j = self._current_j  # jth booster
        if self.device is not None:
            if self.torch_compile:
                grad, hess = _f_obj_nested_torch_compiled(
                    self.labels[self.subsample_idx],
                    self.preds_i_m,
                    self.preds_m,
                    self.num_classes,
                    self.mu,
                    self.nests,
                    self.device,
                    self.rum_structure[j]["utility"],
                )
            else:
                grad, hess = _f_obj_nested_torch(
                    self.labels[self.subsample_idx],
                    self.preds_i_m,
                    self.preds_m,
                    self.num_classes,
                    self.mu,
                    self.nests,
                    self.device,
                    self.rum_structure[j]["utility"],
                )

            grad = grad.cpu().type(torch.float32).numpy()
            hess = hess.cpu().type(torch.float32).numpy()

            if self.subsample_idx.shape[0] < self.num_obs[0]:
                grad_rescaled = np.zeros(
                    (self.num_obs[0], len(self.rum_structure[j]["utility"]))
                )
                hess_rescaled = np.zeros(
                    (self.num_obs[0], len(self.rum_structure[j]["utility"]))
                )
                grad_rescaled[self.subsample_idx.cpu().numpy(), :] = grad
                hess_rescaled[self.subsample_idx.cpu().numpy(), :] = hess

                grad = grad_rescaled
                hess = hess_rescaled

            if not self.rum_structure[j]["shared"]:
                grad = grad.sum(axis=1)
                hess = hess.sum(axis=1)
            elif len(self.rum_structure[j]["variables"]) < len(
                self.rum_structure[j]["utility"]
            ):
                grad = grad.T.reshape(
                    int(
                        len(self.rum_structure[j]["utility"])
                        / len(self.rum_structure[j]["variables"])
                    ),
                    -1,
                ).sum(axis=0)
                hess = hess.T.reshape(
                    int(
                        len(self.rum_structure[j]["utility"])
                        / len(self.rum_structure[j]["variables"])
                    ),
                    -1,
                ).sum(axis=0)

            return grad.T.reshape(-1), hess.T.reshape(-1)

        j = self._current_j
        label = self.labels[self.subsample_idx]
        factor = self.num_classes / (self.num_classes - 1)
        label_nest = self.nest_alt[label]

        shared_ensemble = np.array(self.rum_structure[j]["utility"])

        pred_i_m = self.preds_i_m[
            :, shared_ensemble
        ]  # pred of alternative j knowing nest m
        pred_m = self.preds_m[
            :, self.nest_alt[shared_ensemble]
        ]  # prediction of choosing nest m

        grad = np.where(
            label[:, None] == shared_ensemble[None, :],
            -self.mu[self.nest_alt[shared_ensemble]] * (1 - pred_i_m)
            - pred_i_m * (1 - pred_m),
            np.where(
                label_nest[:, None] == self.nest_alt[shared_ensemble][None, :],
                self.mu[self.nest_alt[shared_ensemble]] * pred_i_m
                - pred_i_m * (1 - pred_m),
                pred_i_m * pred_m,
            ),
        )
        hess = np.where(
            label[:, None] == shared_ensemble[None, :],
            -self.mu[self.nest_alt[shared_ensemble]]
            * pred_i_m
            * (1 - pred_i_m)
            * (1 - self.mu[self.nest_alt[shared_ensemble]] - pred_m)
            + pred_i_m**2 * pred_m * (1 - pred_m),
            np.where(
                label_nest[:, None] == self.nest_alt[shared_ensemble][None, :],
                -self.mu[self.nest_alt[shared_ensemble]]
                * pred_i_m
                * (1 - pred_i_m)
                * (1 - self.mu[self.nest_alt[shared_ensemble]] - pred_m)
                + pred_i_m**2 * pred_m * (1 - pred_m),
                -pred_i_m
                * pred_m
                * (
                    -self.mu[self.nest_alt[shared_ensemble]] * (1 - pred_i_m)
                    - pred_i_m * (1 - pred_m)
                ),
            ),
        )
        hess *= factor

        if self.subsample_idx.size < self.num_obs[0]:
            grad_rescaled = np.zeros(
                (self.num_obs[0], len(self.rum_structure[j]["utility"]))
            )
            hess_rescaled = np.zeros(
                (self.num_obs[0], len(self.rum_structure[j]["utility"]))
            )
            grad_rescaled[self.subsample_idx, :] = grad
            hess_rescaled[self.subsample_idx, :] = hess

            grad = grad_rescaled
            hess = hess_rescaled

        if not self.rum_structure[j]["shared"]:
            grad = grad.sum(axis=1)
            hess = hess.sum(axis=1)
        elif len(self.rum_structure[j]["variables"]) < len(
            self.rum_structure[j]["utility"]
        ):
            grad = grad.T.reshape(
                int(
                    len(self.rum_structure[j]["utility"])
                    / len(self.rum_structure[j]["variables"])
                ),
                -1,
            ).sum(axis=0)
            hess = hess.T.reshape(
                int(
                    len(self.rum_structure[j]["utility"])
                    / len(self.rum_structure[j]["variables"])
                ),
                -1,
            ).sum(axis=0)

        grad = grad.T.reshape(-1)
        hess = hess.T.reshape(-1)

        return grad, hess

    @multiply_grad_hess_by_data
    def f_obj_cross_nested(self, _, __):
        """
        Objective function of the binary classification boosters, for a cross-nested rumboost.

        Returns
        -------
        grad : numpy array
            The gradient with the cross-entropy loss function and cross-nested probabilities.
        hess : numpy array
            The hessian with the cross-entropy loss function and cross-nested probabilities (second derivative approximation rather than the hessian).
        """
        j = self._current_j  # jth booster
        if self.device is not None:
            if self.torch_compile:
                grad, hess = _f_obj_cross_nested_torch_compiled(
                    self.labels[self.subsample_idx],
                    self.preds_i_m,
                    self.preds_m,
                    self._preds,
                    self.num_classes,
                    self.mu,
                    self.device,
                    self.rum_structure[j]["utility"],
                )
            else:
                grad, hess = _f_obj_cross_nested_torch(
                    self.labels[self.subsample_idx],
                    self.preds_i_m,
                    self.preds_m,
                    self._preds,
                    self.num_classes,
                    self.mu,
                    self.device,
                    self.rum_structure[j]["utility"],
                )

            grad = grad.cpu().type(torch.float32).numpy()
            hess = hess.cpu().type(torch.float32).numpy()

            if self.subsample_idx.shape[0] < self.num_obs[0]:
                grad_rescaled = np.zeros(
                    (self.num_obs[0], len(self.rum_structure[j]["utility"]))
                )
                hess_rescaled = np.zeros(
                    (self.num_obs[0], len(self.rum_structure[j]["utility"]))
                )
                grad_rescaled[self.subsample_idx.cpu().numpy(), :] = grad.squeeze()
                hess_rescaled[self.subsample_idx.cpu().numpy(), :] = hess.squeeze()

                grad = grad_rescaled
                hess = hess_rescaled

            if not self.rum_structure[j]["shared"]:
                grad = grad.sum(axis=1)
                hess = hess.sum(axis=1)
            elif len(self.rum_structure[j]["variables"]) < len(
                self.rum_structure[j]["utility"]
            ):
                grad = grad.T.reshape(
                    int(
                        len(self.rum_structure[j]["utility"])
                        / len(self.rum_structure[j]["variables"])
                    ),
                    -1,
                ).sum(axis=0)
                hess = hess.T.reshape(
                    int(
                        len(self.rum_structure[j]["utility"])
                        / len(self.rum_structure[j]["variables"])
                    ),
                    -1,
                ).sum(axis=0)

            return grad.T.reshape(-1), hess.T.reshape(-1)

        j = self._current_j
        label = self.labels[self.subsample_idx]
        data_idx = np.arange(self.preds_i_m.shape[0])
        factor = self.num_classes / (self.num_classes - 1)

        pred_j_m = self.preds_i_m[
            :, self.rum_structure[j]["utility"], :
        ]  # pred of alternative j knowing nest m
        pred_i_m = self.preds_i_m[data_idx, label, :][
            :, None, :
        ]  # prediction of choice i knowing nest m
        pred_m = self.preds_m[:, None, :]  # prediction of choosing nest m
        pred_i = self._preds[data_idx, label][:, None, None]  # pred of choice i
        pred_j = self._preds[:, self.rum_structure[j]["utility"]][
            :, :, None
        ]  # pred of alt j

        pred_i_m_pred_m = pred_i_m * pred_m
        pred_j_m_pred_m = pred_j_m * pred_m
        pred_i_m_pred_i = pred_i_m * pred_i
        pred_i_m_squared = pred_i_m**2
        pred_j_m_squared = pred_j_m**2
        pred_i_squared = pred_i**2
        pred_j_m_pred_j_squared = (pred_j_m - pred_j) ** 2
        pred_i_m_1_mu_mu_pred_i = pred_i_m * (1 - self.mu) + self.mu - pred_i
        pred_j_m_1_mu_pred_j = pred_j_m * (1 - self.mu) - pred_j

        mu_squared = self.mu**2

        d_pred_i_Vi = np.sum(
            (pred_i_m_pred_m * pred_i_m_1_mu_mu_pred_i), axis=2, keepdims=True
        )  # first derivative of pred i with respect to Vi
        d_pred_i_Vj = np.sum(
            (pred_i_m_pred_m * pred_j_m_1_mu_pred_j), axis=2, keepdims=True
        )  # first derivative of pred i with respect to Vj
        d_pred_j_Vj = np.sum(
            (pred_j_m_pred_m * (pred_j_m_1_mu_pred_j + self.mu)),
            axis=2,
            keepdims=True,
        )  # first derivative of pred j with respect to Vj

        mu_3pim2_3pim_2pimpi_pi = self.mu * (
            -3 * pred_i_m_squared + 3 * pred_i_m + 2 * (pred_i_m_pred_i - pred_i)
        )
        pim2_2pimpi_pi2_dpiVi = (
            pred_i_m_squared - 2 * pred_i_m_pred_i + pred_i_squared - d_pred_i_Vi
        )
        mu2_2pim2_3pim_1 = mu_squared * (2 * pred_i_m_squared - 3 * pred_i_m + 1)
        mu2_pjm = mu_squared * (-pred_j_m)
        mu_pjm2_pjm = self.mu * (-pred_j_m_squared + pred_j_m)

        d2_pred_i_Vi = np.sum(
            (
                pred_i_m_pred_m
                * (mu2_2pim2_3pim_1 + mu_3pim2_3pim_2pimpi_pi + pim2_2pimpi_pi2_dpiVi)
            ),
            axis=2,
            keepdims=True,
        )
        d2_pred_i_Vj = np.sum(
            (
                pred_i_m_pred_m
                * (mu2_pjm + mu_pjm2_pjm + pred_j_m_pred_j_squared - d_pred_j_Vj)
            ),
            axis=2,
            keepdims=True,
        )

        # print(d2_pred_i_Vi)
        mask = np.array(self.rum_structure[j]["utility"])[None, :] == label[:, None]
        grad = np.where(
            mask[:, :, None],
            ((-1 / pred_i) * d_pred_i_Vi),
            ((-1 / pred_i) * d_pred_i_Vj),
        )
        hess = np.where(
            mask[:, :, None],
            ((-1 / pred_i**2) * (d2_pred_i_Vi * pred_i - d_pred_i_Vi**2)),
            ((-1 / pred_i**2) * (d2_pred_i_Vj * pred_i - d_pred_i_Vj**2)),
        )
        hess *= factor

        if self.subsample_idx.size < self.num_obs[0]:
            grad_rescaled = np.zeros(
                (self.num_obs[0], len(self.rum_structure[j]["utility"]))
            )
            hess_rescaled = np.zeros(
                (self.num_obs[0], len(self.rum_structure[j]["utility"]))
            )
            grad_rescaled[self.subsample_idx, :] = grad.squeeze()
            hess_rescaled[self.subsample_idx, :] = hess.squeeze()

            grad = grad_rescaled
            hess = hess_rescaled

        if not self.rum_structure[j]["shared"]:
            grad = grad.sum(axis=1)
            hess = hess.sum(axis=1)
        elif len(self.rum_structure[j]["variables"]) < len(
            self.rum_structure[j]["utility"]
        ):
            grad = grad.T.reshape(
                int(
                    len(self.rum_structure[j]["utility"])
                    / len(self.rum_structure[j]["variables"])
                ),
                -1,
            ).sum(axis=0)
            hess = hess.T.reshape(
                int(
                    len(self.rum_structure[j]["utility"])
                    / len(self.rum_structure[j]["variables"])
                ),
                -1,
            ).sum(axis=0)

        grad = grad.T.reshape(-1)
        hess = hess.T.reshape(-1)

        return grad, hess

    @multiply_grad_hess_by_data
    def f_obj_proportional_odds(self, _, __):
        """
        Objective function for a proportional odds rumboost.

        Returns
        -------
        grad : numpy array
            The gradient with the cross-entropy loss function and proportional odds probabilities.
        hess : numpy array
            The hessian with the cross-entropy loss function and proportional odds probabilities (second derivative approximation rather than the hessian).
        """
        if self.device is not None:
            if self.torch_compile:
                grad, hess = _f_obj_proportional_odds_torch_compiled(
                    self.labels[self.subsample_idx],
                    self._preds,
                    self.raw_preds[self.subsample_idx],
                    self.thresholds,
                )
            else:
                grad, hess = _f_obj_proportional_odds_torch(
                    self.labels[self.subsample_idx],
                    self._preds,
                    self.raw_preds[self.subsample_idx],
                    self.thresholds,
                )

            grad = grad.cpu().type(torch.float32).numpy()
            hess = hess.cpu().type(torch.float32).numpy()

            if self.subsample_idx.shape[0] < self.num_obs[0]:
                grad_rescaled = np.zeros((self.num_obs[0], len(self.thresholds) - 1))
                hess_rescaled = np.zeros((self.num_obs[0], len(self.thresholds) - 1))
                grad_rescaled[self.subsample_idx.cpu().numpy(), :] = grad
                hess_rescaled[self.subsample_idx.cpu().numpy(), :] = hess

                grad = grad_rescaled
                hess = hess_rescaled

            return grad.reshape(-1, order="F"), hess.reshape(-1, order="F")

        labels = self.labels[self.subsample_idx]
        preds = self._preds
        thresholds = self.thresholds
        thresholds = np.append(thresholds, 0)
        raw_preds = self.raw_preds[self.subsample_idx]
        grad = np.where(
            labels == 0,
            expit(raw_preds - thresholds[0]),
            np.where(
                labels == len(thresholds),
                preds[:, -1] - 1,
                expit(raw_preds - thresholds[labels - 1])
                + expit(raw_preds - thresholds[labels])
                - 1,
            ),
        )
        hess = np.where(
            labels == 0,
            preds[:, 0] * expit(raw_preds - thresholds[0]),
            np.where(
                labels == len(thresholds),
                preds[:, -1] * (1 - preds[:, -1]),
                expit(raw_preds - thresholds[labels - 1])
                * (1 - expit(raw_preds - thresholds[labels - 1]))
                + expit(raw_preds - thresholds[labels])
                * (1 - expit(raw_preds - thresholds[labels])),
            ),
        )

        if self.subsample_idx.size < self.num_obs[0]:
            grad_rescaled = np.zeros((self.num_obs[0], len(self.thresholds) - 1))
            hess_rescaled = np.zeros((self.num_obs[0], len(self.thresholds) - 1))
            grad_rescaled[self.subsample_idx, :] = grad
            hess_rescaled[self.subsample_idx, :] = hess

            grad = grad_rescaled
            hess = hess_rescaled

        return grad, hess

    @multiply_grad_hess_by_data
    def f_obj_coral(self, _, __):
        """
        Objective function for a coral rumboost.

        Returns
        -------
        grad : numpy array
            The gradient of the weighted binary cross-entropy loss function with coral probabilities.
        hess : numpy array
            The hessian of the weighted binary cross-entropy loss function with coral probabilities (second derivative approximation rather than the hessian).
        """
        if self.device is not None:
            if self.torch_compile:
                grad, hess = _f_obj_coral_torch_compiled(
                    self.labels[self.subsample_idx],
                    self.raw_preds[self.subsample_idx],
                    self.thresholds,
                )
            else:
                grad, hess = _f_obj_coral_torch(
                    self.labels[self.subsample_idx],
                    self.raw_preds[self.subsample_idx],
                    self.thresholds,
                )

            grad = grad.cpu().type(torch.float32).numpy()
            hess = hess.cpu().type(torch.float32).numpy()

            if self.subsample_idx.shape[0] < self.num_obs[0]:
                grad_rescaled = np.zeros((self.num_obs[0], len(self.thresholds) - 1))
                hess_rescaled = np.zeros((self.num_obs[0], len(self.thresholds) - 1))
                grad_rescaled[self.subsample_idx.cpu().numpy(), :] = grad
                hess_rescaled[self.subsample_idx.cpu().numpy(), :] = hess

                grad = grad_rescaled
                hess = hess_rescaled

            return grad.reshape(-1, order="F"), hess.reshape(-1, order="F")

        labels = self.labels[self.subsample_idx]
        thresholds = self.thresholds
        raw_preds = self.raw_preds[self.subsample_idx][:, None]
        sigmoids = expit(raw_preds - thresholds)
        classes = np.arange(thresholds.shape[0])

        grad = np.sum(sigmoids - (labels[:, None] > classes[None, :]), axis=1)

        hess = np.sum(sigmoids * (1 - sigmoids), axis=1)

        if self.subsample_idx.size < self.num_obs[0]:

            grad_rescaled = np.zeros((self.num_obs[0], len(self.thresholds) - 1))
            hess_rescaled = np.zeros((self.num_obs[0], len(self.thresholds) - 1))
            grad_rescaled[self.subsample_idx, :] = grad
            hess_rescaled[self.subsample_idx, :] = hess

            grad = grad_rescaled
            hess = hess_rescaled

        return grad, hess

    @multiply_grad_hess_by_data
    def f_obj_corn(self, _, __):
        """
        Objective function for an ordinal corn rumboost.

        Returns
        -------
        grad : numpy array
            The gradient of the conditional binary cross-entropy loss function with corn probabilities.
        hess : numpy array
            The hessian of the conditional binary cross-entropy loss function with corn probabilities (second derivative approximation rather than the hessian).
        """
        j = self._current_j  # jth booster
        if self.device is not None:
            if self.torch_compile:
                grad, hess = _f_obj_corn_torch_compiled(
                    self.labels[self.subsample_idx],
                    self.raw_preds.view(-1, self.num_obs[0]).T[self.subsample_idx, :],
                    j,
                )
            else:
                grad, hess = _f_obj_corn_torch(
                    self.labels[self.subsample_idx],
                    self.raw_preds.view(-1, self.num_obs[0]).T[self.subsample_idx, :],
                    j,
                )

            grad = grad.cpu().type(torch.float32).numpy()
            hess = hess.cpu().type(torch.float32).numpy()

            if self.subsample_idx.shape[0] < self.num_obs[0]:
                grad_rescaled = np.zeros((self.num_obs[0], len(self.thresholds) - 1))
                hess_rescaled = np.zeros((self.num_obs[0], len(self.thresholds) - 1))
                grad_rescaled[self.subsample_idx.cpu().numpy(), :] = grad
                hess_rescaled[self.subsample_idx.cpu().numpy(), :] = hess

                grad = grad_rescaled
                hess = hess_rescaled

            if (
                not self.rum_structure[j]["shared"]
                and len(self.rum_structure[j]["utility"]) > 1
            ):
                grad = grad.sum(axis=1)
                hess = hess.sum(axis=1)
            elif len(self.rum_structure[j]["variables"]) < len(
                self.rum_structure[j]["utility"]
            ):
                grad = grad.T.reshape(
                    int(
                        len(self.rum_structure[j]["utility"])
                        / len(self.rum_structure[j]["variables"])
                    ),
                    -1,
                ).sum(axis=0)
                hess = hess.T.reshape(
                    int(
                        len(self.rum_structure[j]["utility"])
                        / len(self.rum_structure[j]["variables"])
                    ),
                    -1,
                ).sum(axis=0)

            return grad.reshape(-1, order="F"), hess.reshape(-1, order="F")

        labels = self.labels[self.subsample_idx]
        raw_preds = self.raw_preds.reshape((self.num_obs[0], -1), order="F")[
            self.subsample_idx, :
        ][:, j]
        sigmoids = expit(raw_preds)

        grad = np.where(j < labels, 0, sigmoids - (j > labels))

        hess = np.where(j < labels, 1, sigmoids * (1 - sigmoids))

        if self.subsample_idx.shape[0] < self.num_obs[0]:
            grad_rescaled = np.zeros((self.num_obs[0], len(self.thresholds) - 1))
            hess_rescaled = np.zeros((self.num_obs[0], len(self.thresholds) - 1))
            grad_rescaled[self.subsample_idx.cpu().numpy(), :] = grad
            hess_rescaled[self.subsample_idx.cpu().numpy(), :] = hess

            grad = grad_rescaled
            hess = hess_rescaled

        if (
            not self.rum_structure[j]["shared"]
            and len(self.rum_structure[j]["utility"]) > 1
        ):
            grad = grad.sum(axis=1)
            hess = hess.sum(axis=1)
        elif len(self.rum_structure[j]["variables"]) < len(
            self.rum_structure[j]["utility"]
        ):
            grad = grad.T.reshape(
                int(
                    len(self.rum_structure[j]["utility"])
                    / len(self.rum_structure[j]["variables"])
                ),
                -1,
            ).sum(axis=0)
            hess = hess.T.reshape(
                int(
                    len(self.rum_structure[j]["utility"])
                    / len(self.rum_structure[j]["variables"])
                ),
                -1,
            ).sum(axis=0)

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
        utilities : bool, optional (default=False)
            If True, return raw utilities for each class, without generating probabilities.

        Returns
        -------
        result : numpy array, scipy.sparse or list of scipy.sparse
            Prediction result.
            Can be sparse or a list of sparse objects (each element represents predictions for one class) for feature contributions (when ``pred_contrib=True``).
        """
        # compute utilities with corresponding features
        # split data
        new_data, _ = self._preprocess_data(
            data, return_data=True, free_raw_data=False, construct_datasets=True
        )

        if self.device is not None:

            self.device = torch.device(self.device)
            if isinstance(self.mu, np.ndarray):
                self.mu = (
                    torch.from_numpy(self.mu).type(torch.float32).to(device=self.device)
                )
            if isinstance(self.alphas, np.ndarray):
                self.alphas = (
                    torch.from_numpy(self.alphas)
                    .type(torch.float32)
                    .to(device=self.device)
                )

            booster_preds = [
                (
                    torch.from_numpy(
                        self._monotonise_beta(
                            booster.predict(
                                new_data[k].get_data(),
                                start_iteration,
                                num_iteration,
                                raw_score,
                                pred_leaf,
                                pred_contrib,
                                data_has_header,
                                validate_features,
                            ),
                            k,
                            force_cpu=True,
                        )
                        * new_data[k].get_data().reshape(-1)
                        + self.asc[k]
                    )
                    .type(torch.float32)
                    .to(device=self.device)
                    if self.boost_from_parameter_space[k]
                    else torch.from_numpy(
                        booster.predict(
                            new_data[k].get_data(),
                            start_iteration,
                            num_iteration,
                            raw_score,
                            pred_leaf,
                            pred_contrib,
                            data_has_header,
                            validate_features,
                        )
                    )
                    .type(torch.float32)
                    .to(device=self.device)
                )
                for k, booster in enumerate(self.boosters)
            ]
            if self.num_classes == 2:
                raw_preds = torch.zeros(
                    data.num_data(),
                    device=self.device,
                    dtype=torch.float32,
                )
            else:
                raw_preds = torch.zeros(
                    data.num_data() * self.num_classes,
                    device=self.device,
                    dtype=torch.float32,
                )

            # reshaping raw predictions into num_obs, num_classes array
            for j, struct in enumerate(self.rum_structure):
                if not struct["shared"]:
                    raw_preds[
                        struct["utility"][0]
                        * data.num_data() : (struct["utility"][-1] + 1)
                        * data.num_data()
                    ] += booster_preds[j].repeat(len(struct["utility"]))
                else:
                    raw_preds[
                        struct["utility"][0]
                        * data.num_data() : (struct["utility"][-1] + 1)
                        * data.num_data()
                    ] += booster_preds[j].repeat(
                        int(len(struct["utility"]) / len(struct["variables"]))
                    )

            raw_preds = raw_preds.view(-1, data.num_data()).T
            if self.torch_compile:
                preds, _, _ = _inner_predict_torch_compiled(
                    raw_preds,
                    self.device,
                    self.nests,
                    self.mu,
                    self.alphas,
                    utilities,
                    self.num_classes,
                    self.ord_model,
                    self.thresholds,
                )
            else:
                preds, _, _ = _inner_predict_torch(
                    raw_preds,
                    self.device,
                    self.nests,
                    self.mu,
                    self.alphas,
                    utilities,
                    self.num_classes,
                    self.ord_model,
                    self.thresholds,
                )
            self._free_dataset_memory()

            return preds

        booster_preds = [
            (
                self._monotonise_beta(
                    booster.predict(
                        new_data[k].get_data(),
                        start_iteration,
                        num_iteration,
                        raw_score,
                        pred_leaf,
                        pred_contrib,
                        data_has_header,
                        validate_features,
                    ),
                    k,
                )
                * new_data[k].get_data().reshape(-1)
                + self.asc[k]
                if self.boost_from_parameter_space[k]
                else booster.predict(
                    new_data[k].get_data(),
                    start_iteration,
                    num_iteration,
                    raw_score,
                    pred_leaf,
                    pred_contrib,
                    data_has_header,
                    validate_features,
                )
            )
            for k, booster in enumerate(self.boosters)
        ]

        if self.num_classes == 2:
            raw_preds = np.zeros(data.num_data())
        else:
            raw_preds = np.zeros((data.num_data() * self.num_classes))
        # reshaping raw predictions into num_obs, num_classes array
        for j, struct in enumerate(self.rum_structure):
            idx_ranges = range(
                struct["utility"][0] * data.num_data(),
                (struct["utility"][-1] + 1) * data.num_data(),
            )
            if not struct["shared"]:
                raw_preds[idx_ranges] += np.tile(
                    booster_preds[j], len(struct["utility"])
                )
            else:
                raw_preds[idx_ranges] += np.tile(
                    booster_preds[j],
                    int(len(struct["utility"]) / len(struct["variables"])),
                )
        raw_preds = raw_preds.reshape((data.num_data(), -1), order="F")

        if self.num_classes == 1 and not self.ord_model:  # regression
            return raw_preds

        if not utilities:
            # compute nested probabilities. pred_i_m is predictions of choosing i knowing m, pred_m is prediction of choosing nest m and preds is pred_i_m * pred_m
            if self.nests:
                preds, _, _ = nest_probs(
                    raw_preds, mu=self.mu, nests=self.nests, nest_alt=self.nest_alt
                )

                return preds

            # compute cross-nested probabilities. pred_i_m is predictions of choosing i knowing m, pred_m is prediction of choosing nest m and preds is pred_i_m * pred_m
            if self.alphas is not None:
                preds, _, _ = cross_nested_probs(
                    raw_preds, mu=self.mu, alphas=self.alphas
                )

                return preds

            # ordinal preds
            if self.thresholds is not None:
                if self.ord_model in ["proportional_odds", "coral"]:
                    preds = threshold_preds(raw_preds, self.thresholds)

                return preds

            if self.ord_model == "corn":
                preds = corn_preds(raw_preds)

                return preds

            if self.num_classes == 2:  # binary classification
                preds = expit(raw_preds)

                return preds

            # softmax
            preds = softmax(raw_preds, axis=1)
            return preds

        return raw_preds

    def _inner_predict(
        self,
        data_idx: int = 0,
        utilities: bool = False,
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

        Returns
        -------
        result : numpy array, scipy.sparse or list of scipy.sparse
            Prediction result.
            Can be sparse or a list of sparse objects (each element represents predictions for one class) for feature contributions (when ``pred_contrib=True``).
        """
        # using pytorch if required
        if self.device is not None:
            if data_idx == 0:
                raw_preds = self.raw_preds.view(-1, self.num_obs[data_idx]).T[
                    self.subsample_idx, :
                ]
            else:
                if (
                    self.num_classes == 2
                ):  # binary classification requires only one column
                    raw_preds = torch.zeros(self.num_obs[data_idx], device=self.device)
                else:
                    raw_preds = torch.zeros(
                        self.num_obs[data_idx] * self.num_classes,
                        device=self.device,
                        dtype=torch.float32,
                    )
                for j, _ in enumerate(self.rum_structure):
                    if self.boost_from_parameter_space[j]:
                        raw_preds[
                            self.booster_valid_idx[j][0] : self.booster_valid_idx[j][1]
                        ] += (
                            (
                                torch.from_numpy(
                                    self._monotonise_beta(
                                        self.boosters[j]._Booster__inner_predict(
                                            data_idx
                                        ),
                                        j,
                                        force_cpu=True,
                                    )
                                    * self.valid_sets[data_idx - 1][j].data.reshape(-1)
                                    + self.asc[j]
                                )
                            )
                            .type(dtype=torch.float32)
                            .to(device=self.device)
                        )
                    elif not self.rum_structure[j]["shared"]:
                        raw_preds[
                            self.booster_valid_idx[j][0] : self.booster_valid_idx[j][1]
                        ] += (
                            torch.from_numpy(
                                np.tile(
                                    self.boosters[j]._Booster__inner_predict(data_idx),
                                    len(self.rum_structure[j]["utility"]),
                                )
                            )
                            .type(dtype=torch.float32)
                            .to(device=self.device)
                        )
                    else:
                        raw_preds[
                            self.booster_valid_idx[j][0] : self.booster_valid_idx[j][1]
                        ] += (
                            torch.from_numpy(
                                np.tile(
                                    self.boosters[j]._Booster__inner_predict(data_idx),
                                    int(
                                        len(self.rum_structure[j]["utility"])
                                        / len(self.rum_structure[j]["variables"])
                                    ),
                                )
                            )
                            .type(dtype=torch.float32)
                            .to(device=self.device)
                        )
                raw_preds = raw_preds.view(-1, self.num_obs[data_idx]).T[
                    self.subsample_idx_valid, :
                ]
            if self.torch_compile:
                preds, pred_i_m, pred_m = _inner_predict_torch_compiled(
                    raw_preds,
                    self.device,
                    self.nests,
                    self.mu,
                    self.alphas,
                    utilities,
                    self.num_classes,
                    self.ord_model,
                    self.thresholds,
                )
            else:
                preds, pred_i_m, pred_m = _inner_predict_torch(
                    raw_preds,
                    self.device,
                    self.nests,
                    self.mu,
                    self.alphas,
                    utilities,
                    self.num_classes,
                    self.ord_model,
                    self.thresholds,
                )

            if self.mu is not None and data_idx == 0:
                self.preds_i_m = pred_i_m
                self.preds_m = pred_m

            return preds

        # reshaping raw predictions into num_obs, num_classes array
        if data_idx == 0:
            raw_preds = self.raw_preds.reshape((self.num_obs[data_idx], -1), order="F")[
                self.subsample_idx, :
            ]
        else:
            if self.num_classes == 2:  # binary classification requires only one column
                raw_preds = np.zeros(self.num_obs[data_idx])
            else:
                raw_preds = np.zeros(self.num_obs[data_idx] * self.num_classes)
            for j, _ in enumerate(self.rum_structure):
                if self.boost_from_parameter_space[j]:
                    raw_preds[self.booster_valid_idx[j]] += (
                        self._monotonise_beta(
                            self.boosters[j]._Booster__inner_predict(data_idx), j
                        )
                        * self.valid_sets[data_idx - 1][j].data.reshape(-1)
                        + self.asc[j]
                    )

                elif not self.rum_structure[j]["shared"]:
                    raw_preds[self.booster_valid_idx[j]] += np.tile(
                        self.boosters[j]._Booster__inner_predict(data_idx),
                        len(self.rum_structure[j]["utility"]),
                    )
                else:
                    raw_preds[self.booster_valid_idx[j]] += np.tile(
                        self.boosters[j]._Booster__inner_predict(data_idx),
                        int(
                            len(self.rum_structure[j]["utility"])
                            / len(self.rum_structure[j]["variables"])
                        ),
                    )
            raw_preds = raw_preds.reshape((self.num_obs[data_idx], -1), order="F")

        if self.num_classes == 1 and not self.ord_model:  # regression
            return raw_preds

        if not utilities:
            # compute nested probabilities. pred_i_m is predictions of choosing i knowing m, pred_m is prediction of choosing nest m and preds is pred_i_m * pred_m
            if self.nests:
                preds, pred_i_m, pred_m = nest_probs(
                    raw_preds, mu=self.mu, nests=self.nests, nest_alt=self.nest_alt
                )
                if data_idx == 0:
                    self.preds_i_m = pred_i_m
                    self.preds_m = pred_m

                return preds

            # compute cross-nested probabilities. pred_i_m is predictions of choosing i knowing m, pred_m is prediction of choosing nest m and preds is pred_i_m * pred_m
            if self.alphas is not None:
                preds, pred_i_m, pred_m = cross_nested_probs(
                    raw_preds, mu=self.mu, alphas=self.alphas
                )
                if data_idx == 0:
                    self.preds_i_m = pred_i_m
                    self.preds_m = pred_m

                return preds

            if self.thresholds is not None:
                if self.ord_model in ["proportional_odds", "coral"]:
                    preds = threshold_preds(raw_preds, self.thresholds)

                return preds

            if self.ord_model == "corn":
                preds = corn_preds(raw_preds)

                return preds

            # binary classification
            if self.num_classes == 2:
                preds = expit(raw_preds)
                return preds

            # softmax

            preds = softmax(raw_preds, axis=1)
            return preds

        return raw_preds

    def _preprocess_data(
        self,
        data: Dataset,
        reduced_valid_set=None,
        return_data: bool = False,
        free_raw_data: bool = True,
        construct_datasets: bool = False,
        predictor: list[Booster] = None,
    ):
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
        free_raw_data : bool, optional (default = False)
            If True, the raw data is freed after the datasets are created.
        construct_datasets : bool, optional (default = False)
            If True, the datasets are constructed.
        predictor : list of Booster, optional (default=None)
            The list of predictors to be used for the datasets.

        Returns
        -------
        train_set_J : list[Dataset]
            If return_data is True, return a list with J preprocessed datasets corresponding to the J boosters.
        reduced_valid_sets_J : list[Dataset] or list[list[Dataset]], optional
            If return_data is True, and reduced_valid_set is not None, return one or several list(s) with J preprocessed validation sets corresponding to the J boosters.
        """
        train_set_J = []
        data_to_repeat = []
        reduced_valid_sets_J = []
        self.valid_labels = []

        # to access data
        data.construct()
        self.num_obs = [data.num_data()]  # saving number of observations
        if reduced_valid_set:
            for valid_set in reduced_valid_set:
                valid_set.construct()
                self.num_obs.append(valid_set.num_data())
                self.valid_labels.append(
                    valid_set.get_label().astype(np.int32)
                )  # saving labels

        self.labels = data.get_label().astype(np.int32)  # saving labels
        self.labels_j = (
            self.labels[:, None] == np.array(range(self.num_classes))[None, :]
        ).astype(np.int8)
        val_labels_j = [
            (val_labs[:, None] == np.array(range(self.num_classes))[None, :]).astype(
                np.int8
            )
            for val_labs in self.valid_labels
        ]

        # loop over all J utilities
        for j, struct in enumerate(self.rum_structure):
            if struct:
                if "variables" in struct:
                    train_set_j_data = data.get_data()[
                        struct["variables"]
                    ]  # only relevant features for the jth booster

                    if struct["shared"] == True:
                        new_label = self.labels_j[
                            :, struct["utility"][: len(struct["variables"])]
                        ].reshape(-1, order="F")
                        feature_names = "auto"
                    else:
                        new_label = self.labels_j[:, 0].reshape(-1, order="F")
                        feature_names = struct["variables"]
                    train_set_j = Dataset(
                        train_set_j_data.values.reshape(
                            (len(new_label), -1), order="A"
                        ),
                        label=new_label,
                        free_raw_data=free_raw_data,
                    )  # create and build dataset
                    categorical_feature = struct["boosting_params"].get(
                        "categorical_feature", "auto"
                    )
                    predictor_j = predictor[j] if predictor else None
                    train_set_j._update_params(
                        struct["boosting_params"]
                    )._set_predictor(predictor_j).set_feature_name(
                        feature_names
                    ).set_categorical_feature(
                        categorical_feature
                    )
                    if construct_datasets:
                        train_set_j.construct()

                    if reduced_valid_set is not None:
                        reduced_valid_sets_j = []
                        for i, valid_set in enumerate(reduced_valid_set):
                            # create and build validation sets
                            valid_set.construct()
                            valid_set_j_data = valid_set.get_data()[
                                struct["variables"]
                            ]  # only relevant features for the jth booster

                            if struct["shared"] == True:
                                label_valid = val_labels_j[i][
                                    :, struct["utility"][: len(struct["variables"])]
                                ].reshape(-1, order="F")
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
                                free_raw_data=free_raw_data,
                                reference=train_set_j,
                            )  # create and build dataset
                            valid_set_j._update_params(
                                struct["boosting_params"]
                            )._set_predictor(predictor_j)
                            if construct_datasets:
                                valid_set_j.construct()

                            reduced_valid_sets_j.append(valid_set_j)

                else:
                    # if no alternative specific datasets
                    new_label = np.where(data.get_label() == j, 1, 0)
                    train_set_j = Dataset(
                        data.get_data(), label=new_label, free_raw_data=free_raw_data
                    )
                    if reduced_valid_set is not None:
                        reduced_valid_sets_j = reduced_valid_set[:]

            # store all training and potential validation sets in lists
            train_set_J.append(train_set_j)
            if reduced_valid_set is not None:
                reduced_valid_sets_J.append(reduced_valid_sets_j)

        # store them in the RUMBoost object
        self.train_set = train_set_J
        self.valid_sets = np.array(reduced_valid_sets_J).T.tolist()
        if return_data:
            return train_set_J, reduced_valid_sets_J

    def _preprocess_valids(
        self, train_set: Dataset, params: dict, valid_sets=None, valid_names=None
    ):
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
        # initialise variables
        is_valid_contain_train = False
        train_data_name = "training"
        reduced_valid_sets = []
        name_valid_sets = []

        # finalise validation sets for training
        if valid_sets is not None:
            if isinstance(valid_sets, Dataset):
                valid_sets = [valid_sets]
            if isinstance(valid_names, str):
                valid_names = [valid_names]
            for i, valid_data in enumerate(valid_sets):
                if valid_data is train_set:
                    is_valid_contain_train = (
                        True  # store if train set is in validation set
                    )
                    if valid_names is not None:
                        train_data_name = valid_names[i]
                    continue
                if not isinstance(valid_data, Dataset):
                    raise TypeError("Training only accepts Dataset object")
                reduced_valid_sets.append(valid_data._update_params(params))
                if valid_names is not None and len(valid_names) > i:
                    name_valid_sets.append(valid_names[i])
                else:
                    name_valid_sets.append(f"valid_{i}")

        return (
            reduced_valid_sets,
            name_valid_sets,
            is_valid_contain_train,
            train_data_name,
        )

    def _construct_boosters(
        self,
        train_data_name="Training",
        is_valid_contain_train=False,
        name_valid_sets=["Valid_0"],
    ):
        """Construct boosters of the RUMBoost model with corresponding set of parameters, training datasets, and validation sets and store them in the RUMBoost object.

        Parameters
        ----------
        train_data_name: str, optional (default = 'Training')
            Name of training dataset.
        is_valid_contain_train: bool
            True if the training set is in the validation sets.
        name_valid_sets : list[str]
            List of names of validation sets.
        init_models : list of Booster, optional (default=None)
            The list of initial models to be used for the boosters.
        """
        booster_train_idx = []
        if self.valid_sets is not None:
            booster_valid_idx = []
        for j, struct in enumerate(self.rum_structure):
            # construct booster and perform basic preparations
            try:
                params = copy.deepcopy(struct["boosting_params"])
                if self.boost_from_parameter_space[j]:
                    params["monotone_constraints"] = [
                        0
                    ]  # in case of boosting from parameter, monotonicity is removed
                booster = Booster(
                    params=params,
                    train_set=self.train_set[j],
                )
                if self.device is not None:
                    print(
                        "[RUMBoost] [Warning]: assuming utility is contiguous and sorted"
                    )
                    idx_ranges = [
                        struct["utility"][0] * self.num_obs[0],
                        (struct["utility"][-1] + 1) * self.num_obs[0],
                    ]
                else:
                    idx_ranges = range(
                        struct["utility"][0] * self.num_obs[0],
                        (struct["utility"][-1] + 1) * self.num_obs[0],
                    )
                booster_train_idx.append(idx_ranges)
                if is_valid_contain_train:
                    booster.set_train_data_name(train_data_name)
                if self.valid_sets is not None:
                    for i, (valid_set, name_valid_set) in enumerate(
                        zip(self.valid_sets, name_valid_sets)
                    ):
                        if self.device is not None:
                            idx_ranges = [
                                struct["utility"][0] * self.num_obs[i + 1],
                                (struct["utility"][-1] + 1) * self.num_obs[i + 1],
                            ]
                        else:
                            idx_ranges = range(
                                struct["utility"][0] * self.num_obs[i + 1],
                                (struct["utility"][-1] + 1) * self.num_obs[i + 1],
                            )
                        booster_valid_idx.append(idx_ranges)
                        booster.add_valid(valid_set[j], name_valid_set)
            finally:
                self.train_set[j]._reverse_update_params()
                for valid_set in self.valid_sets:
                    valid_set[j]._reverse_update_params()

            # initialise and store boosters in a list
            booster.best_iteration = 0
            self._append(booster)

        # initialise RUMBoost score information
        self.booster_train_idx = booster_train_idx
        self.booster_valid_idx = (
            booster_valid_idx  # not working for more than one valid set for now
        )
        self.best_iteration = 0
        self.best_score = 1e6
        self.best_score_train = 1e6

    def _find_best_booster(self):
        """Find the best booster(s) to update the raw_predictions accordingly."""

        gains = np.array(self._current_gains)
        max_boost = np.minimum(self.max_booster_to_update, len(gains))
        best_boosters = np.argsort(gains)[::-1]

        selected_indices = []
        utility_counter = np.zeros(self.num_classes)
        # trying to select boosters to update all utilities
        for b in best_boosters:
            utilities = self.rum_structure[b]["utility"]
            if (utility_counter[utilities] > utility_counter.min()).all():
                continue
            utility_counter[utilities] += 1
            selected_indices.append(b)
            if len(selected_indices) == max_boost:
                break

        return selected_indices

    def _update_raw_preds(self, best_boosters):
        """Update the raw predictions of the RUMBoost model with the best booster(s) to update.

        Parameters
        ----------
        best_boosters : list
            List of the best booster(s) to update.
        """
        for j in best_boosters:
            if self.boost_from_parameter_space[j]:
                self._current_preds[j] = self._monotonise_beta(
                    self._current_preds[j], j, force_cpu=True
                ) * self.train_set[j].data.reshape(-1)
                +self.asc[j]
            if self.device is not None:
                self.raw_preds[
                    self.booster_train_idx[j][0] : self.booster_train_idx[j][1]
                ] += (
                    torch.from_numpy(self._current_preds[j])
                    .type(torch.float32)
                    .to(self.device)
                )
            else:
                self.raw_preds[self.booster_train_idx[j]] += self._current_preds[j]

    def _update_mu_or_alphas(self, res, optimise_mu, optimise_alphas, alpha_shape):
        """Update mu or alphas for the cross-nested model.

        Parameters
        ----------
        res : OptimizeResult
            The result of the optimization.
        optimise_mu : bool
            If True, optimise mu.
        optimise_alphas : bool
            If True, optimise alphas.
        alpha_shape : tuple
            The shape of the alphas array.
        """
        if optimise_mu:
            if self.device is not None:
                self.mu.add_(
                    0.1
                    * (
                        torch.tensor(
                            res.x[: len(self.mu)],
                            device=self.device,
                            dtype=torch.float32,
                        )
                        - self.mu
                    )
                )
            else:
                self.mu += 0.1 * (res.x[: len(self.mu)] - self.mu)
        if optimise_alphas:
            if self.device is not None:
                alphas_opt = torch.tensor(
                    res.x[len(self.mu) :].reshape(alpha_shape),
                    device=self.device,
                    dtype=torch.float32,
                )
                alphas_opt = alphas_opt / alphas_opt.sum(dim=1)[:, None]
                self.alphas.add_(0.1 * (alphas_opt - self.alphas))
            else:
                alphas_opt = res.x[len(self.mu) :].reshape(alpha_shape)
                alphas_opt = alphas_opt / alphas_opt.sum(axis=1)[:, None]
                self.alphas += 0.1 * (alphas_opt - self.alphas)

    def _rollback_boosters(self, unchosen_boosters):
        """Rollback the unchosen booster(s)."""
        for j in unchosen_boosters:
            if self._current_gains[j] > 0:
                self.boosters[j].rollback_one_iter()

    def _append(self, booster: Booster) -> None:
        """Add a booster to RUMBoost."""
        self.boosters.append(booster)

    def _free_dataset_memory(self):
        """Remove datasets from RUMBoost object."""
        self.train_set = None
        self.valid_sets = None
        self.labels_j = None
        self.labels = None

    def _from_dict(self, models: Dict[str, Any]) -> None:
        """Load RUMBoost from dict."""
        self.boosters = []
        for model_str in models["boosters"]:
            self._append(Booster(model_str=model_str))
        if "attributes" in models:
            self.__dict__.update(models["attributes"])

    def _to_dict(
        self, num_iteration: Optional[int], start_iteration: int, importance_type: str
    ) -> Dict[str, Any]:
        """Serialize RUMBoost to dict."""
        models_str = []
        for booster in self.boosters:
            models_str.append(
                booster.model_to_string(
                    num_iteration=num_iteration,
                    start_iteration=start_iteration,
                    importance_type=importance_type,
                )
            )
        if self.device is not None:
            if self.labels_j is not None:
                labels_j_numpy = [l.cpu().numpy().tolist() for l in self.labels_j]
            else:
                labels_j_numpy = []
            if self.valid_labels is not None:
                valid_labels_numpy = [
                    v.cpu().numpy().tolist() for v in self.valid_labels
                ]
            else:
                valid_labels_numpy = []
            rumb_to_dict = {
                "boosters": models_str,
                "attributes": {
                    "best_iteration": self.best_iteration,
                    "best_score": float(self.best_score),
                    "best_score_train": float(self.best_score_train),
                    "alphas": (
                        self.alphas.cpu().numpy().tolist()
                        if self.alphas is not None
                        else None
                    ),
                    "mu": (
                        self.mu.cpu().numpy().tolist() if self.mu is not None else None
                    ),
                    "nests": self.nests,
                    "nest_alt": self.nest_alt,
                    "ord_model": self.ord_model,
                    "thresholds": (
                        self.thresholds.tolist()
                        if self.thresholds is not None
                        else None
                    ),
                    "num_classes": self.num_classes,
                    "num_obs": self.num_obs,
                    "labels": self.labels.cpu().numpy().tolist(),
                    "labels_j": labels_j_numpy,
                    "valid_labels": valid_labels_numpy,
                    "device": "cuda" if self.device.type == "cuda" else "cpu",
                    "torch_compile": self.torch_compile,
                    "rum_structure": self.rum_structure,
                    "boost_from_parameter_space": self.boost_from_parameter_space,
                    "asc": self.asc.tolist() if self.asc is not None else None,
                },
            }
        else:
            if self.labels_j is not None:
                labels_j_list = [l.tolist() for l in self.labels_j]
            else:
                labels_j_list = []
            if self.valid_labels is not None:
                valid_labs = [v.tolist() for v in self.valid_labels]
            else:
                valid_labs = []
            rumb_to_dict = {
                "boosters": models_str,
                "attributes": {
                    "best_iteration": self.best_iteration,
                    "best_score": float(self.best_score),
                    "best_score_train": float(self.best_score_train),
                    "alphas": self.alphas.tolist() if self.alphas is not None else None,
                    "mu": self.mu.tolist() if self.mu is not None else None,
                    "nests": self.nests,
                    "nest_alt": self.nest_alt,
                    "ord_model": self.ord_model,
                    "thresholds": (
                        self.thresholds.tolist()
                        if self.thresholds is not None
                        else None
                    ),
                    "num_classes": self.num_classes,
                    "num_obs": self.num_obs,
                    "labels": self.labels.tolist(),
                    "labels_j": labels_j_list,
                    "valid_labels": valid_labs,
                    "device": None,
                    "torch_compile": self.torch_compile,
                    "rum_structure": self.rum_structure,
                    "boost_from_parameter_space": self.boost_from_parameter_space,
                    "asc": self.asc.tolist() if self.asc is not None else None,
                },
            }

        return rumb_to_dict

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
        importance_type: str = "split",
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
        return json.dumps(
            self._to_dict(num_iteration, start_iteration, importance_type)
        )

    def save_model(
        self,
        filename: Union[str, Path],
        num_iteration: Optional[int] = None,
        start_iteration: int = 0,
        importance_type: str = "split",
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
            json.dump(
                self._to_dict(num_iteration, start_iteration, importance_type), file
            )

        return self


def rum_train(
    train_set: Union[Dataset, dict[str, Any]],
    model_specification: dict[str, Any],
    num_boost_round: int = 100,
    valid_sets: Union[Optional[list[Dataset]], Optional[dict]] = None,
    valid_names: Optional[list[str]] = None,
    feval: Optional[
        Union[_LGBM_CustomMetricFunction, list[_LGBM_CustomMetricFunction]]
    ] = None,
    init_models: Optional[Union[list[str], list[Path], list[Booster]]] = None,
    feature_name: Union[list[str], str] = "auto",
    categorical_feature: Union[list[str], list[int], str] = "auto",
    keep_training_booster: bool = False,
    callbacks: Optional[list[Callable]] = None,
    torch_tensors: dict = None,
) -> RUMBoost:
    """Perform the RUM training with given parameters.

    Parameters
    ----------
    train_set : Dataset or dict[int, Any]
        Data to be trained on. Set free_raw_data=False when creating the dataset. If it is
        a dictionary, the key-value pairs should be:
            - "train_sets":  the corresponding preprocessed Dataset.
            - "num_data": the number of observations in the dataset.
            - "labels": the labels of the full dataset.
            - "labels_j": the labels of the dataset for each class (binary).
    model_specification : dict
        Dictionary specifying the model specification. The required keys are:

            - 'general_params': dict
                Dictionary containing the general parameters for the RUMBoost model.
                The dictionary can contain the following keys:
                    - 'num_iterations': int
                        Number of boosting iterations.
                    - 'num_classes': int
                        Number of classes. If equal to 2 and no additional keys are provided,
                        the model will perfomr binary classification. If greater than 2, the model
                        will perform multiclass classification. If equal to 1, the model will perform
                        regression with MSE (other loss functions will be implemented in the future).
                    - 'subsampling': float, optional (default = 1.0)
                        Subsample ratio of gradient when boosting
                    - 'subsampling_freq': int, optional (default = 0)
                        Subsample frequency.
                    - 'subsample_valid': float, optional (default = 1.0)
                        Subsample ratio of validation data.
                    - 'batch_size': int, optional (default = 0)
                        Batch size for the training. The batch size will override the subsampling.
                    - 'early_stopping_rounds': int, optional (default = None)
                        Activates early stopping. The model will train until the validation score stops improving.
                    - 'verbosity': int, optional (default = 1)
                        Verbosity of the model.
                    - 'verbose_interval': int, optional (default = 10)
                        Interval of the verbosity display. only used if verbosity > 1.
                    - 'max_booster_to_update': int, optional (default = num_classes)
                        Maximum number of boosters to update at each round.
                    - 'boost_from_parameter_space': list, optional (default = [])
                        If True, the boosting will be done in the parameter space, as opposed to the utility space.
                        It means that the GBDT algorithm will ouput betas instead of piece-wise constant utility
                        values. The resulting utility functions will be piece-wise linear. Monotonicity
                        is not guaranteed in this case and only one variable per parameter ensemble is allowed.
                    - 'optim_interval': int, optional (default = 1)
                        If all the ensembles are boosted from the parameter space, the interval at which the
                        ASCs are optimised. If 0, the ASCs are fixed.
                    - 'params_monotonic_function': str, optional (default = 'softplus')
                        The monotonicity function used to monotonise the boosters in parameter space.
                        Only used if boost_from_parameter_space is True.
                        "softplus" or "ReLu" are the available options.
                    - 'softplus_strength': float, optional (default = 1.0)
                        Strength of the softplus function used to monotonise the boosters in parameter space.
                        Only used if boost_from_parameter_space is True.
                    - 'save_model_interval': int, optional (default = 0)
                        The interval at which the model will be saved during training.
                    - 'eval_function': func (default = cross_entropy if multi-class, binary_log_loss if binary, mse if regression)
                        The evaluation function to be used.

            -'rum_structure' : list[dict[str, Any]]
                List of dictionaries specifying the variable used to create the parameter ensemble,
                and their monotonicity or interaction. The list must contain one dictionary for each parameter.
                Each dictionary has four required keys:
                    - 'utility': list of alternatives in which the parameter ensemble is used
                    - 'variables': list of columns from the train_set included in that parameter_ensemble
                    - 'boosting_params': dict
                        Dictionary containing the boosting parameters for the parameter ensemble.
                        If num_classes > 2, please specify params['objective'] = 'multiclass'.
                        These parameters are the same than Lightgbm parameters. More information here:
                        https://lightgbm.readthedocs.io/en/latest/Parameters.html.
                    - 'shared': bool
                        If True, the parameter ensemble is shared across all alternatives.

        The other keys are optional and can be:

            - 'nested_logit': dict
                Nested logit model specification. The dictionary must contain:
                    - 'mu': ndarray
                        An array of mu values, the scaling parameters, for each nest.
                        The first value of the array correspond to nest 0, and so on.
                        By default, the value of mu is 1 and is optimised through scipy.minimize.
                        Mu is competing against other parameter ensembles at each round to be
                        selected as the updated parameter ensemble.
                    - 'nests': dict
                        A dictionary representing the nesting structure.
                        Keys are nests, and values are the the list of alternatives in the nest.
                        For example {0: [0, 1], 1: [2, 3]} means that alternative 0 and 1
                        are in nest 0, and alternative 2 and 3 are in nest 1.
                    - 'optimise_mu': bool or list[bool], optional (default = True)
                        If True, the mu values are optimised through scipy.minimize.
                        If a list of booleans, the length must be equal to the number of nests.
                        By example, [True, False] means that mu_0 is optimised and mu_1 is fixed.
                    - 'optim_interval': int, optional (default = 20)
                        Interval at which the mu values are optimised.

            - 'cross_nested_logit': dict
                Cross-nested logit model specification. The dictionary must contain:
                    - 'mu': ndarray
                        An array of mu values, the scaling parameters, for each nest.
                        The first value of the array correspond to nest 0, and so on.
                    - 'alphas': ndarray
                        An array of J (alternatives) by M (nests).
                        alpha_jn represents the degree of membership of alternative j to nest n
                        By example, alpha_12 = 0.5 means that alternative one belongs 50% to nest 2.
                    - 'optimise_mu': bool or list[bool], optional (default = True)
                        If True, the mu values are optimised through scipy.minimize.
                        If a list of booleans, the length must be equal to the number of nests.
                        By example, [True, False] means that mu_0 is optimised and mu_1 is fixed.
                    - 'optimise_alphas': bool or ndarray[bool], optional (default = False)
                        If True, the alphas are optimised through scipy.minimize. This is not recommended
                        for high dimensionality datasets as it can be computationally expensive.
                        If an array of boolean, the array must have the same size than alphas. By example
                        if optimise_alphas_ij = True, alphas_ij will be optimised.
                    - 'optim_interval': int, optional (default = 20)
                        Interval at which the mu and/or alpha values are optimised.

            - 'ordinal_logit': dict
                Ordinal logit model specification.
                The dictionary must contain:
                    - 'model': str, default = 'proportional_odds'
                        The type of ordinal model. It can be:
                            - 'proportional_odds': the proportional odds model.
                            - 'unimodal_poisson': the unimodal poisson model.
                            - 'unimodal_binomial': the unimodal binomial model.
                            - 'coral': a rank consistent binary decomposition model.
                            - 'corn': a more advanced version of the coral model,
                                      allowing for alternative specific utility functions.
                    - 'optim_interval': int, optional (default = 20)
                        Interval at which the thresholds are optimised. This is only
                        used for the proportional odds and the coral models. If 0,
                        the thresholds are fixed.

    num_boost_round : int, optional (default = 100)
        Number of boosting iterations.
    valid_sets : list of Dataset, dict, or None, optional (default = None)
        List of data to be evaluated on during training. If the train_set is passed as
        already preprocessed, it is assumed that valid_sets are also preprocessed. Therefore it
        should be a dictionary following this structure:
            - "valid_sets":  a list of list of corresponding preprocessed validation Datasets.
            - "valid_labels": a list of the valid dataset labels.
            - "num_data": a list of the number of data in validation datasets.
        Note, you can pass several datasets for validation, but only the first one will be used for early stopping.
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
    init_models : list[str], list[pathlib.Path], list[Booster] or None, optional (default = None)
        List of filenames of LightGBM model or Booster instance used for continue training. There
        should be one model for each rum_structure dictionary.
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
    torch_tensors : dict, optional (default=None)
        If a dictionary is passed, torch.Tensors will be used for computing prediction, objective function and cross-entropy calculations.
        This require pytorch to be installed.
        The dictionary should follow the following form:

            'device': 'cpu', 'gpu' or 'cuda'
                The device on which the calculations will be performed.
            'torch_compile': bool
                If True, the prediction, objective function and cross-entropy calculations will be compiled with torch.compile.
                If used with GPU or cuda, it requires to be on a linux os.

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
    # check if general training parameters are in model specification and store them
    if "general_params" not in model_specification:
        raise ValueError("Model specification must contain general_params key")
    params = model_specification["general_params"]

    save_model_interval = params.get("save_model_interval", 0)

    # check if verbosity is in params
    for alias in _ConfigAliases.get("verbosity"):
        if alias in params:
            verbosity = params[alias]
            verbose_interval = params.get("verbose_interval", 10)

    # create predictor first
    params = copy.deepcopy(params)
    params = _choose_param_value(
        main_param_name="objective", params=params, default_value="multiclass"
    )
    fobj: Optional[_LGBM_CustomObjectiveFunction] = None
    if callable(params["objective"]):
        fobj = params["objective"]
        params["objective"] = "none"
    for alias in _ConfigAliases.get("num_iterations"):
        if alias in params:
            num_boost_round = params.pop(alias)
            _log_warning(f"Found `{alias}` in params. Will use it instead of argument")
    params["num_iterations"] = num_boost_round
    # setting early stopping via global params should be possible
    params = _choose_param_value(
        main_param_name="early_stopping_round", params=params, default_value=None
    )
    if params["early_stopping_round"] is None:
        params["early_stopping_round"] = 10000
    first_metric_only = params.get("first_metric_only", False)

    if num_boost_round <= 0:
        raise ValueError("num_boost_round should be greater than zero.")
    predictor: Optional[_InnerPredictor] = None
    if init_models is not None:
        predictor = []
        init_iteration = []
        for j, init_model in enumerate(init_models):
            if isinstance(init_model, (str, Path)):
                predictor.append(
                    _InnerPredictor.from_model_file(
                        model_file=init_model,
                        pred_parameter=model_specification["rum_structure"][j][
                            "boosting_params"
                        ],
                    )
                )
            elif isinstance(init_model, Booster):
                predictor.append(
                    _InnerPredictor.from_booster(
                        booster=init_model,
                        pred_parameter=dict(init_model.params, **params),
                    )
                )

    if predictor is not None:
        init_iteration = np.max(
            [predictor[j].current_iteration() for j in range(len(predictor))]
        )
    else:
        init_iteration = 0

    # process callbacks
    if callbacks is None:
        callbacks_set = set()
    else:
        for i, cb in enumerate(callbacks):
            cb.__dict__.setdefault("order", i - len(callbacks))
        callbacks_set = set(callbacks)

    callbacks_before_iter_set = {
        cb for cb in callbacks_set if getattr(cb, "before_iteration", False)
    }
    callbacks_after_iter_set = callbacks_set - callbacks_before_iter_set
    callbacks_before_iter = sorted(callbacks_before_iter_set, key=attrgetter("order"))
    callbacks_after_iter = sorted(callbacks_after_iter_set, key=attrgetter("order"))

    # construct rumboost object
    rumb = RUMBoost()

    if torch_tensors:
        if not torch_installed:
            raise ImportError(
                "PyTorch is not installed. Please install PyTorch to use torch tensors."
            )
        dev_str = torch_tensors.get("device", "cpu")
        rumb.torch_compile = torch_tensors.get("torch_compile", False)
        if dev_str == "cuda":
            rumb.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif dev_str == "gpu":
            rumb.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            rumb.device = torch.cpu.current_device()
        print(f"Using torch tensors on {rumb.device}")
    else:
        rumb.device = None
        rumb.torch_compile = False

    # check number of classes
    if "num_classes" not in params:
        raise ValueError(
            "Specify the number of classes in the dictionary of parameters with the key num_classes"
        )
    rumb.num_classes = params.get("num_classes")  # saving number of classes

    # checking model specification
    if "rum_structure" not in model_specification:
        raise ValueError("Model specification must contain rum_structure key")
    rumb.rum_structure = model_specification["rum_structure"]
    if not isinstance(rumb.rum_structure, list):
        raise ValueError("rum_structure must be a list")

    if (
        (
            "nested_logit" in model_specification
            and "cross_nested_logit" in model_specification
        )
        or (
            "nested_logit" in model_specification
            and "ordinal_logit" in model_specification
        )
        or (
            "cross_nested_logit" in model_specification
            and "ordinal_logit" in model_specification
        )
    ):
        raise ValueError(
            "Only one model specification can be used at a time. Choose between nested_logit, cross_nested_logit or ordinal_logit"
        )

    rumb.max_booster_to_update = params.get(
        "max_booster_to_update", len(rumb.rum_structure)
    )

    rumb.batch_size = params.get("batch_size", 0)

    # additional parameters to compete for best booster
    rumb.additional_params_idx = []

    # nested logit or cross nested logit or mnl
    if "nested_logit" in model_specification:
        if (
            "nests" not in model_specification["nested_logit"]
            or "mu" not in model_specification["nested_logit"]
        ):
            raise ValueError("Nested logit must contain nests key and mu key")

        # store the nests and mu values
        rumb.nests = model_specification["nested_logit"]["nests"]
        rumb.mu = copy.copy(model_specification["nested_logit"]["mu"])
        rumb.alphas = None
        f_obj = rumb.f_obj_nest

        # mu optimisation initialisaiton
        optimise_mu = copy.copy(
            model_specification["nested_logit"].get("optimise_mu", True)
        )
        if isinstance(optimise_mu, list):
            if len(optimise_mu) != len(rumb.mu):
                raise ValueError(
                    "The length of optimise_mu must be equal to the number of nests"
                )
            bounds = [(1, 2.5) if opt else (1, 1) for opt in optimise_mu]
            optimise_mu = True
            optim_interval = model_specification["nested_logit"].get(
                "optim_interval", 20
            )
        elif optimise_mu:
            bounds = [(1, 2.5)] * len(rumb.mu)
            optim_interval = model_specification["nested_logit"].get(
                "optim_interval", 20
            )
        else:
            bounds = None
            optimise_mu = False
        optimise_alphas = False
        alpha_shape = None

        # store the nest alternative. If torch tensors are used, nest alternatives
        # will be created later.
        if not torch_tensors:
            nest_alt = np.zeros(rumb.num_classes)
            for n, alts in rumb.nests.items():
                nest_alt[alts] = n
            rumb.nest_alt = nest_alt.astype(int)

        # not ordinal logit
        rumb.ord_model = None
        optimise_thresholds = False
        rumb.thresholds = None

        # type checks
        if not isinstance(rumb.mu, np.ndarray):
            raise ValueError("Mu must be a numpy array")
        if not isinstance(rumb.nests, dict):
            raise ValueError("Nests must be a dictionary")
    elif "cross_nested_logit" in model_specification:
        if (
            "alphas" not in model_specification["cross_nested_logit"]
            or "mu" not in model_specification["cross_nested_logit"]
        ):
            raise ValueError("Cross nested logit must contain alphas key and mu key")

        # store the mu and alphas values
        rumb.mu = copy.copy(model_specification["cross_nested_logit"]["mu"])
        rumb.alphas = copy.copy(model_specification["cross_nested_logit"]["alphas"])
        rumb.nests = None
        rumb.nest_alt = None
        f_obj = rumb.f_obj_cross_nested

        optimise_mu = copy.copy(
            model_specification["cross_nested_logit"].get("optimise_mu", True)
        )
        optimise_alphas = copy.copy(
            model_specification["cross_nested_logit"].get("optimise_alphas", False)
        )

        if isinstance(optimise_mu, list):
            if len(optimise_mu) != len(rumb.mu):
                raise ValueError(
                    "The length of optimise_mu must be equal to the number of nests"
                )
            bounds = [(1, 2.5) if opt else (1, 1) for opt in optimise_mu]
            optimise_mu = True
            optim_interval = model_specification["cross_nested_logit"].get(
                "optim_interval", 20
            )
        elif optimise_mu:
            bounds = [(1, 2.5)] * len(rumb.mu)
            optim_interval = model_specification["cross_nested_logit"].get(
                "optim_interval", 20
            )
        else:
            bounds = None
            optimise_mu = False

        if isinstance(optimise_alphas, np.ndarray):
            if optimise_alphas.shape != rumb.alphas.shape:
                raise ValueError(
                    "The shape of optimise_alphas must be equal to the shape of alphas"
                )

            if not bounds:
                bounds = []
            bounds.extend(
                [
                    (
                        (0, 1)
                        if alpha
                        else (rumb.alphas.flatten()[i], rumb.alphas.flatten()[i])
                    )
                    for i, alpha in enumerate(optimise_alphas.flatten().tolist())
                ]
            )
            optimise_alphas = True
            alpha_shape = rumb.alphas.shape
            optim_interval = model_specification["cross_nested_logit"].get(
                "optim_interval", 20
            )
        elif optimise_alphas:
            if not bounds:
                bounds = []
            bounds.extend([(0, 1)] * rumb.alphas.flatten().size)
            alpha_shape = rumb.alphas.shape
            optim_interval = model_specification["cross_nested_logit"].get(
                "optim_interval", 20
            )
        else:
            alpha_shape = None
            optimise_alphas = False

        # not ordinal logit
        rumb.ord_model = None
        optimise_thresholds = False
        rumb.thresholds = None

        # type checks
        if not isinstance(rumb.mu, np.ndarray):
            raise ValueError("Mu must be a numpy array")
        if not isinstance(rumb.alphas, np.ndarray):
            raise ValueError("Alphas must be a numpy array")
    elif "ordinal_logit" in model_specification:
        ordinal_model = model_specification["ordinal_logit"]["model"]
        optim_interval = model_specification["ordinal_logit"].get("optim_interval", 20)

        # some checks
        if ordinal_model not in [
            "proportional_odds",
            "unimodal_poisson",
            "unimodal_binomial",
            "coral",
            "corn",
        ]:
            raise ValueError(
                "Ordinal logit model must be proportional_odds, unimodal_poisson, unimodal_binomial, coral or corn."
            )
        if (
            set([u for rum in rumb.rum_structure for u in rum["utility"]]) != {0}
            and ordinal_model != "corn"
        ):  # check if all utility functions are the same
            raise ValueError(
                "Ordinal logit requires the same utility function for all parameter ensembles or use the CORN method"
            )
        if ordinal_model == "corn" and rumb.num_classes - 1 != len(
            set([u for rum in rumb.rum_structure for u in rum["utility"]])
        ):
            raise ValueError(
                "Ordinal logit with the CORN method requires a utility function for each class"
            )
        if rumb.num_classes == 2:
            raise ValueError("Ordinal logit requires a minimum of 3 classes")
        if "model" not in model_specification["ordinal_logit"]:
            raise ValueError("Ordinal logit must contain model key")

        if ordinal_model == "proportional_odds":
            f_obj = rumb.f_obj_proportional_odds
            optimise_thresholds = optim_interval > 0
            rumb.thresholds = np.arange(rumb.num_classes - 1)
            bounds = [(None, None)]
            bounds.extend([(0, None)] * (rumb.num_classes - 2))
            rumb.num_classes = 1
        elif ordinal_model == "unimodal_poisson":
            f_obj = rumb.f_obj_unimodal_poisson
            optimise_thresholds = False
            rumb.thresholds = None
            rumb.num_classes = 1
        elif ordinal_model == "unimodal_binomial":
            f_obj = rumb.f_obj_unimodal_binomial
            optimise_thresholds = False
            rumb.thresholds = None
            rumb.num_classes = 1
        elif ordinal_model == "coral":
            f_obj = rumb.f_obj_coral
            optimise_thresholds = optim_interval > 0
            rumb.thresholds = np.arange(rumb.num_classes - 1)
            bounds = [(None, None)]
            bounds.extend([(0, None)] * (rumb.num_classes - 2))
            rumb.num_classes = 1
        elif ordinal_model == "corn":
            f_obj = rumb.f_obj_corn
            optimise_thresholds = False
            rumb.thresholds = None
            rumb.num_classes = rumb.num_classes - 1
        rumb.ord_model = ordinal_model

        # no nesting structure
        optimise_mu = False
        optimise_alphas = False
        rumb.mu = None
        rumb.nests = None
        rumb.nest_alt = None
        rumb.alphas = None

    else:
        # no nesting structure nor ordinal logit
        rumb.mu = None
        rumb.nests = None
        rumb.nest_alt = None
        rumb.alphas = None
        rumb.ord_model = None
        rumb.thresholds = None
        if rumb.num_classes == 2:
            f_obj = rumb.f_obj_binary
        elif rumb.num_classes > 2:
            f_obj = rumb.f_obj
        elif rumb.num_classes == 1 and not rumb.ord_model:
            f_obj = rumb.f_obj_mse
        else:
            raise ValueError("Number of classes must be greater than 0")
        optimise_mu = False
        optimise_alphas = False
        optimise_thresholds = False

    if optimise_mu or optimise_alphas:
        opt_mu_or_alpha_idx = len(rumb.rum_structure)
        rumb.additional_params_idx.append(opt_mu_or_alpha_idx)
    else:
        opt_mu_or_alpha_idx = None

    if params.get("boost_from_parameter_space", []):
        if len(params["boost_from_parameter_space"]) != len(rumb.rum_structure):
            raise ValueError(
                "If specified, the length of boost_from_parameter_space must be equal to the number of parameter ensembles."
            )
        rumb.boost_from_parameter_space = params["boost_from_parameter_space"]
        if params.get("params_monotonic_function", "softplus") == "softplus":
            rumb._monotonise_beta = rumb._monotonise_with_softplus
            rumb._monotonise_grad = rumb._monotonise_grad_softplus
            rumb._monotonise_hess = rumb._monotonise_hess_softplus
            rumb.softplus_strength = params.get("softplus_strength", 1.0)
        else:
            rumb._monotonise_beta = rumb._monotonise_with_relu
            rumb._monotonise_grad = rumb._monotonise_grad_relu
            rumb._monotonise_hess = rumb._monotonise_hess_relu
        rumb.asc = np.zeros(len(rumb.rum_structure))

        optim_interval = params.get("optim_interval", 1)
        optimise_ascs = all(rumb.boost_from_parameter_space) and optim_interval > 0

        ascs = np.zeros(rumb.num_classes) if optimise_ascs else None

        # count the number of ensembles boosted from the parameter space
        # for each utility
        rumb.len_params_space = collections.Counter(
            [
                u * params["boost_from_parameter_space"][i]
                for i, rum in enumerate(rumb.rum_structure)
                for u in rum["utility"]
            ]
        )
        for j, struct in enumerate(rumb.rum_structure):
            if params["boost_from_parameter_space"][j]:
                # default regularisation on the hessian because
                # the sum of hessian can be 0 in the parameter space.
                if "lambda_l2" not in struct["boosting_params"]:
                    struct["boosting_params"]["lambda_l2"] = 1e-3
    else:
        rumb.boost_from_parameter_space = [False] * len(rumb.rum_structure)
        optimise_ascs = False
        rumb.asc = np.zeros(len(rumb.rum_structure))

    # check dataset and preprocess it
    if isinstance(train_set, dict):
        if "num_data" not in train_set:
            raise ValueError(
                "The dictionary must contain the number of observations with the key num_data"
            )
        if "labels" not in train_set:
            raise ValueError(
                "The dictionary must contain the labels with the key labels"
            )
        rumb.train_set = train_set[
            "train_sets"
        ]  # assign the J previously preprocessed datasets
        rumb.labels = train_set["labels"]
        if rumb.mu is None and rumb.ord_model is None and rumb.num_classes > 2:
            if not train_set.get("labels_j", None):
                rumb.labels_j = (
                    rumb.labels[:, None] == np.array(range(rumb.num_classes))[None, :]
                ).astype(np.int8)
            else:
                rumb.labels_j = train_set["labels_j"]
        else:
            rumb.labels_j = None
        rumb.num_obs = [train_set["num_data"]]
        if isinstance(valid_sets[0], dict):
            rumb.valid_sets = valid_sets[0]["valid_sets"]
            rumb.valid_labels = valid_sets[0]["valid_labels"]
            rumb.num_obs.extend(valid_sets[0]["num_data"])
        else:
            rumb.valid_sets = []
            rumb.valid_labels = []
        is_valid_contain_train = False
        train_data_name = "training"
        name_valid_sets = ["valid_0"]
        for j, train_set_j in enumerate(rumb.train_set):
            predictor_j = predictor[j] if predictor else None
            train_set_j.construct()
            train_set_j._update_params(
                rumb.rum_structure[j]["boosting_params"]
            )._set_predictor(predictor_j).set_feature_name(
                feature_name
            ).set_categorical_feature(
                categorical_feature
            )
            if rumb.valid_sets:
                for _, valid_set_j in enumerate(rumb.valid_sets):
                    valid_set_j[j]._update_params(
                        rumb.rum_structure[j]["boosting_params"]
                    )._set_predictor(predictor_j)
    elif not isinstance(train_set, Dataset):
        raise TypeError("Training only accepts Dataset object or dictionary")
    else:
        train_set._update_params(params)._set_predictor(None).set_feature_name(
            feature_name
        ).set_categorical_feature(categorical_feature)
        reduced_valid_sets, name_valid_sets, is_valid_contain_train, train_data_name = (
            rumb._preprocess_valids(train_set, params, valid_sets)
        )  # prepare validation sets
        free_raw_data = not any(rumb.boost_from_parameter_space)
        rumb._preprocess_data(
            train_set,
            reduced_valid_sets,
            predictor=predictor,
            free_raw_data=free_raw_data,
        )  # prepare J datasets with relevant features
        if rumb.mu is not None or rumb.ord_model or rumb.num_classes < 3:
            rumb.labels_j = None

    # create J boosters with corresponding params and datasets
    rumb._construct_boosters(train_data_name, is_valid_contain_train, name_valid_sets)

    # free datasets from memory
    if not any(rumb.boost_from_parameter_space):
        rumb.train_set = None
        rumb.valid_sets = None
    train_set = None
    valid_sets = None

    # convert a few numpy arrays to torch tensors if needed
    if torch_tensors:
        rumb.labels = torch.from_numpy(rumb.labels).type(torch.int16).to(rumb.device)
        if rumb.labels_j is not None:
            rumb.labels_j = (
                torch.from_numpy(rumb.labels_j).type(torch.int8).to(rumb.device)
            )
        if rumb.valid_labels:
            rumb.valid_labels = [
                torch.from_numpy(valid_labs).type(torch.int16).to(rumb.device)
                for valid_labs in rumb.valid_labels
            ]
        if rumb.mu is not None:
            rumb.mu = torch.from_numpy(rumb.mu).type(torch.float32).to(rumb.device)
        if rumb.alphas is not None:
            rumb.alphas = (
                torch.from_numpy(rumb.alphas).type(torch.float32).to(rumb.device)
            )

    if "subsampling" in params and rumb.batch_size > 0:
        subsample = params["subsampling"]
        if subsample == 1.0:
            subsample_freq = 0
            if torch_tensors:
                rumb.subsample_idx = torch.arange(
                    rumb.num_obs[0], dtype=torch.int32, device=rumb.device
                )
            else:
                rumb.subsample_idx = np.arange(rumb.num_obs[0])
        else:
            subsample_freq = params.get("subsampling_freq", 0)
            if torch_tensors:
                permutations = torch.randperm(
                    rumb.num_obs[0], device=rumb.device, dtype=torch.int32
                )
                rumb.subsample_idx = permutations[: int(subsample * rumb.num_obs[0])]
            else:
                rumb.subsample_idx = np.random.choice(
                    np.arange(rumb.num_obs[0]),
                    int(subsample * rumb.num_obs[0]),
                    replace=False,
                )
    elif rumb.batch_size > 0:
        subsample = 1.0
        subsample_freq = 0
        permutations = torch.randperm(
            rumb.num_obs[0], device=rumb.device, dtype=torch.int32
        )
        batches = torch.split(permutations, rumb.batch_size)
        rumb.subsample_idx = batches[0]
    else:
        subsample = 1.0
        subsample_freq = 0
        if torch_tensors:
            rumb.subsample_idx = torch.arange(
                rumb.num_obs[0], device=rumb.device, dtype=torch.int32
            )
        else:
            rumb.subsample_idx = np.arange(rumb.num_obs[0])

    if "subsample_valid" in params:
        subsample_valid = params["subsample_valid"]
        if torch_tensors:
            rumb.subsample_idx_valid = torch.randperm(
                rumb.num_obs[1], device=rumb.device
            )[: int(subsample_valid * rumb.num_obs[1])]
        else:
            rumb.subsample_idx_valid = np.random.choice(
                np.arange(rumb.num_obs[1]),
                int(subsample_valid * rumb.num_obs[1]),
                replace=False,
            )
    else:
        subsample_valid = 1.0
        if len(rumb.num_obs) > 1:  # if there are validation sets
            if torch_tensors:
                rumb.subsample_idx_valid = torch.arange(
                    rumb.num_obs[1], device=rumb.device, dtype=torch.int32
                )
            else:
                rumb.subsample_idx_valid = np.arange(rumb.num_obs[1])

    # setting up eval function
    if "eval_func" in params:
        rumb.eval_func = params["eval_func"]
    elif torch_tensors:
        if rumb.torch_compile:
            if rumb.num_classes == 2:
                eval_func = binary_cross_entropy_torch_compiled
            elif rumb.num_classes == 1 and not rumb.ord_model:
                eval_func = mse_torch_compiled
            else:
                eval_func = cross_entropy_torch_compiled
        else:
            if rumb.num_classes == 2:
                eval_func = binary_cross_entropy_torch
            elif rumb.num_classes == 1 and not rumb.ord_model:
                eval_func = mse_torch
            else:
                eval_func = cross_entropy_torch
    else:
        if rumb.num_classes == 2:
            eval_func = binary_cross_entropy
        elif rumb.num_classes == 1 and not rumb.ord_model:
            eval_func = mse
        else:
            eval_func = cross_entropy

    # initial predictions
    if torch_tensors:
        if rumb.num_classes == 2:
            rumb.raw_preds = torch.zeros(
                rumb.num_obs[0], device=rumb.device, dtype=torch.float32
            )
        else:
            rumb.raw_preds = torch.zeros(
                rumb.num_classes * rumb.num_obs[0],
                device=rumb.device,
                dtype=torch.float32,
            )
    else:
        if rumb.num_classes == 2:
            rumb.raw_preds = np.zeros(rumb.num_obs[0])
        else:
            rumb.raw_preds = np.zeros(rumb.num_classes * rumb.num_obs[0])
    if init_models:
        for j, booster in enumerate(rumb.boosters):
            if not rumb.rum_structure[j]["shared"]:
                init_pred = np.tile(
                    booster._Booster__inner_predict(0),
                    len(rumb.rum_structure[j]["utility"]),
                )
            else:
                init_pred = np.tile(
                    booster._Booster__inner_predict(0),
                    int(
                        len(rumb.rum_structure[j]["utility"])
                        / len(rumb.rum_structure[j]["variables"])
                    ),
                )
                if torch_tensors:
                    rumb.raw_preds[
                        rumb.booster_train_idx[j][0] : rumb.booster_train_idx[j][1]
                    ] += (
                        torch.from_numpy(init_pred).type(torch.float32).to(rumb.device)
                    )
                else:
                    rumb.raw_preds[rumb.booster_train_idx[j]] += init_pred

    rumb._preds = rumb._inner_predict()

    # start training
    for i in range(init_iteration, init_iteration + num_boost_round):
        # initialising the current predictions
        rumb._current_preds = []
        rumb._current_gains = []
        # update all binary boosters of the rumb
        for j, booster in enumerate(rumb.boosters):
            for cb in callbacks_before_iter:
                cb(
                    callback.CallbackEnv(
                        model=booster,
                        params=rumb.params[j],
                        iteration=i,
                        begin_iteration=init_iteration,
                        end_iteration=init_iteration + num_boost_round,
                        evaluation_result_list=None,
                    )
                )

            # store current class
            rumb._current_j = j

            # initial gain
            temp_gain = booster.feature_importance("gain").sum()

            # update the booster
            booster.update(train_set=None, fobj=f_obj)

            # store update gain
            rumb._current_gains.append(
                (booster.feature_importance("gain").sum() - temp_gain)
                / (
                    rumb.num_obs[0] * len(rumb.rum_structure[j]["variables"])
                )  # we normalise with the full number of observations even when not predicting for all alternatives since the gradient are summed in the objective function
            )

            # store current predictions
            if not rumb.rum_structure[j]["shared"]:
                rumb._current_preds.append(
                    np.tile(
                        -booster._Booster__inner_predict_buffer[  # this should be first because it is going to be updated when calling __inner_predict()
                            0
                        ]
                        + booster._Booster__inner_predict(0),
                        len(rumb.rum_structure[j]["utility"]),
                    )
                )
            elif len(rumb.rum_structure[j]["variables"]) < len(
                rumb.rum_structure[j]["utility"]
            ):
                rumb._current_preds.append(
                    np.tile(
                        -booster._Booster__inner_predict_buffer[
                            0
                        ]  # this should be first because it is going to be updated when calling __inner_predict()
                        + booster._Booster__inner_predict(0),
                        int(
                            len(rumb.rum_structure[j]["utility"])
                            / len(rumb.rum_structure[j]["variables"])
                        ),
                    )
                )
            else:
                rumb._current_preds.append(
                    -booster._Booster__inner_predict_buffer[
                        0
                    ]  # this should be first because it is going to be updated when calling __inner_predict()
                    + booster._Booster__inner_predict(0)
                )

            # check evaluation result. (from lightGBM initial code, check on all J binary boosters)
            evaluation_result_list = []
            if rumb.valid_labels is not None:
                if is_valid_contain_train:
                    evaluation_result_list.extend(booster.eval_train(feval))
                evaluation_result_list.extend(booster.eval_valid(feval))
            try:
                for cb in callbacks_after_iter:
                    cb(
                        callback.CallbackEnv(
                            model=booster,
                            params=rumb.params[j],
                            iteration=i,
                            begin_iteration=init_iteration,
                            end_iteration=init_iteration + num_boost_round,
                            evaluation_result_list=evaluation_result_list,
                        )
                    )
            except callback.EarlyStopException as earlyStopException:
                booster.best_iteration = earlyStopException.best_iteration + 1
                evaluation_result_list = earlyStopException.best_score

        if (optimise_mu or optimise_alphas) and ((i + 1) % optim_interval == 0):
            params_to_optimise = []

            if optimise_mu:
                if rumb.device is not None:
                    params_to_optimise += rumb.mu.cpu().numpy().tolist()
                else:
                    params_to_optimise += rumb.mu.tolist()
            if optimise_alphas:
                if rumb.device is not None:
                    params_to_optimise += rumb.alphas.cpu().numpy().flatten().tolist()
                else:
                    params_to_optimise += rumb.alphas.flatten().tolist()
            # update mu
            res = minimize(
                optimise_mu_or_alpha,
                np.array(params_to_optimise),
                args=(
                    rumb.labels[rumb.subsample_idx],
                    rumb,
                    optimise_mu,
                    optimise_alphas,
                    alpha_shape,
                ),
                bounds=bounds,
                method="SLSQP",
            )

            rumb._current_gains.append(rumb.best_score_train - res.fun)

        if optimise_thresholds and ((i + 1) % optim_interval == 0):

            thresh_diff = threshold_to_diff(rumb.thresholds)

            if rumb.ord_model == "coral":
                opt_func = optimise_thresholds_coral
            elif rumb.ord_model == "proportional_odds":
                opt_func = optimise_thresholds_proportional_odds

            if rumb.device is not None:
                raw_preds = rumb.raw_preds[rumb.subsample_idx][:, None].cpu().numpy()
                labels = rumb.labels[rumb.subsample_idx].cpu().numpy()
            else:
                raw_preds = rumb.raw_preds[rumb.subsample_idx][:, None]
                labels = rumb.labels[rumb.subsample_idx]

            # update thresholds
            res = minimize(
                opt_func,
                np.array(thresh_diff),
                args=(
                    labels,
                    raw_preds,
                ),
                bounds=bounds,
                method="SLSQP",
            )

            rumb.thresholds = diff_to_threshold(res.x)

        if optimise_ascs and ((i + 1) % optim_interval == 0):
            if rumb.device is not None:
                raw_preds = rumb._inner_predict(utilities=True).cpu().numpy()
                labels = rumb.labels[rumb.subsample_idx].cpu().numpy()
            else:
                raw_preds = rumb._inner_predict(utilities=True)
                labels = rumb.labels[rumb.subsample_idx]
            res = minimize(
                optimise_asc,
                ascs,
                args=(raw_preds, labels),
                method="SLSQP",
            )
            ascs = res.x

            for j, struct in enumerate(rumb.rum_structure):
                if rumb.boost_from_parameter_space[j]:
                    rumb.asc[j] = (
                        res.x[struct["utility"][0]]
                        / rumb.len_params_space[struct["utility"][0]]
                    )  # we share asc across parameters of the same utility function

        # find best booster and update raw predictions
        best_boosters = rumb._find_best_booster()

        if opt_mu_or_alpha_idx in best_boosters:
            best_boosters.remove(opt_mu_or_alpha_idx)
            rumb._update_mu_or_alphas(res, optimise_mu, optimise_alphas, alpha_shape)

        # update raw predictions
        rumb._update_raw_preds(best_boosters)

        # rollback unchosen boosters
        unchosen_boosters = set(range(len(rumb.boosters))) - set(best_boosters)
        rumb._rollback_boosters(unchosen_boosters)

        # reshuffle indices
        if subsample_freq > 0 and (i + 1) % subsample_freq == 0:
            if torch_tensors:
                rumb.subsample_idx = torch.randperm(
                    rumb.num_obs[0], device=rumb.device
                )[: int(subsample * rumb.num_obs[0])]
            else:
                rumb.subsample_idx = np.random.choice(
                    np.arange(rumb.num_obs[0]),
                    int(subsample * rumb.num_obs[0]),
                    replace=False,
                )
        elif rumb.batch_size:
            if (i + 1) % len(batches) == 0:
                permutations = torch.randperm(
                    rumb.num_obs[0], device=rumb.device, dtype=torch.int32
                )
                batches = torch.split(permutations, rumb.batch_size)
            rumb.subsample_idx = batches[(i + 1) % len(batches)]

        if subsample_valid < 1.0 and (i + 1) % 50:
            if torch_tensors:
                rumb.subsample_idx_valid = torch.randperm(
                    rumb.num_obs[1], device=rumb.device
                )[: int(subsample_valid * rumb.num_obs[1])]
            else:
                rumb.subsample_idx_valid = np.random.choice(
                    np.arange(rumb.num_obs[1]),
                    int(subsample_valid * rumb.num_obs[1]),
                    replace=False,
                )

        # make predictions after boosting round to compute new cross entropy and for next iteration grad and hess
        rumb._preds = rumb._inner_predict()

        # compute cross validation on training or validation test
        eval_train = eval_func(rumb._preds, rumb.labels[rumb.subsample_idx])

        if len(rumb.num_obs) > 1:  # only if there are validation sets
            eval_test = []
            for k, val_labels in enumerate(rumb.valid_labels):
                preds_valid = rumb._inner_predict(k + 1)
                eval_test.append(eval_func(preds_valid, val_labels))

            # update best score and best iteration
            if eval_test[0] < rumb.best_score:
                rumb.best_score = eval_test[0]
                rumb.best_iteration = i + 1

        rumb.best_score_train = eval_train

        # verbosity
        if (verbosity >= 1) and (i % verbose_interval == 0):
            print(
                f"[{i+1}]"
                + "-" * (6 - int(np.log10(i + 1)))
                + f"NCE value on train set : {eval_train:.4f}"
            )
            if rumb.valid_labels is not None:
                for k, _ in enumerate(rumb.valid_labels):
                    print(f"---------NCE value on test set {k+1}: {eval_test[k]:.4f}")

        # early stopping
        if (params["early_stopping_round"] != 0) and (
            rumb.best_iteration + params["early_stopping_round"] < i + 1
        ):
            if is_valid_contain_train:
                print(
                    "Early stopping at iteration {}, with a best score of {}".format(
                        rumb.best_iteration, rumb.best_score
                    )
                )
            else:
                print(
                    "Early stopping at iteration {}, with a best score on test set of {}, and on train set of {}".format(
                        rumb.best_iteration, rumb.best_score, rumb.best_score_train
                    )
                )
            break

        # save model
        if save_model_interval > 0 and (i % save_model_interval == 0):
            rumb.save_model(f"models/MTMC_switzerland_CNL_gpu_{i}")

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
        raise NotImplementedError("CVRUMBoost is not implemented yet.")
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


def _make_n_folds(
    full_data,
    folds,
    nfold,
    params,
    seed,
    fpreproc=None,
    stratified=True,
    shuffle=True,
    eval_train_metric=False,
    rum_structure=None,
    biogeme_model=None,
):
    """Make a n-fold list of Booster from random indices."""
    full_data = full_data.construct()
    num_data = full_data.num_data()
    if folds is not None:
        if not hasattr(folds, "__iter__") and not hasattr(folds, "split"):
            raise AttributeError(
                "folds should be a generator or iterator of (train_idx, test_idx) tuples "
                "or scikit-learn splitter object with split method"
            )
        if hasattr(folds, "split"):
            group_info = full_data.get_group()
            if group_info is not None:
                group_info = np.array(group_info, dtype=np.int32, copy=False)
                flatted_group = np.repeat(range(len(group_info)), repeats=group_info)
            else:
                flatted_group = np.zeros(num_data, dtype=np.int32)
            folds = folds.split(
                X=np.empty(num_data), y=full_data.get_label(), groups=flatted_group
            )
    else:
        if any(
            params.get(obj_alias, "")
            in {
                "lambdarank",
                "rank_xendcg",
                "xendcg",
                "xe_ndcg",
                "xe_ndcg_mart",
                "xendcg_mart",
            }
            for obj_alias in _ConfigAliases.get("objective")
        ):
            if not SKLEARN_INSTALLED:
                raise LightGBMError("scikit-learn is required for ranking cv")
            # ranking task, split according to groups
            group_info = np.array(full_data.get_group(), dtype=np.int32, copy=False)
            flatted_group = np.repeat(range(len(group_info)), repeats=group_info)
            group_kfold = _LGBMGroupKFold(n_splits=nfold)
            folds = group_kfold.split(X=np.empty(num_data), groups=flatted_group)
        elif stratified:
            if not SKLEARN_INSTALLED:
                raise LightGBMError("scikit-learn is required for stratified cv")
            skf = _LGBMStratifiedKFold(
                n_splits=nfold, shuffle=shuffle, random_state=seed
            )
            folds = skf.split(X=np.empty(num_data), y=full_data.get_label())
        else:
            if shuffle:
                randidx = np.random.RandomState(seed).permutation(num_data)
            else:
                randidx = np.arange(num_data)
            kstep = int(num_data / nfold)
            test_id = [randidx[i : i + kstep] for i in range(0, num_data, kstep)]
            train_id = [
                np.concatenate([test_id[i] for i in range(nfold) if k != i])
                for k in range(nfold)
            ]
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
        # create RUMBoosts with corresponding training, validation, and parameters sets
        cvbooster = RUMBoost()
        if rum_structure is not None:
            cvbooster.rum_structure = rum_structure  # save utility structure
        else:
            raise ValueError("rum_structure has to be declared")
        reduced_valid_sets, name_valid_sets, is_valid_contain_train, train_data_name = (
            cvbooster._preprocess_valids(train_set, params, valid_set)
        )
        cvbooster._preprocess_data(train_set, reduced_valid_sets)
        cvbooster._preprocess_params(tparam)
        cvbooster._construct_boosters(
            train_data_name, is_valid_contain_train, name_valid_sets=name_valid_sets
        )

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
    return [
        ("cv_agg", k, np.mean(v), metric_type[k], np.std(v)) for k, v in cvmap.items()
    ]


def rum_cv(
    params,
    train_set,
    num_boost_round=100,
    folds=None,
    nfold=5,
    stratified=True,
    shuffle=True,
    metrics=None,
    fobj=None,
    feval=None,
    init_model=None,
    feature_name="auto",
    categorical_feature="auto",
    early_stopping_rounds=None,
    fpreproc=None,
    verbose_eval=None,
    show_stdv=True,
    seed=0,
    callbacks=None,
    eval_train_metric=False,
    return_cvbooster=False,
    rum_structure=None,
    biogeme_model=None,
):
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
        params["objective"] = "none"
    for alias in _ConfigAliases.get("num_iterations"):
        if alias in params:
            _log_warning(f"Found `{alias}` in params. Will use it instead of argument")
            num_boost_round = params.pop(alias)
    params["num_iterations"] = num_boost_round
    if early_stopping_rounds is not None and early_stopping_rounds > 0:
        _log_warning(
            "'early_stopping_rounds' argument is deprecated and will be removed in a future release of LightGBM. "
            "Pass 'early_stopping()' callback via 'callbacks' argument instead."
        )
    for alias in _ConfigAliases.get("early_stopping_round"):
        if alias in params:
            early_stopping_rounds = params.pop(alias)
    params["early_stopping_round"] = early_stopping_rounds
    first_metric_only = params.get("first_metric_only", False)

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
        params["metric"] = metrics

    train_set._update_params(params)._set_predictor(predictor).set_feature_name(
        feature_name
    ).set_categorical_feature(categorical_feature)

    results = collections.defaultdict(list)
    cvfolds = _make_n_folds(
        train_set,
        folds=folds,
        nfold=nfold,
        params=params,
        seed=seed,
        fpreproc=fpreproc,
        stratified=stratified,
        shuffle=shuffle,
        eval_train_metric=eval_train_metric,
        rum_structure=rum_structure,
        biogeme_model=biogeme_model,
    )

    # setup callbacks
    if callbacks is None:
        callbacks = set()
    else:
        for i, cb in enumerate(callbacks):
            cb.__dict__.setdefault("order", i - len(callbacks))
        callbacks = set(callbacks)
    if early_stopping_rounds is not None and early_stopping_rounds > 0:
        callbacks.add(
            callback.early_stopping(
                early_stopping_rounds, first_metric_only, verbose=False
            )
        )
    if verbose_eval is not None:
        _log_warning(
            "'verbose_eval' argument is deprecated and will be removed in a future release of LightGBM. "
            "Pass 'log_evaluation()' callback via 'callbacks' argument instead."
        )
    if verbose_eval is True:
        callbacks.add(callback.log_evaluation(show_stdv=show_stdv))
    elif isinstance(verbose_eval, int):
        callbacks.add(callback.log_evaluation(verbose_eval, show_stdv=show_stdv))

    callbacks_before_iter = {
        cb for cb in callbacks if getattr(cb, "before_iteration", False)
    }
    callbacks_after_iter = callbacks - callbacks_before_iter
    callbacks_before_iter = sorted(callbacks_before_iter, key=attrgetter("order"))
    callbacks_after_iter = sorted(callbacks_after_iter, key=attrgetter("order"))

    for i in range(num_boost_round):
        cross_ent = []
        raw_results = []
        # train all RUMBoosts
        for RUMBoost in cvfolds.RUMBoosts:
            RUMBoost._preds = RUMBoost._inner_predict()
            for j, booster in enumerate(RUMBoost.boosters):
                for cb in callbacks_before_iter:
                    cb(
                        callback.CallbackEnv(
                            model=booster,
                            params=RUMBoost.params[j],
                            iteration=i,
                            begin_iteration=0,
                            end_iteration=num_boost_round,
                            evaluation_result_list=None,
                        )
                    )
                RUMBoost._current_j = j
                booster.update(train_set=RUMBoost.train_set[j], fobj=RUMBoost.f_obj)

            valid_sets = RUMBoost.valid_sets
            for valid_set in valid_sets:
                preds_valid = RUMBoost._inner_predict(data=valid_set)
                raw_results.append(preds_valid)
                cross_ent.append(
                    cross_entropy(preds_valid, valid_set[0].get_label().astype(int))
                )

        results[f"Cross entropy --- mean"].append(np.mean(cross_ent))
        results[f"Cross entropy --- stdv"].append(np.std(cross_ent))
        if verbose_eval:
            print(
                "[{}] -- Cross entropy mean: {}, with std: {}".format(
                    i + 1, np.mean(cross_ent), np.std(cross_ent)
                )
            )

        if np.mean(cross_ent) < cvfolds.best_score:
            cvfolds.best_score = np.mean(cross_ent)
            cvfolds.best_iteration = i + 1

        if (int(params.get("early_stopping_round", 0) or 0) > 0) and (
            cvfolds.best_iteration + params.get("early_stopping_round", 0) < i + 1
        ):
            print(
                "Early stopping at iteration {} with a cross entropy best score of {}".format(
                    cvfolds.best_iteration, cvfolds.best_score
                )
            )
            for k in results:
                results[k] = results[k][: cvfolds.best_iteration]
            break
        # res = _agg_cv_result(raw_results, eval_train_metric)
        # try:
        #    for cb in callbacks_after_iter:
        #        cb(callback.CallbackEnv(model=cvfolds,
        #                                params=params,
        #                                iteration=i,
        #                                begin_iteration=0,
        #                                end_iteration=num_boost_round,
        #                                evaluation_result_list=res))
        # except callback.EarlyStopException as earlyStopException:
        #    cvfolds.best_iteration = earlyStopException.best_iteration + 1
        #    for k in results:
        #        results[k] = results[k][:cvfolds.best_iteration]
        #    break

    if return_cvbooster:
        results["cvbooster"] = cvfolds

    return dict(results)
