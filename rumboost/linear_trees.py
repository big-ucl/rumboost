import json
import numpy as np
import warnings


class LinearTree:
    def __init__(
        self,
        x: np.ndarray = None,
        init_leaf_val: int = 0,
        monotonic_constraint: int = 0,
        max_bin: int = 255,
        learning_rate: float = 0.1,
        lambda_l1: float = 0,
        lambda_l2: float = 0,
        bagging_fraction: float = 1,
        bagging_freq: int = 1,
        min_data_in_leaf: int = 20,
        min_sum_hessian_in_leaf: float = 1e-3,
        min_gain_to_split: int = 0,
        min_data_in_bin: int = 3,
    ):
        """
        Parameters
        ----------
        x : np.ndarray, optional
            Feature values for the training data. If None, the model is initialized without data.
        init_leaf_val : int, optional
            Initial value for the leaves of the tree. Default is 0.
        monotonic_constraint : int, optional
            Monotonic constraint for the tree. 0 means no constraint, 1 means positive, -1 means negative.
        max_bin : int, optional
            Maximum number of bins for histogram-based feature values. Default is 255.
        learning_rate : float, optional
            Learning rate for the model. Default is 0.1.
        lambda_l1 : float, optional
            L1 regularisation term. Default is 0.
        lambda_l2 : float, optional
            L2 regularisation term. Default is 0.
        bagging_fraction : float, optional
            Fraction of data to use for bagging. Default is 1 (no bagging).
        bagging_freq : int, optional
            Frequency of bagging. If > 0, bagging is applied every `bagging_freq` iterations. Default is 1.
        min_data_in_leaf : int, optional
            Minimum number of data points in a leaf. Default is 20.
        min_sum_hessian_in_leaf : float, optional
            Minimum sum of Hessian in a leaf. Default is 1e-3.
        min_gain_to_split : int, optional
            Minimum gain required to perform a split. Default is 0.
        min_data_in_bin : int, optional
            Minimum number of data points in a bin. Default is 3.
        """
        if x is not None:
            self.x = x
            self.bin_edges, self.histograms, self.bin_indices = (
                self.build_lightgbm_style_histogram(x, max_bin, min_data_in_bin)
            )
            self.bin_indices_2d = self.bin_indices.reshape(1, -1).repeat(
                self.bin_edges.shape[0] - 2, axis=0
            )
            self.upper_bound_left = init_leaf_val * np.ones(self.bin_edges.shape[0] - 2)
            self.lower_bound_left = init_leaf_val * np.ones(self.bin_edges.shape[0] - 2)
            self.upper_bound_right = init_leaf_val * np.ones(
                self.bin_edges.shape[0] - 2
            )
            self.lower_bound_right = init_leaf_val * np.ones(
                self.bin_edges.shape[0] - 2
            )
            self.x_minus_bin_edges = self.x[None, :] - self.bin_edges[1:-1, None]
            self.split_and_leaf_values = {
                "splits": self.bin_edges,
                "leaves": init_leaf_val * np.ones(self.bin_edges.shape[0] - 1),
                "value_at_splits": init_leaf_val * self.bin_edges,
            }
            self.bagging_fraction = bagging_fraction
            self.bagging_freq = bagging_freq
            if self.bagging_freq > 0:
                self.bagged_indices = np.random.choice(
                    np.arange(x.shape[0]),
                    size=int((1 - self.bagging_fraction) * x.shape[0]),
                    replace=False,
                )
        self.monotonic_constraint = monotonic_constraint

        self.feature_importance_dict = {"gain": np.array([])}
        self.valid_sets = []
        self.name_valid_sets = []
        self.bin_indices_valid = []
        self.learning_rate = learning_rate
        self.lambda_l1 = lambda_l1
        if self.lambda_l1 > 0:
            warnings.warn(
                "L1 regularisation not implemented yet, ignoring `lambda_l1` value."
            )
        self.lambda_l2 = lambda_l2

        self.min_data_in_leaf = min_data_in_leaf
        self.min_sum_hessian_in_leaf = min_sum_hessian_in_leaf
        self.min_gain_to_split = min_gain_to_split
        self.min_data_in_bin = min_data_in_bin
        self.boosting_count = 0

    def build_lightgbm_style_histogram(
        self, feature_values: np.ndarray, max_bin: int = 255, min_data_in_bin: int = 3
    ):
        """
        Build histogram for feature values similar to LightGBM's histogram-based binning.

        Parameters
        ----------
        feature_values : np.ndarray
            Feature values to be binned.
        max_bin : int, optional
            Maximum number of bins to create. Default is 255.
        min_data_in_bin : int, optional
            Minimum number of data points required in each bin. Default is 3.

        Returns
        -------
        bin_edges : np.ndarray
            Edges of the bins.
        histogram : np.ndarray
            Histogram of the feature values.
        bin_indices : np.ndarray
            Indices of the bins for each feature value.
        """

        while max_bin > 1:
            percentiles = np.linspace(0, 100, max_bin + 1)
            bin_edges = np.unique(np.percentile(feature_values, percentiles))
            bin_indices = np.digitize(feature_values, bins=bin_edges[1:-1], right=True)
            histogram = np.bincount(bin_indices, minlength=len(bin_edges) - 1)
            if (histogram < min_data_in_bin).any():
                warnings.warn(
                    f"[Warning] Not enough data in bins. Reducing max_bin from {max_bin} to {max_bin // 2}."
                )
                max_bin = int(max_bin / 2)  # reduce bins and try again
            else:
                break

        if (histogram < min_data_in_bin).any():
            raise ValueError(
                "[Fatal] Not enough data per bin. Consider reducing `min_data_in_bin`."
            )

        return bin_edges, histogram, bin_indices

    def feature_importance(self, type: str):
        """
        Get the feature importance for the specified type.

        Parameters
        ----------
        type : str
            Type of feature importance to retrieve. Currently only "gain" is supported.

        Returns
        -------
        np.ndarray
            Feature importance values for the specified type.
        """
        return self.feature_importance_dict[type]

    def update(self, train_set, fobj):
        """
        Update the model with new training data and compute the best split.

        Parameters
        ----------
        train_set : Dataset
            Training dataset containing the feature values.
        fobj : callable
            Objective function to compute gradients and Hessians.
        """
        # get gradients and Hessians and scale with data
        grad, hess = fobj(1, 2)
        if self.bagging_freq > 0:
            grad[self.bagged_indices] = 0
            hess[self.bagged_indices] = 0
        grad_x = grad * self.x_minus_bin_edges
        hess_x = hess * self.x_minus_bin_edges**2

        # compute sum of grads and hessians in all bins
        N = self.bin_indices_2d.max() + 1
        id = (
            self.bin_indices_2d + (N * np.arange(self.bin_indices_2d.shape[0]))[:, None]
        )

        grad_x_binned = np.bincount(id.ravel(), weights=grad_x.ravel()).reshape(-1, N)
        hess_x_binned = np.bincount(id.ravel(), weights=hess_x.ravel()).reshape(-1, N)

        # mask for left and right leaves
        arange = np.arange(grad_x_binned.shape[1])
        edgerange = np.arange(grad_x_binned.shape[1] - 1) + 1
        mask = arange[None, :] < edgerange[:, None]

        # compute gains and leaves
        sum_hessian_left = (hess_x_binned * mask).sum(axis=1)
        sum_hessian_right = (hess_x_binned * ~mask).sum(axis=1)

        # left
        left_gain = (grad_x_binned * mask).sum(axis=1) ** 2 / (
            sum_hessian_left + self.lambda_l2
        )
        left_leaf = -(grad_x_binned * mask).sum(axis=1) / (
            sum_hessian_left + self.lambda_l2
        )

        left_gain = np.nan_to_num(left_gain, nan=-np.inf)

        # right
        right_gain = (grad_x_binned * ~mask).sum(axis=1) ** 2 / (
            sum_hessian_right + self.lambda_l2
        )
        right_leaf = -(grad_x_binned * ~mask).sum(axis=1) / (
            sum_hessian_right + self.lambda_l2
        )
        right_gain = np.nan_to_num(right_gain, nan=-np.inf)

        # no split gain
        no_split_gain = grad_x_binned.sum(axis=1) ** 2 / (
            hess_x_binned.sum(axis=1) + self.lambda_l2
        )
        no_split_gain = np.nan_to_num(no_split_gain, nan=-np.inf)

        # apply monotonic constraints and regularisation
        if self.monotonic_constraint == 1:
            left_gain[self.learning_rate * left_leaf < -self.lower_bound_left] = 0
            left_leaf[self.learning_rate * left_leaf < -self.lower_bound_left] = 0
            right_gain[self.learning_rate * right_leaf < -self.lower_bound_right] = 0
            right_leaf[self.learning_rate * right_leaf < -self.lower_bound_right] = 0

        if self.monotonic_constraint == -1:
            left_gain[self.learning_rate * left_leaf > -self.upper_bound_left] = 0
            left_leaf[self.learning_rate * left_leaf > -self.upper_bound_left] = 0
            right_gain[self.learning_rate * right_leaf > -self.upper_bound_right] = 0
            right_leaf[self.learning_rate * right_leaf > -self.upper_bound_right] = 0

        if self.min_sum_hessian_in_leaf > 0:
            left_gain[sum_hessian_left < self.min_sum_hessian_in_leaf] = 0
            left_leaf[sum_hessian_left < self.min_sum_hessian_in_leaf] = 0
            right_gain[sum_hessian_right < self.min_sum_hessian_in_leaf] = 0
            right_leaf[sum_hessian_right < self.min_sum_hessian_in_leaf] = 0

        if self.min_data_in_leaf > 0:
            left_gain[(self.histograms * mask).sum(axis=1) < self.min_data_in_leaf] = 0
            left_leaf[(self.histograms * mask).sum(axis=1) < self.min_data_in_leaf] = 0
            right_gain[
                (self.histograms * ~mask).sum(axis=1) < self.min_data_in_leaf
            ] = 0
            right_leaf[
                (self.histograms * ~mask).sum(axis=1) < self.min_data_in_leaf
            ] = 0

        # gain of split
        gain = left_gain + right_gain - no_split_gain

        gain = np.nan_to_num(gain, nan=-np.inf)

        if self.min_gain_to_split > 0:
            gain[gain < self.min_gain_to_split] = -np.inf

        if (gain <= 0).all():
            warnings.warn("No splits with positive gains. Ignoring boosting round.")
            self.ignore_round = True
        else:
            self.ignore_round = False

        # find best split
        self.best_index = np.argmax(gain)
        self.best_split = self.bin_edges[self.best_index + 1]
        self.best_left_leaf = self.learning_rate * left_leaf[self.best_index]
        self.best_right_leaf = self.learning_rate * right_leaf[self.best_index]
        best_gain = gain[self.best_index]
        self.feature_importance_dict["gain"] = np.concatenate(
            [self.feature_importance_dict["gain"], np.array([best_gain])], axis=0
        )

    def rollback_one_iter(self):
        """Rollback the last boosting iteration."""
        self.best_split = None
        self.best_left_leaf = None
        self.best_right_leaf = None
        self.feature_importance_dict["gain"] = self.feature_importance_dict["gain"][:-1]
        self.boosting_count += 1
        if self.bagging_freq > 0 and self.boosting_count % self.bagging_freq == 0:
            self.bagged_indices = np.random.choice(
                np.arange(self.x.shape[0]),
                size=int((1 - self.bagging_fraction) * self.x.shape[0]),
                replace=False,
            )

    def _inner_predict(self, data_idx: int):
        """
        Predict the values for the given data index.

        Parameters
        ----------
        data_idx : int
            Index of the data to predict. 0 for training data, >0 for validation data.

        Returns
        -------
        np.ndarray
            Predicted values for the specified data index.
        """
        if data_idx == 0: #train set
            x = self.x
            indices = self.bin_indices
        else: # validation set
            x = self.valid_sets[data_idx - 1]
            indices = self.bin_indices_valid[data_idx - 1]

        vas = self.split_and_leaf_values["value_at_splits"][indices]
        s = self.split_and_leaf_values["splits"][indices]
        l = self.split_and_leaf_values["leaves"][indices]
        return vas + (x - s) * l

    def predict(self, x):

        indices = np.digitize(x, bins=self.bin_edges[1:-1], right=True)

        vas = self.split_and_leaf_values["value_at_splits"][indices]
        s = self.split_and_leaf_values["splits"][indices]
        l = self.split_and_leaf_values["leaves"][indices]
        return vas + (x - s) * l

    def _update_linear_constants(self):
        """
        Update the linear constants of the model based on the best split and leaf values.
        """
        if (
            self.best_split is None
            or self.best_left_leaf is None
            or self.best_right_leaf is None
        ):
            return 0

        if self.ignore_round:
            self.rollback_one_iter()
            return 0

        s = self.best_split
        l_0 = self.best_left_leaf
        l_1 = self.best_right_leaf

        self.split_and_leaf_values["leaves"][: self.best_index + 1] += l_0
        self.split_and_leaf_values["leaves"][self.best_index + 1 :] += l_1

        distance_to_s = self.split_and_leaf_values["splits"] - s

        self.split_and_leaf_values["value_at_splits"][: self.best_index + 1] += (
            l_0 * distance_to_s[: self.best_index + 1]
        )
        self.split_and_leaf_values["value_at_splits"][self.best_index + 1 :] += (
            l_1 * distance_to_s[self.best_index + 1 :]
        )

        # update bounds for monotonic constraints
        self.update_bounds()

        self.boosting_count += 1
        if self.bagging_freq > 0 and self.boosting_count % self.bagging_freq == 0:
            self.bagged_indices = np.random.choice(
                np.arange(self.x.shape[0]),
                size=int((1 - self.bagging_fraction) * self.x.shape[0]),
                replace=False,
            )

    def update_bounds(self):
        """
        Update the bounds for the left and right leaves based on the current split and leaf values.
        This is necessary for enforcing monotonic constraints.
        """
        arange = np.arange(self.upper_bound_left.shape[0] + 1)
        edgerange = np.arange(self.upper_bound_left.shape[0]) + 1
        mask = arange[None, :] < edgerange[:, None]

        self.lower_bound_left = np.min(
            self.split_and_leaf_values["leaves"][None, :].repeat(mask.shape[0], axis=0),
            where=mask,
            axis=1,
            initial=np.inf,
        )
        self.upper_bound_left = np.max(
            self.split_and_leaf_values["leaves"][None, :].repeat(mask.shape[0], axis=0),
            where=mask,
            axis=1,
            initial=-np.inf,
        )

        self.lower_bound_right = np.min(
            self.split_and_leaf_values["leaves"][None, :].repeat(mask.shape[0], axis=0),
            where=~mask,
            axis=1,
            initial=np.inf,
        )
        self.upper_bound_right = np.max(
            self.split_and_leaf_values["leaves"][None, :].repeat(mask.shape[0], axis=0),
            where=~mask,
            axis=1,
            initial=-np.inf,
        )

    def eval_train(self, feval):
        return np.zeros(0)

    def eval_valid(self, feval):
        return np.zeros(0)

    def model_to_string(self, **kwargs) -> dict:
        """
        Serialize the model to a JSON string.
        """
        model_dict = {
            "bin_edges": self.bin_edges.tolist(),
            "histograms": self.histograms.tolist(),
            "bin_indices": self.bin_indices.tolist(),
            "split_and_leaf_values": {
                k: v.tolist() for k, v in self.split_and_leaf_values.items()
            },
            "monotonic_constraint": self.monotonic_constraint,
            "upper_bound_left": self.upper_bound_left.tolist(),
            "lower_bound_left": self.lower_bound_left.tolist(),
            "upper_bound_right": self.upper_bound_right.tolist(),
            "lower_bound_right": self.lower_bound_right.tolist(),
            "feature_importance_dict": {
                k: v.tolist() for k, v in self.feature_importance_dict.items()
            },
            "learning_rate": self.learning_rate,
            "lambda_l1": self.lambda_l1,
            "lambda_l2": self.lambda_l2,
            "bagging_fraction": self.bagging_fraction,
            "bagging_freq": self.bagging_freq,
            "min_data_in_leaf": self.min_data_in_leaf,
            "min_sum_hessian_in_leaf": self.min_sum_hessian_in_leaf,
            "min_gain_to_split": self.min_gain_to_split,
            "min_data_in_bin": self.min_data_in_bin,
            "boosting_count": self.boosting_count,
        }
        return model_dict

    def model_from_string(self, s: dict):
        """
        Load the model from a dictionary.
        """
        model_dict = s
        self.bin_edges = np.array(model_dict["bin_edges"])
        self.histograms = np.array(model_dict["histograms"])
        self.bin_indices = np.array(model_dict["bin_indices"])
        self.split_and_leaf_values = {
            k: np.array(v) for k, v in model_dict["split_and_leaf_values"].items()
        }
        self.monotonic_constraint = model_dict["monotonic_constraint"]
        self.upper_bound_left = np.array(model_dict["upper_bound_left"])
        self.lower_bound_left = np.array(model_dict["lower_bound_left"])
        self.upper_bound_right = np.array(model_dict["upper_bound_right"])
        self.lower_bound_right = np.array(model_dict["lower_bound_right"])
        self.feature_importance_dict = {
            k: np.array(v) for k, v in model_dict["feature_importance_dict"].items()
        }
        self.learning_rate = model_dict["learning_rate"]
        self.lambda_l1 = model_dict["lambda_l1"]
        self.lambda_l2 = model_dict["lambda_l2"]
        self.bagging_fraction = model_dict["bagging_fraction"]
        self.bagging_freq = model_dict["bagging_freq"]
        self.min_data_in_leaf = model_dict["min_data_in_leaf"]
        self.min_sum_hessian_in_leaf = model_dict["min_sum_hessian_in_leaf"]
        self.min_gain_to_split = model_dict["min_gain_to_split"]
        self.min_data_in_bin = model_dict["min_data_in_bin"]
        self.boosting_count = model_dict["boosting_count"]
        self.x = None
        self.valid_sets = []
        self.name_valid_sets = []
        self.bin_indices_valid = []

        return self

    def dump_model(self, **kwargs) -> dict:
        """Dump the model to a json string."""
        return self.model_to_string()

    def free_dataset(
        self,
    ):
        self.x = None

    def set_train_data_name(self, name: str) -> "LinearTree":
        """Set the name to the training Dataset.

        Parameters
        ----------
        name : str
            Name for the training Dataset.

        Returns
        -------
        self : Booster
            Booster with set training Dataset name.
        """
        self._train_data_name = name
        return self

    def add_valid(self, data, name: str) -> "LinearTree":
        """Add validation data.

        Parameters
        ----------
        data : Dataset
            Validation data.
        name : str
            Name of validation data.

        Returns
        -------
        self : Booster
            Booster with set validation data.
        """
        if not isinstance(self.valid_sets, list):
            self.valid_sets = []
        if not isinstance(self.name_valid_sets, list):
            self.name_valid_sets = []
        if not isinstance(self.bin_indices_valid, list):
            self.bin_indices_valid = []
        data.construct()
        valid_data = data.get_data().reshape(-1)
        self.valid_sets.append(valid_data)
        self.name_valid_sets.append(name)
        self.bin_indices_valid.append(
            np.digitize(valid_data, bins=self.bin_edges[1:-1], right=True)
        )
        return self
