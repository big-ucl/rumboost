import numpy as np


def accuracy(preds, labels):
    """
    Compute accuracy of the model.

    Parameters
    ----------
    preds : numpy array
        Predictions for all data points and each classes from a softmax function. preds[i, j] correspond
        to the prediction of data point i to belong to class j.
    labels : numpy array
        The labels of the original dataset, as int.

    Returns
    -------
    Accuracy: float
        The computed accuracy, as a float.
    """
    return np.mean(np.argmax(preds, axis=1) == labels)


def cross_entropy(preds, labels):
    """
    Compute negative cross entropy for given predictions and data.

    Parameters
    ----------
    preds: numpy array
        Predictions for all data points and each classes from a softmax function. preds[i, j] correspond
        to the prediction of data point i to belong to class j.
    labels: numpy array
        The labels of the original dataset, as int.

    Returns
    -------
    Cross entropy : float
        The negative cross-entropy, as float.
    """
    num_data = len(labels)
    data_idx = np.arange(num_data)
    return -np.mean(np.log(preds[data_idx, labels]))

def binary_cross_entropy(preds, labels):
    """
    Compute binary cross entropy for given predictions and data.

    Parameters
    ----------
    preds: numpy array
        Predictions for all data points and each classes from a sigmoid function. preds[i, j] correspond
        to the prediction of data point i to belong to class j.
    labels: numpy array
        The labels of the original dataset, as int.

    Returns
    -------
    Cross entropy : float
        The negative cross-entropy, as float.
    """
    preds = preds.reshape(-1)
    return -np.mean(labels * np.log(preds) + (1 - labels) * np.log(1 - preds))

def mse(preds, target):
    """
    Compute mean squared error for given predictions and data.

    Parameters
    ----------
    preds: numpy array
        Predictions for all data points and each classes from a regression model. preds[i, j] correspond
        to the prediction of data point i for class j.
    target: numpy array
        The target values of the original dataset.

    Returns
    -------
    Mean squared error : float
        The mean squared error, as float.
    """
    return np.mean((preds.reshape(-1) - target) ** 2)


def weighted_binary_cross_entropy(preds, labels):
    """
    Compute weighted binary cross entropy for given predictions and data.
    The weights are all ones. This function is used in the ordinal 
    regression model with coral estimation.

    Parameters
    ----------
    preds: numpy array
        Predictions for all data points and each classes from a sigmoid function. preds[i, j] correspond
        to the prediction of data point i to belong to class j.
    labels: numpy array
        The labels of the original dataset, as int.

    Returns
    -------
    Cross entropy : float
        The negative cross-entropy, as float.
    """
    classes = np.arange(np.unique(labels).shape[0] - 1)
    binary_labels = labels[:, None] > classes[None, :]
    return -np.mean(
        np.sum(
            binary_labels * np.log(preds) + (1 - binary_labels) * np.log(1 - preds),
            axis=1,
        )
    )

def safe_softplus(x, beta = 1, threshold = 20):
    """
    Compute the softplus function in a safe way to avoid numerical issues.

    Parameters
    ----------
    x: numpy array
        The input of the softplus function.
    beta: float
        The beta parameter for the softplus function.
    threshold: float
        The threshold for the input of the exponential function.

    Returns
    -------
    Softplus : numpy array
        The softplus function applied to x.
    """
    return np.where(beta * x > threshold, x, 1 / beta * np.logaddexp(0, beta * x))
