import numpy as np
from scipy.special import expit
from rumboost.metrics import cross_entropy, weighted_binary_cross_entropy

def threshold_preds(raw_preds, thresholds):
    """
    Calculate the probabilities of each ordinal class given the raw predictions and thresholds.
    
    Parameters
    ----------
    raw_preds : numpy.ndarray
        List of raw predictions
    thresholds : numpy.ndarray
        List of thresholds
    
    Returns
    -------
    numpy.ndarray
        List of probabilities of each ordinal class
    """
    preds = np.zeros((raw_preds.shape[0], thresholds.shape[0] + 1))
    sigmoids = expit(raw_preds - thresholds)
    preds[:, 0] = 1-sigmoids[:, 0]
    preds[:, 1:-1] = - np.diff(sigmoids, axis=1)
    preds[:, -1] = sigmoids[:, -1]

    return preds

def corn_preds(raw_preds):
    """
    Calculate the probabilities of each ordinal class given the raw predictions.
    
    Parameters
    ----------
    raw_preds : numpy.ndarray
        The raw predictions of the regression classifier.
    
    Returns
    -------
    numpy.ndarray
        List of probabilities of each ordinal class
    """
    preds = np.zeros((raw_preds.shape[0], raw_preds.shape[1] + 1))
    sigmoid = expit(raw_preds)
    cumul_preds = np.cumprod(sigmoid, axis=1)
    preds[:, 0] = 1 - cumul_preds[:, 0]
    preds[:, 1:-1] = - np.diff(cumul_preds, axis=1)
    preds[:, -1] = cumul_preds[:, -1]

    return preds

def threshold_to_diff(thresholds):
    """
    Convert thresholds to differences between thresholds
    
    Parameters
    ----------
    thresholds : numpy.ndarray
        List of thresholds
    
    Returns
    -------
    numpy.ndarray
        List of differences between thresholds, with the first element being the first threshold
    """
    thresh_diff = [thresholds[0]]
    thresh_diff.extend(np.diff(thresholds))
    return np.array(thresh_diff)

def diff_to_threshold(threshold_diff):
    """
    Convert differences between thresholds to thresholds
    
    Parameters
    ----------
    threshold_diff : numpy.ndarray
        List of differences between thresholds, with the first element being the first threshold
    
    Returns
    -------
    numpy.ndarray
        List of thresholds
    """
    return np.cumsum(threshold_diff)


def optimise_thresholds_proportional_odds(thresh_diff, labels, raw_preds):
    """
    Optimise thresholds for ordinal regression, according to the proportional odds model.

    Parameters
    ----------
    thresh_diff : numpy.ndarray
        List of threshold differnces (first element is the first threshold)
    labels : numpy.ndarray
        List of labels
    raw_preds : numpy.ndarray
        List of predictions

    Returns
    -------
    loss : int
        The loss according to the optimisation of thresholds.
    """

    threshold = diff_to_threshold(thresh_diff)
    probs = threshold_preds(raw_preds, threshold)

    loss = cross_entropy(probs, labels)

    return loss

def optimise_thresholds_coral(thresh_diff, labels, raw_preds):
    """
    Optimise thresholds for ordinal regression, with a coral model.

    Parameters
    ----------
    thresh_diff : numpy.ndarray
        List of threshold differnces (first element is the first threshold)
    labels : numpy.ndarray
        List of labels
    raw_preds : numpy.ndarray
        List of predictions

    Returns
    -------
    loss : int
        The loss according to the optimisation of thresholds.
    """

    threshold = diff_to_threshold(thresh_diff)
    sigmoids = expit(raw_preds - threshold)

    loss = weighted_binary_cross_entropy(sigmoids, labels) 

    return loss

