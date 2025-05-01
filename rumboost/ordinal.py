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
    sigmoids = expit(raw_preds - thresholds)
    preds = -np.diff(
        sigmoids, axis=1, prepend=1, append=0
    )

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
    return np.diff(thresholds, prepend=thresholds[0])


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
    logits = -raw_preds + threshold.reshape(1, -1)

    loss = weighted_binary_cross_entropy(logits, labels)

    return loss
