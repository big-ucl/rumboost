import numpy as np

class Constant:
    """
    A class to represent a constant parameter, like ASCs.
    These parameters are not splitted on.

    Attributes
    ----------
    name : str
        The name of the parameter.
    value : float
        The value of the parameter.
            
    Methods
    -------
    __call__():
        Returns the value of the parameter.
    boost(grad, hess, value):

    """

    def __init__(self, name: str, value: float):
        self.name = name
        self.value = value

    def __call__(self):
        """
        Returns the value of the parameter.

        Returns
        -------
        float
            The value of the parameter.
        """
        return self.value

    def boost(self, grad: np.array, hess: np.array):
        """
        Boost the parameter by the given grad and hess.

        Parameters
        ----------
        grad : np.array
            The gradient of the loss function. (n_samples,)
        hess : np.array
            The hessian of the loss function. (n_samples,)
        """
        self.value = self.value - 0.1 * (grad.sum() / hess.sum())

def compute_grad_hess(preds, device, num_classes, labels, labels_j):
    if device is not None:
        preds = preds.cpu().numpy()
    eps = 1e-05

    if num_classes > 2:
        labels = labels_j
        if device is not None:
            labels = labels.cpu().numpy()
        factor = num_classes / (
            num_classes - 1
        )  # factor to correct redundancy (see Friedmann, Greedy Function Approximation)
        grad = preds - labels
        hess = np.maximum(
            factor * preds * (1 - preds), eps
        )  # truncate low values to avoid numerical errors
    elif num_classes == 2:
        preds = preds.reshape(-1)
        if device is not None:
            labels = labels.cpu().numpy()
        grad = preds - labels
        hess = np.maximum(preds * (1 - preds), eps)
        grad = grad.reshape(-1, 1)
        hess = hess.reshape(-1, 1)
    else:
        if device is not None:
            labels = labels.cpu().numpy()
        grad = (preds.reshape(-1) - labels).reshape(-1, 1)
        hess = np.ones_like(preds)

    return grad, hess

