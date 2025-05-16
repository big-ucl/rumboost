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
    learning_rate : float
        The learning rate for the parameter.
        Default is 0.1.

    Methods
    -------
    __call__():
        Returns the value of the parameter.
    boost(grad, hess, value):

    """

    def __init__(self, name: str, value: float, learning_rate: float = 1):
        self.name = name
        self.value = value
        self.learning_rate = learning_rate

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
        self.value = self.value - self.learning_rate * (grad.sum() / hess.sum())

def compute_grad_hess():
    raise NotImplementedError("compute_grad_hess is not implemented yet.")
    preds = rumb._preds
    if rumb.device is not None:
        preds = preds.cpu().numpy()
    eps = 1e-05

    if rumb.num_classes > 2:
        labels = rumb.labels_j[rumb.subsample_idx]
        if rumb.device is not None:
            labels = labels.cpu().numpy()
        factor = rumb.num_classes / (
            rumb.num_classes - 1
        )  # factor to correct redundancy (see Friedmann, Greedy Function Approximation)
        grad = preds - labels
        hess = np.maximum(
            factor * preds * (1 - preds), eps
        )  # truncate low values to avoid numerical errors
    elif rumb.num_classes == 2:
        preds = preds.reshape(-1)
        labels = rumb.labels[rumb.subsample_idx]
        if rumb.device is not None:
            labels = labels.cpu().numpy()
        grad = preds - labels
        hess = np.maximum(preds * (1 - preds), eps)
    else:
        labels = rumb.labels[rumb.subsample_idx]
        if rumb.device is not None:
            labels = labels.cpu().numpy()
        grad = 2 * (preds - labels)
        hess = 2 * np.ones_like(preds)

    for j, cst in enumerate(constant_parameters):
        cst.boost(grad[:, j], hess[:, j])
    if rumb.device is not None:
        rumb.asc = torch.from_numpy(np.array([c() for c in constant_parameters])).type(torch.double).to(rumb.device)
    else:
        rumb.asc = np.array([c() for c in constant_parameters])