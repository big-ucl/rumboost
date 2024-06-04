import numpy as np
from scipy.special import softmax
from rumboost.metrics import cross_entropy

try:
    import torch
    from rumboost.torch_functions import (
        cross_entropy_torch,
        cross_entropy_torch_compiled,
        _nest_probs_torch,
        _nest_probs_torch_compiled,
        _cross_nested_probs_torch,
        _cross_nested_probs_torch_compiled,
    )

    torch_installed = True
except ImportError:
    torch_installed = False


def nest_probs(raw_preds, mu, nests, nest_alt):
    """compute nested predictions.

    Parameters
    ----------

    raw_preds :
        The raw predictions from the booster
    mu :
        The list of mu values for each nest.
        The first value correspond to the first nest and so on.
    nests :
        The dictionary keys are alternatives number and their values are their nest number.
        By example, {0:0, 1:1, 2:0} means that alt 0 and 2 are in nest 0 and alt 1 is in nest 1.
    nest_alt :
        The nest of each alternative. By example, [0, 1, 0] means that alt 0 and 2 are in nest 0 and alt 1 is in nest 1.

    Returns
    -------

    preds.T :
        The nested predictions
    pred_i_m :
        The prediction of choosing alt i knowing nest m
    pred_m :
        The prediction of choosing nest m
    """
    # initialisation
    n_obs, n_alt = raw_preds.shape
    mu_obs = mu[nest_alt]
    nests_array = np.array(list(nests.keys()))
    n_mu = nests_array.shape[0]

    mu_raw_preds_3d = np.exp(mu_obs * raw_preds)[:, :, None]
    # sum_in_nest = np.sum(mu_raw_preds_3d * mask_3d, axis=1)
    mask_3d = (nest_alt[:, None] == nests_array[None, :])[None, :, :]
    sum_in_nest = np.sum(mu_raw_preds_3d * mask_3d, axis=1)

    pred_i_m = np.exp(mu_obs * raw_preds) / np.sum(
        sum_in_nest[:, None, :] * mask_3d, axis=2
    )

    V_tilde_m = 1 / mu * np.log(sum_in_nest)

    # Pred of choosing nest m
    pred_m = softmax(V_tilde_m, axis=1)

    # Final predictions for choosing i
    preds = pred_i_m * np.sum(pred_m[:, None, :] * mask_3d, axis=2)

    return preds, pred_i_m, pred_m


def cross_nested_probs(raw_preds, mu, alphas):
    """Compute nested predictions.

    Parameters
    ----------
    raw_preds : numpy.ndarray
        The raw predictions from the booster
    mu : list
        The list of mu values for each nest.
        The first value corresponds to the first nest and so on.
    alphas : numpy.ndarray
        An array of J (alternatives) by M (nests).
        alpha_jn represents the degree of membership of alternative j to nest n.
        For example, alpha_12 = 0.5 means that alternative one belongs 50% to nest 2.

    Returns
    -------
    preds : numpy.ndarray
        The cross nested predictions
    pred_i_m : numpy.ndarray
        The prediction of choosing alt i knowing nest m
    pred_m : numpy.ndarray
        The prediction of choosing nest m
    """
    # scaling and exponential of raw_preds, following by degree of memberships
    raw_preds_mu_alpha_3d = (alphas**mu)[None, :, :] * np.exp(
        mu[None, None, :] * raw_preds[:, :, None]
    )
    # storing sum of utilities in nests
    sum_in_nest = np.sum(raw_preds_mu_alpha_3d, axis=1) ** (1 / mu)[None, :]

    # pred of choosing i knowing m.
    pred_i_m = raw_preds_mu_alpha_3d / np.sum(
        raw_preds_mu_alpha_3d, axis=1, keepdims=True
    )

    # pred of choosing m
    pred_m = sum_in_nest / np.sum(sum_in_nest, axis=1, keepdims=True)

    # final predictions for choosing i
    preds = np.sum(pred_i_m * pred_m[:, None, :], axis=2)

    return preds, pred_i_m, pred_m


def optimise_mu_or_alpha(
    params_to_optimise,
    labels,
    rumb,
    optimise_mu,
    optimise_alpha,
    alpha_shape,
):
    """
    Optimize mu or alpha values for a given dataset.

    Parameters
    ----------
    params_to_optimise : list
        The list of mu or alpha values to optimize.
    labels : numpy.ndarray, optional (default=None)
        The labels of the original dataset, as int.
    rumb : RUMBoost, optional (default=None)
        A trained RUMBoost object.
    optimise_mu : bool, optional (default=False)
        Whether to optimize mu values.
    optimise_alpha : bool, optional (default=False)
        Whether to optimize alpha values.
    alpha_shape : tuple
        The shape of the alpha values.
    data_idx : numpy.ndarray
        The indices of the dataset to optimize.
    lambda_l1 : float, optional (default=0)
        The L1 regularization parameter.
    lambda_l2 : float, optional (default=0)
        The L2 regularization parameter.
    previous_ce : float, optional (default=0)
        The cross-entropy loss of the previous iteration.

    Returns
    -------
    loss : int
        The loss according to the optimization of mu or alpha values.
    """
    if rumb.device is not None:
        if not torch_installed:
            raise ImportError(
                "PyTorch is not installed. Please install PyTorch to use the GPU."
            )
        if optimise_mu:
            mu = (
                torch.from_numpy(params_to_optimise[: rumb.mu.shape[0]])
                .type(torch.float64)
                .to(rumb.device)
            )
            if optimise_alpha:
                alphas = (
                    torch.from_numpy(params_to_optimise[rumb.mu.shape[0] :])
                    .type(torch.float64)
                    .view(alpha_shape)
                    .to(rumb.device)
                )
                alphas = alphas / alphas.sum(dim=1, keepdim=True)
            else:
                alphas = rumb.alphas.type(torch.float64)
        elif optimise_alpha:
            alphas = (
                torch.from_numpy(params_to_optimise)
                .type(torch.float64)
                .view(alpha_shape)
                .to(rumb.device)
            )
            alphas = alphas / alphas.sum(dim=1, keepdim=True)
            mu = rumb.mu.type(torch.float64)
        if rumb.nests:
            if rumb.torch_compile:
                new_preds, _, _ = _nest_probs_torch_compiled(
                    rumb.raw_preds.view(-1, rumb.num_obs[0]).T[rumb.subsample_idx, :].type(torch.float64),
                    mu,
                    rumb.nests,
                    rumb.device,
                )
            else:
                new_preds, _, _ = _nest_probs_torch(
                    rumb.raw_preds.view(-1, rumb.num_obs[0]).T[rumb.subsample_idx, :].type(torch.float64),
                    mu,
                    rumb.nests,
                    rumb.device,
                )
        else:
            if rumb.torch_compile:
                new_preds, _, _ = _cross_nested_probs_torch_compiled(
                    rumb.raw_preds.view(-1, rumb.num_obs[0]).T[rumb.subsample_idx, :].type(torch.float64),
                    mu,
                    alphas,
                    rumb.device,
                )
            else:
                new_preds, _, _ = _cross_nested_probs_torch(
                    rumb.raw_preds.view(-1, rumb.num_obs[0]).T[rumb.subsample_idx, :].type(torch.float64),
                    mu,
                    alphas,
                    rumb.device,
                )
        if rumb.torch_compile:
            loss = cross_entropy_torch_compiled(new_preds, labels)
        else:
            loss = cross_entropy_torch(new_preds, labels)
    else:
        if optimise_mu:
            mu = params_to_optimise[: rumb.mu.shape[0]]
            if optimise_alpha:
                alphas = params_to_optimise[rumb.mu.shape[0] :].reshape(alpha_shape)
                alphas = alphas / alphas.sum(axis=1, keepdims=True)
            else:
                alphas = rumb.alphas
        elif optimise_alpha:
            alphas = params_to_optimise.reshape(alpha_shape)
            alphas = alphas / alphas.sum(axis=1, keepdims=True)
            mu = rumb.mu

        if rumb.nests:
            new_preds, _, _ = nest_probs(
                rumb.raw_preds.reshape(rumb.num_obs[0], -1, order="F")[
                    rumb.subsample_idx, :
                ],
                mu,
                rumb.nests,
                rumb.nest_alt,
            )
        else:
            new_preds, _, _ = cross_nested_probs(
                rumb.raw_preds.reshape(rumb.num_obs[0], -1, order="F")[
                    rumb.subsample_idx, :
                ],
                mu,
                alphas,
            )
        loss = cross_entropy(new_preds, labels)

    return loss
