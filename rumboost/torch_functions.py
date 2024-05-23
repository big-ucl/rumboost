import numpy as np
import sys

try:
    import torch
    torch_installed = True
    if sys.version_info > (3, 11):
        compile_decorator = lambda func: func
        Warning("RuntimeError: Torch.compile is not supported in python 3.12. Running functions without compilation.")
    else:
        compile_decorator = torch.compile
except ImportError:
    torch_installed = False
    compile_decorator = lambda func: func

def _predict_torch(
    raw_preds,
    shared_ensembles=None,
    num_obs=None,
    num_classes=None,
    device=None,
    shared_start_idx=None,
    functional_effects=False,
    nests=None,
    mu=None,
    alphas=None,
    utilities=False,
):
    """
    Predict function for RUMBoost class, with torch tensors.
    """
    if not torch_installed:
        raise ImportError("Pytorch is not installed. Please install it to run rumboost on torch tensors.")

    # if shared ensembles, get the shared predictions out and reorder them for easier addition later
    if shared_ensembles:
        raw_shared_preds = torch.zeros(
            size=(num_obs, num_classes), dtype=torch.float64, device=device
        )
        for i, arr in enumerate(raw_preds[shared_start_idx:]):
            raw_shared_preds[:, shared_ensembles[i + shared_start_idx]] = (
                raw_shared_preds[:, shared_ensembles[i + shared_start_idx]].add(
                    arr.view(-1, num_obs).T
                )
            )
        if shared_start_idx == 0:
            raw_preds = torch.zeros(
                size=(num_obs, num_classes), dtype=torch.float64, device=device
            )
        else:
            raw_preds = torch.stack(raw_preds[:shared_start_idx]).T
    else:
        raw_preds = torch.stack(raw_preds).T

    # if functional effect, sum the two ensembles (of attributes and socio-economic characteristics) of each alternative
    if functional_effects:
        raw_preds = raw_preds.view(-1, num_classes, 2).sum(dim=2)

    # if shared ensembles, add the shared ensembles to the individual specific ensembles
    if shared_ensembles:
        raw_preds.add_(raw_shared_preds)

    # compute nested probabilities. pred_i_m is predictions of choosing i knowing m, pred_m is prediction of choosing nest m and preds is pred_i_m * pred_m
    if nests:
        preds, pred_i_m, pred_m = _nest_probs_torch(
            raw_preds, mu=mu, nests=nests, device=device
        )

        return preds, pred_i_m, pred_m

    # compute cross-nested probabilities. pred_i_m is predictions of choosing i knowing m, pred_m is prediction of choosing nest m and preds is pred_i_m * pred_m
    if alphas is not None:
        preds, pred_i_m, pred_m = _cross_nested_probs_torch(
            raw_preds, mu=mu, alphas=alphas, device=device
        )

        return preds, pred_i_m, pred_m

    # softmax
    if not utilities:
        preds = torch.nn.functional.softmax(raw_preds, dim=1)
        return preds, None, None

    return raw_preds, None, None


@compile_decorator
def _predict_torch_compiled(
    raw_preds,
    shared_ensembles=None,
    num_obs=None,
    num_classes=None,
    device=None,
    shared_start_idx=None,
    functional_effects=False,
    nests=None,
    mu=None,
    alphas=None,
    utilities=False,
):
    """
    Compiled predict function for RUMBoost with torch tensors.
    """
    if not torch_installed:
        raise ImportError("Pytorch is not installed. Please install it to run rumboost on torch tensors.")

    # if shared ensembles, get the shared predictions out and reorder them for easier addition later
    if shared_ensembles:
        raw_shared_preds = torch.zeros(
            size=(num_obs, num_classes), dtype=torch.float64, device=device
        )
        for i, arr in enumerate(raw_preds[shared_start_idx:]):
            raw_shared_preds[:, shared_ensembles[i + shared_start_idx]] = (
                raw_shared_preds[:, shared_ensembles[i + shared_start_idx]].add(
                    arr.view(-1, num_obs).T
                )
            )
        if shared_start_idx == 0:
            raw_preds = torch.zeros(
                size=(num_obs, num_classes), dtype=torch.float64, device=device
            )
        else:
            raw_preds = torch.stack(raw_preds[:shared_start_idx]).T
    else:
        raw_preds = torch.stack(raw_preds).T

    # if functional effect, sum the two ensembles (of attributes and socio-economic characteristics) of each alternative
    if functional_effects:
        raw_preds = raw_preds.view(-1, num_classes, 2).sum(dim=2)

    # if shared ensembles, add the shared ensembles to the individual specific ensembles
    if shared_ensembles:
        raw_preds.add_(raw_shared_preds)

    # compute nested probabilities. pred_i_m is predictions of choosing i knowing m, pred_m is prediction of choosing nest m and preds is pred_i_m * pred_m
    if nests:
        preds, pred_i_m, pred_m = _nest_probs_torch(
            raw_preds, mu=mu, nests=nests, device=device
        )

        return preds, pred_i_m, pred_m

    # compute cross-nested probabilities. pred_i_m is predictions of choosing i knowing m, pred_m is prediction of choosing nest m and preds is pred_i_m * pred_m
    if alphas is not None:
        preds, pred_i_m, pred_m = _cross_nested_probs_torch(
            raw_preds, mu=mu, alphas=alphas, device=device
        )

        return preds, pred_i_m, pred_m

    # softmax
    if not utilities:
        preds = torch.nn.functional.softmax(raw_preds, dim=1)
        return preds, None, None

    return raw_preds, None, None


def _inner_predict_torch(
    raw_preds,
    shared_ensembles=None,
    num_obs=None,
    num_classes=None,
    device=None,
    shared_start_idx=None,
    functional_effects=False,
    nests=None,
    mu=None,
    alphas=None,
    data_idx=0,
    utilities=False,
):
    """
    Inner predict function for RUMBoost with torch tensors.
    """
    if not torch_installed:
        raise ImportError("Pytorch is not installed. Please install it to run rumboost on torch tensors.")
    
    # if shared ensembles, get the shared predictions out and reorder them for easier addition later
    if shared_ensembles:
        raw_shared_preds = torch.zeros(
            size=(num_obs[data_idx], num_classes), dtype=torch.float64, device=device
        )
        for i, arr in enumerate(raw_preds[shared_start_idx:]):
            raw_shared_preds[:, shared_ensembles[i + shared_start_idx]] = (
                raw_shared_preds[:, shared_ensembles[i + shared_start_idx]].add(
                    arr.view(-1, num_obs[data_idx]).T
                )
            )
        if shared_start_idx == 0:
            raw_preds = torch.zeros(
                size=(num_obs[data_idx], num_classes),
                dtype=torch.float64,
                device=device,
            )
        else:
            raw_preds = torch.stack(raw_preds[:shared_start_idx]).T
    else:
        raw_preds = torch.stack(raw_preds).T

    # if functional effect, sum the two ensembles (of attributes and socio-economic characteristics) of each alternative
    if functional_effects:
        raw_preds = raw_preds.view(-1, num_classes, 2).sum(dim=2)

    # if shared ensembles, add the shared ensembles to the individual specific ensembles
    if shared_ensembles:
        raw_preds.add_(raw_shared_preds)

    # compute nested probabilities. pred_i_m is predictions of choosing i knowing m, pred_m is prediction of choosing nest m and preds is pred_i_m * pred_m
    if nests:
        preds, pred_i_m, pred_m = _nest_probs_torch(
            raw_preds, mu=mu, nests=nests, device=device
        )

        return preds, pred_i_m, pred_m

    # compute cross-nested probabilities. pred_i_m is predictions of choosing i knowing m, pred_m is prediction of choosing nest m and preds is pred_i_m * pred_m
    if alphas is not None:
        preds, pred_i_m, pred_m = _cross_nested_probs_torch(
            raw_preds, mu=mu, alphas=alphas, device=device
        )

        return preds, pred_i_m, pred_m

    # softmax
    if not utilities:
        preds = torch.nn.functional.softmax(raw_preds, dim=1)
        return preds, None, None

    return raw_preds, None, None


@compile_decorator
def _inner_predict_torch_compiled(
    raw_preds,
    shared_ensembles=None,
    num_obs=None,
    num_classes=None,
    device=None,
    shared_start_idx=None,
    functional_effects=False,
    nests=None,
    mu=None,
    alphas=None,
    data_idx=0,
    utilities=False,
):
    """
    Inner compiled predict function for RUMBoost with torch tensors.
    """
    if not torch_installed:
        raise ImportError("Pytorch is not installed. Please install it to run rumboost on torch tensors.")
    
    # if shared ensembles, get the shared predictions out and reorder them for easier addition later
    if shared_ensembles:
        raw_shared_preds = torch.zeros(
            size=(num_obs[data_idx], num_classes), dtype=torch.float64, device=device
        )
        for i, arr in enumerate(raw_preds[shared_start_idx:]):
            raw_shared_preds[:, shared_ensembles[i + shared_start_idx]] = (
                raw_shared_preds[:, shared_ensembles[i + shared_start_idx]].add(
                    arr.view(-1, num_obs[data_idx]).T
                )
            )
        if shared_start_idx == 0:
            raw_preds = torch.zeros(
                size=(num_obs[data_idx], num_classes),
                dtype=torch.float64,
                device=device,
            )
        else:
            raw_preds = torch.stack(raw_preds[:shared_start_idx]).T
    else:
        raw_preds = torch.stack(raw_preds).T

    # if functional effect, sum the two ensembles (of attributes and socio-economic characteristics) of each alternative
    if functional_effects:
        raw_preds = raw_preds.view(-1, num_classes, 2).sum(dim=2)

    # if shared ensembles, add the shared ensembles to the individual specific ensembles
    if shared_ensembles:
        raw_preds.add_(raw_shared_preds)

    # compute nested probabilities. pred_i_m is predictions of choosing i knowing m, pred_m is prediction of choosing nest m and preds is pred_i_m * pred_m
    if nests:
        preds, pred_i_m, pred_m = _nest_probs_torch_compiled(
            raw_preds, mu=mu, nests=nests, device=device
        )

        return preds, pred_i_m, pred_m

    # compute cross-nested probabilities. pred_i_m is predictions of choosing i knowing m, pred_m is prediction of choosing nest m and preds is pred_i_m * pred_m
    if alphas is not None:
        preds, pred_i_m, pred_m = _cross_nested_probs_torch_compiled(
            raw_preds, mu=mu, alphas=alphas, device=device
        )

        return preds, pred_i_m, pred_m

    # softmax
    if not utilities:
        preds = torch.nn.functional.softmax(raw_preds, dim=1)
        return preds, None, None

    return raw_preds, None, None


def _nest_probs_torch(raw_preds, mu, nests, device):
    """compute nested predictions.

    Parameters
    ----------
    raw_preds :
        The raw predictions from the booster
    mu :
        The list of mu values for each nest.
        The first value correspond to the first nest and so on.
    nests :
        The dictionary keys are the nest numbers and its values are the
        list of alternatives in the nest.
        By example, {0:[0,2], 1:[1]} means that the nest 0 contains the
        alternatives 0 and 2 and the nest 1 contains the alternative 1.
    device :
        The device to use for the computation

    Returns
    -------
    preds.T :
        The nested predictions
    pred_i_m :
        The prediction of choosing alt i knowing nest m
    pred_m :
        The prediction of choosing nest m
    """
    if not torch_installed:
        raise ImportError("Pytorch is not installed. Please install it to run rumboost on torch tensors.")
    
    n_obs = raw_preds.size(0)
    n_alt = raw_preds.size(1)
    pred_i_m = torch.zeros((n_obs, n_alt), device=device)
    V_tilde_m = torch.zeros((n_obs, len(mu)), device=device)
    nest_alt = torch.zeros(n_alt, dtype=torch.int64, device=device)
    nests_tensor = torch.tensor(list(nests.keys())).to(device)
    for a, n in nests.items():
        nest_alt[n] = a
    mu_tensor = mu[nest_alt]

    mu_raw_preds_3d = torch.exp(mu_tensor * raw_preds)[:, :, None].repeat(
        1, 1, nests_tensor.size(0)
    )
    mask_3d = torch.where(
        (nest_alt[:, None] == nests_tensor[:, None].T).repeat(n_obs, 1, 1), 1, 0
    )
    sum_in_nest = torch.sum(mu_raw_preds_3d * mask_3d, dim=1)

    pred_i_m = torch.exp(mu_tensor * raw_preds) / torch.sum(
        sum_in_nest[:, None, :].repeat(1, n_alt, 1) * mask_3d, dim=2
    )

    V_tilde_m = 1 / mu * torch.log(sum_in_nest)

    # Pred of choosing nest m
    pred_m = torch.nn.functional.softmax(V_tilde_m, dim=1)

    # Final predictions for choosing i
    preds = pred_i_m * torch.sum(
        pred_m[:, None, :].repeat(1, n_alt, 1) * mask_3d, dim=2
    )

    return preds, pred_i_m, pred_m


@compile_decorator
def _nest_probs_torch_compiled(raw_preds, mu, nests, device):
    """compute nested predictions.

    Parameters
    ----------
    raw_preds :
        The raw predictions from the booster as a torch tensor
    mu :
        The list of mu values for each nest as a torch tensor
        The first value correspond to the first nest and so on.
    nests :
        The dictionary keys are the nest numbers and its values are the
        list of alternatives in the nest.
        By example, {0:[0,2], 1:[1]} means that the nest 0 contains the
        alternatives 0 and 2 and the nest 1 contains the alternative 1.
    device :
        The device to use for the computation

    Returns
    -------
    preds.T :
        The nested predictions
    pred_i_m :
        The prediction of choosing alt i knowing nest m
    pred_m :
        The prediction of choosing nest m
    """
    if not torch_installed:
        raise ImportError("Pytorch is not installed. Please install it to run rumboost on torch tensors.")
    
    n_obs = raw_preds.size(0)
    n_alt = raw_preds.size(1)
    pred_i_m = torch.zeros((n_obs, n_alt), device=device)
    V_tilde_m = torch.zeros((n_obs, len(mu)), device=device)
    nest_alt = torch.zeros(n_alt, dtype=torch.int64, device=device)
    nests_tensor = torch.tensor(list(nests.keys())).to(device)
    for a, n in nests.items():
        nest_alt[n] = a
    mu_tensor = mu[nest_alt]

    mu_raw_preds_3d = torch.exp(mu_tensor * raw_preds)[:, :, None].repeat(
        1, 1, nests_tensor.size(0)
    )
    mask_3d = torch.where(
        (nest_alt[:, None] == nests_tensor[:, None].T).repeat(n_obs, 1, 1), 1, 0
    )
    sum_in_nest = torch.sum(mu_raw_preds_3d * mask_3d, dim=1)

    pred_i_m = torch.exp(mu_tensor * raw_preds) / torch.sum(
        sum_in_nest[:, None, :].repeat(1, n_alt, 1) * mask_3d, dim=2
    )

    V_tilde_m = 1 / mu * torch.log(sum_in_nest)

    # Pred of choosing nest m
    pred_m = torch.nn.functional.softmax(V_tilde_m, dim=1)

    # Final predictions for choosing i
    preds = pred_i_m * torch.sum(
        pred_m[:, None, :].repeat(1, n_alt, 1) * mask_3d, dim=2
    )

    return preds, pred_i_m, pred_m


def _cross_nested_probs_torch(raw_preds, mu, alphas, device):
    """compute cross nested predictions.

    Parameters
    ----------
    raw_preds :
        The raw predictions from the booster
    mu :
        The list of mu values for each nest.
        The first value correspond to the first nest and so on.
    alphas :
        The list of alpha values for each nest.
        The first value correspond to the first nest and so on.
    device :
        The device to use for the computation

    Returns
    -------
    preds :
        The cross nested predictions
    pred_i_m :
        The prediction of choosing alt i knowing nest m
    pred_m :
        The prediction of choosing nest m
    """
    if not torch_installed:
        raise ImportError("Pytorch is not installed. Please install it to run rumboost on torch tensors.")
    
    n_obs, n_alt = raw_preds.shape[0], raw_preds.shape[1]
    pred_i_m = torch.zeros((n_obs, n_alt, mu.shape[0]), device=device)
    pred_m = torch.zeros((n_obs, n_alt, mu.shape[0]), device=device)

    # scaling and exponential of raw_preds, following by degree of memberships
    raw_preds_mu_alpha_3d = (alphas**mu)[None, :, :] * torch.exp(
        mu[None, None, :] * raw_preds[:, :, None]
    )
    # storing sum of utilities in nests
    sum_in_nest = torch.sum(raw_preds_mu_alpha_3d, dim=1) ** (1 / mu)[None, :]

    # pred of choosing i knowing m.
    pred_i_m = raw_preds_mu_alpha_3d / torch.sum(
        raw_preds_mu_alpha_3d, dim=1, keepdim=True
    )

    # pred of choosing m
    pred_m = sum_in_nest / torch.sum(sum_in_nest, dim=1, keepdim=True)

    # final predictions for choosing i
    preds = torch.sum(pred_i_m * pred_m[:, None, :], dim=2)

    return preds, pred_i_m, pred_m


@compile_decorator
def _cross_nested_probs_torch_compiled(raw_preds, mu, alphas, device):
    """compute cross nested predictions.

    Parameters
    ----------
    raw_preds :
        The raw predictions from the booster
    mu :
        The list of mu values for each nest.
        The first value correspond to the first nest and so on.
    alphas :
        The list of alpha values for each nest.
        The first value correspond to the first nest and so on.
    device :
        The device to use for the computation

    Returns
    -------
    preds :
        The cross nested predictions
    pred_i_m :
        The prediction of choosing alt i knowing nest m
    pred_m :
        The prediction of choosing nest m
    """
    if not torch_installed:
        raise ImportError("Pytorch is not installed. Please install it to run rumboost on torch tensors.")
    
    n_obs, n_alt = raw_preds.shape[0], raw_preds.shape[1]
    pred_i_m = torch.zeros((n_obs, n_alt, mu.shape[0]), device=device)
    pred_m = torch.zeros((n_obs, n_alt, mu.shape[0]), device=device)

    # scaling and exponential of raw_preds, following by degree of memberships
    raw_preds_mu_alpha_3d = (alphas**mu)[None, :, :] * torch.exp(
        mu[None, None, :] * raw_preds[:, :, None]
    )
    # storing sum of utilities in nests
    sum_in_nest = torch.sum(raw_preds_mu_alpha_3d, dim=1) ** (1 / mu)[None, :]

    # pred of choosing i knowing m.
    pred_i_m = raw_preds_mu_alpha_3d / torch.sum(
        raw_preds_mu_alpha_3d, dim=1, keepdim=True
    )

    # pred of choosing m
    pred_m = sum_in_nest / torch.sum(sum_in_nest, dim=1, keepdim=True)

    # final predictions for choosing i
    preds = torch.sum(pred_i_m * pred_m[:, None, :], dim=2)

    return preds, pred_i_m, pred_m


def _f_obj_torch(
    current_j,
    preds,
    num_classes,
    device,
    shared_ensembles=None,
    shared_start_idx=None,
    labels_j=None,
    labels=None,
):
    if not torch_installed:
        raise ImportError("Pytorch is not installed. Please install it to run rumboost on torch tensors.")
    
    j = current_j  # jth booster
    if shared_ensembles and j >= shared_start_idx:
        pred = preds.T[shared_ensembles[j], :].view(-1) # corresponding predictions
    else:
        pred = preds[:, j]  # corresponding predictions
    factor = num_classes / (
        num_classes - 1
    )  # factor to correct redundancy (see Friedmann, Greedy Function Approximation)
    eps = 1e-6
    if labels_j:
        labels = labels_j[j]
    else:
        labels = (
            (
                torch.from_numpy(np.array(shared_ensembles[j])).to(device)[:, None]
                == labels[None, :]
            )
            .int()
            .view(-1)
        )
    grad = pred - labels
    factor_times_one_minus_pred = factor * (1 - pred)
    hess = (pred * factor_times_one_minus_pred).clamp_(min=eps)  # truncate low values to avoid numerical errors

    return grad, hess


@compile_decorator
def _f_obj_torch_compiled(
    current_j,
    preds,
    num_classes,
    device,
    shared_ensembles=None,
    shared_start_idx=None,
    labels_j=None,
    labels=None,
):
    if not torch_installed:
        raise ImportError("Pytorch is not installed. Please install it to run rumboost on torch tensors.")
    
    j = current_j  # jth booster
    if shared_ensembles and j >= shared_start_idx:
        pred = preds.T[shared_ensembles[j], :].view(-1)  # corresponding predictions
    else:
        pred = preds[:, j]  # corresponding predictions
    factor = num_classes / (
        num_classes - 1
    )  # factor to correct redundancy (see Friedmann, Greedy Function Approximation)
    eps = 1e-6
    if labels_j:
        labels = labels_j[j]
    else:
        labels = (
            (
                torch.from_numpy(np.array(shared_ensembles[j])).to(device)[:, None]
                == labels[None, :]
            )
            .int()
            .view(-1)
        )
    grad = pred - labels
    factor_times_one_minus_pred = factor * (1 - pred)
    hess = (pred * factor_times_one_minus_pred).clamp_(min=eps)

    return grad, hess

def _f_obj_nested_torch(
    current_j,
    labels,
    preds_i_m,
    preds_m,
    num_classes,
    mu,
    nests,
    device,
    shared_ensembles=None,
    shared_start_idx=None,
):
    if not torch_installed:
        raise ImportError("Pytorch is not installed. Please install it to run rumboost on torch tensors.")
    j = current_j
    label = labels
    n_obs = preds_i_m.shape[0]
    data_idx = torch.arange(n_obs, device=device)
    factor = num_classes / (num_classes - 1)
    n_alt = preds_i_m.shape[1]
    nest_alt = torch.zeros(n_alt, dtype=torch.int64, device=device)
    for a, n in nests.items():
        nest_alt[n] = a
    label_nest = nest_alt[None, :].repeat(n_obs, 1)[data_idx, label]

    if shared_ensembles and j >= shared_start_idx:

        shared_ensembles_tensor = torch.from_numpy(np.array(shared_ensembles[j])).to(
            device
        )

        pred_i_m = preds_i_m[
            :, shared_ensembles_tensor
        ]  # pred of alternative j knowing nest m
        pred_m = preds_m[:, :, None].repeat(1, 1, n_alt)[
            :, nest_alt[shared_ensembles_tensor], shared_ensembles_tensor
        ]  # prediction of choosing nest m

        mu_reps = mu[:, None].repeat(1, n_alt)

        grad = torch.where(
            label[:, None] == shared_ensembles_tensor[None, :],
            -mu_reps[nest_alt[shared_ensembles_tensor], shared_ensembles_tensor]
            * (1 - pred_i_m)
            - pred_i_m * (1 - pred_m),
            torch.where(
                label_nest[:, None] == nest_alt[shared_ensembles_tensor][None, :],
                mu_reps[nest_alt[shared_ensembles_tensor], shared_ensembles_tensor]
                * pred_i_m
                - pred_i_m * (1 - pred_m),
                pred_i_m * pred_m,
            ),
        )
        hess = torch.where(
            label[:, None] == shared_ensembles_tensor[None, :],
            -mu_reps[nest_alt[shared_ensembles_tensor], shared_ensembles_tensor]
            * pred_i_m
            * (1 - pred_i_m)
            * (
                1
                - mu_reps[nest_alt[shared_ensembles_tensor], shared_ensembles_tensor]
                - pred_m
            )
            + pred_i_m**2 * pred_m * (1 - pred_m),
            torch.where(
                label_nest[:, None] == nest_alt[shared_ensembles_tensor][None, :],
                -mu_reps[nest_alt[shared_ensembles_tensor], shared_ensembles_tensor]
                * pred_i_m
                * (1 - pred_i_m)
                * (
                    1
                    - mu_reps[
                        nest_alt[shared_ensembles_tensor], shared_ensembles_tensor
                    ]
                    - pred_m
                )
                + pred_i_m**2 * pred_m * (1 - pred_m),
                -pred_i_m
                * pred_m
                * (
                    -mu_reps[nest_alt[shared_ensembles_tensor], shared_ensembles_tensor]
                    * (1 - pred_i_m)
                    - pred_i_m * (1 - pred_m)
                ),
            ),
        )
        hess.mul_(factor)

        grad = grad.T.reshape(-1)
        hess = hess.T.reshape(-1)
    else:
        pred_i_m = preds_i_m[:, j]  # pred of alternative j knowing nest m
        # pred_i_m = preds_i_m[data_idx, label]  #prediction of choice i knowing nest m
        pred_m = preds_m[:, nest_alt[j]]  # prediction of choosing nest m

        grad = torch.where(
            label == j,
            -mu[nest_alt[j]] * (1 - pred_i_m) - pred_i_m * (1 - pred_m),
            torch.where(
                label_nest == nest_alt[j],
                mu[nest_alt[j]] * pred_i_m - pred_i_m * (1 - pred_m),
                pred_i_m * pred_m,
            ),
        )
        hess = torch.where(
            label == j,
            -mu[nest_alt[j]]
            * pred_i_m
            * (1 - pred_i_m)
            * (1 - mu[nest_alt[j]] - pred_m)
            + pred_i_m**2 * pred_m * (1 - pred_m),
            torch.where(
                label_nest == nest_alt[j],
                -mu[nest_alt[j]]
                * pred_i_m
                * (1 - pred_i_m)
                * (1 - mu[nest_alt[j]] - pred_m)
                + pred_i_m**2 * pred_m * (1 - pred_m),
                -pred_i_m
                * pred_m
                * (-mu[nest_alt[j]] * (1 - pred_i_m) - pred_i_m * (1 - pred_m)),
            ),
        )

        hess.mul_(factor)

        grad = grad.view(-1)
        hess = hess.view(-1)

    return grad, hess


@compile_decorator
def _f_obj_nested_torch_compiled(
    current_j,
    labels,
    preds_i_m,
    preds_m,
    num_classes,
    mu,
    nests,
    device,
    shared_ensembles=None,
    shared_start_idx=None,
):
    if not torch_installed:
        raise ImportError("Pytorch is not installed. Please install it to run rumboost on torch tensors.")
    
    j = current_j
    label = labels
    n_obs = preds_i_m.shape[0]
    data_idx = torch.arange(n_obs, device=device)
    factor = num_classes / (num_classes - 1)
    n_alt = preds_i_m.shape[1]
    nest_alt = torch.zeros(n_alt, dtype=torch.int64, device=device)
    for a, n in nests.items():
        nest_alt[n] = a
    label_nest = nest_alt[None, :].repeat(n_obs, 1)[data_idx, label]

    if shared_ensembles and j >= shared_start_idx:

        shared_ensembles_tensor = torch.from_numpy(np.array(shared_ensembles[j])).to(
            device
        )

        pred_i_m = preds_i_m[
            :, shared_ensembles_tensor
        ]  # pred of alternative j knowing nest m
        pred_m = preds_m[:, :, None].repeat(1, 1, n_alt)[
            :, nest_alt[shared_ensembles_tensor], shared_ensembles_tensor
        ]  # prediction of choosing nest m

        mu_reps = mu[:, None].repeat(1, n_alt)

        grad = torch.where(
            label[:, None] == shared_ensembles_tensor[None, :],
            -mu_reps[nest_alt[shared_ensembles_tensor], shared_ensembles_tensor]
            * (1 - pred_i_m)
            - pred_i_m * (1 - pred_m),
            torch.where(
                label_nest[:, None] == nest_alt[shared_ensembles_tensor][None, :],
                mu_reps[nest_alt[shared_ensembles_tensor], shared_ensembles_tensor]
                * pred_i_m
                - pred_i_m * (1 - pred_m),
                pred_i_m * pred_m,
            ),
        )
        hess = torch.where(
            label[:, None] == shared_ensembles_tensor[None, :],
            -mu_reps[nest_alt[shared_ensembles_tensor], shared_ensembles_tensor]
            * pred_i_m
            * (1 - pred_i_m)
            * (
                1
                - mu_reps[nest_alt[shared_ensembles_tensor], shared_ensembles_tensor]
                - pred_m
            )
            + pred_i_m**2 * pred_m * (1 - pred_m),
            torch.where(
                label_nest[:, None] == nest_alt[shared_ensembles_tensor][None, :],
                -mu_reps[nest_alt[shared_ensembles_tensor], shared_ensembles_tensor]
                * pred_i_m
                * (1 - pred_i_m)
                * (
                    1
                    - mu_reps[
                        nest_alt[shared_ensembles_tensor], shared_ensembles_tensor
                    ]
                    - pred_m
                )
                + pred_i_m**2 * pred_m * (1 - pred_m),
                -pred_i_m
                * pred_m
                * (
                    -mu_reps[nest_alt[shared_ensembles_tensor], shared_ensembles_tensor]
                    * (1 - pred_i_m)
                    - pred_i_m * (1 - pred_m)
                ),
            ),
        )
        hess.mul_(factor)

        grad = grad.T.reshape(-1)
        hess = hess.T.reshape(-1)
    else:
        pred_i_m = preds_i_m[:, j]  # pred of alternative j knowing nest m
        # pred_i_m = preds_i_m[data_idx, label]  #prediction of choice i knowing nest m
        pred_m = preds_m[:, nest_alt[j]]  # prediction of choosing nest m

        grad = torch.where(
            label == j,
            -mu[nest_alt[j]] * (1 - pred_i_m) - pred_i_m * (1 - pred_m),
            torch.where(
                label_nest == nest_alt[j],
                mu[nest_alt[j]] * pred_i_m - pred_i_m * (1 - pred_m),
                pred_i_m * pred_m,
            ),
        )
        hess = torch.where(
            label == j,
            -mu[nest_alt[j]]
            * pred_i_m
            * (1 - pred_i_m)
            * (1 - mu[nest_alt[j]] - pred_m)
            + pred_i_m**2 * pred_m * (1 - pred_m),
            torch.where(
                label_nest == nest_alt[j],
                -mu[nest_alt[j]]
                * pred_i_m
                * (1 - pred_i_m)
                * (1 - mu[nest_alt[j]] - pred_m)
                + pred_i_m**2 * pred_m * (1 - pred_m),
                -pred_i_m
                * pred_m
                * (-mu[nest_alt[j]] * (1 - pred_i_m) - pred_i_m * (1 - pred_m)),
            ),
        )

        hess.mul_(factor)

        grad = grad.view(-1)
        hess = hess.view(-1)

    return grad, hess


def _f_obj_cross_nested_torch(
    current_j,
    labels,
    preds_i_m,
    preds_m,
    preds,
    num_classes,
    mu,
    device,
    shared_ensembles=None,
    shared_start_idx=None,
):
    if not torch_installed:
        raise ImportError("Pytorch is not installed. Please install it to run rumboost on torch tensors.")
    j = current_j
    label = labels
    data_idx = torch.arange(preds_i_m.shape[0], device=device)
    factor = num_classes / (num_classes - 1)

    if shared_ensembles and j >= shared_start_idx:
        pred_j_m = preds_i_m[
            :, shared_ensembles[j], :
        ]  # pred of alternative j knowing nest m
        pred_i_m = preds_i_m[data_idx, label, :][
            :, None, :
        ]  # prediction of choice i knowing nest m
        pred_m = preds_m[:, None, :]  # prediction of choosing nest m
        pred_i = preds[data_idx, label][:, None, None]  # pred of choice i
        pred_j = preds[:, shared_ensembles[j]][:, :, None]  # pred of alt j

        pred_i_m_pred_m = pred_i_m * pred_m
        pred_j_m_pred_m = pred_j_m * pred_m
        pred_i_m_pred_i = pred_i_m * pred_i
        pred_i_m_squared = pred_i_m**2
        pred_j_m_squared = pred_j_m**2
        pred_i_squared = pred_i**2
        pred_j_m_pred_j_squared = (pred_j_m - pred_j) ** 2
        pred_i_m_1_mu_mu_pred_i = pred_i_m * (1 - mu) + mu - pred_i
        pred_j_m_1_mu_pred_j = pred_j_m * (1 - mu) - pred_j

        mu_squared = mu**2

        d_pred_i_Vi = torch.sum(
            (pred_i_m_pred_m * pred_i_m_1_mu_mu_pred_i), dim=2, keepdim=True
        )  # first derivative of pred i with respect to Vi
        d_pred_i_Vj = torch.sum(
            (pred_i_m_pred_m * pred_j_m_1_mu_pred_j), dim=2, keepdim=True
        )  # first derivative of pred i with respect to Vj
        d_pred_j_Vj = torch.sum(
            (pred_j_m_pred_m * (pred_j_m_1_mu_pred_j + mu)), dim=2, keepdim=True
        )  # first derivative of pred j with respect to Vj

        mu_3pim2_3pim_2pimpi_pi = mu * (
            -3 * pred_i_m_squared + 3 * pred_i_m + 2 * (pred_i_m_pred_i - pred_i)
        )
        pim2_2pimpi_pi2_dpiVi = (
            pred_i_m_squared - 2 * pred_i_m_pred_i + pred_i_squared - d_pred_i_Vi
        )
        mu2_2pim2_3pim_1 = mu_squared * (2 * pred_i_m_squared - 3 * pred_i_m + 1)
        mu2_pjm = mu_squared * (-pred_j_m)
        mu_pjm2_pjm = mu * (-pred_j_m_squared + pred_j_m)

        d2_pred_i_Vi = torch.sum(
            (
                pred_i_m_pred_m
                * (mu2_2pim2_3pim_1 + mu_3pim2_3pim_2pimpi_pi + pim2_2pimpi_pi2_dpiVi)
            ),
            dim=2,
            keepdim=True,
        )
        d2_pred_i_Vj = torch.sum(
            (
                pred_i_m_pred_m
                * (mu2_pjm + mu_pjm2_pjm + pred_j_m_pred_j_squared - d_pred_j_Vj)
            ),
            dim=2,
            keepdim=True,
        )

        # print(d2_pred_i_Vi)
        mask = (
            torch.from_numpy(np.array(shared_ensembles[j])).to(device)[None, :]
            == label[:, None]
        )
        grad = torch.where(
            mask[:, :, None],
            ((-1 / pred_i) * d_pred_i_Vi),
            ((-1 / pred_i) * d_pred_i_Vj),
        )
        hess = torch.where(
            mask[:, :, None],
            ((-1 / pred_i**2) * (d2_pred_i_Vi * pred_i - d_pred_i_Vi**2)),
            ((-1 / pred_i**2) * (d2_pred_i_Vj * pred_i - d_pred_i_Vj**2)),
        )
        hess.mul_(factor)

        grad = grad.T.reshape(-1)
        hess = hess.T.reshape(-1)
    else:
        pred_j_m = preds_i_m[:, j, :]  # pred of alternative j knowing nest m
        pred_i_m = preds_i_m[
            data_idx, label, :
        ]  # prediction of choice i knowing nest m
        pred_m = preds_m[:, :]  # prediction of choosing nest m
        pred_i = preds[data_idx, label][:, None]  # pred of choice i
        pred_j = preds[:, j][:, None]  # pred of alt j

        pred_i_m_pred_m = pred_i_m * pred_m
        pred_j_m_pred_m = pred_j_m * pred_m
        pred_i_m_pred_i = pred_i_m * pred_i
        pred_i_m_squared = pred_i_m**2
        pred_j_m_squared = pred_j_m**2
        pred_i_squared = pred_i**2
        pred_j_m_pred_j_squared = (pred_j_m - pred_j) ** 2
        pred_i_m_1_mu_mu_pred_i = pred_i_m * (1 - mu) + mu - pred_i
        pred_j_m_1_mu_pred_j = pred_j_m * (1 - mu) - pred_j

        mu_squared = mu**2

        d_pred_i_Vi = torch.sum(
            (pred_i_m_pred_m * pred_i_m_1_mu_mu_pred_i), dim=1, keepdim=True
        )  # first derivative of pred i with respect to Vi
        d_pred_i_Vj = torch.sum(
            (pred_i_m_pred_m * pred_j_m_1_mu_pred_j), dim=1, keepdim=True
        )  # first derivative of pred i with respect to Vj
        d_pred_j_Vj = torch.sum(
            (pred_j_m_pred_m * (pred_j_m_1_mu_pred_j + mu)), dim=1, keepdim=True
        )  # first derivative of pred j with respect to Vj

        mu_3pim2_3pim_2pimpi_pi = mu * (
            -3 * pred_i_m_squared + 3 * pred_i_m + 2 * (pred_i_m_pred_i - pred_i)
        )
        pim2_2pimpi_pi2_dpiVi = (
            pred_i_m_squared - 2 * pred_i_m_pred_i + pred_i_squared - d_pred_i_Vi
        )
        mu2_2pim2_3pim_1 = mu_squared * (2 * pred_i_m_squared - 3 * pred_i_m + 1)
        mu2_pjm = mu_squared * (-pred_j_m)
        mu_pjm2_pjm = mu * (-pred_j_m_squared + pred_j_m)

        d2_pred_i_Vi = torch.sum(
            (
                pred_i_m_pred_m
                * (mu2_2pim2_3pim_1 + mu_3pim2_3pim_2pimpi_pi + pim2_2pimpi_pi2_dpiVi)
            ),
            dim=1,
            keepdim=True,
        )
        d2_pred_i_Vj = torch.sum(
            (
                pred_i_m_pred_m
                * (mu2_pjm + mu_pjm2_pjm + pred_j_m_pred_j_squared - d_pred_j_Vj)
            ),
            dim=1,
            keepdim=True,
        )

        # two cases: 1. alt j is choice i, 2. alt j is not choice i
        grad = torch.where(
            (label == j).view(-1,1), (-1 / pred_i) * d_pred_i_Vi, (-1 / pred_i) * d_pred_i_Vj
        )
        hess = torch.where(
            (label == j).view(-1,1),
            (-1 / pred_i**2) * (d2_pred_i_Vi * pred_i - d_pred_i_Vi**2),
            (-1 / pred_i**2) * (d2_pred_i_Vj * pred_i - d_pred_i_Vj**2),
        )

        hess.mul_(factor)

        grad = grad.view(-1)
        hess = hess.view(-1)

    return grad, hess


@compile_decorator
def _f_obj_cross_nested_torch_compiled(
    current_j,
    labels,
    preds_i_m,
    preds_m,
    preds,
    num_classes,
    mu,
    device,
    shared_ensembles=None,
    shared_start_idx=None,
):
    if not torch_installed:
        raise ImportError("Pytorch is not installed. Please install it to run rumboost on torch tensors.")
    
    j = current_j
    label = labels
    data_idx = torch.arange(preds_i_m.shape[0], device=device)
    factor = num_classes / (num_classes - 1)

    if shared_ensembles and j >= shared_start_idx:
        pred_j_m = preds_i_m[
            :, shared_ensembles[j], :
        ]  # pred of alternative j knowing nest m
        pred_i_m = preds_i_m[data_idx, label, :][
            :, None, :
        ]  # prediction of choice i knowing nest m
        pred_m = preds_m[:, None, :]  # prediction of choosing nest m
        pred_i = preds[data_idx, label][:, None, None]  # pred of choice i
        pred_j = preds[:, shared_ensembles[j]][:, :, None]  # pred of alt j

        pred_i_m_pred_m = pred_i_m * pred_m
        pred_j_m_pred_m = pred_j_m * pred_m
        pred_i_m_pred_i = pred_i_m * pred_i
        pred_i_m_squared = pred_i_m**2
        pred_j_m_squared = pred_j_m**2
        pred_i_squared = pred_i**2
        pred_j_m_pred_j_squared = (pred_j_m - pred_j) ** 2
        pred_i_m_1_mu_mu_pred_i = pred_i_m * (1 - mu) + mu - pred_i
        pred_j_m_1_mu_pred_j = pred_j_m * (1 - mu) - pred_j

        mu_squared = mu**2

        d_pred_i_Vi = torch.sum(
            (pred_i_m_pred_m * pred_i_m_1_mu_mu_pred_i), dim=2, keepdim=True
        )  # first derivative of pred i with respect to Vi
        d_pred_i_Vj = torch.sum(
            (pred_i_m_pred_m * pred_j_m_1_mu_pred_j), dim=2, keepdim=True
        )  # first derivative of pred i with respect to Vj
        d_pred_j_Vj = torch.sum(
            (pred_j_m_pred_m * (pred_j_m_1_mu_pred_j + mu)), dim=2, keepdim=True
        )  # first derivative of pred j with respect to Vj

        mu_3pim2_3pim_2pimpi_pi = mu * (
            -3 * pred_i_m_squared + 3 * pred_i_m + 2 * (pred_i_m_pred_i - pred_i)
        )
        pim2_2pimpi_pi2_dpiVi = (
            pred_i_m_squared - 2 * pred_i_m_pred_i + pred_i_squared - d_pred_i_Vi
        )
        mu2_2pim2_3pim_1 = mu_squared * (2 * pred_i_m_squared - 3 * pred_i_m + 1)
        mu2_pjm = mu_squared * (-pred_j_m)
        mu_pjm2_pjm = mu * (-pred_j_m_squared + pred_j_m)

        d2_pred_i_Vi = torch.sum(
            (
                pred_i_m_pred_m
                * (mu2_2pim2_3pim_1 + mu_3pim2_3pim_2pimpi_pi + pim2_2pimpi_pi2_dpiVi)
            ),
            dim=2,
            keepdim=True,
        )
        d2_pred_i_Vj = torch.sum(
            (
                pred_i_m_pred_m
                * (mu2_pjm + mu_pjm2_pjm + pred_j_m_pred_j_squared - d_pred_j_Vj)
            ),
            dim=2,
            keepdim=True,
        )

        # print(d2_pred_i_Vi)
        mask = (
            torch.from_numpy(np.array(shared_ensembles[j])).to(device)[None, :]
            == label[:, None]
        )
        grad = torch.where(
            mask[:, :, None],
            ((-1 / pred_i) * d_pred_i_Vi),
            ((-1 / pred_i) * d_pred_i_Vj),
        )
        hess = torch.where(
            mask[:, :, None],
            ((-1 / pred_i**2) * (d2_pred_i_Vi * pred_i - d_pred_i_Vi**2)),
            ((-1 / pred_i**2) * (d2_pred_i_Vj * pred_i - d_pred_i_Vj**2)),
        )
        hess.mul_(factor)

        grad = grad.T.reshape(-1)
        hess = hess.T.reshape(-1)
    else:
        pred_j_m = preds_i_m[:, j, :]  # pred of alternative j knowing nest m
        pred_i_m = preds_i_m[
            data_idx, label, :
        ]  # prediction of choice i knowing nest m
        pred_m = preds_m[:, :]  # prediction of choosing nest m
        pred_i = preds[data_idx, label][:, None]  # pred of choice i
        pred_j = preds[:, j][:, None]  # pred of alt j

        pred_i_m_pred_m = pred_i_m * pred_m
        pred_j_m_pred_m = pred_j_m * pred_m
        pred_i_m_pred_i = pred_i_m * pred_i
        pred_i_m_squared = pred_i_m**2
        pred_j_m_squared = pred_j_m**2
        pred_i_squared = pred_i**2
        pred_j_m_pred_j_squared = (pred_j_m - pred_j) ** 2
        pred_i_m_1_mu_mu_pred_i = pred_i_m * (1 - mu) + mu - pred_i
        pred_j_m_1_mu_pred_j = pred_j_m * (1 - mu) - pred_j

        mu_squared = mu**2

        d_pred_i_Vi = torch.sum(
            (pred_i_m_pred_m * pred_i_m_1_mu_mu_pred_i), dim=1, keepdim=True
        )  # first derivative of pred i with respect to Vi
        d_pred_i_Vj = torch.sum(
            (pred_i_m_pred_m * pred_j_m_1_mu_pred_j), dim=1, keepdim=True
        )  # first derivative of pred i with respect to Vj
        d_pred_j_Vj = torch.sum(
            (pred_j_m_pred_m * (pred_j_m_1_mu_pred_j + mu)), dim=1, keepdim=True
        )  # first derivative of pred j with respect to Vj

        mu_3pim2_3pim_2pimpi_pi = mu * (
            -3 * pred_i_m_squared + 3 * pred_i_m + 2 * (pred_i_m_pred_i - pred_i)
        )
        pim2_2pimpi_pi2_dpiVi = (
            pred_i_m_squared - 2 * pred_i_m_pred_i + pred_i_squared - d_pred_i_Vi
        )
        mu2_2pim2_3pim_1 = mu_squared * (2 * pred_i_m_squared - 3 * pred_i_m + 1)
        mu2_pjm = mu_squared * (-pred_j_m)
        mu_pjm2_pjm = mu * (-pred_j_m_squared + pred_j_m)

        d2_pred_i_Vi = torch.sum(
            (
                pred_i_m_pred_m
                * (mu2_2pim2_3pim_1 + mu_3pim2_3pim_2pimpi_pi + pim2_2pimpi_pi2_dpiVi)
            ),
            dim=1,
            keepdim=True,
        )
        d2_pred_i_Vj = torch.sum(
            (
                pred_i_m_pred_m
                * (mu2_pjm + mu_pjm2_pjm + pred_j_m_pred_j_squared - d_pred_j_Vj)
            ),
            dim=1,
            keepdim=True,
        )

        # two cases: 1. alt j is choice i, 2. alt j is not choice i
        grad = torch.where(
            (label == j).view(-1,1), (-1 / pred_i) * d_pred_i_Vi, (-1 / pred_i) * d_pred_i_Vj
        )
        hess = torch.where(
            (label == j).view(-1,1),
            (-1 / pred_i**2) * (d2_pred_i_Vi * pred_i - d_pred_i_Vi**2),
            (-1 / pred_i**2) * (d2_pred_i_Vj * pred_i - d_pred_i_Vj**2),
        )
        hess.mul_(factor)

        grad = grad.view(-1)
        hess = hess.view(-1)

    return grad, hess


def cross_entropy_torch(preds, labels):
    """
    Compute negative cross entropy for given predictions and data.

    Parameters
    ----------
    preds: torch.Tensor
        Predictions for all data points and each classes from a softmax function. preds[i, j] correspond
        to the prediction of data point i to belong to class j.
    labels: torch.Tensor
        The labels of the original dataset, as int.

    Returns
    -------
    Cross entropy : float
        The negative cross-entropy, as float.
    """
    if not torch_installed:
        raise ImportError("Pytorch is not installed. Please install it to run rumboost on torch tensors.")
    return (
        -torch.mean(
            torch.log(
                preds[torch.arange(labels.shape[0], device=labels.device), labels]
            )
        )
        .cpu()
        .numpy()
    )


@compile_decorator
def cross_entropy_torch_compiled(preds, labels):
    """
    Compute negative cross entropy for given predictions and data.

    Parameters
    ----------
    preds: torch.Tensor
        Predictions for all data points and each classes from a softmax function. preds[i, j] correspond
        to the prediction of data point i to belong to class j.
    labels: torch.Tensor
        The labels of the original dataset, as int.

    Returns
    -------
    Cross entropy : float
        The negative cross-entropy, as float.
    """
    if not torch_installed:
        raise ImportError("Pytorch is not installed. Please install it to run rumboost on torch tensors.")
    return (
        -torch.mean(
            torch.log(
                preds[torch.arange(labels.shape[0], device=labels.device), labels]
            )
        )
        .cpu()
        .numpy()
    )
