import logging

import numpy as np
import pandas as pd
from sklearn.decomposition import non_negative_factorization

from microbiome_gpt.utils.metrics import cos_sim, exp_var

log = logging.getLogger(__name__)


def nmf(X, n_components, init="nndsvdar", update_H=True, H=None, random_state=None):
    """Wrapper around non_negative_factorization with fixed settings.

    Returns (W, H, n_iter). Fixed params: solver=mu, beta_loss=kullback-leibler, max_iter=2000.
    """
    W, H, n_iter = non_negative_factorization(
        X,
        n_components=n_components,
        init=init,
        solver="mu",
        beta_loss="kullback-leibler",
        max_iter=2000,
        update_H=update_H,
        H=H,
        random_state=random_state,
    )
    return W, H, n_iter


def run_nmf_loop(X_train: pd.DataFrame, X_val: pd.DataFrame, n_signatures: int, n_runs: int):
    """Train NMF with multiple random seeds, return best model and per-run history.

    NMF is non-convex and can get stuck in local minima — multiple random
    initializations are used to find the best solution. Best is selected by
    validation explained variance and cosine similarity.

    Returns:
        best    — dict with keys: exp_var, cos_sim, model, H, W
        history — list of per-run dicts with train/val metrics
    """
    best = {
        "exp_var": -np.inf,
        "cos_sim": -np.inf,
        "H":       None,
        "W":       None,
        "W_val":   None,
    }

    history = []

    for seed in range(1, n_runs + 1):
        W_train, H, _ = nmf(X_train, n_signatures, random_state=seed)
        W_val,   _, _ = nmf(X_val,   n_signatures, init="custom", update_H=False, H=H)

        ev_train = exp_var(X_train, W_train @ H)
        cs_train = cos_sim(X_train, W_train @ H)
        ev_val   = exp_var(X_val,   W_val   @ H)
        cs_val   = cos_sim(X_val,   W_val   @ H)

        history.append({
            "seed":          seed,
            "exp_var_train": ev_train,
            "cos_sim_train": cs_train,
            "exp_var_val":   ev_val,
            "cos_sim_val":   cs_val,
        })

        log.info(
            "Seed %d/%d | train: exp_var=%.4f cos_sim=%.4f | val: exp_var=%.4f cos_sim=%.4f",
            seed, n_runs, ev_train, cs_train, ev_val, cs_val,
        )

        if ev_val > best["exp_var"] and cs_val > best["cos_sim"]:
            best["exp_var"] = ev_val
            best["cos_sim"] = cs_val
            best["H"]       = H
            best["W"]       = W_train
            best["W_val"]   = W_val

    log.info("Best run | exp_var_val: %.4f | cos_sim_val: %.4f", best["exp_var"], best["cos_sim"])
    return best, history
