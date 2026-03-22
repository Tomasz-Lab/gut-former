import logging
import argparse

import numpy as np
import pandas as pd
from sklearn.decomposition import NMF

from utils.project_paths import find_data_path, find_output_path
from utils.log_config import setup_logging
from utils.data_prep import train_val_split
from utils.metrics import cos_sim, exp_var

log = logging.getLogger(__name__)


def main():
    # Setting up project
    setup_logging()

    log.info("Starting NMF training script..\n ")

    # Handling Args
    p = argparse.ArgumentParser()
    p.add_argument("--dataset",      type=str,   default="sample", help="TODO")
    p.add_argument("--n_signatures", type=int,   default=8,        help="TODO")
    p.add_argument("--n_runs",       type=int,   default=100,      help="TODO")
    args = p.parse_args()

    # Loading & Preparing Data
    data_path = find_data_path()
    Xp_df = pd.read_csv(f"{data_path}/pathways_{args.dataset}.csv", index_col=[0], low_memory=False).fillna(0).sort_index() * 100

    Xp_train, Xp_val = train_val_split(Xp_df)

    log.info("Data loaded | train: %s | val: %s", Xp_train.shape, Xp_val.shape)

    # Running NMF
    best = {
        "exp_var": -np.inf,
        "cos_sim": -np.inf,
        "model":   None,
        "H":       None,
        "W":       None,
    }

    history = []

    for seed in range(1, args.n_runs + 1):
        # Each run uses its index as seed — NMF is non-convex and can get stuck in local minima,
        # so multiple random initializations are used to find the best solution
        model = NMF(
            n_components=args.n_signatures,
            init="nndsvdar",
            solver="mu",
            beta_loss="kullback-leibler",
            max_iter=2000,
            random_state=seed,
        )

        W_train = model.fit_transform(Xp_train)
        H       = model.components_

        Xp_train_rec = W_train @ H
        ev_train = exp_var(Xp_train, Xp_train_rec)
        cs_train = cos_sim(Xp_train, Xp_train_rec)

        W_val      = model.transform(Xp_val)
        Xp_val_rec = W_val @ H
        ev_val  = exp_var(Xp_val, Xp_val_rec)
        cs_val  = cos_sim(Xp_val, Xp_val_rec)

        history.append({
            "seed":          seed,
            "exp_var_train": ev_train,
            "cos_sim_train": cs_train,
            "exp_var_val":   ev_val,
            "cos_sim_val":   cs_val,
        })

        log.info(
            "Seed %d/%d | train: exp_var=%.4f cos_sim=%.4f | val: exp_var=%.4f cos_sim=%.4f",
            seed, args.n_runs, ev_train, cs_train, ev_val, cs_val,
        )

        if ev_val > best["exp_var"] and cs_val > best["cos_sim"]:
            best["exp_var"] = ev_val
            best["cos_sim"] = cs_val
            best["model"]   = model
            best["H"]       = H
            best["W"]       = W_train

    log.info("Best run | exp_var_val: %.4f | cos_sim_val: %.4f", best["exp_var"], best["cos_sim"])

    # Saving outputs
    output_path = find_output_path() / "nmf"
    output_path.mkdir(parents=True, exist_ok=True)

    n  = args.n_signatures
    ds = args.dataset
    W_val_best = best["model"].transform(Xp_val)

    pd.DataFrame(best["H"], columns=Xp_train.columns).to_csv(output_path / f"H_{n}_{ds}.csv")
    pd.DataFrame(best["W"], index=Xp_train.index)    .to_csv(output_path / f"W_train_{n}_{ds}.csv")
    pd.DataFrame(W_val_best, index=Xp_val.index)     .to_csv(output_path / f"W_val_{n}_{ds}.csv")
    pd.DataFrame(history)                             .to_csv(output_path / f"nmf_stats_{n}_{ds}.csv", index=False)

    log.info(
        "NMF complete. Saved to %s:\n"
        "  - H_%s_%d.csv\n"
        "  - W_train_%s_%d.csv\n"
        "  - W_val_%s_%d.csv\n"
        "  - nmf_stats_%s_%d.csv",
        output_path, ds, n, ds, n, ds, n, ds, n,
    )


if __name__ == "__main__":
    main()
