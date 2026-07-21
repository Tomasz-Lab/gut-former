import argparse
import logging
import time

import numpy as np
import pandas as pd

from gut_former.nmf.nmf_core import nmf
from gut_former.utils.log_config import setup_logging
from gut_former.utils.metrics import cos_sim, exp_var
from gut_former.utils.metrics import l2norm as calc_l2norm
from gut_former.utils.metrics import rss as calc_rss
from gut_former.utils.project_paths import find_data_path, find_output_path

log = logging.getLogger(__name__)


def split_into_abcd(m):
    """Split a rearranged matrix into 4 parts: A (validation), B, C, D (training).

    The cut is made at 1/3 of rows and 1/3 of cols, so:
        A = top-left  (1/9 of matrix) — held-out, never seen during training
        B = top-right (2/9 of matrix) — same samples as A, different pathways
        C = bot-left  (2/9 of matrix) — same pathways as A, different samples
        D = bot-right (4/9 of matrix) — NMF is trained here to predict A
    """
    n_rows, n_cols = m.shape
    row_cut = n_rows // 3
    col_cut = n_cols // 3

    A = m[:row_cut,  :col_cut]
    B = m[:row_cut,  col_cut:]
    C = m[row_cut:,  :col_cut]
    D = m[row_cut:,  col_cut:]

    return A, B, C, D


def rearrange_blocks(blocks):
    """Rearrange 9 blocks into 9 configurations, each with a different block in position A (top-left).

    In each configuration the matrix is structured as:
        ┌─────┬──────────┐
        │  A  │    B     │   A — held-out validation block (never seen during training)
        ├─────┼──────────┤   B — same samples as A, different pathways
        │  C  │    D     │   C — same pathways as A, different samples
        └─────┴──────────┘   D — NMF is trained here to predict A
    """
    def assemble(order):
        rows = [np.hstack([blocks[order[r*3 + c]] for c in range(3)]) for r in range(3)]
        return np.vstack(rows)

    arrangements = [
        [1,2,3, 4,5,6, 7,8,9],  # A=1
        [3,1,2, 6,4,5, 9,7,8],  # A=3
        [2,3,1, 5,6,4, 8,9,7],  # A=2
        [7,8,9, 1,2,3, 4,5,6],  # A=7
        [4,5,6, 7,8,9, 1,2,3],  # A=4
        [6,4,5, 9,7,8, 3,1,2],  # A=6
        [9,7,8, 3,1,2, 6,4,5],  # A=9
        [8,9,7, 2,3,1, 5,6,4],  # A=8
        [5,6,4, 8,9,7, 2,3,1],  # A=5
    ]

    return {idx: assemble(order) for idx, order in enumerate(arrangements, start=1)}


def shuffle_and_split(X):
    """Shuffle matrix rows and columns, then cut into h×h submatrices."""
    # h=3 is the standard in biCV (Owen & Perry, 2009): gives 9 folds where
    # each held-out block A is 1/9 of the matrix — small enough to be a real
    # test, large enough to be meaningful.
    h = 3

    M = X.copy()
    M = M[np.random.permutation(M.shape[0]), :]  # shuffle rows (samples)
    M = M[:, np.random.permutation(M.shape[1])]  # shuffle cols (pathways)

    n_samples, n_features = M.shape
    row_splits = [i * (n_samples  // h) for i in range(h)] + [n_samples]
    col_splits = [i * (n_features // h) for i in range(h)] + [n_features]

    blocks = {}
    idx = 1
    for r in range(h):
        for c in range(h):
            blocks[idx] = M[row_splits[r]:row_splits[r+1], col_splits[c]:col_splits[c+1]]
            idx += 1

    return blocks


def run_cv(folds, signatures, outdir, n_runs=50):
    """Run biCV for all folds, signatures and runs. Save all results to a single CSV."""
    results = []

    for fold_idx, (A, B, C, D) in folds.items():
        log.info("Fold %d/9", fold_idx)

        for signature in signatures:
            log.info("  Signature %d | %d runs", signature, n_runs)

            for seed in range(1, n_runs + 1):

                # Step 1: Train NMF on D → W_d (samples_D x signature), H_d (signature x features_D)
                W_d, H_d, _ = nmf(D, signature, random_state=seed)
                D_pred = W_d @ H_d

                # Step 2: Fix H_d, solve B ≈ W_a @ H_d
                # B has same samples as A, different features (features of D)
                W_a, _, _ = nmf(B, signature, init="custom", update_H=False, H=H_d)
                B_pred = W_a @ H_d

                # Step 3: Fix W_d, solve C ≈ W_d @ H_a
                # C has same features as A, different samples (samples of D)
                # Transpose trick: C.T ≈ H_a.T @ W_d.T
                H_a_T, _, _ = nmf(C.T, signature, init="custom", update_H=False, H=W_d.T)
                C_pred = W_d @ H_a_T.T   # H_a_T is (features_A x signature), need signature x features_A

                # Step 4: Predict A from W_a (samples) and H_a (features) — never seen during training
                A_pred = W_a @ H_a_T.T

                # Record metrics for all 4 blocks
                for block, true, pred in [("A", A, A_pred), ("B", B, B_pred),
                                          ("C", C, C_pred), ("D", D, D_pred)]:
                    results.append({
                        "fold":    fold_idx,
                        "signature":    signature,
                        "run":     seed,
                        "block":   block,
                        "evar":    exp_var(true, pred),
                        "cos_sim": cos_sim(true, pred),
                        "rss":     calc_rss(true, pred),
                        "l2norm":  calc_l2norm(true, pred),
                    })

    df = pd.DataFrame(results)
    out_file = outdir / "bicv_nmf_stats.csv"
    df.to_csv(out_file, index=False)
    log.info("Results saved to %s", out_file)

    summary = (
        df[df["block"] == "A"]
        .groupby("signature")[["evar", "cos_sim"]]
        .agg(["mean", "std"])
        .round(3)
    )
    summary.columns = ["evar_mean", "evar_std", "cos_sim_mean", "cos_sim_std"]
    log.info("Block A summary (held-out validation):\n%s", summary.to_string())

    best = summary["evar_mean"].idxmax()
    log.info("Best signature: %d (evar_mean=%.3f)", best, summary.loc[best, "evar_mean"])


def main():
    setup_logging()

    log.info("Starting biCV script..")

    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, default="sample", help="Dataset name — loads pathways_{dataset}.csv")
    p.add_argument("--n_signatures_min",   type=int, default=2,        help="Minimum number of NMF components to test")
    p.add_argument("--n_signatures_max",   type=int, default=15,       help="Maximum number of NMF components to test")
    p.add_argument("--n_runs",  type=int, default=50,       help="Number of NMF runs per fold per signature")
    args = p.parse_args()

    # Loading data
    data_path = find_data_path()
    X = pd.read_csv(f"{data_path}/pathways_{args.dataset}.csv", index_col=[0], low_memory=False).fillna(0).sort_index()

    log.info("Data loaded | shape: %s", X.shape)

    # Output directory
    output_path = find_output_path() / "bicv"
    output_path.mkdir(parents=True, exist_ok=True)

    log.info("Output directory: %s", output_path)

    signatures = list(range(args.n_signatures_min, args.n_signatures_max + 1))
    log.info("Signatures to test: %s", signatures)

    blocks   = shuffle_and_split(X.values)
    matrices = rearrange_blocks(blocks)
    folds_by_idx   = {idx: split_into_abcd(m) for idx, m in matrices.items()}

    t_start = time.time()
    run_cv(folds_by_idx, signatures, output_path, n_runs=args.n_runs)
    elapsed = time.time() - t_start
    log.info("Done in %dm %ds", elapsed // 60, elapsed % 60)


if __name__ == "__main__":
    main()
