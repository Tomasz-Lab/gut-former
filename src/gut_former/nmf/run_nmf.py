import logging
import argparse

import pandas as pd

from gut_former.utils.project_paths import find_data_path, find_output_path
from gut_former.utils.log_config import setup_logging
from gut_former.utils.data_prep import train_val_split, collapse_to_genus, filter_samples
from gut_former.nmf.nmf_core import run_nmf_loop

log = logging.getLogger(__name__)


def load_data(data_type: str, dataset: str, subset: str) -> pd.DataFrame:
    """Load and preprocess input data based on data type."""
    data_path = find_data_path()
    metadata = pd.read_csv(f"{data_path}/metadata_{dataset}.csv", index_col=0, low_memory=False)
    if "sample_id" in metadata.columns:
        metadata = metadata.set_index("sample_id")
    metadata = metadata.sort_index()

    if data_type == "pathways":
        df = pd.read_csv(f"{data_path}/pathways_{dataset}.csv", index_col=0, low_memory=False).fillna(0).sort_index() * 100
    else:  # taxonomy
        df = pd.read_csv(f"{data_path}/taxonomy_{dataset}.csv", index_col=0, low_memory=False).fillna(0).sort_index()

    df = filter_samples(df, metadata, subset)

    if data_type == "taxonomy":
        df = collapse_to_genus(df)

    return df


def main():
    setup_logging()

    log.info("Starting NMF script..")

    p = argparse.ArgumentParser()
    p.add_argument("--type",         type=str, required=True,   help="Data type: pathways | taxonomy", choices=["pathways", "taxonomy"])
    p.add_argument("--dataset",      type=str, default="sample", help="Dataset name")
    p.add_argument("--n_signatures", type=int, default=8,        help="Number of NMF components")
    p.add_argument("--n_runs",       type=int, default=100,      help="Number of NMF runs per seed")
    p.add_argument("--filter",       type=str, default="all",    help="Sample subset: all | healthy | non-healthy",
                   choices=["all", "healthy", "non-healthy"])
    args = p.parse_args()

    # Load & preprocess
    X = load_data(args.type, args.dataset, args.filter)
    log.info("Data loaded | shape: %s", X.shape)

    # Train / val split
    X_train, X_val = train_val_split(X)
    log.info("Train: %s | Val: %s", X_train.shape, X_val.shape)

    # Run NMF
    best, history = run_nmf_loop(X_train, X_val, args.n_signatures, args.n_runs)

    # Save outputs
    output_path = find_output_path() / "nmf"
    output_path.mkdir(parents=True, exist_ok=True)

    n   = args.n_signatures
    ds  = args.dataset
    tag = f"{args.type}_{args.filter}"
    pd.DataFrame(best["H"],     columns=X_train.columns).to_csv(output_path / f"H_{n}_{ds}_{tag}.csv")
    pd.DataFrame(best["W"],     index=X_train.index)    .to_csv(output_path / f"W_train_{n}_{ds}_{tag}.csv")
    pd.DataFrame(best["W_val"], index=X_val.index)      .to_csv(output_path / f"W_val_{n}_{ds}_{tag}.csv")
    pd.DataFrame(history)                           .to_csv(output_path / f"nmf_stats_{n}_{ds}_{tag}.csv", index=False)

    log.info(
        "Done. Saved to %s:\n"
        "  - H_%d_%s_%s.csv\n"
        "  - W_train_%d_%s_%s.csv\n"
        "  - W_val_%d_%s_%s.csv\n"
        "  - nmf_stats_%d_%s_%s.csv",
        output_path, n, ds, tag, n, ds, tag, n, ds, tag, n, ds, tag,
    )


if __name__ == "__main__":
    main()
