import argparse
import logging

import pandas as pd
from utils.project_paths import find_data_path, find_output_path
from utils.log_config import setup_logging

log = logging.getLogger(__name__)

def main():
    # Setting up project
    setup_logging()

    log.info("Starting data preparation script..\n ")
    # Handling Args
    p = argparse.ArgumentParser()

    p.add_argument("--dataset", type=str, default="raw", help="TODO")

    args = p.parse_args()

    data_path = find_data_path()

    # Loading Data
    T_raw = pd.read_csv(f"{data_path}/taxonomy_relab_{args.dataset}.csv", index_col=[0], low_memory=False)
    P_raw = pd.read_csv(f"{data_path}/pathways_relab_{args.dataset}.csv", index_col=[0], low_memory=False)
    M_raw = pd.read_csv(f"{data_path}/metadata_{args.dataset}.csv", index_col=0, low_memory=False)

    log.info(f"T_raw: (rows={T_raw.shape[0]}, cols={T_raw.shape[1]}) | "
          f"P_raw: (rows={P_raw.shape[0]}, cols={P_raw.shape[1]}) | "
          f"M_raw: (rows={M_raw.shape[0]}, cols={M_raw.shape[1]})")

if __name__ == "__main__":
    main()