import argparse
import logging

import pandas as pd
import torch

from gut_former.config import InferenceConfig
from gut_former.model import BacteriaModel
from gut_former.utils.log_config import setup_logging
from gut_former.utils.project_paths import find_data_path, find_output_path

log = logging.getLogger(__name__)


def main():
    # Setting up project
    setup_logging()

    log.info("Starting inference script..\n ")

    # Handling Args
    config = InferenceConfig()
    p = argparse.ArgumentParser()

    p.add_argument("--dataset", type=str, default=config.data.dataset, help="TODO")
    p.add_argument("--embedding_dim", type=int, default=config.model.embedding_dim, help="TODO")
    p.add_argument("--latent_dim", type=int, default=config.model.latent_dim, help="TODO")
    output_path = find_output_path()
    checkpoint = config.checkpoint or f"{output_path}/checkpoint_sample.pt"
    p.add_argument("--checkpoint", type=str, default=checkpoint, help="TODO")
    p.add_argument(
        "--run-name", type=str, default=None, help="Output file suffix (defaults to --dataset)"
    )
    p.add_argument("--batch_size", type=int, default=64, help="Inference batch size")
    args = p.parse_args()

    run_name = args.run_name or args.dataset

    # Loading & Preparing Data
    data_path = find_data_path()
    t_df = (
        pd.read_csv(f"{data_path}/taxonomy_{args.dataset}.csv", index_col=[0], low_memory=False)
        .fillna(0)
        .sort_index()
        * 100
    )
    p_df = (
        pd.read_csv(f"{data_path}/pathways_{args.dataset}.csv", index_col=[0], low_memory=False)
        .fillna(0)
        .sort_index()
        * 100
    )

    t_tensor = torch.tensor(t_df.values, dtype=torch.float32).unsqueeze(-1)
    p_tensor = torch.tensor(p_df.values, dtype=torch.float32).unsqueeze(-1)

    # Recreating Model
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BacteriaModel(
        p_df.shape[1],
        t_df.shape[1],
        args.embedding_dim,
        args.latent_dim,
    ).to(DEVICE)

    log.info("Recreating model from checkpoint: %s\n", args.checkpoint)
    checkpoint_dict = torch.load(args.checkpoint, map_location=DEVICE, weights_only=True)
    model.load_state_dict(checkpoint_dict, strict=True)
    model.eval()

    # Inference
    t_chunks, p_chunks, z_chunks = [], [], []
    with torch.no_grad():
        for start in range(0, t_tensor.size(0), args.batch_size):
            stop = start + args.batch_size
            t_b, p_b = t_tensor[start:stop].to(DEVICE), p_tensor[start:stop].to(DEVICE)
            t_out, p_out, z_out = model(p_b, t_b)
            t_chunks.append(t_out.squeeze(-1).cpu())
            p_chunks.append(p_out.squeeze(-1).cpu())
            z_chunks.append(z_out.cpu())
    t_pred, p_pred, latent = torch.cat(t_chunks), torch.cat(p_chunks), torch.cat(z_chunks)
    log.info("Inference done for %d samples", latent.shape[0])

    # Storing outputs
    t_pred_df = pd.DataFrame(t_pred.numpy(), index=t_df.index, columns=t_df.columns)
    t_pred_df.index.name = t_df.index.name
    t_output_path = f"{output_path}/pred_taxonomy_{run_name}.csv"
    t_pred_df.to_csv(t_output_path, index=True)

    p_pred_df = pd.DataFrame(p_pred.numpy(), index=p_df.index, columns=p_df.columns)
    p_pred_df.index.name = p_df.index.name
    p_output_path = f"{output_path}/pred_pathways_{run_name}.csv"
    p_pred_df.to_csv(p_output_path, index=True)

    latent_cols = [f"z{i}" for i in range(args.latent_dim)]
    latent_df = pd.DataFrame(latent.numpy(), index=p_df.index, columns=latent_cols)
    latent_df.index.name = t_df.index.name
    latent_output_path = f"{output_path}/latent_{run_name}.csv"
    latent_df.to_csv(latent_output_path, index=True)

    log.info(
        "Inference completed.\nSaved outputs:\n  - Latent: %s\n  - P_pred:  %s\n  - T_pred:  %s",
        latent_output_path,
        p_output_path,
        t_output_path,
    )


if __name__ == "__main__":
    main()
