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
    args = p.parse_args()

    # Loading & Preparing Data
    data_path = find_data_path()
    t_df = pd.read_csv(f"{data_path}/taxonomy_{args.dataset}.csv", index_col=[0], low_memory=False).fillna(0).sort_index() * 100
    p_df = pd.read_csv(f"{data_path}/pathways_{args.dataset}.csv", index_col=[0], low_memory=False).fillna(0).sort_index() * 100

    t_tensor = torch.tensor(t_df.values, dtype=torch.float32).unsqueeze(-1)
    p_tensor = torch.tensor(p_df.values, dtype=torch.float32).unsqueeze(-1)

    # Recreating Model
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BacteriaModel(
        p_df.shape[1], t_df.shape[1], args.embedding_dim, args.latent_dim,
    ).to(DEVICE)

    log.info("Recreating model from checkpoint: %s\n", args.checkpoint)
    checkpoint_dict = torch.load(args.checkpoint, weights_only=True)
    model.load_state_dict(checkpoint_dict, strict=False)
    model.eval()
    model.cpu()

    # Inference
    t_pred, p_pred, latent = model(p_tensor, t_tensor)

    # Storing outputs
    t_pred_df = pd.DataFrame(t_pred.detach().cpu().numpy().squeeze(-1), index=t_df.index, columns=t_df.columns)
    t_pred_df.index.name = t_df.index.name
    t_output_path = f"{output_path}/pred_taxonomy_{args.dataset}.csv"
    t_pred_df.to_csv(t_output_path, index=True)

    p_pred_df = pd.DataFrame(p_pred.detach().cpu().numpy().squeeze(-1), index=p_df.index, columns=p_df.columns)
    p_pred_df.index.name = p_df.index.name
    p_output_path = f"{output_path}/pred_pathways_{args.dataset}.csv"
    p_pred_df.to_csv(p_output_path, index=True)

    latent_cols = [f"z{i}" for i in range(args.latent_dim)]
    latent_df = pd.DataFrame(latent.detach().cpu().numpy(), index=p_df.index, columns=latent_cols)
    latent_df.index.name = t_df.index.name
    latent_output_path = f"{output_path}/latent_{args.dataset}.csv"
    latent_df.to_csv(latent_output_path, index=True)

    log.info("Inference completed.\n"
             "Saved outputs:\n"
             "  - Latent: %s\n"
             "  - P_pred:  %s\n"
             "  - T_pred:  %s",
             latent_output_path, p_output_path, t_output_path)


if __name__ == "__main__":
    main()
