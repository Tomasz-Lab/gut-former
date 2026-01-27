import json
import torch
import argparse

import pandas as pd
from model import BacteriaModel

def main():
    # Handling Args
    p = argparse.ArgumentParser()

    p.add_argument("--dataset", type=str, default="sample_train", help="TODO")
    p.add_argument("--embedding_dim", type=int, default=128, help="TODO")
    p.add_argument("--latent_dim", type=int, default=64, help="TODO")
    p.add_argument("--checkpoint", type=str, default="../../data/checkpoint_sample.pt", help="TODO")
    args = p.parse_args()

    # Loading & Preparing Data
    data_path = f"../../data"
    t_df = pd.read_csv(f"{data_path}/taxonomy_{args.dataset}.csv", index_col=[0], low_memory=False).fillna(0).sort_index() * 100
    p_df = pd.read_csv(f"{data_path}/pathways_{args.dataset}.csv", index_col=[0], low_memory=False).fillna(0).sort_index() * 100

    t_tensor = torch.tensor(t_df.values, dtype=torch.float32).unsqueeze(-1)
    p_tensor = torch.tensor(p_df.values, dtype=torch.float32).unsqueeze(-1)

    # Recreating Model
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BacteriaModel(
        p_df.shape[1], t_df.shape[1], args.embedding_dim, args.latent_dim,
    ).to(DEVICE)

    checkpoint_dict = torch.load(args.checkpoint, weights_only=True)
    model.load_state_dict(checkpoint_dict, strict=False)
    model.eval()
    model.cpu()

    # Inference
    t_pred, p_pred, latent = model(p_tensor, t_tensor)

    # Storing predictions
    t_pred_df = pd.DataFrame(t_pred.detach().cpu().numpy().squeeze(-1), index=t_df.index)
    t_pred_df.index.name = t_df.index.name
    t_pred_df.to_csv(f"{data_path}/output/pred_taxonomy_{args.dataset}.csv", index=True)

    p_pred_df = pd.DataFrame(p_pred.detach().cpu().numpy().squeeze(-1), index=t_df.index)
    p_pred_df.index.name = p_df.index.name
    p_pred_df.to_csv(f"{data_path}/output/pred_pathways_{args.dataset}.csv", index=True)

if __name__ == "__main__":
    main()