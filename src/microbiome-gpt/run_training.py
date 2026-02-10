import json
import torch
import argparse
import logging

import numpy as np
import pandas as pd
from datetime import date

from loss import CompositeLoss
from model import BacteriaModel
from dataset import BacteriaDataset
from utils.project_paths import find_data_path, find_output_path
from utils.log_config import setup_logging

from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.nn.functional import mse_loss
from sklearn.model_selection import train_test_split

log = logging.getLogger(__name__)

def main():
    # Setting up project
    setup_logging()

    log.info("Starting training script..\n ")
    # Handling Args

    p = argparse.ArgumentParser()

    p.add_argument("--dataset", type=str, default="sample_train", help="TODO")
    p.add_argument("--embedding_dim", type=int, default=128, help="TODO")
    p.add_argument("--latent_dim", type=int, default=64, help="TODO")
    p.add_argument("--batch_size", type=int, default=16, help="TODO")
    p.add_argument("--learning_rate", type=float, default=1.893292917167e-4, help="TODO")
    p.add_argument("--epochs", type=int, default=55, help="TODO")
    p.add_argument("--verbose", type=bool, default=True, help="TODO")

    today_str = date.today().strftime("%Y%m%d")
    output_path = find_output_path()
    p.add_argument("--checkpoint", type=str, default=f"{output_path}/{today_str}_checkpoint.pt", help="TODO")
    args = p.parse_args()

    # Loading & Preparing Data
    data_path = find_data_path()
    Xt_df = pd.read_csv(f"{data_path}/taxonomy_{args.dataset}.csv", index_col=[0], low_memory=False).fillna(0).sort_index() * 100
    Xp_df = pd.read_csv(f"{data_path}/pathways_{args.dataset}.csv", index_col=[0], low_memory=False).fillna(0).sort_index() * 100

    idx_train, idx_test = train_test_split(Xt_df.index, test_size=0.1, random_state=0)

    def split_df(df):
        return df.loc[idx_train], df.loc[idx_test]

    Xt_train, Xt_test = split_df(Xt_df)
    Xp_train, Xp_test = split_df(Xp_df)

    train_dataset = BacteriaDataset(Xt_train, Xp_train)
    train_dloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    test_dataset = BacteriaDataset(Xt_test, Xp_test)
    test_dloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Defining Model
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BacteriaModel(
        Xp_df.shape[1], Xt_df.shape[1], args.embedding_dim, args.latent_dim,
    ).to(DEVICE)

    optimizer = Adam(model.parameters(), lr=args.learning_rate, weight_decay=0.001)
    composite_loss = CompositeLoss()

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    log.info(
        "Model summary:\n"
        "  - Total params:      %d\n"
        "  - Trainable params:  %d\n",
        total_params, trainable_params
    )

    # Training Model
    history = []
    for epoch in range(args.epochs):
        model.train()

        train_loss = 0.0
        train_n_samples = 0
        train_t_mse = 0.0
        train_p_mse = 0.0
        for Xt_b, Xp_b in train_dloader:
            batch_size = Xt_b.size(0)
            train_n_samples += batch_size

            Xt_b = Xt_b.to(DEVICE)
            Xp_b = Xp_b.to(DEVICE)

            optimizer.zero_grad()

            Xt_pred, Xp_pred, latent = model(Xp_b, Xt_b)

            loss = composite_loss(Xt_b, Xt_pred, Xp_b, Xp_pred)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * batch_size
            train_t_mse += mse_loss(Xt_pred, Xt_b).item() * batch_size
            train_p_mse += mse_loss(Xp_pred, Xp_b).item() * batch_size

        model.eval()

        with torch.no_grad():
            test_loss = 0.0
            test_n_samples = 0
            test_t_mse = 0.0
            test_p_mse = 0.0
            for Xt_b, Xp_b in test_dloader:
                batch_size = Xt_b.size(0)
                test_n_samples += batch_size

                Xt_b = Xt_b.to(DEVICE)
                Xp_b = Xp_b.to(DEVICE)

                Xt_pred, Xp_pred, latent = model(Xp_b, Xt_b)
                loss = composite_loss(Xt_b, Xt_pred, Xp_b, Xp_pred)

                test_loss += loss.item() * batch_size
                test_t_mse += mse_loss(Xt_pred, Xt_b).item() * batch_size
                test_p_mse += mse_loss(Xp_pred, Xp_b).item() * batch_size

        epoch_metrics = {
            "epoch": epoch,
            "train_loss": train_loss / train_n_samples,
            "train_taxonomy_mse": train_t_mse / train_n_samples,
            "train_pathways_mse": train_p_mse / train_n_samples,
            "test_loss": test_loss / test_n_samples,
            "test_taxonomy_mse": test_t_mse / test_n_samples,
            "test_pathways_mse": test_p_mse / test_n_samples,  # <- note: use test_p_mse here
        }
        history.append(epoch_metrics)

        if epoch % 10 == 1:
            log.info(f"Epoch: {epoch} -> Test Loss: {test_loss:.3f}")
            torch.save(model.state_dict, args.checkpoint)

    stats_path = f"{output_path}/{today_str}_training_stats.csv"
    stats_cols = [
        "epoch", "train_loss", "train_taxonomy_mse", "train_pathways_mse",
        "test_loss", "test_taxonomy_mse", "test_pathways_mse",
    ]

    stats_df = pd.DataFrame(history, columns=stats_cols)
    stats_df.to_csv(stats_path, index=False)

    torch.save(model.state_dict, args.checkpoint)
    log.info("\nTraining completed.\n"
             "Saved outputs:\n"
             "  - Training stats: %s\n"
             "  - Model checkpoint: %s\n",
             stats_path, args.checkpoint)

if __name__ == "__main__":
    main()