import json
import torch
import argparse

import numpy as np
import pandas as pd


from loss import CompositeLoss
from model import BacteriaModel
from dataset import BacteriaDataset

from datetime import date
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.nn.functional import mse_loss
from sklearn.model_selection import train_test_split

def main():
    # Handling Args
    p = argparse.ArgumentParser()

    p.add_argument("--dataset", type=str, default="sample_train", help="TODO")
    p.add_argument("--embedding_dim", type=int, default=128, help="TODO")
    p.add_argument("--latent_dim", type=int, default=64, help="TODO")
    p.add_argument("--batch_size", type=int, default=16, help="TODO")
    p.add_argument("--learning_rate", type=float, default=1.893292917167e-4, help="TODO")
    p.add_argument("--epochs", type=int, default=55, help="TODO")

    today_str = date.today().strftime("%Y%m%d")
    p.add_argument("--checkpoint", type=str, default=f"../../data/{today_str}_checkpoint.pt", help="TODO")
    args = p.parse_args()

    # Loading & Preparing Data
    data_path = f"../../data"
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

    # Training Model
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


        if epoch % 10 == 1:
            print(f"Epoch: {epoch}")
            print(f"Train Loss: {train_loss / train_n_samples:.3f}")
            print(f"Train Taxonomy MSE: {train_t_mse / train_n_samples:.3f}")
            print(f"Train Pathways MSE: {train_p_mse / train_n_samples:.3f}")
            print(f"Test Loss: {test_loss / test_n_samples:.3f}")
            print(f"Test Taxonomy MSE: {test_t_mse / test_n_samples:.3f}")
            print(f"Test Pathways MSE: {test_t_mse / test_n_samples:.3f}")

            torch.save(model.state_dict, args.checkpoint)

    torch.save(model.state_dict, args.checkpoint)

if __name__ == "__main__":
    main()