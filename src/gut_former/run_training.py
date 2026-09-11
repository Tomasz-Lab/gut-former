import argparse
import logging
from datetime import date

import pandas as pd
import torch
from torch.nn.functional import mse_loss
from torch.optim import Adam
from torch.utils.data import DataLoader

from gut_former.config import TrainingConfig
from gut_former.dataset import BacteriaDataset
from gut_former.loss import CompositeLoss
from gut_former.model import BacteriaModel
from gut_former.utils.data_prep import train_val_split
from gut_former.utils.log_config import setup_logging
from gut_former.utils.project_paths import find_data_path, find_output_path

log = logging.getLogger(__name__)


def main():
    # Handling Args

    config = TrainingConfig()
    p = argparse.ArgumentParser()

    p.add_argument("--dataset", type=str, default=config.data.dataset, help="TODO")
    p.add_argument(
        "--taxonomy-file", type=str, default=None, help="Path to taxonomy CSV (overrides --dataset)"
    )
    p.add_argument(
        "--pathways-file", type=str, default=None, help="Path to pathways CSV (overrides --dataset)"
    )
    p.add_argument("--split-file", type=str, default=None, help="Path to CSV with split columns")
    p.add_argument(
        "--split-column",
        type=str,
        default=None,
        help="Column in --split-file with train/test labels",
    )
    p.add_argument(
        "--run-name", type=str, default=None, help="Output file prefix (defaults to date_dataset)"
    )
    p.add_argument("--embedding_dim", type=int, default=config.model.embedding_dim, help="TODO")
    p.add_argument("--latent_dim", type=int, default=config.model.latent_dim, help="TODO")
    p.add_argument("--batch_size", type=int, default=config.batch_size, help="TODO")
    p.add_argument("--learning_rate", type=float, default=config.learning_rate, help="TODO")
    p.add_argument("--epochs", type=int, default=config.epochs, help="Maximum number of epochs")
    p.add_argument(
        "--patience",
        type=int,
        default=100,
        help="Early stopping patience (epochs without improvement)",
    )
    p.add_argument(
        "--verbose", action=argparse.BooleanOptionalAction, default=config.verbose, help="TODO"
    )

    p.add_argument(
        "--checkpoint",
        type=str,
        default=config.checkpoint_path,
        help="Path to checkpoint to resume training from",
    )
    args = p.parse_args()

    # Setting up project
    setup_logging(logging.INFO if args.verbose else logging.WARNING)

    log.info("Starting training script..\n ")

    today_str = date.today().strftime("%Y%m%d")
    output_path = find_output_path()
    run_name = args.run_name or f"{today_str}_{args.dataset}_{config.model.model_version}"
    checkpoint_path = f"{output_path}/{run_name}_checkpoint.pt"
    stats_path = f"{output_path}/{run_name}_training_stats.csv"

    # Loading & Preparing Data
    data_path = find_data_path()
    taxonomy_file = args.taxonomy_file or f"{data_path}/taxonomy_{args.dataset}.csv"
    pathways_file = args.pathways_file or f"{data_path}/pathways_{args.dataset}.csv"
    Xt_df = pd.read_csv(taxonomy_file, index_col=[0], low_memory=False).fillna(0).sort_index() * 100
    Xp_df = pd.read_csv(pathways_file, index_col=[0], low_memory=False).fillna(0).sort_index() * 100

    if bool(args.split_file) != bool(args.split_column):
        raise ValueError("--split-file and --split-column must be given together")

    split = None
    if args.split_file:
        split = pd.read_csv(args.split_file, index_col=0, low_memory=False)[args.split_column]
        log.info(
            "Using predefined split %r from %s: %d train, %d test",
            args.split_column,
            args.split_file,
            (split == "train").sum(),
            (split == "test").sum(),
        )

    Xt_train, Xt_test = train_val_split(Xt_df, split=split)
    Xp_train, Xp_test = train_val_split(Xp_df, split=split)

    train_dataset = BacteriaDataset(Xt_train, Xp_train)
    train_dloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    test_dataset = BacteriaDataset(Xt_test, Xp_test)
    test_dloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Defining Model
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BacteriaModel(
        Xp_df.shape[1],
        Xt_df.shape[1],
        args.embedding_dim,
        args.latent_dim,
    ).to(DEVICE)

    optimizer = Adam(model.parameters(), lr=args.learning_rate, weight_decay=0.001)
    composite_loss = CompositeLoss()

    if args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint, map_location=DEVICE))
        log.info("Resumed from checkpoint: %s", args.checkpoint)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    log.info(
        "Model summary:\n  - Total params:      %d\n  - Trainable params:  %d\n",
        total_params,
        trainable_params,
    )

    # Training Model
    history = []
    best_val_loss = float("inf")
    epochs_without_improvement = 0
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

        if epoch_metrics["test_loss"] < best_val_loss:
            best_val_loss = epoch_metrics["test_loss"]
            epochs_without_improvement = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                log.info(
                    "Early stopping at epoch %d (best validation loss: %.6f)", epoch, best_val_loss
                )
                break

    stats_cols = [
        "epoch",
        "train_loss",
        "train_taxonomy_mse",
        "train_pathways_mse",
        "test_loss",
        "test_taxonomy_mse",
        "test_pathways_mse",
    ]

    stats_df = pd.DataFrame(history, columns=stats_cols)
    stats_df.to_csv(stats_path, index=False)

    log.info(
        "\nTraining completed.\nSaved outputs:\n  - Training stats: %s\n  - Model checkpoint: %s\n",
        stats_path,
        checkpoint_path,
    )


if __name__ == "__main__":
    main()
