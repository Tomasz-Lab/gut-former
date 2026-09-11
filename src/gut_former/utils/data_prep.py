import logging

import pandas as pd
from sklearn.model_selection import train_test_split

log = logging.getLogger(__name__)


def train_val_split(
    df: pd.DataFrame, split: pd.Series | None = None, test_size: float = 0.1, random_state: int = 0
):

    if split is not None:
        idx_train = split[split == "train"].index
        idx_val = split[split == "test"].index
        return df.loc[idx_train], df.loc[idx_val]

    idx_train, idx_val = train_test_split(df.index, test_size=test_size, random_state=random_state)
    return df.loc[idx_train], df.loc[idx_val]


def collapse_to_genus(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse species-level taxonomy columns to genus level, renormalized to relative abundance.

    Expects columns in pipe-separated format: k__|p__|c__|o__|f__|g__|s__
    Genus is extracted from position 5 (0-indexed).
    """
    genus_names = df.columns.str.split("|").str[5]
    genus_df = df.copy()
    genus_df.columns = genus_names
    genus_df = genus_df.T.groupby(level=0).sum().T  # sum duplicate genera
    genus_df = genus_df.div(genus_df.sum(axis=1), axis=0)  # renormalize to relative abundance
    return genus_df.dropna()


def filter_samples(df: pd.DataFrame, metadata: pd.DataFrame, subset: str) -> pd.DataFrame:
    """Filter DataFrame rows by health status using study_condition column in metadata.

    subset:
        "all"         → no filter
        "healthy"     → keep study_condition == "control"
        "non-healthy" → keep study_condition != "control"
    """
    if subset == "all":
        return df

    common = df.index.intersection(metadata.index)
    if len(common) == 0:
        raise ValueError("No common samples between data and metadata — check index alignment.")

    meta = metadata.loc[common]
    if subset == "healthy":
        keep = meta[meta["study_condition"] == "control"].index
    else:  # non-healthy
        keep = meta[meta["study_condition"] != "control"].index

    log.info("Filter '%s': %d/%d samples kept", subset, len(keep), len(df))
    return df.loc[keep]
