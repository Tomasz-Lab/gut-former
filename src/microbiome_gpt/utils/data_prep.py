import pandas as pd
from sklearn.model_selection import train_test_split


def train_val_split(df: pd.DataFrame, test_size: float = 0.1, random_state: int = 0):
    idx_train, idx_val = train_test_split(df.index, test_size=test_size, random_state=random_state)
    return df.loc[idx_train], df.loc[idx_val]
