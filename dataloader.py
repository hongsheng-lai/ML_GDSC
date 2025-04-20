import pandas as pd
import numpy as np
import os
from kagglehub import kagglehub

def load_gdsc(
    path: str = None,
    dataset_name: str = "samiraalipour/genomics-of-drug-sensitivity-in-cancer-gdsc",
    excluded_columns: list = None,
) -> pd.DataFrame:
    """
    Load or download the GDSC dataset, drop missing values, and winsorize LN_IC50.
    Returns the cleaned DataFrame.
    """
    if excluded_columns is None:
        excluded_columns = [
            "LN_IC50", "AUC", "Z_SCORE",
            "DRUG_ID", "COSMIC_ID", "DRUG_NAME", "CELL_LINE_NAME",
        ]
    # Load
    if path and os.path.exists(path):
        df = pd.read_csv(path)
    else:
        archive = kagglehub.dataset_download(dataset_name)
        df = pd.read_csv(os.path.join(archive, "GDSC_DATASET.csv"))
    # Clean
    df = df.dropna()
    # Winsorize
    q1 = df["LN_IC50"].quantile(0.25)
    q3 = df["LN_IC50"].quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    df["LN_IC50"] = df["LN_IC50"].clip(lower=lower, upper=upper)
    return df


def prepare_features(
    df: pd.DataFrame,
    excluded_columns: list = None,
    encode_dummies: bool = True,
    drop_first: bool = False,
) -> (pd.DataFrame, pd.Series):
    """
    From cleaned DataFrame, produce X and y.
    If encode_dummies: one-hot encode categoricals,
    else: label-encode each categorical column.
    """
    if excluded_columns is None:
        excluded_columns = [
            "LN_IC50", "AUC", "Z_SCORE",
            "DRUG_ID", "COSMIC_ID", "DRUG_NAME", "CELL_LINE_NAME",
        ]
    y = df["LN_IC50"].reset_index(drop=True)
    X = df.drop(columns=excluded_columns).reset_index(drop=True)
    # identify categoricals
    cats = X.select_dtypes(include=["object", "category"]).columns.tolist()
    if encode_dummies:
        X = pd.get_dummies(X, columns=cats, drop_first=drop_first)
    else:
        for col in cats:
            X[col] = pd.factorize(X[col])[0]
    return X, y


def split_data(
    X1: pd.DataFrame,
    X2: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """
    Split indices once, returning train/test sets
    for both feature DataFrames X1 and X2, and y.
    """
    n = len(y)
    idx = np.arange(n)
    np.random.seed(random_state)
    np.random.shuffle(idx)
    split = int((1 - test_size) * n)
    train_idx, test_idx = idx[:split], idx[split:]
    return (
        X1.iloc[train_idx], X1.iloc[test_idx],
        X2.iloc[train_idx], X2.iloc[test_idx],
        y.iloc[train_idx], y.iloc[test_idx]
    )
