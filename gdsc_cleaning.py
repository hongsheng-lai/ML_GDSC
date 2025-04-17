# gdsc_cleaning.py

import os
from typing import Optional

import numpy as np
import pandas as pd

import kagglehub


__all__ = ["load_and_clean_gdsc"]


def load_and_clean_gdsc(
    path: Optional[str] = None,
    dataset_name: str = "samiraalipour/genomics-of-drug-sensitivity-in-cancer-gdsc",
    excluded_columns: Optional[list] = None,
    drop_first: bool = False,
) -> pd.DataFrame:
    """
    Download (or load) the GDSC dataset, drop missing values, winsorize LN_IC50,
    and one-hot encode categorical features.

    Parameters
    ----------
    path : str, optional
        Local path to GDSC_DATASET.csv. If None, will download via kagglehub.
    dataset_name : str
        Identifier for kagglehub download.
    excluded_columns : list of str, optional
        Columns to leave out of dummy encoding. Defaults to:
        ['LN_IC50', 'AUC', 'Z_SCORE', 'DRUG_ID', 'COSMIC_ID', 'DRUG_NAME', 'CELL_LINE_NAME']
    drop_first : bool
        If True, drop the first level of each categorical feature to avoid collinearity.

    Returns
    -------
    pd.DataFrame
        The cleaned and encoded DataFrame.
    """
    # defaults
    if excluded_columns is None:
        excluded_columns = [
            "LN_IC50",
            "AUC",
            "Z_SCORE",
            "DRUG_ID",
            "COSMIC_ID",
            "DRUG_NAME",
            "CELL_LINE_NAME",
        ]

    # 1) Load
    if path and os.path.exists(path):
        df = pd.read_csv(path)
    else:
        # download archive & extract
        archive_path = kagglehub.dataset_download(dataset_name)
        # inside the archive we expect 'GDSC_DATASET.csv'
        df = pd.read_csv(os.path.join(archive_path, "GDSC_DATASET.csv"))

    # 2) Drop rows with any missing data
    df = df.dropna()

    # 3) Winsorize LN_IC50 via IQR
    q1 = df["LN_IC50"].quantile(0.25)
    q3 = df["LN_IC50"].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    df["LN_IC50"] = df["LN_IC50"].clip(lower=lower, upper=upper)

    # 4) One-hot encode all object (categorical) cols except excluded
    cats = [
        col
        for col in df.select_dtypes(include=["object", "category"]).columns
        if col not in excluded_columns
    ]
    if cats:
        dummies = pd.get_dummies(df[cats], drop_first=drop_first, dtype=np.uint8)
        df = pd.concat([df.drop(columns=cats), dummies], axis=1)

    return df


if __name__ == "__main__":
    # Example usage: cleans and shows basic info
    cleaned = load_and_clean_gdsc()
    print(f"Cleaned DataFrame: {cleaned.shape[0]} rows × {cleaned.shape[1]} cols")
