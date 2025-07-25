from typing import Any, Dict, List, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def compute_auc_from_df(df: pd.DataFrame, true_labels: list | pd.Series) -> float:
    """
    Compute AUC for the 'rule_violation' column in the DataFrame.

    Parameters:
    - df: DataFrame with at least 'rule_violation' column containing prediction scores.
    - true_labels: List or Series with true binary labels corresponding to df rows.

    Returns:
    - auc_score: float, computed AUC score.
    """
    # Extract predicted scores from 'rule_violation' column
    y_scores = df["rule_violation"]

    # Compute AUC using sklearn
    auc_score = roc_auc_score(true_labels, y_scores)
    return auc_score
