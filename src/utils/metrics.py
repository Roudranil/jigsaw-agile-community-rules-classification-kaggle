from typing import Any, Dict, List, Tuple, Union

import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.metrics import roc_auc_score


def column_averaged_auc(eval_pred, logger):
    logger.info(f"{type(eval_pred) = }")
    logits, labels = eval_pred
    logger.info(
        f"{type(logits) = } | {type(labels) = } | {logits.shape = } | {labels.shape = }"
    )
    probs = softmax(logits, axis=1)[:, 1]
    auc = roc_auc_score(y_true=labels, y_score=probs)
    return {"auc": auc}
