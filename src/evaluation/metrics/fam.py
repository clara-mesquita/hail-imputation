# # Malu's Fuzzy Accuracy Metric 

import numpy as np
import pandas as pd
import os

BINS_MBPS = np.arange(0, 2100, 100)
# LABELS = "Vazao"

def fuzzy_weight(distance):
    """
    A Malu disse que esses pesos são o que devem ser ajustados
    """
    return {0: 1.0, 1: 0.75, 2: 0.5}.get(distance, 0.0)

def compute_fuzzy_accuracy(y_true, y_pred):
    df = pd.DataFrame({'y_test': y_true, 'y_pred': y_pred}).dropna()
    if df.empty:
        return np.nan

    df['y_test_mbps'] = df['y_test'] / 1e6
    df['y_pred_mbps'] = df['y_pred'] / 1e6

    # df['true_bin'] = pd.cut(df['y_test_mbps'], bins=BINS_MBPS, labels=False, right=False)
    # df['pred_bin'] = pd.cut(df['y_pred_mbps'], bins=BINS_MBPS, labels=False, right=False)

    df['true_idx'] = pd.cut(df['y_test_mbps'], bins=BINS_MBPS, labels=False, right=False)
    df['pred_idx'] = pd.cut(df['y_pred_mbps'], bins=BINS_MBPS, labels=False, right=False)

    # label_to_index = {label: idx for idx, label in enumerate(False)}
    # df['true_idx'] = df['true_bin'].map(label_to_index)
    # df['pred_idx'] = df['pred_bin'].map(label_to_index)

    df = df.dropna(subset=['true_idx', 'pred_idx'])
    if df.empty:
        return np.nan

    df['distance'] = (df['true_idx'].astype(int) - df['pred_idx'].astype(int)).abs()
    df['fuzzy_weight'] = df['distance'].apply(fuzzy_weight)

    return df['fuzzy_weight'].mean() * 100
