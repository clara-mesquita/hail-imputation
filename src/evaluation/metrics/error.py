# For NRMSE RMSE MAPE and MAE 
# Existe a opção de adicionar um parâmetro "mask", que, quando tiver no cálculo, calcula só a diff dos na máscara 
# Ou fazer isso antes de chamar as funções de erro

import numpy as np 

def rmse(y_true: np.array, y_pred: np.array):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def nrmse(y_true, y_pred, mode="range"):
    if mode == "range":
        denom = np.max(y_true) - np.min(y_true)
    elif mode == "max":
        denom = (np.max(y_true)) 
    else:
        denom = np.std(y_true)
    return rmse(y_true, y_pred) / denom if denom != 0 else np.nan

def mape(y_true, y_pred):
    # evitar divisão por 0
    non_zero_mask = y_true != 0
    if np.sum(non_zero_mask) == 0:
        return np.nan
    
    y_true_filtered = y_true[non_zero_mask]
    y_pred_filtered = y_pred[non_zero_mask]

    return np.mean(np.abs((y_true_filtered - y_pred_filtered) / y_true_filtered)) * 100

def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / ss_tot if ss_tot != 0 else np.nan