"""
Funções para avaliar outras questões além do erro médio
Usar o DTW como uma métrica de avaliação -> forma da série temporal é importante 
(ou mais importante que erro ponto a ponto, como é o caso do RMSE)
O DTW indica que as séries têm forma (shape) e padrão similares, mesmo que haja desalinhamento
"""

import numpy as np

from sklearn.preprocessing import MinMaxScaler
from tslearn.metrics import dtw

def dtw_distance(y_true: np.array, y_pred: np.array, normalize_for_shape: bool = False) -> float:
    """
    Calcula a distância Dynamic Time Warping (DTW) entre duas séries.
    
    :param normalize_for_shape: Se True, normaliza ambas as séries para [0, 1] 
                                antes de calcular o DTW. Isso avalia *apenas*
                                a forma, ignorando a amplitude.
    """
    if normalize_for_shape:
        # Usa MinMaxScaler para normalizar cada segmento individualmente
        # (reshape -1, 1) é necessário para o scaler
        scaler = MinMaxScaler()
        
        # Garante que não haja NaNs ou Infs (embora não devam existir aqui)
        # y_true_safe = np.nan_to_num(y_true.reshape(-1, 1))
        # y_pred_safe = np.nan_to_num(y_pred.reshape(-1, 1))

        y_true_norm = scaler.fit_transform(y_true)
        y_pred_norm = scaler.fit_transform(y_pred)

        return dtw(y_true_norm, y_pred_norm)
    
    else:
        # Calcula o DTW nos valores brutos (sensível à amplitude)
        return dtw(y_true, y_pred)