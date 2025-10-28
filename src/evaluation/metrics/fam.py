import numpy as np
import pandas as pd
import os

# --- Definição dos Bins ---

# 1. Bins de 0-100, 100-200, ..., 1900-2000
BINS_MBPS_BASE = np.arange(0, 2100, 100)

# 2. Labels para esses bins (ex: "0-100", "100-200")
LABELS_BASE = [f'{BINS_MBPS_BASE[i]}-{BINS_MBPS_BASE[i+1]}' for i in range(len(BINS_MBPS_BASE)-1)]

# 3. CORREÇÃO: Adiciona o bin "Infinito" para capturar erros graves (> 2000)
bins_list = BINS_MBPS_BASE.tolist()
bins_list.append(np.inf)
BINS_MBPS = np.array(bins_list) # Nome final da constante [0, 100, ..., 2000, inf]

labels_list = LABELS_BASE.copy()
labels_list.append(f'2000-inf')
LABELS = labels_list # Nome final da constante ["0-100", ..., "2000-inf"]

# --- Fim da Definição dos Bins ---


def fuzzy_weight(distance):
    """
    Pesos difusos baseados na distância categórica, conforme o artigo.
    Distância 0 = 1.0, 1 = 0.75, 2 = 0.5, 3+ = 0.0
    """
    return {0: 1.0, 1: 0.75, 2: 0.5}.get(distance, 0.0)

def compute_fuzzy_accuracy(y_true, y_pred):
    """
    Calcula a Métrica de Acurácia Difusa (FAM).
    
    Retorna *sempre* um float (o score percentual) ou np.nan em caso de falha.
    """
    df = pd.DataFrame({'y_test': y_true, 'y_pred': y_pred}).dropna()
    if df.empty:
        # CORREÇÃO: Retorna np.nan (float), e não uma tupla
        return np.nan

    # Os dados de vazão estão em bps, a métrica usa Mbps
    df['y_test_mbps'] = df['y_test'] / 1e6
    df['y_pred_mbps'] = df['y_pred'] / 1e6

    # Categoriza usando os Bins (incluindo o bin 'inf')
    df['true_bin'] = pd.cut(df['y_test_mbps'], bins=BINS_MBPS, labels=LABELS, right=False)
    df['pred_bin'] = pd.cut(df['y_pred_mbps'], bins=BINS_MBPS, labels=LABELS, right=False)

    # Converte os labels (ex: '0-100') para índices (0, 1, 2...)
    label_to_index = {label: idx for idx, label in enumerate(LABELS)}
    df['true_idx'] = df['true_bin'].map(label_to_index)
    df['pred_idx'] = df['pred_bin'].map(label_to_index)

    # Remove NaNs (geralmente se algum valor for negativo)
    # Valores > 2000 são capturados pelo bin 'inf' e NÃO são removidos
    df = df.dropna(subset=['true_idx', 'pred_idx'])
    
    if df.empty:
        print("   - Aviso (FAM): Nenhum dado válido encontrado após binning.")
        # CORREÇÃO: Retorna np.nan (float)
        return np.nan

    # Calcula a distância categórica (ex: |bin 5 - bin 7| = 2)
    df['distance'] = (df['true_idx'].astype(int) - df['pred_idx'].astype(int)).abs()
    
    # Aplica os pesos difusos a cada amostra
    df['fuzzy_weight'] = df['distance'].apply(fuzzy_weight)

    # Retorna a média dos pesos, como um percentual
    return df['fuzzy_weight'].mean() * 100