import pandas as pd
import numpy as np

# --- ADAPTAÇÃO ---
# Bins originais eram 0-2400 *Mbps*.
# Nossos dados estão em Gbps, então adaptamos os bins para 0-2400 *Gbps*.
BINS_GBPS = np.arange(0, 2500, 100) # De 0 a 2400, com passo de 100 Gbps
LABELS = [f'{BINS_GBPS[i]}-{BINS_GBPS[i+1]} Gbps' for i in range(len(BINS_GBPS)-1)]
# --- FIM DA ADAPTAÇÃO ---

def fuzzy_weight(distance):
    """A lógica de peso (0, 1, 2) não muda."""
    return {0: 1.0, 1: 0.75, 2: 0.5}.get(distance, 0.0)

def compute_fuzzy_accuracy(y_true, y_pred):
    """
    Calcula a Fuzzy Accuracy em uma escala de GBPS.
    Espera os valores de entrada BRUTOS (em Kbps) para fazer a conversão.
    """
    df = pd.DataFrame({'y_test': y_true, 'y_pred': y_pred}).dropna()
    if df.empty:
        return np.nan

    # --- ADAPTAÇÃO ---
    # Converte de Kbps (bruto) para Gbps (nossa unidade de análise)
    # df['y_test_gbps'] = df['y_test'] / 1_000_000
    # df['y_pred_gbps'] = df['y_pred'] / 1_000_000
    # --- FIM DA ADAPTAÇÃO ---

    # Agora, usamos os bins em GBPS
    df['true_bin'] = pd.cut(df['y_test'], bins=BINS_GBPS, labels=LABELS, right=False)
    df['pred_bin'] = pd.cut(df['y_pred'], bins=BINS_GBPS, labels=LABELS, right=False)

    label_to_index = {label: idx for idx, label in enumerate(LABELS)}
    df['true_idx'] = df['true_bin'].map(label_to_index)
    df['pred_idx'] = df['pred_bin'].map(label_to_index)

    # Se valores caírem fora dos bins (ex: > 2400 Gbps), eles se tornarão NaN.
    # Você pode querer verificar isso. Por enquanto, dropamos.
    df = df.dropna(subset=['true_idx', 'pred_idx'])
    if df.empty:
        # Isso pode acontecer se TODOS os seus dados forem maiores que 2400 Gbps
        print("Aviso: Todos os dados caíram fora dos bins definidos (0-2400 Gbps).")
        return np.nan

    df['distance'] = (df['true_idx'].astype(int) - df['pred_idx'].astype(int)).abs()
    df['fuzzy_weight'] = df['distance'].apply(fuzzy_weight)

    return df['fuzzy_weight'].mean() * 100