# ============================================================
# ARIMA/SARIMA Time Series Trainer with Safe CV
# ============================================================
import os
import json
import pickle
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from itertools import product
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error

from statsmodels.tsa.statespace.sarimax import SARIMAX

# ============================================
# CONFIGURAÇÕES - AJUSTE AQUI
# ============================================
INPUT_FOLDER = './imputation_results_original'
OUTPUT_MODEL_FOLDER = './modelo_salvo_arima'     # pode ser None para não salvar
OUTPUT_JSON = 'evaluation_rmse_mae_arima.json'
TRAIN_TEST_SPLIT = 0.8

GRID_SEARCH_ENABLED = True
# Grades pequenas primeiro (rápido). Depois você pode expandir.
PARAM_GRID = {
    'p': [0, 1, 2],
    'd': [0, 1],
    'q': [0, 1, 2],
}

# Sazonalidade (opcional). Se ligar, o script pode estimar período via FFT quando SEASONAL_PERIOD=None.
SEASONAL_ENABLED = False
SEASONAL_PERIOD = None     # Ex.: 24 para dados horários com ciclo diário; 7 para diário com ciclo semanal
SPARAM_GRID = {
    'P': [0, 1],
    'D': [0, 1],
    'Q': [0, 1],
}

N_SPLITS = 2  # validação temporal (2 folds para ficar leve)
MAX_TRAIN_TRIES = 1  # tentativas de refit por fold (mantenha 1 para velocidade)

# ============================================
# Utilidades
# ============================================
def estimate_period_fft(y, min_period=4, max_period=None):
    """
    Estima um período sazonal dominante via FFT de forma bem simples.
    Retorna None se não conseguir estimar algo razoável.
    """
    y = np.asarray(y, dtype=float).flatten()
    n = len(y)
    if n < 2*min_period:
        return None

    if max_period is None:
        max_period = max(7, min(n // 4, 1000))

    # preenche NaN linearmente para análise espectral
    s = pd.Series(y)
    y_filled = s.interpolate(limit_direction="both").bfill().ffill().to_numpy()

    # remove média
    y0 = y_filled - np.nanmean(y_filled)

    # FFT
    fft = np.fft.rfft(y0)
    freqs = np.fft.rfftfreq(n, d=1.0)

    # ignora frequência zero
    freqs[0] = np.nan
    power = np.abs(fft)

    # converte frequência em período
    with np.errstate(divide='ignore', invalid='ignore'):
        periods = 1.0 / freqs

    mask = (periods >= min_period) & (periods <= max_period) & np.isfinite(periods)
    if not np.any(mask):
        return None

    idx = np.nanargmax(power[mask])
    period_est = int(round(periods[mask][idx]))
    return period_est if period_est >= min_period else None


def select_value_column(df):
    # tenta heurística semelhante à do seu GRU
    date_cols = [c for c in df.columns if c.lower() in ['data','date','timestamp','datetime']]
    value_cols = [c for c in df.columns if c.lower() in ['vazao','throughput','value','valor','flow']]

    cols = df.columns.tolist()
    if '0' in cols:
        df = df.drop(columns=['0'])

    target_col = None
    if value_cols:
        target_col = value_cols[0]
    else:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [c for c in numeric_cols if c not in date_cols]
        if not numeric_cols:
            raise ValueError("Nenhuma coluna numérica encontrada.")
        target_col = numeric_cols[0]
    return df[[target_col]].dropna(), target_col


def ts_cv_score(endog, order, seasonal_order=None, n_splits=2, max_train_tries=1):
    """
    Faz validação cruzada temporal para uma tupla (order, seasonal_order).
    Retorna o MSE médio nos folds de validação.
    """
    y = np.asarray(endog, dtype=float).flatten()
    tscv = TimeSeriesSplit(n_splits=n_splits)
    val_mse = []

    for fold, (tr_idx, val_idx) in enumerate(tscv.split(y), 1):
        y_tr = y[tr_idx]
        y_val = y[val_idx]

        # Ajusta modelo no fold
        attempt = 0
        fitted = None
        while attempt < max_train_tries and fitted is None:
            try:
                model = SARIMAX(
                    y_tr,
                    order=order,
                    seasonal_order=seasonal_order if seasonal_order else (0,0,0,0),
                    enforce_stationarity=False,
                    enforce_invertibility=False
                )
                res = model.fit(disp=False)
                fitted = res
            except Exception:
                attempt += 1
                fitted = None

        if fitted is None:
            # penaliza combinações que não convergem
            return np.inf

        # Prevê horizonte = len(y_val) (one-shot) para manter simples/rápido
        try:
            fc = fitted.forecast(steps=len(y_val))
        except Exception:
            return np.inf

        mse = mean_squared_error(y_val, fc)
        val_mse.append(mse)

    return float(np.mean(val_mse)) if val_mse else np.inf


def grid_search_arima(train_series, param_grid, seasonal_enabled=False,
                      seasonal_period=None, sparam_grid=None, n_splits=2):
    """
    Varre combinações de (p,d,q) e, se sazonal, de (P,D,Q,m).
    Usa TimeSeriesSplit para obter MSE médio na validação.
    """
    p_list = param_grid.get('p', [0,1,2])
    d_list = param_grid.get('d', [0,1])
    q_list = param_grid.get('q', [0,1,2])

    if seasonal_enabled:
        # detecta período se não fornecido
        m = seasonal_period if seasonal_period else estimate_period_fft(train_series)
        if not m or m < 2:
            # se não conseguiu estimar um período bom, desliga sazonalidade para esta série
            seasonal_enabled = False
            m = 0
        else:
            m = int(m)
    else:
        m = 0

    results = []
    best_score = np.inf
    best_params = None

    if seasonal_enabled:
        P_list = sparam_grid.get('P', [0,1])
        D_list = sparam_grid.get('D', [0,1])
        Q_list = sparam_grid.get('Q', [0,1])
        combos = product(p_list, d_list, q_list, P_list, D_list, Q_list)
    else:
        combos = product(p_list, d_list, q_list)

    for combo in combos:
        if seasonal_enabled:
            p,d,q,P,D,Q = combo
            order = (p,d,q)
            seasonal_order = (P,D,Q,m)
            label = {'order': order, 'seasonal_order': seasonal_order}
        else:
            p,d,q = combo
            order = (p,d,q)
            seasonal_order = None
            label = {'order': order}

        try:
            score = ts_cv_score(train_series, order, seasonal_order, n_splits=N_SPLITS, max_train_tries=MAX_TRAIN_TRIES)
        except Exception as e:
            score = np.inf

        results.append({'params': label, 'val_mse': float(score)})

        if score < best_score:
            best_score = score
            best_params = label

    return best_params, results


def fit_final_and_evaluate(train_series, test_series, best_params):
    """
    Ajusta o modelo final no treino e avalia no teste.
    Usa forecast de múltiplos passos (steps=len(test)).
    """
    order = tuple(best_params['order'])
    seasonal_order = tuple(best_params['seasonal_order']) if 'seasonal_order' in best_params else (0,0,0,0)

    model = SARIMAX(
        train_series,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    res = model.fit(disp=False)

    # previsão no horizonte de teste
    fc = res.forecast(steps=len(test_series))

    rmse = float(np.sqrt(mean_squared_error(test_series, fc)))
    mae = float(mean_absolute_error(test_series, fc))

    return res, {'rmse': rmse, 'mae': mae, 'predictions': fc.tolist()}


def save_model(result_model, directory, filename):
    if not directory:
        return
    Path(directory).mkdir(parents=True, exist_ok=True)
    file_path = os.path.join(directory, f"{filename}_SARIMAX.pkl")
    with open(file_path, "wb") as f:
        pickle.dump(result_model, f)
    print(f"Modelo salvo em: '{file_path}'")


def process_file(filepath, filename, output_model_dir, use_grid_search):
    print("\n" + "="*60)
    print(f"Processando: {filename}")
    print("="*60)

    df = pd.read_csv(filepath)
    print(f"Colunas detectadas: {list(df.columns)}")

    df_clean, target_col = select_value_column(df)
    print(f"Coluna de valores: '{target_col}'")
    print(f"Registros após limpeza: {len(df_clean)}")

    if len(df_clean) < 50:
        raise ValueError(f"Dados insuficientes após limpeza: {len(df_clean)} registros")

    # Split treino/teste
    split_idx = int(len(df_clean) * TRAIN_TEST_SPLIT)
    train = df_clean.iloc[:split_idx, 0].to_numpy(dtype=float)
    test  = df_clean.iloc[split_idx:, 0].to_numpy(dtype=float)

    print(f"Tamanho treino: {len(train)}, Tamanho teste: {len(test)}")

    if use_grid_search:
        best_params, grid_results = grid_search_arima(
            train,
            PARAM_GRID,
            seasonal_enabled=SEASONAL_ENABLED,
            seasonal_period=SEASONAL_PERIOD,
            sparam_grid=SPARAM_GRID,
            n_splits=N_SPLITS
        )
        if best_params is None:
            raise RuntimeError("Grid search não encontrou parâmetros válidos.")
    else:
        # fallback básico
        best_params = {'order': (1,1,1)}
        if SEASONAL_ENABLED and (SEASONAL_PERIOD or estimate_period_fft(train)):
            m = SEASONAL_PERIOD or estimate_period_fft(train)
            if m and m >= 2:
                best_params['seasonal_order'] = (0,1,1,int(m))
        grid_results = None

    print(f"Melhores parâmetros: {best_params}")

    final_model, metrics = fit_final_and_evaluate(train, test, best_params)
    metrics['best_params'] = best_params
    if grid_results:
        metrics['grid_search_results'] = grid_results

    print(f"RMSE (teste): {metrics['rmse']:.4f}")
    print(f"MAE  (teste): {metrics['mae']:.4f}")

    save_model(final_model, output_model_dir, filename.replace('.csv',''))

    return metrics


def arima_prediction(source_dir, output_model_dir=None, output_json='evaluation_rmse_mae_arima.json',
                     use_grid_search=True):
    evaluation = {}

    print("\n" + "="*70)
    print("INÍCIO DO PROCESSAMENTO (ARIMA/SARIMA)")
    print(f"Pasta de entrada: {source_dir}")
    print(f"Grid Search: {'ATIVADO' if use_grid_search else 'DESATIVADO'}")
    print("="*70 + "\n")

    # coleta CSVs
    csv_files = []
    for root, _, files in os.walk(source_dir):
        for file in files:
            if file.endswith(".csv"):
                csv_files.append((root, file))

    print(f"Encontrados {len(csv_files)} arquivos CSV\n")

    for idx, (root, file) in enumerate(csv_files, 1):
        filepath = os.path.join(root, file)
        print(f"\n[{idx}/{len(csv_files)}] {file}")

        try:
            results = process_file(filepath, file, output_model_dir, use_grid_search)
            evaluation[file] = results
        except Exception as e:
            print(f"ERRO ao processar {file}: {e}")
            evaluation[file] = {'error': str(e)}

    with open(output_json, 'w') as f:
        json.dump(evaluation, f, indent=4)

    print("\n" + "="*70)
    print("PROCESSAMENTO CONCLUÍDO")
    print(f"Resultados salvos em: {output_json}")
    print("="*70 + "\n")

    # resumo
    print("RESUMO DOS RESULTADOS:")
    print("="*70)
    successful, total_rmse, total_mae = 0, 0.0, 0.0
    for file, metrics in evaluation.items():
        if 'error' in metrics:
            print(f"❌ {file}: ERRO - {metrics['error']}")
        else:
            successful += 1
            total_rmse += metrics['rmse']
            total_mae += metrics['mae']
            print(f"✓ {file}: RMSE={metrics['rmse']:.4f}, MAE={metrics['mae']:.4f}")

    if successful > 0:
        print("\n" + "="*70)
        print(f"Média RMSE: {total_rmse/successful:.4f}")
        print(f"Média MAE: {total_mae/successful:.4f}")
        print(f"Taxa de sucesso: {successful}/{len(evaluation)}")
        print("="*70 + "\n")

    return evaluation


if __name__ == "__main__":
    results = arima_prediction(
        source_dir=INPUT_FOLDER,
        output_model_dir=OUTPUT_MODEL_FOLDER,
        output_json=OUTPUT_JSON,
        use_grid_search=GRID_SEARCH_ENABLED
    )
