import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
from scipy import signal
from sklearn.impute import KNNImputer
from statsmodels.tsa.statespace.structural import UnobservedComponents
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from fancyimpute import SoftImpute
from fancyimpute import IterativeSVD
import re

BASE_DIR = Path(__file__).resolve().parent.parent.parent

DATA_IMPUTATION_DIR = BASE_DIR / "data" / "imputation"
MISSING_DIR = DATA_IMPUTATION_DIR / "missing_sample_data"
SAMPLES_DIR = DATA_IMPUTATION_DIR / "sample_data"
IMPUTED_DATASETS_FOLDER = DATA_IMPUTATION_DIR / "imputed_sample_data"

IMPUTATION_RESULTS_DIR = BASE_DIR / "results" / "imputation" / "imputed_sample_data" 
FINAL_REPORT_FILE = IMPUTATION_RESULTS_DIR / "sample_imputation_initial_results.txt"
FIGURES_FOLDER = IMPUTATION_RESULTS_DIR / "figures"

TIMESTAMP_COL = "Data"
THROUGHPUT_COLUMN = "Vazao"

MISSING_SENTINEL = np.nan
RANDOM_SEED = 42

MISSING_RATES = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4]
MISSING_RATES_FOLDERS = [str(int(rate * 100)) for rate in MISSING_RATES]



################## HAIL IMPUTATION: DON'T CHANGE ######################

# ============================================================================
# FFT Period Estimation
# ============================================================================

def estimate_period_fft(y, min_period=4, max_period=None):
    """Estimate period using FFT spectral analysis"""
    # pega a série como numpy array
    y = np.asarray(y, dtype=float)
    n = len(y)
    
    #essa parte faz sentido mesmo se os dados não chegam a mil?
    # não buscar períodos maiores que 1/4 tamanho da série mas tb nunca acima de mil 
    if max_period is None:
        max_period = max(7, min(n // 4, 1000))
    
    # deixando mais robusto evitando problemas com séries muito curtas 
    if n < min_period * 2:
        return max(min_period, min(n, 8))
    
    # é feita uma interpoção apenas para evitar problemas com NaNs e para o cálculo do FFT
    # mas a série original não é alterada
    y_series = pd.Series(y)
    y_filled = y_series.interpolate(limit_direction="both").ffill().bfill().to_numpy()

    # Verifica se a série preenchida é constante (se a diferença entre os valores não é mt pequena)
    if np.std(y_filled) < 1e-10:
        return min_period

    # Detrend e normaliza para capturar o fft sem influência de tendência ou escala    
    y_detrended = signal.detrend(y_filled)
    y_normalized = (y_detrended - np.mean(y_detrended)) / (np.std(y_detrended) + 1e-10)
    
    # Cálculo do FFT

    # já que a série é real e não complexa, é feita a fft apenas para a metade positiva do espectro - one-dimensional Discrete Fourier Transform (DFT) 
    # ffet_vals é um array complexo de tamanho n/2 + 1 (freq senoidal q corresponde ao sinal)
    fft_vals = np.fft.rfft(y_normalized)
    # power_spectrum é um vetor real e positivo com o quanto cada frequência contribui para o sinal.
    power_spectrum = np.abs(fft_vals) ** 2
    # frequências associadas a cada componente do espectro
    frequencies = np.fft.rfftfreq(n)
    
    # converte frequencia > periodo
    with np.errstate(divide='ignore', invalid='ignore'):
        periods = 1.0 / frequencies
        periods[0] = np.inf
    
    # ignora frequencias mt baixas
    valid_mask = (periods >= min_period) & (periods <= max_period)
    
    if not np.any(valid_mask):
        return min_period
    
    # pega o período onde o espectro da potência é máximo
    valid_power = power_spectrum[valid_mask]
    valid_periods = periods[valid_mask]
    best_period = valid_periods[np.argmax(valid_power)]
    
    return int(np.round(best_period))

# ============================================================================
# Matrix Construction and SVD Utilities
# ============================================================================

def fold_series_to_matrix(y, period):
    """Fold 1D series into (n_blocks, period) matrix"""
    y = np.asarray(y, dtype=float)
    n = len(y)
    n_blocks = int(np.ceil(n / period))
    pad_len = n_blocks * period - n
    
    if pad_len > 0:
        y = np.concatenate([y, np.full(pad_len, np.nan)])
    
    M = y.reshape(n_blocks, period)
    return M, n

def unfold_matrix_to_series(M, original_len):
    """Unfold matrix back to series"""
    y = M.reshape(-1)
    return y[:original_len]

# Transformação na matriz de hankel -> matriz de trajetória
# Uma matriz de Hankel tem valores constantes ao longo de suas antidiagonais 
def build_hankel_matrix(y, window_size):
    """Build Hankel matrix from time series"""
    y = np.asarray(y, dtype=float)
    n = len(y)
    
    if window_size > n:
        window_size = n
    
    K = n - window_size + 1
    L = window_size
    H = np.zeros((K, L))
    
    for i in range(K):
        H[i, :] = y[i:i+L]
    
    return H

# Reconstructing the series from the Hankel matrix
def hankel_to_series(H, method='diagonal_average'):
    """Reconstruct series from Hankel matrix"""
    K, L = H.shape
    n = K + L - 1
    
    if method == 'diagonal_average':
        y = np.zeros(n)
        counts = np.zeros(n)
        
        for i in range(K):
            for j in range(L):
                idx = i + j
                y[idx] += H[i, j]
                counts[idx] += 1
        
        y = y / counts
        
    elif method == 'first_row_col':
        y = np.concatenate([H[0, :], H[1:, -1]])
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return y

# Entendimento do espaço latente via SVD
# o svd "divide" a matriz original em camadas (estrutura principal -> padrão temporal/espacial)
def svd_rank(M_filled, energy=0.9):
    """Compute SVD and choose rank by cumulative energy"""
    U, s, Vt = np.linalg.svd(M_filled, full_matrices=False)

    # energia -> soma dos valores singulares (que vem de uma amtriz diagonal)
    # o uso de soma ao invés e soma quadrática ajuda a não evidenciar tanto os primeiros componentes
    # pode-se dizer que foi principalmente uma decisão empírica tanto porque performou melhor como foi mais consistente
    # e tb pode-se dizer que a ideia foi usar si como peso de cada padrão e não como potência
    total_energy = np.sum(s)
    # verifica se a energia é muito pequena 
    if total_energy < 1e-12:
        return U, s, Vt, 1

    # energia acumulada (igualzin o svd, tem a ver com a redução de dimensionalidade)
    cum_energy = np.cumsum(s) / total_energy
    r = int(np.searchsorted(cum_energy, energy) + 1)
    r = max(1, min(r, min(M_filled.shape)))
    
    return U, s, Vt, r

# imputando inicialmente 
# para poder aplicar o svd
def _initial_fill(M):
    """Column-wise median fill for initial SVD"""
    M_filled = M.copy()
    col_medians = np.nanmedian(M_filled, axis=0)
    # foi usado mediana porque é mais robusta para outliers
    global_median = np.nanmedian(M_filled)
    
    if np.isnan(global_median):
        global_median = 0.0
    
    # apenas imputando na mão
    col_medians = np.where(np.isnan(col_medians), global_median, col_medians)
    nan_indices = np.where(np.isnan(M_filled))
    M_filled[nan_indices] = np.take(col_medians, nan_indices[1])
    
    return M_filled


def knn_in_latent(M, k=5, energy=0.9, allow_future=True):
    """Impute missing values using KNN in SVD latent space"""
    M_filled = _initial_fill(M)
    U, s, Vt, r = svd_rank(M_filled, energy=energy)
    Z = U[:, :r] * s[:r]
    
    M_imputed = M.copy()
    T, P = M.shape
    observed_mask = ~np.isnan(M)
    
    # percorre cada linha i que tenha valores faltantes
    for i in range(T):
        missing_cols = np.where(~observed_mask[i])[0]
        if len(missing_cols) == 0:
            continue
        
        # ("vetor resumo")
        zi = Z[i]
        candidates = np.arange(T) if allow_future else np.arange(0, i)
        
        if len(candidates) == 0:
            candidates = np.arange(T)
        
        for j in missing_cols:
            valid_candidates = candidates[observed_mask[candidates, j]]
            
            if len(valid_candidates) == 0:
                continue
            
            distances = np.linalg.norm(Z[valid_candidates] - zi[None, :], axis=1) + 1e-8
            
            if len(distances) > k:
                nearest_indices = np.argpartition(distances, k)[:k]
                neighbor_rows = valid_candidates[nearest_indices]
                neighbor_distances = distances[nearest_indices]
            else:
                neighbor_rows = valid_candidates
                neighbor_distances = distances
            
            weights = 1.0 / neighbor_distances
            values = M[neighbor_rows, j]
            valid_mask = ~np.isnan(values)
            
            values = values[valid_mask]
            weights = weights[valid_mask]
            
            if len(values) > 0:
                M_imputed[i, j] = np.sum(weights * values) / np.sum(weights)
    
    return M_imputed

################## ANOTHER IMPUTATION METHODS ######################

def impute_svd_knn(df, col=THROUGHPUT_COLUMN, min_period=4, max_period=None, 
                   energy=0.9, k=5, allow_future=True, use_hankel=False):
    """SVD-KNN imputation with optional Hankel matrix"""
    y = df[col].to_numpy(dtype=float)
    period = estimate_period_fft(y, min_period=min_period, max_period=max_period)
    
    if use_hankel:
        M = build_hankel_matrix(y, window_size=period)
        M_imputed = knn_in_latent(M, k=k, energy=energy, allow_future=allow_future)
        y_imputed = hankel_to_series(M_imputed, method='diagonal_average')
    else:
        M, orig_len = fold_series_to_matrix(y, period=period)
        M_imputed = knn_in_latent(M, k=k, energy=energy, allow_future=allow_future)
        y_imputed = unfold_matrix_to_series(M_imputed, original_len=orig_len)
    
    df_imputed = df.copy()
    df_imputed[col] = y_imputed
    return df_imputed

def impute_linear(df, col=THROUGHPUT_COLUMN):
    """Linear interpolation"""
    df_imputed = df.copy()
    df_imputed[col] = df_imputed[col].interpolate(method="linear", limit_direction="both")
    return df_imputed

def impute_knn_sklearn(df, k=5, col=THROUGHPUT_COLUMN):
    """sklearn KNN imputer"""
    df_imputed = df.copy()
    imputer = KNNImputer(n_neighbors=k, weights="uniform")
    imputed_values = imputer.fit_transform(df_imputed[[col]])
    df_imputed[col] = imputed_values[:, 0]
    return df_imputed

def impute_spline(df, col=THROUGHPUT_COLUMN, order=3):
    """Spline interpolation"""
    df_imputed = df.copy()
    df_imputed[col] = df_imputed[col].interpolate(method='spline', order=order, limit_direction='both')
    return df_imputed

def impute_mean(df, col=THROUGHPUT_COLUMN):
    """Mean imputation"""
    df_imputed = df.copy()
    mean_value = df_imputed[col].mean()
    df_imputed[col] = df_imputed[col].fillna(mean_value)
    return df_imputed

def impute_locf(df, col=THROUGHPUT_COLUMN):
    """Last Observation Carried Forward"""
    df_imputed = df.copy()
    df_imputed[col] = df_imputed[col].ffill()
    return df_imputed

def impute_kalman(df, col=THROUGHPUT_COLUMN, min_period=4, max_period=None):
    """Kalman Filter imputation using UnobservedComponents"""
    df_imputed = df.copy()
    y = df_imputed[col].copy()
    
    # Estimate seasonal period
    y_temp = y.interpolate(limit_direction="both").ffill().bfill()
    period = estimate_period_fft(y_temp.values, min_period=min_period, max_period=max_period)
    period = max(2, min(period, len(y) // 3))
    
    try:
        # Fit UnobservedComponents model with trend and seasonality
        mod = UnobservedComponents(
            y, 
            level="local linear trend", 
            seasonal=period,
            irregular=True
        )
        res = mod.fit(disp=False, maxiter=100)
        
        # Use Kalman smoother estimates
        df_imputed[col] = y.fillna(res.fittedvalues)
        
    except Exception as e:
        # Fallback to simpler model if fitting fails
        try:
            mod = UnobservedComponents(y, level="local level", irregular=True)
            res = mod.fit(disp=False, maxiter=50)
            df_imputed[col] = y.fillna(res.fittedvalues)
        except:
            # Final fallback to linear interpolation
            df_imputed[col] = y.interpolate(limit_direction="both")
    
    return df_imputed

def impute_arima(df, col=THROUGHPUT_COLUMN, order=(1,1,1)):
    """ARIMA imputation"""
    df_imputed = df.copy()
    y = df_imputed[col].copy()
    
    # Initial fill for ARIMA fitting
    y_filled = y.interpolate(limit_direction="both").ffill().bfill()
    
    try:
        # Fit ARIMA model
        model = ARIMA(y_filled, order=order)
        fit = model.fit()
        
        # Get fitted values
        fitted = fit.fittedvalues
        
        # Fill missing values with fitted values
        missing_mask = y.isna()
        df_imputed.loc[missing_mask, col] = fitted[missing_mask]
        
    except Exception as e:
        # Fallback to linear interpolation
        df_imputed[col] = y.interpolate(limit_direction="both")
    
    return df_imputed

def impute_holtwinters(df, col=THROUGHPUT_COLUMN, min_period=4, max_period=None):
    """Holt-Winters Exponential Smoothing imputation"""
    df_imputed = df.copy()
    y = df_imputed[col].copy()
    
    # Estimate seasonal period
    y_temp = y.interpolate(limit_direction="both").ffill().bfill()
    period = estimate_period_fft(y_temp.values, min_period=min_period, max_period=max_period)
    period = max(2, min(period, len(y) // 3))
    
    # Need at least 2 full seasons
    if len(y_temp) < 2 * period:
        df_imputed[col] = y.interpolate(limit_direction="both")
        return df_imputed
    
    try:
        # Fit Holt-Winters model
        model = ExponentialSmoothing(
            y_temp, 
            trend="add", 
            seasonal="add", 
            seasonal_periods=period
        )
        fit = model.fit(optimized=True)
        
        # Fill missing values with fitted values
        missing_mask = y.isna()
        df_imputed.loc[missing_mask, col] = fit.fittedvalues[missing_mask]
        
    except Exception as e:
        # Fallback to additive trend only
        try:
            model = ExponentialSmoothing(y_temp, trend="add", seasonal=None)
            fit = model.fit(optimized=True)
            missing_mask = y.isna()
            df_imputed.loc[missing_mask, col] = fit.fittedvalues[missing_mask]
        except:
            # Final fallback
            df_imputed[col] = y.interpolate(limit_direction="both")
    
    return df_imputed

# def impute_softimpute(df, col=THROUGHPUT_COLUMN, max_rank=5):
#     """SoftImpute matrix completion"""
    
#     df_imputed = df.copy()
#     y = df_imputed[col].values.reshape(-1, 1).astype(float)
    
#     try:
#         # Apply SoftImpute
#         imputer = SoftImpute(max_rank=max_rank, verbose=False)
#         y_imputed = imputer.fit_transform(y)
#         df_imputed[col] = y_imputed.ravel()
        
#     except Exception as e:
#         print("Not possible to soft impute impute")
#         return SyntaxError
    
#     return df_imputed

# def impute_iterativesvd(df, col=THROUGHPUT_COLUMN, rank=5):
#     """IterativeSVD matrix completion (similar to SoftImpute)"""

#     df_imputed = df.copy()
#     y = df_imputed[col].values.reshape(-1, 1).astype(float)
    
#     imputer = IterativeSVD(rank=rank, verbose=False)
#     y_imputed = imputer.fit_transform(y)
#     df_imputed[col] = y_imputed.ravel()
    
#     return df_imputed



def rmse(a, b):
    """Root Mean Squared Error"""
    diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    return float(np.sqrt(np.nanmean(diff**2)))

def mae(a, b):
    """Mean Absolute Error"""
    diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    return float(np.nanmean(np.abs(diff)))

def evaluate_imputation(df_original, df_imputed, missing_mask):
    """Calculate RMSE and MAE for imputed values"""
    orig_vals = df_original[THROUGHPUT_COLUMN].to_numpy()[missing_mask]
    imp_vals = df_imputed[THROUGHPUT_COLUMN].to_numpy()[missing_mask]
    
    return {
        'rmse': rmse(orig_vals, imp_vals),
        'mae': mae(orig_vals, imp_vals),
        'n_points': int(missing_mask.sum())
    }

# def visualize_imputation(df_original, df_missing, df_imputed, method_name, 
#                         dataset_name, missing_rate, save_path):
#     """Visualize original, missing, and imputed time series"""
#     fig, ax = plt.subplots(figsize=(14, 5))
    
#     missing_mask = (df_missing[VALUE_COL] == MISSING_SENTINEL) | df_missing[VALUE_COL].isna()
    
#     ax.plot(df_original.index, df_original[VALUE_COL], 'b-', 
#             label='Original', alpha=0.7, linewidth=1.5)
#     ax.plot(df_imputed.index, df_imputed[VALUE_COL], 'r--', 
#             label=f'Imputed ({method_name})', alpha=0.7, linewidth=1.2)
#     ax.scatter(df_original.index[missing_mask], df_original[VALUE_COL][missing_mask], 
#                c='orange', s=30, label='Missing Points', zorder=5, alpha=0.6)
    
#     ax.set_title(f'{dataset_name} - {method_name} - Missing Rate: {missing_rate*100:.0f}%')
#     ax.set_xlabel('Time Index')
#     ax.set_ylabel(VALUE_COL)
#     ax.legend()
#     ax.grid(True, alpha=0.3)
    
#     fig.tight_layout()
#     fig.savefig(save_path, dpi=150, bbox_inches='tight')
#     plt.close(fig)

def setup_folders():
    """Cria todas as pastas de saída necessárias, incluindo subpastas de taxa"""
    os.makedirs(IMPUTED_DATASETS_FOLDER, exist_ok=True)
    os.makedirs(FIGURES_FOLDER, exist_ok=True)
    
    # Cria as subpastas para cada taxa
    for rate_folder in MISSING_RATES_FOLDERS:
        os.makedirs(os.path.join(IMPUTED_DATASETS_FOLDER, rate_folder), exist_ok=True)
        os.makedirs(os.path.join(FIGURES_FOLDER, rate_folder), exist_ok=True)

    print(f"Subpastas de CSVs criadas em: {IMPUTED_DATASETS_FOLDER}/[taxa]")
    print(f"Subpastas de Figuras criadas em: {FIGURES_FOLDER}/[taxa]")

def process_all_datasets():
    """Processa todos os datasets com todos os métodos de imputação"""
    setup_folders()
    
    # Define todos os métodos de imputação
    imputation_methods = {
        'SVD-KNN': lambda df: impute_svd_knn(df, k=10, use_hankel=False),
        'SVD-KNN-Hankel': lambda df: impute_svd_knn(df, k=10, use_hankel=True),
        'Kalman': impute_kalman,
        'ARIMA': impute_arima,
        'HoltWinters': impute_holtwinters,
        'Linear': impute_linear,
        'KNN-Sklearn': lambda df: impute_knn_sklearn(df, k=5),
        'Spline': impute_spline,
        'Mean': impute_mean,
        'LOCF': impute_locf,
        # 'IterativeSVD': lambda df: impute_iterativesvd(df, rank=5),
    }
    
    all_results = []
    
    with open(FINAL_REPORT_FILE, 'w', encoding='utf-8') as report:
        report.write("="*80 + "\n")
        report.write("RELATÓRIO DE AVALIAÇÃO DE IMPUTAÇÃO\n")
        report.write("="*80 + "\n\n")
        
        for rate_folder in MISSING_RATES_FOLDERS:
            current_rate_dir = os.path.join(MISSING_DIR, rate_folder)
            
            if not os.path.exists(current_rate_dir):
                print(f"Aviso: Diretório de taxa {current_rate_dir} não encontrado. Pulando.")
                continue
                
            print(f"\nProcessando pasta de taxa: {rate_folder}%")
            
            # Itera sobre os arquivos dentro da subpasta (ex: 'dados_mr10.csv')
            for missing_file in sorted(os.listdir(current_rate_dir)):
                if not missing_file.endswith('.csv'):
                    continue
                
                base_name_with_mr, extension = os.path.splitext(missing_file) 
                
                # Remove o sufixo _mrXX para obter o nome base
                # 'dados_mr10' -> 'dados'
                base_name = re.sub(f"_mr{rate_folder}$", "", base_name_with_mr)
                
                missing_rate_str = rate_folder
                missing_rate = int(missing_rate_str) / 100.0
                
                # Monta o nome do arquivo original (ex: 'dados_sample.csv')
                original_file = f"{base_name}_sample{extension}"
                original_path = os.path.join(SAMPLES_DIR, original_file)
                
                # Caminho completo para o arquivo com falhas
                missing_path = os.path.join(current_rate_dir, missing_file)
                                
                if not os.path.exists(original_path):
                    report.write(f"AVISO: Arquivo original '{original_file}' não encontrado em {SAMPLES_DIR} para {missing_file}\n\n")
                    print(f"AVISO: Arquivo original '{original_file}' não encontrado em {SAMPLES_DIR} para {missing_file}")
                    continue
                
                report.write("\n" + "="*80 + "\n")
                report.write(f"Dataset: {base_name}\n")
                report.write(f"Taxa de Falha: {missing_rate*100:.0f}%\n")
                report.write("="*80 + "\n\n")
                
                try:
                    df_original = pd.read_csv(original_path)
                    df_missing = pd.read_csv(missing_path)

                    df_missing_processed = df_missing.copy()
                    missing_mask = df_missing_processed[THROUGHPUT_COLUMN].isna()
                    
                    if not missing_mask.any():
                        report.write("AVISO: Nenhum dado faltante encontrado neste arquivo (mask.sum() == 0). Pulando.\n")
                        print(f"AVISO: {missing_file} não continha dados faltantes.")
                        continue
                        
                    report.write(f"Shape Original: {df_original.shape}\n")
                    report.write(f"Pontos Faltantes: {missing_mask.sum()} ({missing_mask.sum()/len(df_missing)*100:.1f}%)\n\n")
                    
                    report.write("Performance dos Métodos:\n")
                    report.write("-" * 60 + "\n")
                    report.write(f"{'Método':<20} {'RMSE':>12} {'MAE':>12} {'Pontos':>10}\n")
                    report.write("-" * 60 + "\n")
                    
                    for method_name, impute_func in imputation_methods.items():
                        try:
                            df_imputed = impute_func(df_missing_processed)

                            # Salva o dataset imputado
                            safe_method = method_name.replace(" ", "").replace("/", "_")
                            # Nome do arquivo (ex: 'dados_mr10_SVD_KNN_imputed.csv')
                            imputed_fname = f"{base_name}_mr{int(missing_rate*100)}_{safe_method}_imputed.csv"
                            
                            # Salva dentro da subpasta da taxa (ex: /imputed_sample_data/10/)
                            imputed_fpath = os.path.join(IMPUTED_DATASETS_FOLDER, rate_folder, imputed_fname)
                            df_imputed.to_csv(imputed_fpath, index=False)

                            # Avalia e loga
                            metrics = evaluate_imputation(df_original, df_imputed, missing_mask)
                            report.write(f"{method_name:<20} {metrics['rmse']:>12.4f} "
                                         f"{metrics['mae']:>12.4f} {metrics['n_points']:>10}\n")

                            all_results.append({
                                'dataset': base_name,
                                'missing_rate': missing_rate,
                                'method': method_name,
                                'rmse': metrics['rmse'],
                                'mae': metrics['mae'],
                                'n_points': metrics['n_points'],
                                'imputed_csv': imputed_fpath
                            })

                            # # Exporta a figura (Comentado a pedido)
                            # fig_name = f"{base_name}_mr{int(missing_rate*100)}_{safe_method}.png"
                            # # Salva dentro da subpasta da taxa (ex: /figures/10/)
                            # fig_path = os.path.join(FIGURES_FOLDER, rate_folder, fig_name)
                            # visualize_imputation(df_original, df_missing, df_imputed,
                            #                      method_name, base_name, missing_rate, fig_path)

                        except Exception as e:
                            error_msg = f"{method_name:<20} ERRO: {str(e)}"
                            report.write(f"{error_msg}\n")
                            print(f"Erro no método {method_name} para {missing_file}: {str(e)}")

                except Exception as e:
                    error_msg = f"ERRO ao processar arquivo {missing_file}: {str(e)}"
                    report.write(f"{error_msg}\n\n")
                    print(error_msg)
                    continue
        
        results_df = pd.DataFrame(all_results)
        
        if len(results_df) > 0:
            report.write("\n" + "="*80 + "\n")
            report.write("ESTATÍSTICAS RESUMO\n")
            report.write("="*80 + "\n\n")
            
            # Agrupa por taxa de falha
            for missing_rate_val in sorted(results_df['missing_rate'].unique()):
                report.write(f"\nTaxa de Falha: {missing_rate_val*100:.0f}%\n")
                report.write("-" * 60 + "\n")
                
                rate_data = results_df[results_df['missing_rate'] == missing_rate_val]
                # Agrupa por método para esta taxa
                summary = rate_data.groupby('method').agg({
                    'rmse': ['mean', 'std'],
                    'mae': ['mean', 'std']
                }).round(4)
                
                report.write(summary.to_string())
                report.write("\n\n")
                
                # Encontra o melhor para esta taxa (baseado na média)
                summary_mean = rate_data.groupby('method').agg({'rmse': 'mean', 'mae': 'mean'})
                
                if not summary_mean.empty:
                    best_rmse_method = summary_mean['rmse'].idxmin()
                    best_mae_method = summary_mean['mae'].idxmin()
                    report.write(f"Melhor Média RMSE: {best_rmse_method} ({summary_mean.loc[best_rmse_method, 'rmse']:.4f})\n")
                    report.write(f"Melhor Média MAE : {best_mae_method} ({summary_mean.loc[best_mae_method, 'mae']:.4f})\n")
                else:
                    report.write("Nenhum resultado de método bem-sucedido para esta taxa.\n")
            
            csv_path = os.path.join(IMPUTATION_RESULTS_DIR, "detailed_results.csv")
            results_df.to_csv(csv_path, index=False)
            report.write(f"\n\nResultados detalhados salvos em: {csv_path}\n")
        
        else:
            report.write("Nenhum resultado foi gerado.\n")
    
    print(f"\nProcessamento concluído!")
    print(f"Relatório salvo em: {FINAL_REPORT_FILE}")
    print(f"Figuras (desativadas) salvas em: {FIGURES_FOLDER}")
    
    if len(all_results) > 0:
        print(f"Resultados detalhados salvos em: {os.path.join(IMPUTATION_RESULTS_DIR, 'detailed_results.csv')}")
    else:
        print("Nenhum resultado detalhado foi salvo.")

if __name__ == "__main__":
    process_all_datasets()