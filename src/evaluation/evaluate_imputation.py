import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
import re

# Importe suas métricas
from metrics.error import rmse, mae, nrmse, mape, r2_score
from metrics.dtw import dtw_distance
from metrics.fam import compute_fuzzy_accuracy

# Use Path() para compatibilidade entre Windows, Mac e Linux
# (Assumindo que este script está em um subdiretório como 'src/evaluation')
try:
    BASE_DIR = Path(__file__).resolve().parent.parent.parent
except NameError:
    # Fallback se executado interativamente (ex: Jupyter)
    BASE_DIR = Path.cwd().parent 

# Diretório BASE contendo os arquivos y_pred (predições do modelo)
BASE_PREDICTION_DIR = BASE_DIR/"data"/"imputation"/"imputed_sample_data"
# Diretório BASE contendo os arquivos y_true (dados de teste reais)
BASE_TEST_DIR = BASE_DIR/"data"/"imputation"/"sample_data"
# Diretório BASE onde os CSVs de métricas serão salvos
BASE_OUTPUT_DIR = BASE_DIR/"results"/"imputation"/"imputed_sample_data"

# Defina a lista de subdiretórios para processar
# Para processar ...
# 10 e ...
# 20:
SUB_DIRS = [str(i) for i in range(5, 41, 5)]
# Para processar APENAS ...
#  (comportamento original), use:
# SUB_DIRS = [""] 


IMPUTATION_TECHNIQUES = [
    "ARIMA",
    "HoltWinters", 
    # "IterativeSVD",  
    "Kalman",
    "KNN-Sklearn",
    "Linear",
    "LOCF",
    "Mean",
    # "SoftImpute",   
    "Spline",
    "SVD-KNN-Hankel",
    "SVD-KNN"
]

COL_TRUE = 'Vazao'
COL_PRED = 'Vazao'


def calculate_metrics(y_true, y_pred):
    """
    Calcula todas as métricas e retorna um dicionário.
    (Versão robusta com try/except individuais para evitar falhas)
    """
    mask = ~ (np.isnan(y_true) | np.isnan(y_pred))
    y_true_clean = y_true[mask]
    y_pred_clean = y_pred[mask]

    y_true_gbps = y_true_clean / 1_000_000
    y_pred_gbps = y_pred_clean / 1_000_000

    # Inicializa todas as métricas com NaN
    metrics = {
        "rmse": np.nan,
        "nrmse_range": np.nan,
        "nrmse_std": np.nan,
        "mae": np.nan,
        "mape": np.nan,
        "r2": np.nan,
        "fam": np.nan,
        "dtw_magnitude": np.nan, 
        "dtw_shape": np.nan,
        "valid_pairs": len(y_true_gbps)
    }

    if metrics["valid_pairs"] == 0:
        print("   - Aviso: Não há dados válidos (não-NaN) para comparar.")
        return metrics

    try:
        metrics["rmse"] = rmse(y_true_gbps, y_pred_gbps)
        metrics["mae"] = mae(y_true_gbps, y_pred_gbps)
        metrics["r2"] = r2_score(y_true_gbps, y_pred_gbps)
    except Exception as e:
        print(f"   - Erro (RMSE/MAE/R2): {e}")

    try:
        metrics["mape"] = mape(y_true_gbps, y_pred_gbps)
    except ZeroDivisionError:
        print(f"   - Aviso: Divisão por zero no MAPE (y_true contém zeros). Definindo como NaN.")
    except Exception as e:
        print(f"   - Erro (MAPE): {e}")

    try:
        metrics["nrmse_range"] = nrmse(y_true_gbps, y_pred_gbps, mode="range")
    except ZeroDivisionError:
        print(f"   - Aviso: Divisão por zero no NRMSE-Range (range=0). Definindo como NaN.")
    except Exception as e:
        print(f"   - Erro (NRMSE-Range): {e}")

    try:
        metrics["nrmse_std"] = nrmse(y_true_gbps, y_pred_gbps, mode="std")
    except ZeroDivisionError:
        print(f"   - Aviso: Divisão por zero no NRMSE-Std (std=0). Definindo como NaN.")
    except Exception as e:
        print(f"   - Erro (NRMSE-Std): {e}")
        
    try:
        metrics["dtw_magnitude"] = dtw_distance(y_true_gbps, y_pred_gbps, normalize_for_shape=False)
        metrics["dtw_shape"] = dtw_distance(y_true_gbps, y_pred_gbps, normalize_for_shape=True)
    except Exception as e:
        print(f"   - Erro (DTW): {e}")

    try:
        metrics["fam"] = compute_fuzzy_accuracy(y_true_gbps, y_pred_gbps)
    except ValueError as e:
        # Captura especificamente o erro de binning
        if "Bin labels" in str(e) or "cannot be equal" in str(e):
            print(f"   - Aviso: Erro 'Bin labels' na FAM (y_true constante?). Definindo como NaN.")
        else:
            print(f"   - Erro (FAM): {e}")
    except Exception as e:
         print(f"   - Erro (FAM): {e}")

    return metrics

def main():
    
    # Ordena técnicas da mais longa para a mais curta para evitar correspondências parciais
    sorted_techniques = sorted(IMPUTATION_TECHNIQUES, key=len, reverse=True)
    
    for sub_dir in SUB_DIRS:
        # Define os caminhos dinâmicos
        current_prediction_dir = BASE_PREDICTION_DIR / sub_dir
        current_test_dir = BASE_TEST_DIR # O diretório de teste é sempre o mesmo
        current_output_dir = BASE_OUTPUT_DIR / sub_dir

        print(f"current_prediction_dir: {current_prediction_dir}")
        print(f"current_test_dir: {current_test_dir}")
        print(f"current_output_dir: {current_output_dir}")

        print(f"\n--- Processando Subdiretório: '{sub_dir if sub_dir else 'BASE'}' ---")

        current_output_dir.mkdir(parents=True, exist_ok=True)
        
        all_results = defaultdict(list)
        
        # --- INÍCIO DA CORREÇÃO LÓGICA ---
        
        print(f"Buscando arquivos de PREDIÇÃO (imputados) em: {current_prediction_dir}")
        
        # 1. Itera sobre os arquivos de PREDIÇÃO (imputados), não de teste
        # Filtra para pegar apenas os arquivos do missing rate atual
        prediction_files = list(current_prediction_dir.glob(f"*_mr{sub_dir}_*_imputed.csv"))

        if not prediction_files:
            print(f"Aviso: Nenhum arquivo '*_mr{sub_dir}_*_imputed.csv' encontrado em {current_prediction_dir}")
            continue # Pula para o próximo sub_dir

        # Itera sobre cada arquivo de predição encontrado
        for pred_path in prediction_files:
            pred_name = pred_path.name
            
            # 2. Tenta extrair a técnica e o nome original do ARQUIVO DE PREDIÇÃO
            found_match = False
            original_name = ""
            imputation_tech = ""
            
            for tech in sorted_techniques:
                # Sufixo que esperamos encontrar no arquivo de predição
                suffix_to_find = f"_6h_mr{sub_dir}_{tech}_imputed.csv"
                
                if pred_name.endswith(suffix_to_find):
                    # Encontramos! Extrai o nome base
                    original_name = pred_name.removesuffix(suffix_to_find)
                    imputation_tech = tech
                    found_match = True
                    break # Para o loop de técnicas
            
            if not found_match:
                print(f"Aviso: Ignorando arquivo (não foi possível parsear técnica): {pred_name}")
                continue # Pula para o próximo arquivo de predição

            # 3. Monta o nome do arquivo de TESTE (ground truth) correspondente
            test_file_name = f"{original_name}_6h_sample.csv"
            test_path = current_test_dir / test_file_name
            
            # 4. Verifica se o arquivo de teste existe
            if not test_path.exists():
                print(f"Aviso: Arquivo de TESTE (ground truth) não encontrado: {test_file_name} em {current_test_dir}")
                continue # Pula para o próximo arquivo de predição
            
            # Se chegamos aqui, temos os dois arquivos (pred_path e test_path)
            print(f"Processando: {original_name} | {imputation_tech} | {sub_dir}")
            
            try:
                df_test = pd.read_csv(test_path)
                df_pred = pd.read_csv(pred_path)
                
                if COL_TRUE not in df_test.columns:
                    print(f"   Erro: Coluna '{COL_TRUE}' não encontrada em {test_path}")
                    continue
                if COL_PRED not in df_pred.columns:
                    print(f"   Erro: Coluna '{COL_PRED}' não encontrada em {pred_path}")
                    continue

                y_true = df_test[COL_TRUE].values
                y_pred = df_pred[COL_PRED].values

                if len(y_true) != len(y_pred):
                    print(f"   Erro: Tamanhos diferentes em {test_path.name} ({len(y_true)}) e {pred_path.name} ({len(y_pred)}). Pulando...")
                    # CORRIGIDO: Usando 'continue' como seu comentário sugeria, 
                    # em vez de 'raise IndexError' que pararia o script.
                    continue 
                
                metrics = calculate_metrics(y_true, y_pred)
                
                result_row = {
                    "imputation_technique": imputation_tech,
                    "missing_rate": sub_dir,
                    **metrics  
                }

                all_results[original_name].append(result_row)
                
            except Exception as e:
                print(f"   ERRO CRÍTICO ao processar {pred_path}: {e}")
        
        # --- FIM DA CORREÇÃO LÓGICA ---

        print(f"\nProcessamento do subdiretório '{sub_dir if sub_dir else 'BASE'}' concluído. Salvando resultados...")
        
        if not all_results:
            print("Nenhum resultado foi calculado para este subdiretório.")
            continue 

        for original_name, results_list in all_results.items():
            if not results_list:
                continue
                
            df_out = pd.DataFrame(results_list)
            
            # (O resto do seu script para salvar o CSV está correto)
            columns_order = [
                "imputation_technique", "missing_rate", # <-- Corrigi "model" para "missing_rate"
                # Métricas de Erro (↓ Menor é melhor)
                "rmse", "nrmse_range", "nrmse_std", "mae", "mape", 
                # Métrica de Acurácia (↑ Maior é melhor)
                 "fam",
                # Métricas de Forma/Estrutura (↓ Menor é melhor)
                "dtw_magnitude", "dtw_shape",
                # Contagem
                "valid_pairs"
            ]
            
            final_cols = [col for col in columns_order if col in df_out.columns]
            df_out = df_out[final_cols]
            
            output_filename = f"{original_name}_metrics.csv"
            output_path = current_output_dir / output_filename
            
            df_out.to_csv(output_path, index=False, float_format='%.6f')
            print(f"Resultados para '{original_name}' salvos em: {output_path}")
        
if __name__ == "__main__":
    main()