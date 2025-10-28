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

MODEL = "gru"
# MODEL = "sarima"

# --- MODIFICAÇÃO INÍCIO ---
# 1. Renomeie os diretórios para servirem como "BASE"
# Diretório BASE contendo os arquivos y_pred (predições do modelo)
BASE_PREDICTION_DIR = BASE_DIR/"data"/"prediction"/MODEL
# Diretório BASE contendo os arquivos y_true (dados de teste reais)
BASE_TEST_DIR = BASE_DIR/"data"/"prediction"/"test"/MODEL
# Diretório BASE onde os CSVs de métricas serão salvos
BASE_OUTPUT_DIR = BASE_DIR/"results"/"prediction"/MODEL

# 2. Defina a lista de subdiretórios para processar
# Para processar .../MODEL/10 e .../MODEL/20:
SUB_DIRS = [""] 
# Para processar APENAS .../MODEL/ (comportamento original), use:
# SUB_DIRS = [""] 
# --- MODIFICAÇÃO FIM ---




IMPUTATION_TECHNIQUES = [
    "ARIMA",
    "HoltWinters", 
    # "IterativeSVD",  
    "Kalman",
    "KNN_Sklearn",
    "Linear",
    "LOCF",
    "Mean",
    # "SoftImpute",   
    "Spline",
    "SVD_KNN_Hankel",
    "SVD_KNN"
]

COL_TRUE = 'Vazao'
COL_PRED = 'prediction'


def calculate_metrics(y_true, y_pred):
    """
    Calcula todas as métricas e retorna um dicionário.
    (Versão robusta com try/except individuais para evitar falhas)
    """
    mask = ~ (np.isnan(y_true) | np.isnan(y_pred))
    y_true_clean = y_true[mask]
    y_pred_clean = y_pred[mask]
    
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
        "valid_pairs": len(y_true_clean)
    }

    if metrics["valid_pairs"] == 0:
        print("   - Aviso: Não há dados válidos (não-NaN) para comparar.")
        return metrics

    try:
        metrics["rmse"] = rmse(y_true_clean, y_pred_clean)
        metrics["mae"] = mae(y_true_clean, y_pred_clean)
        metrics["r2"] = r2_score(y_true_clean, y_pred_clean)
    except Exception as e:
        print(f"   - Erro (RMSE/MAE/R2): {e}")

    try:
        metrics["mape"] = mape(y_true_clean, y_pred_clean)
    except ZeroDivisionError:
        print(f"   - Aviso: Divisão por zero no MAPE (y_true contém zeros). Definindo como NaN.")
    except Exception as e:
        print(f"   - Erro (MAPE): {e}")

    try:
        metrics["nrmse_range"] = nrmse(y_true_clean, y_pred_clean, mode="range")
    except ZeroDivisionError:
        print(f"   - Aviso: Divisão por zero no NRMSE-Range (range=0). Definindo como NaN.")
    except Exception as e:
        print(f"   - Erro (NRMSE-Range): {e}")

    try:
        metrics["nrmse_std"] = nrmse(y_true_clean, y_pred_clean, mode="std")
    except ZeroDivisionError:
        print(f"   - Aviso: Divisão por zero no NRMSE-Std (std=0). Definindo como NaN.")
    except Exception as e:
        print(f"   - Erro (NRMSE-Std): {e}")
        
    try:
        metrics["dtw_magnitude"] = dtw_distance(y_true_clean, y_pred_clean, normalize_for_shape=False)
        metrics["dtw_shape"] = dtw_distance(y_true_clean, y_pred_clean, normalize_for_shape=True)
    except Exception as e:
        print(f"   - Erro (DTW): {e}")

    try:
        metrics["fam"] = compute_fuzzy_accuracy(y_true_clean, y_pred_clean)
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
    # (ex: "SVD_KNN" deve ser verificado antes de "KNN")
    sorted_techniques = sorted(IMPUTATION_TECHNIQUES, key=len, reverse=True)
    
    # --- MODIFICAÇÃO INÍCIO ---
    # 3. Adiciona loop principal para iterar sobre os subdiretórios
    for sub_dir in SUB_DIRS:
        # Define os caminhos dinâmicos para esta iteração
        # (Se sub_dir for "", os caminhos serão os mesmos que os BASE)
        current_prediction_dir = BASE_PREDICTION_DIR / sub_dir
        current_test_dir = BASE_TEST_DIR / sub_dir
        current_output_dir = BASE_OUTPUT_DIR / sub_dir

        print(f"\n--- Processando Subdiretório: '{sub_dir if sub_dir else 'BASE'}' ---")

        # Cria o diretório de saída específico
        current_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Dicionário para armazenar os resultados (resetado para cada sub_dir)
        all_results = defaultdict(list)
        
        print(f"Buscando arquivos de teste em: {current_test_dir}")
        
        # Itera sobre os arquivos de TESTE (ground truth) no diretório ATUAL
        test_files = list(current_test_dir.glob("*_imputed_test_gru.csv"))
        # test_files = list(current_test_dir.glob("*_imputed_test.csv"))
    # --- MODIFICAÇÃO FIM --- (O restante do código fica dentro deste novo loop)

        if not test_files:
            print(f"Aviso: Nenhum arquivo '*_imputed_test_gru.csv' encontrado em {current_test_dir}")
            # print(f"Aviso: Nenhum arquivo '*_imputed_test.csv' encontrado em {current_test_dir}")
            continue # Pula para o próximo sub_dir

        for test_path in test_files:
            test_name = test_path.name
            
            base_identifier = test_name.removesuffix('_imputed_test_gru.csv')
            # base_identifier = test_name.removesuffix('_imputed_test.csv')

            # Encontra a qual dataset e técnica este arquivo pertence
            found_match = False
            original_name = ""
            imputation_tech = ""
            
            for tech in sorted_techniques:
                suffix = f"_6h_{tech}"
                if base_identifier.endswith(suffix):
                    original_name = base_identifier.removesuffix(suffix)
                    imputation_tech = tech
                    found_match = True
                    break
            
            if not found_match:
                print(f"Aviso: Ignorando arquivo (não foi possível parsear técnica): {test_name}")
                continue

            # Monta o nome do arquivo de predição esperado
            pred_file_name = f"{original_name}_6h_{imputation_tech}_imputed_{MODEL}_prediction.csv"
            
            # --- MODIFICAÇÃO INÍCIO ---
            # 4. Usa o diretório de predição ATUAL
            pred_path = current_prediction_dir / pred_file_name
            # --- MODIFICAÇÃO FIM ---
            
            if not pred_path.exists():
                print(f"Aviso: Arquivo de predição não encontrado: {pred_file_name} em {current_prediction_dir}")
                continue
            
            print(f"Processando: {original_name} | {imputation_tech} | {MODEL}")
            
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

                # Garante que os vetores tenham o mesmo tamanho
                if len(y_true) != len(y_pred):
                    # CORRIGIDO: 'continue' em vez de 'return' para não parar o script
                    print(f"   Erro: Tamanhos diferentes em {test_path.name} ({len(y_true)}) e {pred_path.name} ({len(y_pred)}). Pulando...")
                    # continue
                    raise IndexError
                
                # Calcula todas as métricas
                metrics = calculate_metrics(y_true, y_pred)
                
                # Adiciona as chaves de identificação
                result_row = {
                    "imputation_technique": imputation_tech,
                    "model": MODEL,
                    **metrics  
                }

                all_results[original_name].append(result_row)
                
            except Exception as e:
                print(f"   ERRO CRÍTICO ao processar {pred_path}: {e}")

        print(f"\nProcessamento do subdiretório '{sub_dir if sub_dir else 'BASE'}' concluído. Salvando resultados...")
        
        if not all_results:
            print("Nenhum resultado foi calculado para este subdiretório.")
            continue # Pula para o próximo sub_dir

        for original_name, results_list in all_results.items():
            if not results_list:
                continue
                
            # Converte a lista de resultados deste dataset em um DataFrame
            df_out = pd.DataFrame(results_list)
            
            # Define a ordem final das colunas (baseado nas chaves da nova função calculate_metrics)
            columns_order = [
                "imputation_technique", "model", 
                # Métricas de Erro (↓ Menor é melhor)
                "rmse", "nrmse_range", "nrmse_std", "mae", "mape", 
                # Métrica de Acurácia (↑ Maior é melhor)
                "r2", "fam",
                # Métricas de Forma/Estrutura (↓ Menor é melhor)
                "dtw_magnitude", "dtw_shape",
                # Contagem
                "valid_pairs"
            ]
            
            # Garante que apenas colunas existentes e na ordem correta sejam usadas
            final_cols = [col for col in columns_order if col in df_out.columns]
            df_out = df_out[final_cols]
            
            # Salva o CSV
            output_filename = f"{original_name}_metrics.csv"
            
            # --- MODIFICAÇÃO INÍCIO ---
            # 5. Salva no diretório de saída ATUAL
            output_path = current_output_dir / output_filename
            # --- MODIFICAÇÃO FIM ---
            
            df_out.to_csv(output_path, index=False, float_format='%.6f')
            print(f"Resultados para '{original_name}' salvos em: {output_path}")

if __name__ == "__main__":
    main()