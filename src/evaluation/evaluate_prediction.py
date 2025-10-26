# Generate csv for each file with all the evaluation metrics calculated (for prediction)

import pandas as pd
import numpy as np
from pathlib import Path
import sys
from collections import defaultdict
import re

from metrics.error import rmse, mae, nrmse, mape, r2_score

# Use Path() para compatibilidade entre Windows, Mac e Linux
BASE_DIR = Path(__file__).resolve().parent.parent.parent # Aponta para a raiz do projeto (acima de 'evaluation e src')

# MODEL = "gru"
MODEL = "sarima"
# Diretório contendo os arquivos y_pred (predições do modelo)
PREDICTION_DIR = BASE_DIR/"data"/"prediction"/MODEL

# Diretório contendo os arquivos y_true (dados de teste reais)
TEST_DIR = BASE_DIR/"data"/"prediction"/"test"/MODEL

# Diretório onde os CSVs de métricas serão salvos
OUTPUT_DIR = BASE_DIR/"results"/"prediction"/MODEL

# 2. Defina as listas de técnicas e modelos
# !! IMPORTANTE !! Forneça a lista exata de nomes de técnicas de imputação.
# A ordem não importa, o script vai ordenar por comprimento.
IMPUTATION_TECHNIQUES = [
    "ARIMA",
    "HoltWinters", 
    "IterativeSVD",
    "Kalman",
    "KNN_Sklearn",
    "Linear",
    "LOCF",
    "Mean",
    "SoftImpute",
    "Spline",
    "SVD_KNN_Hankel",
    "SVD_KNN"
]

# Nomes dos modelos usados para gerar as predições
MODEL_NAMES = [ "sarima"]

# 3. Nomes das colunas
COL_TRUE = 'Vazao'
COL_PRED = 'prediction'

def sanitize_filename(name):
    """Remove caracteres inválidos para nomes de arquivo."""
    name = re.sub(r'[\\/*?:"<>|]', "", name)
    name = name.replace(" ", "_")
    return name

def calculate_metrics(y_true, y_pred):
    """
    Calcula todas as métricas e retorna um dicionário.
    Assume que as funções de métrica (rmse, mae, etc.) estão no escopo global.
    
    IMPORTANTE: Esta função assume que seu 'mape' foi corrigido
    para lidar com y_true == 0, como discutido anteriormente.
    """
    
    # 1. Limpeza de NaN: Calcula métricas apenas onde ambos são válidos
    # mask = ~ (np.isnan(y_true) | np.isnan(y_pred))
    # y_true_clean = y_true[mask]
    # y_pred_clean = y_pred[mask]
    
    if len(y_true) == 0:
        print("Aviso: Não há dados válidos (não-NaN) para comparar.")
        return {
            "rmse": np.nan,
            "nrmse_range": np.nan,
            "nrmse_std": np.nan,
            "mae": np.nan,
            "mape": np.nan,
            "r2": np.nan,
        }

    # 2. Cálculo
    try:
        metrics = {
            "rmse": rmse(y_true, y_pred),
            "nrmse_range": nrmse(y_true, y_pred, mode="range"),
            "nrmse_std": nrmse(y_true, y_pred, mode="std"),
            "mae": mae(y_true, y_pred),
            "mape": mape(y_true, y_pred),
            "r2": r2_score(y_true, y_pred),
            "valid_pairs": len(y_true)
        }
    except Exception as e:
        print(f"Erro no cálculo de métricas: {e}")
        metrics = {
            "rmse": np.nan, "nrmse_range": np.nan, "nrmse_std": np.nan,
            "mae": np.nan, "mape": np.nan, "r2": np.nan, "valid_pairs": 0
        }
        
    return metrics

def main():
    """
    Função principal para orquestrar a avaliação.
    """
    print("Iniciando avaliação de predições...")
    
    # # Garante que o diretório de saída exista
    # OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Ordena técnicas da mais longa para a mais curta para evitar correspondências parciais
    # (ex: "SVD_KNN" deve ser verificado antes de "KNN")
    sorted_techniques = sorted(IMPUTATION_TECHNIQUES, key=len, reverse=True)
    
    # Dicionário para armazenar os resultados. Chave: original_name, Valor: lista de dicts (linhas)
    all_results = defaultdict(list)
    
    print(f"Buscando arquivos de teste em: {TEST_DIR}")
    
    # 1. Itera sobre os arquivos de TESTE (ground truth)
    # test_files = list(TEST_DIR.glob("*_imputed_test_gru.csv"))
    test_files = list(TEST_DIR.glob("*_imputed_test.csv"))

    if not test_files:
        # print(f"Aviso: Nenhum arquivo '*_imputed_test_gru.csv' encontrado em {TEST_DIR}")
        print(f"Aviso: Nenhum arquivo '*_imputed_test.csv' encontrado em {TEST_DIR}")
        return

    for test_path in test_files:
        test_name = test_path.name
        
        # Extrai o 'base_identifier' (ex: "dataset_6h_ARIMA")
        # base_identifier = test_name.removesuffix('_imputed_test_gru.csv')
        base_identifier = test_name.removesuffix('_imputed_test.csv')

        
        # 2. Encontra a qual dataset e técnica este arquivo pertence
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
            
        # 3. Para este par (dataset, técnica), procure predições de TODOS os modelos
        for model_name in MODEL_NAMES:
            
            # Monta o nome do arquivo de predição esperado
            # {original_name}_6h_{imputation_tech}_imputed.csv_predictions_{model_name}.csv
            pred_file_name = f"{original_name}_6h_{imputation_tech}_imputed_{model_name}_prediction.csv"
            pred_path = PREDICTION_DIR / pred_file_name
            
            if not pred_path.exists():
                print(f"Aviso: Arquivo de predição não encontrado: {pred_file_name}")
                continue
            
            # 4. TEMOS UM PAR! (test_path e pred_path). Calcular métricas.
            print(f"Processando: {original_name} | {imputation_tech} | {model_name}")
            
            try:
                df_test = pd.read_csv(test_path)
                df_pred = pd.read_csv(pred_path)
                
                if COL_TRUE not in df_test.columns:
                    print(f"  Erro: Coluna '{COL_TRUE}' não encontrada em {test_path}")
                    continue
                if COL_PRED not in df_pred.columns:
                    print(f"  Erro: Coluna '{COL_PRED}' não encontrada em {pred_path}")
                    continue

                y_true = df_test[COL_TRUE].values
                y_pred = df_pred[COL_PRED].values

                # Garante que os vetores tenham o mesmo tamanho (caso contrário, usa o menor)
                min_len = min(len(y_true), len(y_pred))
                if len(y_true) != len(y_pred):
                    # print("Erro crítico: arrays com tamanhos diferentes")
                    # return
                    print(f"  Aviso: Comprimentos diferentes. y_true={len(y_true)}, y_pred={len(y_pred)}. Usando min={min_len}.")
                    y_true = y_true[:min_len]
                    y_pred = y_pred[:min_len]
                
                # Calcula todas as métricas
                metrics = calculate_metrics(y_true, y_pred)
                
                # Adiciona as chaves de identificação
                result_row = {
                    "imputation_technique": imputation_tech,
                    "model": model_name,
                    **metrics  # Adiciona todos os pares (ex: "rmse": 0.5)
                }
                
                # Armazena na lista do dataset original
                all_results[original_name].append(result_row)
                
            except Exception as e:
                print(f"  ERRO CRÍTICO ao processar {pred_path}: {e}")

    # 5. Salva todos os resultados coletados em CSVs separados
    print("\nProcessamento concluído. Salvando resultados...")
    
    if not all_results:
        print("Nenhum resultado foi calculado.")
        return

    for original_name, results_list in all_results.items():
        if not results_list:
            continue
            
        # Converte a lista de resultados deste dataset em um DataFrame
        df_out = pd.DataFrame(results_list)
        
        # Define a ordem das colunas
        columns_order = [
            "imputation_technique", "model", "rmse", "nrmse_range", 
            "nrmse_std", "mae", "mape", "r2", "valid_pairs"
        ]
        # Garante que apenas colunas existentes sejam usadas
        df_out = df_out[[col for col in columns_order if col in df_out.columns]]
        
        # Salva o CSV
        safe_name = sanitize_filename(original_name)
        output_filename = f"{safe_name}_metrics.csv"
        output_path = OUTPUT_DIR / output_filename
        
        df_out.to_csv(output_path, index=False, float_format='%.6f')
        print(f"Resultados para '{original_name}' salvos em: {output_path}")

if __name__ == "__main__":
    # Verifica se as funções de métrica foram corrigidas (especialmente MAPE)
    print("Lembrete: Certifique-se que sua função 'mape' em 'metrics/error.py' "
          "lida corretamente com divisão por zero (y_true == 0).\n")
    main()