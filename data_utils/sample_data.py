"""
Este script faz duas coisas principais:
    1. Separa intervalos mais longos com poucas falhas (por default, duas falhas) e salva em csvs chamados "sample data"
    2. Cria gaps artificiais (de forma aleatória, guiado pela porcentagem) nessas amostras sem muitas falhas
"""

import os
import pandas as pd
import numpy as np
import glob
from pathlib import Path

#### CONFIGURATION ####

BASE_DIR = Path(__file__).resolve().parent.parent

# Diretório contendo os datasets originais (agregados em 6h com falhas)
ORIGINAL_DATA_DIR = BASE_DIR / "data" / "data"
# Diretório para onde salvar as amostras (o "intervalo mais longo")
SAMPLES_DIR = BASE_DIR / "data" / "imputation" / "sample_data"
# Diretório para onde salvar as amostras com falhas introduzidas
MISSING_DIR = BASE_DIR / "data" / "imputation" / "missing_sample_data"

THROUGHPUT_COL = "Vazao"

#### LONGEST INTERVAL FINDING CONFIG ####
MAX_FAILURES_ALLOWED = 2 # Número máximo de falhas permitidas no intervalo
ALLOW_CONSECUTIVE = False # Permite falhas consecutivas?

#### GAP CREATION CONFIG ####
# MISSING_RATES = [0.1, 0.2, 0.3, 0.4] 
MISSING_RATES = [0.05, 0.15, 0.25] 

RANDOM_SEED = 42


def introduce_missing_data(df, missing_rate, seed=42, column="Vazao"):
    rng = np.random.default_rng(seed)
    df_missing = df.copy()
    
    if column not in df_missing.columns:
        print(f"  [Erro] Coluna '{column}' não encontrada para introduzir falhas.")
        raise ValueError 
        
    mask = rng.random(len(df_missing)) < missing_rate
    df_missing.loc[mask, column] = np.nan
    return df_missing

def find_longest_interval(flow, max_failures=2, max_missing_percentage=None, 
                          allow_consecutive_failures=False):
    """
    Encontra o intervalo mais longo contendo no máximo:
    - um número específico de falhas, OU
    - uma porcentagem máxima de dados ausentes
    
    Falhas são valores NaN ou -1.
    
    Retorna:
        (start_idx, end_idx, size, total_failures, failure_percentage)
    """
    v = flow.copy().replace(-1, np.nan)
    is_fail = v.isna().to_numpy()
    n = len(is_fail)

    best_size = 0
    best_start = 0
    best_end = 0
    best_failures = 0

    for i in range(n):
        failures = 0
        has_consecutive = False
        last_was_fail = False

        for j in range(i, n):
            if is_fail[j]:
                failures += 1
                if not allow_consecutive_failures and last_was_fail:
                    has_consecutive = True
                last_was_fail = True
            else:
                last_was_fail = False

            current_size = j - i + 1
            current_missing_pct = failures / current_size if current_size > 0 else 0
            
            if max_missing_percentage is not None:
                if current_missing_pct > max_missing_percentage or (not allow_consecutive_failures and has_consecutive):
                    break
            else:
                if failures > max_failures or (not allow_consecutive_failures and has_consecutive):
                    break

            if current_size > best_size:
                best_size = current_size
                best_start = i
                best_end = j
                best_failures = failures

    failure_percentage = best_failures / best_size if best_size > 0 else 0
    
    # +1 no best_end para ser compatível com slicing (ex: .iloc[start:end])
    return best_start, best_end, best_size, best_failures, failure_percentage


def process_datasets(only_missing, imputation = "mean"):
    print(f"Diretório de Entrada: {ORIGINAL_DATA_DIR}")
    print(f"Diretório de Amostras: {SAMPLES_DIR}")
    print(f"Diretório de Falhas:  {MISSING_DIR}")

    os.makedirs(MISSING_DIR, exist_ok=True)

    if only_missing:
        print("Fazendo a criação de gaps (usando amostras existentes)...")
        
        # 1. Inicializa a lista que será populada
        sample_files_to_process = []
        
        files_suffix = os.path.join(SAMPLES_DIR, "*_sample.csv")
        existing_sample_files = glob.glob(files_suffix)
        print(f"Arquivos de amostra encontrados: {existing_sample_files}")

        # 2. Itera pelos arquivos de amostra encontrados
        for sample_path in existing_sample_files:

            # Carrega o dataframe da amostra
            df_sample = pd.read_csv(sample_path)
            
            # Extrai o nome do arquivo (ex: "dados_sample.csv")
            sample_filename = os.path.basename(sample_path)
            
            # Separa a base com sufixo da extensão (ex: "dados_sample", ".csv")
            base_name_with_suffix, extension = os.path.splitext(sample_filename)
            
            # Remove o sufixo "_sample" para obter o base_name original (ex: "dados")
            # Usamos .removesuffix() que é mais seguro que .replace()
            if not base_name_with_suffix.endswith("_sample"):
                print(f"[Aviso] Ignorando arquivo com nome inesperado: {sample_filename}")
                continue
                
            base_name = base_name_with_suffix.removesuffix("_sample")
            
            # 3. Adiciona na lista no formato esperado
            sample_files_to_process.append(((base_name, extension), df_sample))  

    else:
        os.makedirs(SAMPLES_DIR, exist_ok=True)
        
        for rate in MISSING_RATES:
            rate_str = str(int(rate * 100))
            os.makedirs(os.path.join(MISSING_DIR, rate_str), exist_ok=True)
            
        sample_files_to_process = [] 
        
        for filename in os.listdir(ORIGINAL_DATA_DIR):
            if not filename.endswith(".csv"):
                continue
                
            input_path = os.path.join(ORIGINAL_DATA_DIR, filename)
            
            try:
                df_original = pd.read_csv(input_path)
            except Exception as e:
                print(f"[Erro] Falha ao ler {filename}: {e}")
                raise NameError
                
            if THROUGHPUT_COL not in df_original.columns:
                print(f"[Erro]: Coluna '{THROUGHPUT_COL}' não encontrada em {filename}.")
                raise NameError
                
            start_idx, end_idx, size, failures, _ = find_longest_interval(
                df_original[THROUGHPUT_COL],
                max_failures=MAX_FAILURES_ALLOWED,
                allow_consecutive_failures=ALLOW_CONSECUTIVE
            )
            
            if size == 0:
                print(f"[Erro]: Nenhum intervalo válido encontrado para {filename}")
                raise IndexError
                
            print(f"  -> {filename}: Melhor intervalo [{start_idx}:{end_idx}] (Tam: {size}, Falhas: {failures})")
            
            # Extrair a amostra (usamos end_idx + 1 pois iloc é exclusivo no fim)
            df_sample = df_original.iloc[start_idx:end_idx + 1].copy()

            if imputation == "mean":
                df_sample.loc[:, THROUGHPUT_COL] = df_sample[THROUGHPUT_COL].replace(-1, np.nan)
                mean_value = df_sample[THROUGHPUT_COL].mean()
                df_sample.loc[:, THROUGHPUT_COL] = df_sample[THROUGHPUT_COL].fillna(mean_value)

            # Divide o nome do arquivo em base e extensão
            # Ex: "dados.csv" -> base_name="dados", extension=".csv"
            base_name, extension = os.path.splitext(filename)
            
            # Cria o novo nome para a amostra (ex: "dados_sample.csv")
            sample_filename = f"{base_name}_sample{extension}"
            sample_output_path = os.path.join(SAMPLES_DIR, sample_filename)
            df_sample.to_csv(sample_output_path, index=False)
            
            # Guarda o NOME BASE e a EXTENSÃO (e o dataframe) para a próxima etapa
            sample_files_to_process.append(((base_name, extension), df_sample))

        if not sample_files_to_process:
            print("Nenhuma amostra foi gerada na Etapa 1. Processo encerrado.")
            return    

    for (base_name, extension), df_sample in sample_files_to_process:
        print(f"  -> Processando amostra base: {base_name}{extension}")
        
        for rate in MISSING_RATES:
            rate_str = str(int(rate * 100))
            
            df_missing = introduce_missing_data(
                df_sample, 
                missing_rate=rate, 
                seed=RANDOM_SEED,
                column=THROUGHPUT_COL
            )
            
            output_subdir = os.path.join(MISSING_DIR, rate_str)
            os.makedirs(output_subdir, exist_ok=True)
            
            missing_filename = f"{base_name}_mr{rate_str}{extension}"
            missing_output_path = os.path.join(output_subdir, missing_filename)
            
            df_missing.to_csv(missing_output_path, index=False)
            print(f"    - {rate_str}% salvo em: {missing_output_path}")

if __name__ == "__main__":
    process_datasets(only_missing=True)