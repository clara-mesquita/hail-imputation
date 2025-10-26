import os
import pandas as pd
import numpy as np
import warnings

# Diretório contendo os datasets originais (agregados em 6h com falhas)
INPUT_DIR = "./data/data"

# Diretório para onde salvar as amostras (o "intervalo mais longo")
SAMPLES_DIR = "./data/imputation/sample_data"

# Diretório para onde salvar as amostras com falhas introduzidas
MISSING_DIR = "./data/imputation/missing_sample_data"

# Coluna alvo para análise de falhas e introdução de novos NaNs
TARGET_COLUMN = "Vazao"

MAX_FAILURES_ALLOWED = 2 # Número máximo de falhas permitidas no intervalo
ALLOW_CONSECUTIVE = False # Permite falhas consecutivas?

MISSING_RATES = [0.1, 0.2, 0.3, 0.4] # Taxas de falhas a serem criadas
RANDOM_SEED = 42 # Seed para reprodutibilidade

def introduce_missing_data(df, missing_rate, seed=42, column="Vazao"):
    # Suprime o SettingWithCopyWarning, pois estamos cientes de que 
    # estamos modificando uma cópia.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", pd.errors.SettingWithCopyWarning)
        
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


def process_datasets(imputation = "mean"):
    print("Iniciando o processo...")
    print(f"Diretório de Entrada: {INPUT_DIR}")
    print(f"Diretório de Amostras: {SAMPLES_DIR}")
    print(f"Diretório de Falhas:  {MISSING_DIR}")
    print("=" * 30)

    os.makedirs(SAMPLES_DIR, exist_ok=True)
    os.makedirs(MISSING_DIR, exist_ok=True)
    
    for rate in MISSING_RATES:
        rate_str = str(int(rate * 100))
        os.makedirs(os.path.join(MISSING_DIR, rate_str), exist_ok=True)
        
        
    sample_files_to_process = [] # Lista para guardar os caminhos das amostras
    
    for filename in os.listdir(INPUT_DIR):
        if not filename.endswith(".csv"):
            continue
            
        input_path = os.path.join(INPUT_DIR, filename)
        
        try:
            df_original = pd.read_csv(input_path)
        except Exception as e:
            print(f"[Erro] Falha ao ler {filename}: {e}")
            raise NameError
            
        if TARGET_COLUMN not in df_original.columns:
            print(f"[Erro]: Coluna '{TARGET_COLUMN}' não encontrada em {filename}.")
            raise NameError
            
        start_idx, end_idx, size, failures, _ = find_longest_interval(
            df_original[TARGET_COLUMN],
            max_failures=MAX_FAILURES_ALLOWED,
            allow_consecutive_failures=ALLOW_CONSECUTIVE
        )
        
        if size == 0:
            print(f"Erro!: Nenhum intervalo válido encontrado para {filename}")
            raise IndexError
            
        print(f"  -> {filename}: Melhor intervalo [{start_idx}:{end_idx}] (Tam: {size}, Falhas: {failures})")
        
        # Extrair a amostra (usamos end_idx + 1 pois iloc é exclusivo no fim)
        df_sample = df_original.iloc[start_idx:end_idx + 1].copy()

        if imputation == "mean":
            df_sample.loc[:, TARGET_COLUMN] = df_sample[TARGET_COLUMN].replace(-1, np.nan)
            mean_value = df_sample[TARGET_COLUMN].mean()
            df_sample.loc[:, TARGET_COLUMN] = df_sample[TARGET_COLUMN].fillna(mean_value)

        # --- MODIFICADO: Lógica de Nomes (Etapa 1) ---
        # Divide o nome do arquivo em base e extensão
        # Ex: "dados.csv" -> base_name="dados", extension=".csv"
        base_name, extension = os.path.splitext(filename)
        
        # Cria o novo nome para a amostra (ex: "dados_sample.csv")
        sample_filename = f"{base_name}_sample{extension}"
        sample_output_path = os.path.join(SAMPLES_DIR, sample_filename)
        df_sample.to_csv(sample_output_path, index=False)
        
        # Guarda o NOME BASE e a EXTENSÃO (e o dataframe) para a próxima etapa
        sample_files_to_process.append(((base_name, extension), df_sample))
        # --- FIM DA MODIFICAÇÃO ---

    if not sample_files_to_process:
        print("Nenhuma amostra foi gerada na Etapa 1. Processo encerrado.")
        return

    # --- MODIFICADO: Lógica de Nomes (Etapa 2) ---
    # Lê o nome base e a extensão que salvamos na lista
    for (base_name, extension), df_sample in sample_files_to_process:
        # Usa o nome base + extensão para o print
        print(f"  -> Processando amostra base: {base_name}{extension}")
    # --- FIM DA MODIFICAÇÃO ---
        
        for rate in MISSING_RATES:
            rate_str = str(int(rate * 100))
            
            df_missing = introduce_missing_data(
                df_sample, 
                missing_rate=rate, 
                seed=RANDOM_SEED,
                column=TARGET_COLUMN
            )
            
            # --- MODIFICADO: Lógica de Nomes (Etapa 2 - Saída) ---
            output_subdir = os.path.join(MISSING_DIR, rate_str)
            
            # Cria o novo nome do arquivo com missing rate (ex: "dados_mr10.csv")
            missing_filename = f"{base_name}_mr{rate_str}{extension}"
            missing_output_path = os.path.join(output_subdir, missing_filename)
            
            df_missing.to_csv(missing_output_path, index=False)
            print(f"    - {rate_str}% salvo em: {missing_output_path}")
            # --- FIM DA MODIFICAÇÃO ---

    print("\nProcesso concluído.")

if __name__ == "__main__":
    process_datasets()