import pandas as pd
from pathlib import Path

# =============================================================================
# --- CONFIGURAÇÃO ---
# =============================================================================

# Assume que este script está na pasta 'evaluation/'
# BASE_DIR aponta para a raiz do projeto (um nível acima de 'evaluation')
try:
    BASE_DIR = Path(__file__).resolve().parent.parent.parent
except NameError:
    # Fallback para ambientes interativos (como Jupyter) onde __file__ não é definido
    print("Aviso: __file__ não definido. Usando o diretório de trabalho atual como base.")
    BASE_DIR = Path.cwd().parent # Ajuste se necessário

# 1. Diretório de ENTRADA (conforme sua solicitação)
# Contém todos os CSVs de métricas do modelo 'gru'
# INPUT_DIR = BASE_DIR / "results" / "prediction" / "gru"
INPUT_DIR = BASE_DIR / "results" / "prediction" / "sarima"


# 2. Arquivo de SAÍDA
# Onde o sumário final será salvo
OUTPUT_FILE = INPUT_DIR / "_SUMARIO_SARIMA_por_tecnica.csv" # Começar com '_' coloca o arquivo no topo da pasta

# 3. Métrica de Ordenação
# Qual métrica usar para definir a "melhor" técnica.
# Usaremos 'rmse_mean' (a média do RMSE).
SORT_METRIC = 'nrmse_range_mean'
# Como RMSE é um erro, 'True' (ascendente) significa que o menor (melhor) vem primeiro.
SORT_ASCENDING = True

# =============================================================================
# --- LÓGICA DO SCRIPT ---
# =============================================================================

def main():
    print(f"Iniciando sumarização de resultados...")
    print(f"Lendo arquivos de: {INPUT_DIR}")

    # 1. Encontra todos os arquivos de métricas no diretório
    # (Ignora o próprio arquivo de sumário se ele já existir)
    all_metric_files = [
        f for f in INPUT_DIR.glob("*_metrics.csv") 
        if f.name != OUTPUT_FILE.name
    ]

    if not all_metric_files:
        print(f"Erro: Nenhum arquivo '*_metrics.csv' encontrado em {INPUT_DIR}.")
        print("Certifique-se que o caminho está correto e que o script anterior foi executado.")
        return

    print(f"Encontrados {len(all_metric_files)} arquivos de métricas. Carregando...")

    # 2. Carrega todos os arquivos em uma lista de DataFrames
    df_list = []
    for f_path in all_metric_files:
        try:
            df_list.append(pd.read_csv(f_path))
        except Exception as e:
            print(f"Aviso: Falha ao ler {f_path.name}: {e}")

    if not df_list:
        print("Erro: Nenhum arquivo foi lido com sucesso.")
        return

    # 3. Concatena todos em um único DataFrame
    df_all = pd.concat(df_list, ignore_index=True)

    # 4. Identifica as colunas de métrica
    # (exclui colunas de ID como 'imputation_technique' e 'model')
    cols_to_agg = [
        'rmse', 'nrmse_range', 'nrmse_std', 
        'mae', 'mape', 'r2', 'valid_pairs'
    ]
    
    # Filtra caso alguma coluna não exista
    existing_metric_cols = [
        col for col in cols_to_agg if col in df_all.columns
    ]

    if 'imputation_technique' not in df_all.columns:
        print("Erro: A coluna 'imputation_technique' é necessária e não foi encontrada.")
        return
        
    if not existing_metric_cols:
        print("Erro: Nenhuma coluna de métrica (rmse, mae, etc.) foi encontrada.")
        return

    print(f"Calculando 'mean' e 'std' por 'imputation_technique'...")

    # 5. Agrupa por técnica e calcula 'mean' e 'std'
    df_summary = df_all.groupby('imputation_technique')[existing_metric_cols].agg(['mean', 'std'])

    # 6. "Achata" as colunas MultiIndex
    # (ex: ('rmse', 'mean') vira 'rmse_mean')
    df_summary.columns = ['_'.join(col).strip() for col in df_summary.columns.values]

    # 7. Adiciona uma contagem de quantos datasets foram usados para cada técnica
    df_summary['dataset_count'] = df_all.groupby('imputation_technique').size()

    # Traz 'imputation_technique' de volta como uma coluna
    df_summary = df_summary.reset_index()

    # 8. Ordena os resultados pela métrica definida
    try:
        print(f"Ordenando por '{SORT_METRIC}' (Ascendente={SORT_ASCENDING})...")
        df_summary = df_summary.sort_values(by=SORT_METRIC, ascending=SORT_ASCENDING)
    except KeyError:
        print(f"Aviso: Métrica de ordenação '{SORT_METRIC}' não encontrada. O arquivo não será ordenado.")

    # 9. Salva o arquivo CSV final
    try:
        df_summary.to_csv(OUTPUT_FILE, index=False, float_format='%.6f')
        print("\n" + "="*50)
        print(f"Sumário salvo com sucesso em: {OUTPUT_FILE}")
        print("="*50)
        print("\nPreview das 5 melhores técnicas:")
        print(df_summary[['imputation_technique', SORT_METRIC, 'r2_mean', 'mae_mean']].head())
        
    except Exception as e:
        print(f"\nErro ao salvar o arquivo: {e}")

if __name__ == "__main__":
    main()