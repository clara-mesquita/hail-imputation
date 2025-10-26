# Extract Last 20% (Test Split) from Datasets
import os
import pandas as pd
import numpy as np
import json  # <-- Import json

INPUT_DIR = "./data/imputation/imputed_data"
# I recommend a new output dir to avoid mixing with SARIMA tests
OUTPUT_DIR = "./data/prediction/test/gru" 
TRAIN_TEST_SPLIT = 0.8
# Path to the JSON file created by your GRU script
GRU_JSON_PATH = './results/prediction/evaluation_rmse_mae_gru.json' 

def extract_gru_test_split(input_dir, output_dir, gru_json_path, split_ratio=TRAIN_TEST_SPLIT):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"✅ Diretório de saída criado: {output_dir}")

    if os.path.isdir(input_dir):
        print("✅ Diretório de entrada encontrado!")
    else:
        print(f"❌ Diretório de entrada não encontrado: {input_dir}")
        return 
        
    # --- Load the GRU results JSON ---
    try:
        with open(gru_json_path, 'r') as f:
            gru_results = json.load(f)
        print(f"✅ Resultados do GRU (com look_back) lidos de: {gru_json_path}")
    except FileNotFoundError:
        print(f"❌ Arquivo JSON de resultados do GRU não encontrado: {gru_json_path}")
        print("   Este arquivo é necessário para saber o 'look_back' de cada CSV.")
        return
    # -----------------------------------

    csv_files = []
    for root, _, files in os.walk(input_dir):
        for f in files:
            if f.endswith(".csv"):
                csv_files.append(os.path.join(root, f))

    print(f"Encontrados {len(csv_files)} arquivos.\n")

    for idx, filepath in enumerate(csv_files, 1):
        filename = os.path.basename(filepath)
        print(f"[{idx}/{len(csv_files)}] Processando: {filename}")

        # --- Get the look_back for this file ---
        file_results = gru_results.get(filename)
        if not file_results or 'error' in file_results:
            print(f" ⚠️  Resultados não encontrados ou com erro no JSON para {filename}, pulando.")
            continue
            
        look_back = file_results.get('best_params', {}).get('look_back')
        if look_back is None:
            print(f" ⚠️  'look_back' não encontrado nos best_params para {filename}, pulando.")
            continue
        print(f" ℹ️  'look_back' detectado para este arquivo: {look_back}")
        # -----------------------------------------

        try:
            df = pd.read_csv(filepath)
            if df.empty:
                print("  ⚠️ Arquivo vazio, pulando.")
                continue

            if '0' in df.columns:
                df = df.drop(columns=['0'])

            # Detecta coluna de valores
            value_cols = [c for c in df.columns if c.lower() in 
                          ['vazao', 'throughput', 'value', 'valor', 'flow']]
            if value_cols:
                target_col = value_cols[0]
            else:
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                numeric_cols = [c for c in numeric_cols if c.lower() not in
                                ['data', 'date', 'timestamp', 'datetime']]
                if not numeric_cols:
                    print("  ⚠️ Nenhuma coluna numérica detectada, pulando.")
                    continue
                target_col = numeric_cols[0]

            # Cleaning (exatamente como no script GRU)
            df_clean = df[[target_col]].dropna().reset_index(drop=True)
            
            if len(df_clean) < 50:
                print(f"  ⚠️ Dados insuficientes após limpeza: {len(df_clean)}, pulando.")
                continue

            # Split (últimos 20%)
            split_idx = int(len(df_clean) * split_ratio)

            # --- !!! A CORREÇÃO !!! ---
            # O y_test real começa APÓS o split_idx + look_back
            # Usamos .iloc para pegar por índice inteiro,
            # pois df_clean foi resetado
            gru_y_test_start_idx = split_idx + look_back
            
            if gru_y_test_start_idx >= len(df_clean):
                print(f"  ⚠️  Erro: 'split_idx' ({split_idx}) + 'look_back' ({look_back}) é maior que o tamanho dos dados ({len(df_clean)}).")
                print("     Isso significa que o conjunto de teste original era muito pequeno para este look_back.")
                continue
                
            test_data = df_clean.iloc[gru_y_test_start_idx:]
            # ---------------------------

            # Salvar arquivo
            out_path = os.path.join(output_dir, f"{filename.replace('.csv','')}_test_gru.csv")
            test_data.to_csv(out_path, index=False)
            print(f"  ✅ Teste (GRU) salvo em: {out_path} (Tamanho: {len(test_data)})")

        except Exception as e:
            print(f"  ❌ Erro ao processar {filename}: {e}")

    print("\nProcesso concluído!")

if __name__ == "__main__":
    extract_gru_test_split(INPUT_DIR, OUTPUT_DIR, GRU_JSON_PATH)