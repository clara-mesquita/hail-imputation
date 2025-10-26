# Extract Last 20% (Test Split) from Datasets
import os
import pandas as pd
import numpy as np

INPUT_DIR = "./data/imputation/imputed_data"     # pasta com os imputed data
OUTPUT_DIR = "./data/prediction/test"          # pasta onde salvar os testes
TRAIN_TEST_SPLIT = 0.8                          # 80/20 split

def extract_test_split(input_dir, output_dir, split_ratio=TRAIN_TEST_SPLIT):

    if os.path.isdir(input_dir):
        print("✅ Diretório encontrado!")
    else:
        print("❌ Diretório não encontrado!")
        return 

    csv_files = []
    for root, _, files in os.walk(input_dir):
        for f in files:
            print(f)
            if f.endswith(".csv"):
                csv_files.append(os.path.join(root, f))

    print(f"Encontrados {len(csv_files)} arquivos.\n")

    for idx, filepath in enumerate(csv_files, 1):
        filename = os.path.basename(filepath)
        print(f"[{idx}/{len(csv_files)}] Processando: {filename}")

        try:
            df = pd.read_csv(filepath)
            if df.empty:
                print("  ⚠️ Arquivo vazio, pulando.")
                continue

            # Remove coluna '0' se existir
            if '0' in df.columns:
                df = df.drop(columns=['0'])

            # Detecta coluna de valores (similar ao seu GRU)
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

            # Cleaning (drop NaNs)
            # This step is important because is used on prediction code 
            df_clean = df[[target_col]].dropna().reset_index(drop=True)

            # Split (últimos 20%)
            split_idx = int(len(df_clean) * split_ratio)
            test_data = df_clean[split_idx:]  # mantém a ordem original

            # Salvar arquivo
            out_path = os.path.join(output_dir, f"{filename.replace('.csv','')}_test.csv")
            test_data.to_csv(out_path, index=False)
            print(f"  ✅ Teste salvo em: {out_path}")

        except Exception as e:
            print(f"  ❌ Erro ao processar {filename}: {e}")

    print("\nProcesso concluído!")

if __name__ == "__main__":
    extract_test_split(INPUT_DIR, OUTPUT_DIR)
