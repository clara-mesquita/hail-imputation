# Extrai "predictions" de JSONs e salva CSVs
import os
import json
import re
from pathlib import Path
import pandas as pd

# MODEL = 'gru'
MODEL = 'sarima'
INPUT_DIR  = "./results/prediction"           # pasta que contém os JSONs
OUTPUT_DIR = f"./data/prediction/{MODEL}"   # para onde salvar os CSVs extraídos
# Opcional: filtrar só arquivos que começam com 'evaluation_rmse_mae_'
FILE_PREFIX = f"evaluation_rmse_mae_{MODEL}.json"        # deixe "" para pegar todos os .json

def extract_predictions_from_json(json_path: str, output_dir: str) -> int:
    """
    Lê um JSON de resultados (dicionário de objetos), extrai 'predictions'
    de cada entrada e salva CSVs individuais no output_dir.
    Retorna quantos CSVs foram gerados a partir desse JSON.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        print(f"⚠️  {json_path}: conteúdo não é um dicionário no topo. Pulando.")
        return 0

    os.makedirs(output_dir, exist_ok=True)
    generated = 0

    for key, obj in data.items():
        if not isinstance(obj, dict):
            continue

        preds = obj.get("predictions", None)

        if preds is None:
            # Alguns itens podem ter 'error' ou não conter predictions
            print(f"Erro: O arquivo {key} não contém predictions")
            return 

        if not isinstance(preds, list) or len(preds) == 0:
            # Se vier vazio ou em formato inesperado, ignorar
            print(f"Ocorreu um erro inesperado com {key}")
            return

        df = pd.DataFrame({
            "prediction": preds
        })

        key = key.replace(".csv", "")
        out_name = key + f"_{MODEL}_prediction.csv"
        out_path = output_dir + "/" + out_name
        print(out_path)
        df.to_csv(out_path, index=False)
        generated += 1

    return generated

def extract_all_predictions(json_path:str, output_dir: str):
    """
    Percorre todos os JSONs (com prefixo opcional) no input_dir e gera CSVs
    de 'predictions' em output_dir.
    """
    total_files = 0
    total_csvs = 0


    total_files += 1
    print(f"📄 Lendo: {json_path}")
    count = extract_predictions_from_json(json_path, output_dir)
    print(f"   → {count} CSV(s) gerado(s) a partir deste JSON.\n")
    total_csvs += count

    print("======================================")
    print(f"JSONs processados: {total_files}")
    print(f"CSVs gerados:      {total_csvs}")
    print("======================================")

# ==========================
# EXECUÇÃO DIRETA
# ==========================
if __name__ == "__main__":
    json_path = INPUT_DIR + "/" + FILE_PREFIX
    print(f"json_path: {json_path}")
    extract_all_predictions(json_path, OUTPUT_DIR)
