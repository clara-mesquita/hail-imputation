# ============================================================
# GRU Time Series Trainer with Safe CV, Pop!_OS Stabilizers
# ============================================================

import os

# --- Pop!_OS / Linux stability & performance guards (set BEFORE TF import) ---
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")   # silence TF info logs
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "4")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "2")
# If CUDA is problematic on your Pop!_OS install, uncomment the next line to force CPU:
# os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import TimeSeriesSplit
import json
from itertools import product
import warnings
warnings.filterwarnings('ignore')

# --- Additional TF runtime setup (safe for GPU/CPU) ---
try:
    gpus = tf.config.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
except Exception:
    pass

tf.config.threading.set_intra_op_parallelism_threads(4)
tf.config.threading.set_inter_op_parallelism_threads(2)

# ============================================
# CONFIGURAÇÕES - AJUSTE AQUI
# ============================================
INPUT_FOLDER = './imputation_results_original'
OUTPUT_MODEL_FOLDER = './modelo_salvo'
OUTPUT_JSON = 'evaluation_rmse_mae.json'
TRAIN_TEST_SPLIT = 0.8
LOOK_BACK = 5                # default look_back (used when GRID_SEARCH_ENABLED=False)
EPOCHS = 50                  # cut down for stability; adjust later
BATCH_SIZE = 32
PATIENCE = 6
N_SPLITS = 2                 # 2 folds to reduce compute

# Grid Search Parameters
GRID_SEARCH_ENABLED = True
PARAM_GRID = {
    'gru_units': [64],              # start lean; expand once stable (e.g., [32,64,128])
    'learning_rate': [1e-3, 1e-4],
    'look_back': [5],               # try [3,5,7] later
    'dropout_rate': [0.0, 0.2]
}
# ============================================

def create_dataset(X, look_back):
    """
    Cria dataset com janela temporal para séries temporais.
    Retorna y como 2D (N,1) para facilitar inverse_transform e Keras.
    """
    Xs, ys = [], []
    for i in range(len(X) - look_back):
        Xs.append(X[i:i+look_back])
        ys.append(X[i+look_back])
    Xs = np.array(Xs)
    ys = np.array(ys).reshape(-1, 1)   # ensure 2D
    return Xs, ys

def create_gru_model(units, train_shape, learning_rate, dropout_rate=0.0):
    """
    Cria modelo GRU com opção de dropout (stacked GRU many-to-one).
    train_shape: (N, look_back, n_features)
    """
    model = Sequential()
    model.add(GRU(units=units, return_sequences=True,
                  input_shape=[train_shape[1], train_shape[2]]))
    if dropout_rate > 0:
        model.add(Dropout(dropout_rate))
    model.add(GRU(units=units))
    if dropout_rate > 0:
        model.add(Dropout(dropout_rate))
    model.add(Dense(1))
    model.compile(
        loss=MeanSquaredError(),
        optimizer=Adam(learning_rate=learning_rate),
        metrics=[RootMeanSquaredError()]
    )
    return model

def fit_model_with_cross_validation(model_builder, xtrain, ytrain,
                                    patience, epochs, batch_size, n_splits):
    """
    Treina com validação cruzada temporal, RECRIANDO o modelo a cada fold.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    val_losses = []

    for fold_idx, (train_index, val_index) in enumerate(tscv.split(xtrain)):
        print(f"  Fold {fold_idx + 1}/{n_splits}")
        x_train_fold, x_val_fold = xtrain[train_index], xtrain[val_index]
        y_train_fold, y_val_fold = ytrain[train_index], ytrain[val_index]

        # Modelo fresco por fold (evita leakage e estados acumulados)
        model = model_builder()

        early_stop = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=0
        )
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=max(2, patience//2),
            min_lr=1e-6, verbose=0
        )

        history = model.fit(
            x_train_fold, y_train_fold,
            epochs=epochs,
            validation_data=(x_val_fold, y_val_fold),
            batch_size=min(batch_size, len(x_train_fold)),
            callbacks=[early_stop, reduce_lr],
            verbose=0
        )

        val_losses.append(float(np.min(history.history['val_loss'])))

        # libera memória do fold
        del model
        tf.keras.backend.clear_session()

    mean_val_loss = float(np.mean(val_losses))
    return mean_val_loss

def grid_search(train_scaled, param_grid, epochs, batch_size, patience, n_splits):
    """
    Realiza grid search para encontrar melhores hiperparâmetros.
    """
    print("\n" + "="*60)
    print("INICIANDO GRID SEARCH")
    print("="*60)

    param_combinations = [dict(zip(param_grid.keys(), v))
                          for v in product(*param_grid.values())]

    best_score = float('inf')
    best_params = None
    results = []

    for idx, params in enumerate(param_combinations):
        print(f"\nTestando combinação {idx + 1}/{len(param_combinations)}: {params}")

        try:
            # Cria dataset com look_back específico
            X_train, y_train = create_dataset(train_scaled, params['look_back'])

            if len(X_train) < 10:
                raise ValueError("Dataset de janelas muito pequeno para CV.")

            def model_builder():
                return create_gru_model(
                    units=params['gru_units'],
                    train_shape=X_train.shape,
                    learning_rate=params['learning_rate'],
                    dropout_rate=params['dropout_rate']
                )

            # Treina e avalia (modelo novo por fold)
            val_loss = fit_model_with_cross_validation(
                model_builder, X_train, y_train, patience, epochs, batch_size, n_splits
            )

            print(f"  Validation Loss: {val_loss:.6f}")

            results.append({'params': params, 'val_loss': val_loss})

            # Atualiza melhores parâmetros
            if val_loss < best_score:
                best_score = val_loss
                best_params = params
                print("  *** Novo melhor resultado! ***")

            tf.keras.backend.clear_session()

        except Exception as e:
            print(f"  Erro: {e}")
            results.append({'params': params, 'error': str(e)})
            tf.keras.backend.clear_session()

    print("\n" + "="*60)
    print("GRID SEARCH CONCLUÍDO")
    print(f"Melhores parâmetros: {best_params}")
    print(f"Melhor validation loss: {best_score:.6f}")
    print("="*60 + "\n")

    return best_params, results

def save_model(model, directory, filename):
    """Salva modelo treinado"""
    if directory:
        if not os.path.exists(directory):
            os.makedirs(directory)
        file_path = os.path.join(directory, f'{filename}_GRU.keras')
        model.save(file_path)
        print(f"Modelo salvo em: '{file_path}'")

def predict_and_evaluate(model, xtest, ytest, scaler):
    """Faz predições e calcula métricas (com shapes seguros p/ inverse_transform)"""
    predictions = model.predict(xtest, verbose=0)  # (N,1)
    predictions_inv = scaler.inverse_transform(predictions)

    # ytest deve ser 2D (N,1) – garantir aqui:
    if ytest.ndim == 1:
        ytest_2d = ytest.reshape(-1, 1)
    else:
        ytest_2d = ytest
    ytest_inv = scaler.inverse_transform(ytest_2d)

    errors = predictions_inv - ytest_inv
    mse = float(np.mean(np.square(errors)))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(errors)))

    return {
        'rmse': rmse,
        'mae': mae,
        'predictions': predictions_inv.flatten().tolist()
    }

def process_file(filepath, filename, output_model_dir, use_grid_search):
    """Processa um arquivo CSV individual"""
    print(f"\n{'='*60}")
    print(f"Processando: {filename}")
    print('='*60)

    # Carrega dados sem definir index
    df = pd.read_csv(filepath)

    print(f"Colunas detectadas: {list(df.columns)}")

    # Detecta coluna de data/timestamp e remove (não é necessária para o modelo)
    date_cols = [col for col in df.columns if col.lower() in ['data', 'date', 'timestamp', 'datetime']]
    if date_cols:
        print(f"Coluna de data encontrada: '{date_cols[0]}' (será ignorada)")

    # Remove coluna '0' se existir
    if '0' in df.columns:
        df.drop(columns=['0'], inplace=True)

    # Detecta coluna de valores
    value_cols = [col for col in df.columns if col.lower() in ['vazao', 'throughput', 'value', 'valor', 'flow']]
    if value_cols:
        target_col = value_cols[0]
    else:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [c for c in numeric_cols if c.lower() not in ['data', 'date', 'timestamp', 'datetime']]
        if len(numeric_cols) == 0:
            raise ValueError(f"Nenhuma coluna numérica encontrada em {filename}")
        target_col = numeric_cols[0]

    print(f"Coluna de valores detectada: '{target_col}'")
    print(f"Total de registros: {len(df)}")

    # Remove valores NaN ou inválidos
    df_clean = df[[target_col]].dropna()
    print(f"Registros após limpeza: {len(df_clean)}")

    if len(df_clean) < 50:
        raise ValueError(f"Dados insuficientes após limpeza: {len(df_clean)} registros")

    # Split treino/teste
    split_idx = int(len(df_clean) * TRAIN_TEST_SPLIT)
    train_data = df_clean[:split_idx][target_col].values.reshape(-1, 1)
    test_data  = df_clean[split_idx:][target_col].values.reshape(-1, 1)

    print(f"Tamanho treino: {len(train_data)}, Tamanho teste: {len(test_data)}")

    # Normalização
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_data)
    test_scaled = scaler.transform(test_data)

    # Sanity check: dados suficientes para janela
    def ensure_windows_ok(arr, look_back):
        if len(arr) <= look_back:
            raise ValueError(f"Série muito curta ({len(arr)}) para look_back={look_back}.")

    # Grid Search ou parâmetros padrão
    if use_grid_search:
        best_params, grid_results = grid_search(
            train_scaled, PARAM_GRID, EPOCHS, BATCH_SIZE, PATIENCE, N_SPLITS
        )
        if best_params is None:
            raise RuntimeError("Grid search não encontrou parâmetros válidos.")
    else:
        best_params = {
            'gru_units': 64,
            'learning_rate': 1e-4,
            'look_back': LOOK_BACK,
            'dropout_rate': 0.0
        }
        grid_results = None

    ensure_windows_ok(train_scaled, best_params['look_back'])
    X_train, y_train = create_dataset(train_scaled, best_params['look_back'])
    # Teste pode ser curto; se ficar vazio, reduza look_back automaticamente
    if len(test_scaled) <= best_params['look_back']:
        print("Aviso: test set muito curto para look_back; reduzindo look_back apenas para teste.")
        lb_test = max(1, len(test_scaled) - 1)
    else:
        lb_test = best_params['look_back']
    X_test, y_test = create_dataset(test_scaled, lb_test)

    if len(X_test) == 0:
        raise ValueError("Conjunto de teste não possui janelas após ajuste de look_back.")

    print(f"\nTreinando modelo final com melhores parâmetros...")
    print(f"Parâmetros: {best_params}")

    # Treina modelo final
    final_model = create_gru_model(
        units=best_params['gru_units'],
        train_shape=X_train.shape,
        learning_rate=best_params['learning_rate'],
        dropout_rate=best_params['dropout_rate']
    )

    early_stop = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=PATIENCE,
        restore_best_weights=True,
        verbose=1
    )
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=max(2, PATIENCE//2),
        min_lr=1e-6, verbose=1
    )

    # Usa último split para validação
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    train_idx, val_idx = list(tscv.split(X_train))[-1]

    history = final_model.fit(
        X_train[train_idx], y_train[train_idx],
        epochs=EPOCHS,
        validation_data=(X_train[val_idx], y_train[val_idx]),
        batch_size=min(BATCH_SIZE, len(X_train[train_idx])),
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )

    # Salva modelo
    if output_model_dir:
        save_model(final_model, output_model_dir, filename.replace('.csv', ''))

    # Avalia no conjunto de teste
    results = predict_and_evaluate(final_model, X_test, y_test, scaler)
    results['best_params'] = best_params
    if grid_results:
        results['grid_search_results'] = grid_results

    print(f"\nResultados no conjunto de teste:")
    print(f"RMSE: {results['rmse']:.4f}")
    print(f"MAE: {results['mae']:.4f}")

    # Limpa memória
    del final_model
    tf.keras.backend.clear_session()

    return results

def gru_prediction(source_dir, output_model_dir=None, output_json='evaluation_rmse_mae.json',
                   use_grid_search=True):
    """Função principal para processar todos os arquivos"""
    evaluation = {}

    print(f"\n{'='*70}")
    print(f"INÍCIO DO PROCESSAMENTO")
    print(f"Pasta de entrada: {source_dir}")
    print(f"Grid Search: {'ATIVADO' if use_grid_search else 'DESATIVADO'}")
    print(f"{'='*70}\n")

    # Processa cada arquivo CSV
    csv_files = []
    for root, _, files in os.walk(source_dir):
        for file in files:
            if file.endswith('.csv'):
                csv_files.append((root, file))

    print(f"Encontrados {len(csv_files)} arquivos CSV\n")

    for idx, (root, file) in enumerate(csv_files, 1):
        filepath = os.path.join(root, file)
        print(f"\n[{idx}/{len(csv_files)}]")

        try:
            results = process_file(filepath, file, output_model_dir, use_grid_search)
            evaluation[file] = results
        except Exception as e:
            print(f"ERRO ao processar {file}: {e}")
            evaluation[file] = {'error': str(e)}
        finally:
            tf.keras.backend.clear_session()

    # Salva resultados em JSON
    with open(output_json, 'w') as f:
        json.dump(evaluation, f, indent=4)

    print(f"\n{'='*70}")
    print(f"PROCESSAMENTO CONCLUÍDO")
    print(f"Resultados salvos em: {output_json}")
    print(f"{'='*70}\n")

    # Resumo
    print("\nRESUMO DOS RESULTADOS:")
    print("="*70)
    successful = 0
    total_rmse = 0.0
    total_mae = 0.0

    for file, metrics in evaluation.items():
        if 'error' in metrics:
            print(f"❌ {file}: ERRO - {metrics['error']}")
        else:
            successful += 1
            total_rmse += metrics['rmse']
            total_mae += metrics['mae']
            print(f"✓ {file}: RMSE={metrics['rmse']:.4f}, MAE={metrics['mae']:.4f}")

    if successful > 0:
        print(f"\n{'='*70}")
        print(f"Média RMSE: {total_rmse/successful:.4f}")
        print(f"Média MAE: {total_mae/successful:.4f}")
        print(f"Taxa de sucesso: {successful}/{len(evaluation)}")
        print(f"{'='*70}\n")

    return evaluation

if __name__ == "__main__":
    results = gru_prediction(
        source_dir=INPUT_FOLDER,
        output_model_dir=OUTPUT_MODEL_FOLDER,
        output_json=OUTPUT_JSON,
        use_grid_search=GRID_SEARCH_ENABLED
    )
