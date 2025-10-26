# Malu's Fuzzy Accuracy Metric 

import numpy as np
import pandas as pd

BINS_MBPS = np.arange(0, 2100, 100)
LABELS = [f'{BINS_MBPS[i]}-{BINS_MBPS[i+1]}' for i in range(len(BINS_MBPS)-1)]

def fuzzy_weight(distance):
    return {0: 1.0, 1: 0.75, 2: 0.5}.get(distance, 0.0)

def compute_fuzzy_accuracy(y_true, y_pred):
    df = pd.DataFrame({'y_test': y_true, 'y_pred': y_pred}).dropna()
    if df.empty:
        return None, 0

    df['y_test_mbps'] = df['y_test'] / 1e6
    df['y_pred_mbps'] = df['y_pred'] / 1e6

    df['true_bin'] = pd.cut(df['y_test_mbps'], bins=BINS_MBPS, labels=LABELS, right=False)
    df['pred_bin'] = pd.cut(df['y_pred_mbps'], bins=BINS_MBPS, labels=LABELS, right=False)

    label_to_index = {label: idx for idx, label in enumerate(LABELS)}
    df['true_idx'] = df['true_bin'].map(label_to_index)
    df['pred_idx'] = df['pred_bin'].map(label_to_index)

    df = df.dropna(subset=['true_idx', 'pred_idx'])
    if df.empty:
        return None, 0

    df['distance'] = (df['true_idx'].astype(int) - df['pred_idx'].astype(int)).abs()
    df['fuzzy_weight'] = df['distance'].apply(fuzzy_weight)

    return df['fuzzy_weight'].mean(), len(df)

def load_predictions(base_file, folder):
    df_base = pd.read_csv(os.path.join(folder, base_file))
    df_base.columns = df_base.columns.str.strip()

    if 'y_test' not in df_base.columns:
        return None, {}

    y_test = df_base['y_test']
    predictions = {
        'StackingRegressor': df_base.get('y_predict_StackingRegressor'),
        'KNeighborsRegressor': df_base.get('y_predict_KNeighborsRegressor'),
        'GradientBoostingRegressor': df_base.get('y_predict_GradientBoostingRegressor'),
        'XGBRegressor': df_base.get('y_predict_XGBRegressor'),
        'RandomForestRegressor': df_base.get('y_predict_RandomForestRegressor'),
        'ElasticNet': df_base.get('y_predict_ElasticNet')
    }

    for model_name in ['GRU', 'ARIMA', 'HoltWinters']:
        model_file = base_file.replace('.csv', f'_{model_name}.csv')
        model_path = os.path.join(folder, model_file)
        if os.path.exists(model_path):
            df_model = pd.read_csv(model_path)
            df_model.columns = df_model.columns.str.strip()
            col_pred = f'y_predict_{model_name}'
            if col_pred in df_model.columns:
                predictions[model_name] = df_model[col_pred]

    return y_test, predictions

def evaluate_fuzzy_accuracy(folder):
    results = []
    base_files = [f for f in os.listdir(folder)
                  if f.endswith('.csv') and not any(f.endswith(suf) for suf in ['GRU.csv', 'ARIMA.csv', 'HoltWinters.csv'])]

    for file in base_files:
        y_test, predictions = load_predictions(file, folder)
        if y_test is None:
            continue

        pop_name = file.replace('Vazao_', '').replace('.csv', '').upper()
        for model_name, y_pred in predictions.items():
            if y_pred is None:
                continue

            accuracy, sample_count = compute_fuzzy_accuracy(y_test, y_pred)
            if accuracy is not None:
                results.append({
                    'Pop': pop_name,
                    'Model': model_name,
                    'Fuzzy_Accuracy': accuracy * 100,
                    'Samples': sample_count
                })

    return pd.DataFrame(results)

df_fuzzy = evaluate_fuzzy_accuracy(CSV_FOLDER)
df_pivot = df_fuzzy.pivot(index='Pop', columns='Model', values='Fuzzy_Accuracy').reset_index()
df_counts = df_fuzzy.groupby('Pop')['Samples'].sum().reset_index()
df_result = pd.merge(df_pivot, df_counts, on='Pop').round(2)

mean_values = df_result.drop(columns=['Pop']).mean()
overall_row = {**mean_values.to_dict(), 'Pop': 'overall'}
df_result = pd.concat([df_result, pd.DataFrame([overall_row])], ignore_index=True)

pd.options.display.float_format = '{:.2f}'.format

selected_columns = ['Pop', 'StackingRegressor', 'ARIMA', 'GRU', 'HoltWinters']

df_result[selected_columns]
