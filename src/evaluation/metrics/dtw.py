# Calcular a similaridade DTW entre dois vetores

def analyze_dtw_vs_original(predictions_folder=PREDICTIONS_FOLDER, save_plot="dtw_vs_original_gru.png"):
    """Analisa DTW das predições GRU vs originais e gera plots e resumo."""
    results = []

    pred_files = sorted(predictions_folder.glob("*_6h_*_gru.csv"))
    if not pred_files:
        print("Nenhum arquivo de predição encontrado.")
        return None

    for pred_path in pred_files:
        try:
            df_pred = pd.read_csv(pred_path)
            pred_vals = df_pred.select_dtypes(include=[np.number]).iloc[:, 0].dropna().values
        except Exception as e:
            warnings.warn(f"Falha lendo {pred_path}: {e}")
            continue

        method_match = re.search(r"_6h_(.+?)\.csv_gru\.csv$", pred_path.name)
        imputation_method = method_match.group(1) if method_match else "Unknown"

        ref_series, ref_path = find_original_series(pred_path, ORIGINALS_FOLDER)
        if ref_series is None:
            results.append({
                "filename": pred_path.name,
                "imputation_method": imputation_method,
                "ref_found": False,
                "dtw_distance": np.nan,
                "normalized_dtw": np.nan
            })
            continue

        ref_vals = ref_series.values.astype(float)
        n = len(pred_vals)
        if len(ref_vals) >= n:
            ref_segment = ref_vals[-n:]
        else:
            pred_vals = pred_vals[-len(ref_vals):]
            ref_segment = ref_vals

        try:
            dist, _ = fastdtw(pred_vals.reshape(-1, 1), ref_segment.reshape(-1, 1), dist=euclidean)
            norm = dist / len(pred_vals)
        except Exception as e:
            warnings.warn(f"DTW falhou em {pred_path.name}: {e}")
            dist, norm = np.nan, np.nan

        results.append({
            "filename": pred_path.name,
            "imputation_method": imputation_method,
            "ref_found": True,
            "ref_path": str(ref_path),
            "dtw_distance": dist,
            "normalized_dtw": norm,
            "sequence_length": len(pred_vals)
        })

    dtw_df = pd.DataFrame(results)
    valid = dtw_df.dropna(subset=["normalized_dtw"])
    if valid.empty:
        print("Sem dados válidos para DTW.")
        return dtw_df

    summary = (valid.groupby("imputation_method")["normalized_dtw"]
               .agg(["count", "mean", "std", "min", "max"])
               .sort_values("mean").round(4))

    # ==============================
    # PLOTS
    # ==============================
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1) Heatmap média por método
    sns.heatmap(summary[["mean"]], annot=True, fmt=".4f", cmap="YlGnBu",
                cbar_kws={"label": "Avg Normalized DTW"}, ax=axes[0, 0])
    axes[0, 0].set_title("Fidelidade ao Original por Método (menor = melhor)")

    # 2) Histograma global
    axes[0, 1].hist(valid["normalized_dtw"], bins=30, edgecolor="black", alpha=0.7)
    axes[0, 1].set_xlabel("Normalized DTW")
    axes[0, 1].set_ylabel("Frequência")
    axes[0, 1].set_title("Distribuição de DTW Normalizado (Predição vs Original)")
    axes[0, 1].grid(axis="y", alpha=0.3)

    # 3) Top 10 métodos
    top10 = summary.head(10)
    axes[1, 0].barh(top10.index, top10["mean"])
    axes[1, 0].invert_yaxis()
    axes[1, 0].set_xlabel("Avg Normalized DTW")
    axes[1, 0].set_title("Top 10 Métodos Mais Fiéis (menor DTW)")
    axes[1, 0].grid(axis="x", alpha=0.3)

    # 4) Correlação DTW x RMSE (se existir coluna rmse)
    if "rmse" in df_pred.columns:
        try:
            merged = valid[["imputation_method", "normalized_dtw"]].copy()
            merged["rmse"] = df_pred["rmse"].mean()  # valor médio, placeholder
            axes[1, 1].scatter(merged["normalized_dtw"], merged["rmse"], alpha=0.6, s=50)
            axes[1, 1].set_xlabel("Normalized DTW")
            axes[1, 1].set_ylabel("RMSE")
            axes[1, 1].set_title("DTW (vs original) x RMSE")
            axes[1, 1].grid(alpha=0.3)
        except Exception:
            axes[1, 1].text(0.5, 0.5, "Sem RMSE disponível", ha="center", va="center")
    else:
        axes[1, 1].axis("off")
        axes[1, 1].text(0.5, 0.5, "Sem RMSE disponível", ha="center", va="center")

    plt.tight_layout()
    plt.savefig(save_plot, dpi=300, bbox_inches="tight")
    print(f"DTW vs original salvo em {save_plot}")

    return dtw_df, summary, fig

dtw_df, summary, fig = analyze_dtw_vs_original(
    predictions_folder=Path("predictions_gru"),
    save_plot="dtw_vs_original_gru.png"
)