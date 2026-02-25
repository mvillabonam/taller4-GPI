import os
import pandas as pd
import numpy as np

def main():
    # Cargar datos procesados
    base_dir = os.path.join(os.path.dirname(__file__), "..")
    data_path = os.path.join(base_dir, "data", "processed", "combined_data.csv")

    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"No existe {data_path}. Corre primero el pipeline de procesamiento."
        )

    df = pd.read_csv(data_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # Elegimos un target simple: el primer retorno disponible
    return_cols = [c for c in df.columns if c.endswith("_Return")]
    if len(return_cols) == 0:
        raise ValueError("No hay columnas *_Return en combined_data.csv para modelar.")

    y_col = return_cols[0]

    # Features: lags + rolling stats del primer activo (o de los que existan)
    feature_cols = [c for c in df.columns if ("_lag_" in c) or ("_rolling_" in c)]
    if len(feature_cols) == 0:
        raise ValueError("No hay features lag/rolling en combined_data.csv.")

    # Dataset para modelar
    X = df[feature_cols].astype(float).values
    y = df[y_col].astype(float).values

    # Split temporal 80/20
    split = int(0.8 * len(df))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Modelo: regresión lineal por mínimos cuadrados (sin sklearn)
    # beta = (X'X)^-1 X'y  usando pseudo-inversa para estabilidad
    X_train_aug = np.c_[np.ones(len(X_train)), X_train]
    X_test_aug  = np.c_[np.ones(len(X_test)),  X_test]

    beta = np.linalg.pinv(X_train_aug) @ y_train
    y_hat = X_test_aug @ beta

    # Métricas
    mae = np.mean(np.abs(y_test - y_hat))
    rmse = np.sqrt(np.mean((y_test - y_hat) ** 2))

    print("="*70)
    print("MODELADO: REGRESION LINEAL (BASELINE)")
    print("="*70)
    print(f"Target: {y_col}")
    print(f"N train: {len(y_train)} | N test: {len(y_test)}")
    print(f"Features: {len(feature_cols)}")
    print(f"MAE:  {mae:.6f}")
    print(f"RMSE: {rmse:.6f}")

    # Guardar resultados
    out_dir = os.path.join(base_dir, "results", "tables")
    os.makedirs(out_dir, exist_ok=True)

    out_path = os.path.join(out_dir, "modelado_metricas.csv")
    pd.DataFrame([{
        "target": y_col,
        "n_train": len(y_train),
        "n_test": len(y_test),
        "n_features": len(feature_cols),
        "mae": mae,
        "rmse": rmse
    }]).to_csv(out_path, index=False)

    print(f"✓ Guardado: {out_path}")

if __name__ == "__main__":
    main()