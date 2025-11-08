import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import joblib


def cargar_datos():
    """Carga los datos limpios"""
    df = pd.read_csv('../datos/historico_limpio.csv')
    df['time'] = pd.to_datetime(df['time'])
    df = df.sort_values('time')
    return df


def crear_features(df, horizonte=24):
    """Crea features para predicción multihorizonte"""
    df = df.copy()

    # Features temporales
    df['hora'] = df['time'].dt.hour
    df['dia'] = df['time'].dt.day
    df['mes'] = df['time'].dt.month
    df['dia_semana'] = df['time'].dt.dayofweek

    # Lags para temperatura (variable objetivo)
    for i in range(1, horizonte + 1):
        df[f'temp_lag_{i}'] = df['temp'].shift(i)

    # Eliminar filas con NaN generados por los lags
    df = df.dropna()

    return df


def entrenar_modelo(X_train, y_train):
    """Entrena el modelo con cross-validation"""
    print("Entrenando modelo con TimeSeriesSplit...")

    tscv = TimeSeriesSplit(n_splits=5)
    modelo = HistGradientBoostingRegressor(max_iter=200, random_state=42)

    scores = []
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train), 1):
        X_t, X_v = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_t, y_v = y_train.iloc[train_idx], y_train.iloc[val_idx]

        modelo.fit(X_t, y_t)
        y_pred = modelo.predict(X_v)

        mse = mean_squared_error(y_v, y_pred)
        mae = mean_absolute_error(y_v, y_pred)
        r2 = r2_score(y_v, y_pred)

        scores.append({'fold': fold, 'mse': mse, 'mae': mae, 'r2': r2})
        print(f"Fold {fold} - MSE: {mse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")

    # Entrenamiento final con todos los datos
    modelo.fit(X_train, y_train)

    return modelo, pd.DataFrame(scores)


def generar_pronostico(modelo, X_test, horizonte=24):
    """Genera pronóstico multihorizonte"""
    print(f"\nGenerando pronóstico para las próximas {horizonte} horas...")
    predicciones = modelo.predict(X_test)
    return predicciones


def visualizar_resultados(y_test, predicciones, scores):
    """Visualiza los resultados del modelo"""
    fig, axes = plt.subplots(2, 1, figsize=(15, 10))

    # Gráfico 1: Predicción vs Real
    axes[0].plot(y_test.values[:500], label='Real', alpha=0.7)
    axes[0].plot(predicciones[:500], label='Predicción', alpha=0.7)
    axes[0].set_title('Temperatura: Real vs Predicción (primeras 500 horas)')
    axes[0].set_xlabel('Horas')
    axes[0].set_ylabel('Temperatura (°C)')
    axes[0].legend()
    axes[0].grid(True)

    # Gráfico 2: Métricas por fold
    axes[1].plot(scores['fold'], scores['mae'], marker='o', label='MAE')
    axes[1].set_title('Error Absoluto Medio por Fold')
    axes[1].set_xlabel('Fold')
    axes[1].set_ylabel('MAE')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig('resultados_modelo.png')
    print("✓ Gráficos guardados en resultados_modelo.png")


def guardar_resultados(modelo, scores, y_test, predicciones):
    """Guarda el modelo y resultados"""
    # Guardar modelo
    joblib.dump(modelo, 'modelo_temperatura.pkl')
    print("✓ Modelo guardado en modelo_temperatura.pkl")

    # Guardar métricas
    scores.to_csv('metricas_cv.csv', index=False)
    print("✓ Métricas guardadas en metricas_cv.csv")

    # Guardar predicciones
    resultados = pd.DataFrame({
        'temperatura_real': y_test.values,
        'temperatura_predicha': predicciones
    })
    resultados.to_csv('predicciones.csv', index=False)
    print("✓ Predicciones guardadas en predicciones.csv")


def main():
    # Cargar datos
    df = cargar_datos()
    print(f"Datos cargados: {len(df)} registros")

    # Crear features
    df = crear_features(df, horizonte=24)

    # Dividir datos
    train_size = int(len(df) * 0.8)
    train_df = df[:train_size]
    test_df = df[train_size:]

    # Variables independientes y dependientes
    feature_cols = [col for col in df.columns if col not in ['time', 'temp']]
    X_train = train_df[feature_cols]
    y_train = train_df['temp']
    X_test = test_df[feature_cols]
    y_test = test_df['temp']

    print(f"\nTamaño entrenamiento: {len(X_train)}")
    print(f"Tamaño prueba: {len(X_test)}")

    # Entrenar modelo
    modelo, scores = entrenar_modelo(X_train, y_train)

    # Generar pronóstico
    predicciones = generar_pronostico(modelo, X_test)

    # Métricas finales
    mse = mean_squared_error(y_test, predicciones)
    mae = mean_absolute_error(y_test, predicciones)
    r2 = r2_score(y_test, predicciones)

    print(f"\n=== MÉTRICAS FINALES ===")
    print(f"MSE: {mse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R2: {r2:.4f}")

    # Visualizar
    visualizar_resultados(y_test, predicciones, scores)

    # Guardar resultados
    guardar_resultados(modelo, scores, y_test, predicciones)


if __name__ == "__main__":
    main()