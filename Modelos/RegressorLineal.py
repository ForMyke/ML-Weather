import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import joblib
from datetime import datetime, timedelta
from meteostat import Hourly, Point
import requests

def cargar_datos():
    """Carga y ordena datos limpios"""
    df = pd.read_csv('datos/historico_limpio.csv')
    df['time'] = pd.to_datetime(df['time'])
    return df.sort_values('time')


def crear_features(df, modo='entrenamiento'):
    """Crea features para predicción"""
    df = df.copy()

    # Features temporales
    df['hora'] = df['time'].dt.hour
    df['dia'] = df['time'].dt.day
    df['mes'] = df['time'].dt.month
    df['dia_semana'] = df['time'].dt.dayofweek

    # Cíclicas
    df['hora_sin'] = np.sin(2 * np.pi * df['hora'] / 24)
    df['hora_cos'] = np.cos(2 * np.pi * df['hora'] / 24)
    df['mes_sin'] = np.sin(2 * np.pi * df['mes'] / 12)
    df['mes_cos'] = np.cos(2 * np.pi * df['mes'] / 12)

    # Lags de temperatura
    lags = [1, 2, 3, 4, 5, 6, 12, 24]
    if modo == 'prediccion':
        lags = [i for i in lags if i <= len(df) - 1]

    for lag in lags:
        df[f'temp_lag_{lag}h'] = df['temp'].shift(lag).fillna(df['temp'].mean())

    # Promedios móviles y desviaciones
    for window in [2, 3, 6, 12, 24]:
        w = min(window, len(df))
        if w > 0:
            df[f'temp_media_{window}h'] = df['temp'].rolling(w, min_periods=1).mean()
            if w > 1:
                df[f'temp_std_{window}h'] = df['temp'].rolling(w, min_periods=1).std().fillna(0)

    # Diferencias y tasas de cambio
    for lag in [1, 2, 3]:
        df[f'temp_diff_{lag}h'] = df['temp'].diff(lag).fillna(0)
        df[f'temp_cambio_{lag}h'] = (df['temp'] - df['temp'].shift(lag)) / (df['temp'].shift(lag).abs() + 1e-5)
        df[f'temp_cambio_{lag}h'] = df[f'temp_cambio_{lag}h'].fillna(0)

    # Máximos y mínimos
    for window in [3, 6, 12]:
        w = min(window, len(df))
        if w > 0:
            df[f'temp_max_{window}h'] = df['temp'].rolling(w, min_periods=1).max()
            df[f'temp_min_{window}h'] = df['temp'].rolling(w, min_periods=1).min()

    # Features de otras variables
    for col in ['dwpt', 'rhum', 'prcp', 'wdir', 'wspd', 'pres']:
        if col in df.columns:
            for lag in [1, 2, 3]:
                df[f'{col}_lag_{lag}h'] = df[col].shift(lag).fillna(df[col].mean())
            df[f'{col}_diff_1h'] = df[col].diff(1).fillna(0)

    return df.dropna() if modo == 'entrenamiento' else df


def obtener_clima_actual():
    try:
        # Coordenadas Gustavo A. Madero, CDMX: 19.50, -99.08
        url = "https://api.open-meteo.com/v1/forecast?latitude=19.50&longitude=-99.08&current=temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m,wind_direction_10m,pressure_msl&timezone=America/Mexico_City"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            current = response.json()['current']
            return {
                'time': datetime.now(),
                'temp': float(current['temperature_2m']),
                'rhum': float(current['relative_humidity_2m']),
                'prcp': float(current.get('precipitation', 0)),
                'wdir': float(current.get('wind_direction_10m', 0)),
                'wspd': float(current.get('wind_speed_10m', 0)),
                'pres': float(current.get('pressure_msl', 0))
            }
    except Exception as e:
        print(f"Error obteniendo clima: {e}")
    return None

def obtener_datos_historicos():
    punto = Point(19.40, -99.15, 2240)
    ahora = datetime.now()
    hace_800_dias = ahora - timedelta(days=800)

    datos = Hourly(punto, hace_800_dias, ahora).fetch()
    return datos.reset_index() if len(datos) > 0 else pd.DataFrame()


def entrenar_modelo(X_train, y_train):
    tscv = TimeSeriesSplit(n_splits=5)
    modelo = HistGradientBoostingRegressor(
        max_iter=500, max_depth=15, learning_rate=0.02,
        min_samples_leaf=3, l2_regularization=0.1, random_state=42
    )

    scores = []
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train), 1):
        X_t, X_v = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_t, y_v = y_train.iloc[train_idx], y_train.iloc[val_idx]

        modelo.fit(X_t, y_t)
        y_pred = modelo.predict(X_v)

        scores.append({
            'fold': fold,
            'mse': mean_squared_error(y_v, y_pred),
            'mae': mean_absolute_error(y_v, y_pred),
            'r2': r2_score(y_v, y_pred)
        })

    modelo.fit(X_train, y_train)
    return modelo, pd.DataFrame(scores)


def predecir_hora_actual(modelo, columnas_features, df_historico):
    clima_actual = obtener_clima_actual()
    if clima_actual is None:
        print("No se pudo obtener clima en tiempo real")
        return None

    df_completo = df_historico.copy()
    df_completo = pd.concat([df_completo, pd.DataFrame([clima_actual])], ignore_index=True)

    df_procesado = crear_features(df_completo, modo='prediccion')
    for col in columnas_features:
        if col not in df_procesado.columns:
            df_procesado[col] = 0

    X_pred = df_procesado[columnas_features].iloc[-1:].fillna(0)
    temp_predicha = modelo.predict(X_pred)[0]

    return temp_predicha, clima_actual['temp']


def guardar_y_visualizar(modelo, scores, y_test, predicciones, feature_cols, df_historico):
    joblib.dump(modelo, 'modelo_temperatura.pkl')
    scores.to_csv('metricas_cv.csv', index=False)

    resultados = pd.DataFrame({
        'temperatura_real': y_test.values,
        'temperatura_predicha': predicciones
    })
    resultados.to_csv('predicciones.csv', index=False)

    # Gráfico 1: Predicciones
    fig, axes = plt.subplots(2, 1, figsize=(15, 10))
    axes[0].plot(y_test.values[:500], label='Actual', alpha=0.7)
    axes[0].plot(predicciones[:500], label='Predicción', alpha=0.7)
    axes[0].set_title('Temperatura: Real vs Predicción')
    axes[0].legend()
    axes[0].grid(True)

    # Gráfico 2: Temperatura actual
    resultado = predecir_hora_actual(modelo, feature_cols, df_historico)
    if resultado:
        temp_pred, temp_real = resultado
        axes[1].bar(['Real', 'Predicha'], [temp_real, temp_pred], color=['#FF6B6B', '#4ECDC4'], alpha=0.7)
        axes[1].set_title(f'Temperatura Actual - Diferencia: {abs(temp_real - temp_pred):.1f}°C')
        axes[1].set_ylabel('°C')

    plt.tight_layout()
    plt.savefig('resultados.png', dpi=150)

def main():
    hora = datetime.now()
    hora_actual = hora.strftime('%H:%M:%S')
    hora_siguiente = (hora + timedelta(hours=1)).strftime('%H:%M:%S')
    df = cargar_datos()
    print(f"Datos: {len(df)} registros")

    df = crear_features(df)
    train_size = int(len(df) * 0.8)

    feature_cols = [col for col in df.columns if col not in ['time', 'temp']]
    X_train, y_train = df[:train_size][feature_cols], df[:train_size]['temp']
    X_test, y_test = df[train_size:][feature_cols], df[train_size:]['temp']

    print(f"Train: {len(X_train)} | Test: {len(X_test)} | Features: {len(feature_cols)}")

    modelo, scores = entrenar_modelo(X_train, y_train)
    predicciones = modelo.predict(X_test)

    print(f"\nMSE: {mean_squared_error(y_test, predicciones):.4f}")
    print(f"MAE: {mean_absolute_error(y_test, predicciones):.4f}")
    print(f"R2: {r2_score(y_test, predicciones):.4f}")

    resultado = predecir_hora_actual(modelo, feature_cols, df)
    if resultado:
        temp_pred, temp_real = resultado
        print(f"\nResultados")
        print(f"Temperatura para las {hora_actual}: {temp_real:.1f}°C")
        print(f"Temperatura para las {hora_siguiente} : {temp_pred:.1f}°C")
        print(f"La temperatura cambiara: {abs(temp_pred - temp_real):.1f}°C")

    guardar_y_visualizar(modelo, scores, y_test, predicciones, feature_cols, df)

