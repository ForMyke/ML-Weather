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
    """Carga los datos limpios"""
    df = pd.read_csv('../datos/historico_limpio.csv')
    df['time'] = pd.to_datetime(df['time'])
    df = df.sort_values('time')
    return df


def crear_features(df, modo='entrenamiento'):
    """Crea features para predicción"""
    df = df.copy()

    # Features temporales
    df['hora'] = df['time'].dt.hour
    df['minuto'] = df['time'].dt.minute
    df['dia'] = df['time'].dt.day
    df['mes'] = df['time'].dt.month
    df['dia_semana'] = df['time'].dt.dayofweek
    df['dia_año'] = df['time'].dt.dayofyear

    # Características cíclicas para capturar mejor patrones horarios
    df['hora_sin'] = np.sin(2 * np.pi * df['hora'] / 24)
    df['hora_cos'] = np.cos(2 * np.pi * df['hora'] / 24)
    df['mes_sin'] = np.sin(2 * np.pi * df['mes'] / 12)
    df['mes_cos'] = np.cos(2 * np.pi * df['mes'] / 12)

    # Lags de temperatura - MÁS CORTOS para capturar cambios recientes
    lags = [1, 2, 3, 4, 5, 6, 12, 24]
    if modo == 'prediccion':
        max_lag = min(24, len(df) - 1)
        lags = [i for i in lags if i <= max_lag]

    for i in lags:
        df[f'temp_lag_{i}h'] = df['temp'].shift(i)

    # Rellenar lags faltantes
    for i in [1, 2, 3, 4, 5, 6, 12, 24]:
        col_name = f'temp_lag_{i}h'
        if col_name not in df.columns:
            df[col_name] = df['temp'].mean() if len(df) > 0 else 20.0
        elif df[col_name].isna().any():
            df[col_name] = df[col_name].fillna(df['temp'].mean())

    # Promedios móviles - ventanas más cortas
    for window in [2, 3, 6, 12, 24]:
        window_size = min(window, len(df))
        if window_size > 0:
            df[f'temp_media_{window}h'] = df['temp'].rolling(window=window_size, min_periods=1).mean()
        else:
            df[f'temp_media_{window}h'] = df['temp'].mean() if len(df) > 0 else 20.0

    # Desviación estándar
    for window in [3, 6, 12, 24]:
        window_size = min(window, len(df))
        if window_size > 1:
            df[f'temp_std_{window}h'] = df['temp'].rolling(window=window_size, min_periods=1).std().fillna(0)
        else:
            df[f'temp_std_{window}h'] = 0

    # Diferencias - CLAVE para detectar cambios bruscos
    for lag in [1, 2, 3, 6]:
        df[f'temp_diff_{lag}h'] = df['temp'].diff(lag).fillna(0)

    # Tasa de cambio - indica si la temperatura está subiendo o bajando rápidamente
    for lag in [1, 2, 3]:
        df[f'temp_cambio_{lag}h'] = (df['temp'] - df['temp'].shift(lag)) / (df['temp'].shift(lag).abs() + 1e-5)
        df[f'temp_cambio_{lag}h'] = df[f'temp_cambio_{lag}h'].fillna(0)

    # Máximo y mínimo de las últimas horas
    for window in [3, 6, 12]:
        window_size = min(window, len(df))
        if window_size > 0:
            df[f'temp_max_{window}h'] = df['temp'].rolling(window=window_size, min_periods=1).max()
            df[f'temp_min_{window}h'] = df['temp'].rolling(window=window_size, min_periods=1).min()
        else:
            df[f'temp_max_{window}h'] = df['temp']
            df[f'temp_min_{window}h'] = df['temp']

    # Features de otras variables meteorológicas
    for col in ['dwpt', 'rhum', 'prcp', 'wdir', 'wspd', 'pres']:
        if col in df.columns:
            # Lags
            for lag in [1, 2, 3]:
                df[f'{col}_lag_{lag}h'] = df[col].shift(lag).fillna(df[col].mean() if len(df) > 0 else 0)

            # Diferencias
            df[f'{col}_diff_1h'] = df[col].diff(1).fillna(0)
        else:
            for lag in [1, 2, 3]:
                df[f'{col}_lag_{lag}h'] = 0
            df[f'{col}_diff_1h'] = 0

    if modo == 'entrenamiento':
        df = df.dropna()

    return df


def entrenar_modelo(X_train, y_train):
    """Entrena el modelo con hiperparámetros optimizados"""
    print("Entrenando modelo con TimeSeriesSplit...")

    tscv = TimeSeriesSplit(n_splits=5)

    # Modelo más profundo para capturar patrones complejos
    modelo = HistGradientBoostingRegressor(
        max_iter=500,
        max_depth=15,
        learning_rate=0.02,
        min_samples_leaf=3,
        l2_regularization=0.1,
        random_state=42
    )

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

    modelo.fit(X_train, y_train)

    return modelo, pd.DataFrame(scores)


def obtener_clima_wttr():
    """Obtiene clima actual de wttr.in"""
    try:
        url = "https://wttr.in/Mexico_City?format=j1"
        response = requests.get(url, timeout=10)

        if response.status_code == 200:
            data = response.json()
            current = data['current_condition'][0]

            clima_actual = {
                'time': datetime.now(),
                'temp': float(current['temp_C']),
                'rhum': float(current['humidity']),
                'prcp': float(current.get('precipMM', 0)),
                'wdir': float(current.get('winddirDegree', 0)),
                'wspd': float(current['windspeedKmph']),
                'pres': float(current['pressure'])
            }

            print(f"✓ Clima actual de wttr.in: {clima_actual['temp']:.1f}°C")
            return clima_actual
        else:
            print(f"Error en wttr.in: {response.status_code}")
            return None
    except Exception as e:
        print(f"Error obteniendo clima: {e}")
        return None


def obtener_datos_recientes():
    """Obtiene datos recientes de Meteostat"""
    punto = Point(19.40, -99.15, 2240)
    ahora = datetime.now()
    hace_5dias = ahora - timedelta(days=5)

    print(f"Descargando datos históricos desde {hace_5dias.date()}")
    datos = Hourly(punto, hace_5dias, ahora)
    df_actual = datos.fetch()

    if len(df_actual) > 0:
        df_actual = df_actual.reset_index()
        print(f"Datos históricos obtenidos: {len(df_actual)} registros")

    return df_actual


def predecir_hora_actual(modelo, columnas_features):
    """Predice temperatura actual usando datos en tiempo real"""
    ahora = datetime.now()
    print(f"\n=== PREDICCIÓN PARA: {ahora.strftime('%Y-%m-%d %H:%M')} ===")

    # Obtener clima actual
    clima_actual = obtener_clima_wttr()

    if clima_actual is None:
        print("No se pudo obtener clima en tiempo real")
        return None

    # Obtener datos históricos
    df_historico = obtener_datos_recientes()

    if len(df_historico) == 0:
        print("No hay datos históricos disponibles")
        return None

    # Crear múltiples registros interpolados entre el último histórico y ahora
    ultimo_historico = df_historico.iloc[-1]
    tiempo_ultimo = ultimo_historico['time']
    temp_ultimo = ultimo_historico['temp']

    horas_diferencia = (ahora - tiempo_ultimo).total_seconds() / 3600

    # Interpolar temperaturas entre último histórico y clima actual
    registros_interpolados = []
    if horas_diferencia > 0:
        pasos = int(horas_diferencia) + 1
        for i in range(1, pasos):
            factor = i / pasos
            tiempo_interp = tiempo_ultimo + timedelta(hours=i)
            temp_interp = temp_ultimo + (clima_actual['temp'] - temp_ultimo) * factor

            registro = ultimo_historico.copy()
            registro['time'] = tiempo_interp
            registro['temp'] = temp_interp

            # Interpolar otras variables también
            for col in ['rhum', 'wspd', 'pres']:
                if col in clima_actual and col in registro:
                    registro[col] = ultimo_historico[col] + (clima_actual[col] - ultimo_historico[col]) * factor

            registros_interpolados.append(registro)

    # Agregar clima actual
    nuevo_registro = clima_actual.copy()
    if 'dwpt' not in nuevo_registro:
        nuevo_registro['dwpt'] = ultimo_historico.get('dwpt', None)

    # Combinar todo
    df_completo = df_historico.copy()
    if len(registros_interpolados) > 0:
        df_completo = pd.concat([df_completo, pd.DataFrame(registros_interpolados)], ignore_index=True)
    df_completo = pd.concat([df_completo, pd.DataFrame([nuevo_registro])], ignore_index=True)
    df_completo = df_completo.sort_values('time')

    print(f"Último dato histórico: {tiempo_ultimo} - {temp_ultimo:.1f}°C")
    print(f"Clima actual (tiempo real): {clima_actual['temp']:.1f}°C")
    print(f"Cambio de temperatura: {clima_actual['temp'] - temp_ultimo:.1f}°C en {horas_diferencia:.1f} horas")

    # Crear features
    df_procesado = crear_features(df_completo, modo='prediccion')

    if len(df_procesado) == 0:
        print("No se pudieron crear features")
        return None

    # Rellenar columnas faltantes
    for col in columnas_features:
        if col not in df_procesado.columns:
            df_procesado[col] = 0

    # Tomar último registro
    X_pred = df_procesado[columnas_features].iloc[-1:].fillna(0)

    # Predecir
    temp_predicha = modelo.predict(X_pred)[0]

    return temp_predicha, clima_actual['temp']


def visualizar_resultados(y_test, predicciones, scores):
    """Visualiza los resultados del modelo"""
    fig, axes = plt.subplots(2, 1, figsize=(15, 10))

    axes[0].plot(y_test.values[:500], label='Real', alpha=0.7)
    axes[0].plot(predicciones[:500], label='Predicción', alpha=0.7)
    axes[0].set_title('Temperatura: Real vs Predicción (primeras 500 horas)')
    axes[0].set_xlabel('Horas')
    axes[0].set_ylabel('Temperatura (°C)')
    axes[0].legend()
    axes[0].grid(True)

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
    joblib.dump(modelo, 'modelo_temperatura.pkl')
    print("✓ Modelo guardado en modelo_temperatura.pkl")

    scores.to_csv('metricas_cv.csv', index=False)
    print("✓ Métricas guardadas en metricas_cv.csv")

    resultados = pd.DataFrame({
        'temperatura_real': y_test.values,
        'temperatura_predicha': predicciones
    })
    resultados.to_csv('predicciones.csv', index=False)
    print("✓ Predicciones guardadas en predicciones.csv")


def visualizar_predicciones_csv():
    """Visualiza las predicciones guardadas en predicciones.csv"""
    # Leer archivo
    df = pd.read_csv('predicciones.csv')

    fig, axes = plt.subplots(2, 1, figsize=(15, 10))

    # Gráfico 1: Todas las predicciones
    axes[0].plot(df['temperatura_real'], label='Real', alpha=0.7, linewidth=1)
    axes[0].plot(df['temperatura_predicha'], label='Predicción', alpha=0.7, linewidth=1)
    axes[0].set_title('Temperatura: Real vs Predicción (conjunto de prueba completo)')
    axes[0].set_xlabel('Índice')
    axes[0].set_ylabel('Temperatura (°C)')
    axes[0].legend()
    axes[0].grid(True)

    # Gráfico 2: Primeras 200 predicciones (zoom)
    n_zoom = min(200, len(df))
    axes[1].plot(df['temperatura_real'][:n_zoom], label='Real', alpha=0.7, linewidth=2)
    axes[1].plot(df['temperatura_predicha'][:n_zoom], label='Predicción', alpha=0.7, linewidth=2)
    axes[1].set_title(f'Temperatura: Real vs Predicción (primeras {n_zoom} horas - zoom)')
    axes[1].set_xlabel('Horas')
    axes[1].set_ylabel('Temperatura (°C)')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig('visualizacion_predicciones.png', dpi=150)
    print("✓ Visualización guardada en visualizacion_predicciones.png")
    plt.close()


def visualizar_temperatura_actual(temp_real, temp_predicha):
    """Visualiza comparación entre temperatura real y predicha actual"""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    ahora = datetime.now()
    categorias = ['Temperatura Real', 'Temperatura Predicha']
    temperaturas = [temp_real, temp_predicha]
    colores = ['#FF6B6B', '#4ECDC4']

    # Gráfico de barras
    barras = ax.bar(categorias, temperaturas, color=colores, alpha=0.7, edgecolor='black', linewidth=2)

    # Agregar valores encima de las barras
    for i, (barra, temp) in enumerate(zip(barras, temperaturas)):
        altura = barra.get_height()
        ax.text(barra.get_x() + barra.get_width() / 2., altura,
                f'{temp:.1f}°C',
                ha='center', va='bottom', fontsize=14, fontweight='bold')

    # Agregar línea de diferencia
    diferencia = abs(temp_real - temp_predicha)
    ax.plot([0, 1], [temp_real, temp_predicha], 'k--', alpha=0.3, linewidth=1)

    # Configuración del gráfico
    ax.set_ylabel('Temperatura (°C)', fontsize=12, fontweight='bold')
    ax.set_title(f'Comparación Temperatura Actual\n{ahora.strftime("%Y-%m-%d %H:%M")}\nDiferencia: {diferencia:.1f}°C',
                 fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([min(temperaturas) - 2, max(temperaturas) + 3])

    plt.tight_layout()
    plt.savefig('temperatura_actual.png', dpi=150)
    print("✓ Comparación de temperatura actual guardada en temperatura_actual.png")
    plt.close()


# Agregar estas funciones al final del main():
def main():
    df = cargar_datos()
    print(f"Datos limpios cargados: {len(df)} registros")

    df = crear_features(df, modo='entrenamiento')

    train_size = int(len(df) * 0.8)
    train_df = df[:train_size]
    test_df = df[train_size:]

    feature_cols = [col for col in df.columns if col not in ['time', 'temp']]
    X_train = train_df[feature_cols]
    y_train = train_df['temp']
    X_test = test_df[feature_cols]
    y_test = test_df['temp']

    print(f"\nTamaño entrenamiento: {len(X_train)}")
    print(f"Tamaño prueba: {len(X_test)}")
    print(f"Número de features: {len(feature_cols)}")

    modelo, scores = entrenar_modelo(X_train, y_train)

    predicciones = modelo.predict(X_test)

    mse = mean_squared_error(y_test, predicciones)
    mae = mean_absolute_error(y_test, predicciones)
    r2 = r2_score(y_test, predicciones)

    print(f"\n=== MÉTRICAS FINALES ===")
    print(f"MSE: {mse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R2: {r2:.4f}")

    visualizar_resultados(y_test, predicciones, scores)
    guardar_resultados(modelo, scores, y_test, predicciones)

    # NUEVA: Visualizar predicciones.csv
    print("\n=== VISUALIZANDO PREDICCIONES ===")
    visualizar_predicciones_csv()

    resultado = predecir_hora_actual(modelo, feature_cols)

    if resultado and isinstance(resultado, tuple):
        temp_pred, temp_real = resultado
        print(f"\n✓ Temperatura REAL (wttr.in): {temp_real:.1f}°C")
        print(f"✓ Temperatura PREDICHA: {temp_pred:.1f}°C")
        print(f"Diferencia: {abs(temp_real - temp_pred):.1f}°C")

        # NUEVA: Visualizar temperatura actual
        print("\n=== VISUALIZANDO TEMPERATURA ACTUAL ===")
        visualizar_temperatura_actual(temp_real, temp_pred)


if __name__ == "__main__":
    main()