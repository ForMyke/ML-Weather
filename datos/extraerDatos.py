from meteostat import Hourly, Point
from datetime import datetime
import pandas as pd

# Configuración
LATITUD = 19.40
LONGITUD = -99.15
ALTITUD = 2240


def descargar_datos():
    """Descarga datos climáticos de los últimos 20 años"""
    punto = Point(LATITUD, LONGITUD, ALTITUD)
    fecha_fin = datetime.now()
    fecha_inicio = datetime(fecha_fin.year - 10, fecha_fin.month, fecha_fin.day)

    print(f"Descargando datos de {fecha_inicio.date()} a {fecha_fin.date()}...")

    datos = Hourly(punto, fecha_inicio, fecha_fin)
    df = datos.fetch()

    df.to_csv('historico.csv')
    print(f"✓ Datos guardados en historico.csv")
    print(f"Total de registros: {len(df)}")

    return df


def limpiar_datos():
    """Limpia los datos con interpolación en lugar de eliminación"""
    df = pd.read_csv('historico.csv')

    print(f"\nRegistros originales: {len(df)}")
    print(f"Valores nulos por columna:\n{df.isnull().sum()}\n")

    # Eliminar columnas con demasiados nulos (más del 80%)
    porcentaje_nulos = df.isnull().sum() / len(df)
    columnas_mantener = porcentaje_nulos[porcentaje_nulos < 0.8].index
    df = df[columnas_mantener]

    print(f"Columnas después de filtrar: {list(df.columns)}")

    # Interpolación lineal para rellenar nulos
    columnas_numericas = df.select_dtypes(include=['float64', 'int64']).columns
    for col in columnas_numericas:
        df[col] = df[col].interpolate(method='linear', limit_direction='both')

    # Si quedan nulos después de interpolación, rellenar con media
    for col in columnas_numericas:
        if df[col].isnull().any():
            df[col].fillna(df[col].mean(), inplace=True)

    # Eliminar solo si aún hay nulos después de interpolación
    df = df.dropna()

    df.to_csv('historico_limpio.csv', index=False)

    print(f"\nRegistros después de limpieza: {len(df)}")
    print(f"Valores nulos restantes:\n{df.isnull().sum()}\n")
    print("✓ Datos limpios guardados en historico_limpio.csv")


