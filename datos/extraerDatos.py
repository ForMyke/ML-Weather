from meteostat import Hourly, Point
from datetime import datetime
import pandas as pd

# Configuración
LATITUD = 19.504610156330326
LONGITUD = -99.15299168364929
ALTITUD = 2240


def descargar_datos():
    punto = Point(LATITUD, LONGITUD, ALTITUD)
    fecha_fin = datetime.now()
    fecha_inicio = datetime(fecha_fin.year - 10, fecha_fin.month, fecha_fin.day)
    datos = Hourly(punto, fecha_inicio, fecha_fin)
    df = datos.fetch()
    df.to_csv('historico.csv')
    return df
def limpiar_datos():
    df = pd.read_csv('historico.csv')
    # Eliminar columnas con demasiados nulos (más del 80%)
    porcentaje_nulos = df.isnull().sum() / len(df)
    columnas_mantener = porcentaje_nulos[porcentaje_nulos < 0.8].index
    df = df[columnas_mantener]
    # Interpolación lineal para rellenar nulos
    columnas_numericas = df.select_dtypes(include=['float64', 'int64']).columns
    for col in columnas_numericas:
        df[col] = df[col].interpolate(method='linear', limit_direction='both')
    # Si quedan nulos despues de interpolacion, rellenar con media
    for col in columnas_numericas:
        if df[col].isnull().any():
            df[col].fillna(df[col].mean(), inplace=True)

    # Eliminar solo si aún hay nulos después de interpolación
    df = df.dropna()

    df.to_csv('historico_limpio.csv', index=False)



