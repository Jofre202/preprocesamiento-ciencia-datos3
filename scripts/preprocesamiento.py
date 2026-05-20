import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def cargar_datos(ruta):
    return pd.read_csv(ruta)

def eliminar_duplicados(df):
    return df.drop_duplicates()

def manejar_nulos(df):
    return df.fillna(df.mean(numeric_only=True))

def normalizar_datos(df, columnas):
    scaler = MinMaxScaler()
    df[columnas] = scaler.fit_transform(df[columnas])
    return df

def codificar_categoricas(df):
    return pd.get_dummies(df)

def guardar_datos(df, ruta):
    df.to_csv(ruta, index=False)

print("Preprocesamiento completado")