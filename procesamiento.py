# procesamiento.py

import pandas as pd
from sklearn.model_selection import train_test_split


def cargar_y_preprocesar(ruta_archivo='creditcard.csv', n_muestras=None):
    df = pd.read_csv(ruta_archivo)

    #Exploración solicitada en la pauta
    n_duplicados = df.duplicated().sum()  # se calcula una sola vez (es costoso en 284k filas)
    print("--- EXPLORACIÓN INICIAL ---")
    print(f"Registros totales: {df.shape[0]}, Características: {df.shape[1]}")
    print(f"Valores faltantes totales: {df.isnull().sum().sum()}")
    print(f"Filas duplicadas: {n_duplicados}")

    # Limpieza: Eliminar duplicados si existen (justificación: transacciones idénticas al mismo milisegundo suelen ser errores de captura)
    if n_duplicados > 0:
        df = df.drop_duplicates()
        print(f"Registros tras eliminar duplicados: {df.shape[0]}")

    # La pauta permite usar un subconjunto (mínimo 10.000 registros).
    # Es la optimización más importante: SVM y k-NN crecen más que linealmente con las filas.
    if n_muestras and n_muestras < len(df):
        # Tomamos una muestra manteniendo la proporción de clases
        df, _ = train_test_split(df, train_size=n_muestras, stratify=df['Class'], random_state=42)
        print(f"Se extrajo una muestra de {n_muestras} registros.")

    # Distribución de clases
    conteo = df['Class'].value_counts()
    legitimas, fraudes = conteo[0], conteo[1]
    print(f"Transacciones legítimas (0): {legitimas} ({legitimas/len(df)*100:.3f}%)")
    print(f"Transacciones fraudulentas (1): {fraudes} ({fraudes/len(df)*100:.3f}%)")
    print("---------------------------\n")

    # Separar variables predictoras de la variable objetivo
    X = df.drop(columns=['Class'])
    y = df['Class']

    # Dividir en entrenamiento (70%) y prueba (30%) de forma estratificada
    # El parámetro stratify=y garantiza que el 30% de prueba conserve la misma proporción real de fraudes
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=42
    )

    return X_train, X_test, y_train, y_test
