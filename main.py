# main.py

import time
from procesamiento import cargar_y_preprocesar
from entrenamiento import ejecutar_experimentos, evaluar_en_prueba

# Cantidad de registros a usar (la pauta exige al menos 10.000).
# Es la perilla principal de velocidad: más registros = más fraudes y resultados más estables,
# pero SVM y k-NN se vuelven mucho más lentos. Usar None para trabajar con el dataset completo.
N_MUESTRAS = 50000


def main():
    inicio = time.perf_counter()
    print("--- INICIANDO PROYECTO DE DETECCIÓN DE FRAUDE ---")

    # 1. Cargar datos y hacer la partición 70/30 estratificada
    print("\n1. Cargando datos y dividiendo en Train/Test...")
    # Se asume que el dataset está en la misma carpeta
    X_train, X_test, y_train, y_test = cargar_y_preprocesar('creditcard.csv', n_muestras=N_MUESTRAS)

    # 2. Entrenar y ajustar hiperparámetros (Validación Cruzada)
    print("\n2. Ejecutando los 8 experimentos (CV de 5 particiones)...")
    # Esto devolverá un diccionario con los 8 mejores modelos ya entrenados
    modelos_entrenados = ejecutar_experimentos(X_train, y_train)

    # 3. Evaluación final en el conjunto de prueba
    print("\n3. Evaluando modelos en el conjunto de prueba...")
    for nombre_modelo, modelo in modelos_entrenados.items():
        print(f"\n--- Evaluando: {nombre_modelo} ---")
        evaluar_en_prueba(modelo, X_test, y_test)

    print(f"\n--- PROCESO COMPLETADO en {time.perf_counter() - inicio:.1f} s ---")


if __name__ == "__main__":
    main()
