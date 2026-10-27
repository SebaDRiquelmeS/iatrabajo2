# Detección de fraude con aprendizaje supervisado

Problema 2. Compara k-NN, Árbol de Decisión, SVM y un
Random Forest de implementación propia, con y sin SMOTE. El criterio de selección es la
sensibilidad de la clase fraude, la precisión se usa solo para desempatar.

## Archivos

| Archivo | Contenido |
|---|---|
| `main.py` | Punto de entrada: ejecuta todo el procedimiento en orden |
| `procesamiento.py` | Carga, exploración, limpieza y partición 70/30 estratificada |
| `entrenamiento.py` | Pipelines, Grid Search con validación cruzada y evaluación en 
| `mi_random_forest.py` | Random Forest propio (bootstrap, `max_features` y votación) |
| `requirements.txt` | Dependencias con sus versiones |

## Dependencias y versiones

- Python 3.15.0
- numpy 2.5.4
- pandas 3.1.0
- scikit-learn 1.9.1
- imbalanced-learn 0.14.2
## Base de datos 
- https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
```
python -m pip install -r requirements.txt
```

## Dataset

Descargar `creditcard.csv` (Credit Card Fraud Detection, Kaggle) y dejarlo en la misma
carpeta que `main.py`. El dataset no se incluye en la entrega.

## Ejecución

```
python main.py
```

El programa imprime la exploración de datos, los hiperparámetros elegidos y las métricas de
validación cruzada de los 8 experimentos, y luego las métricas de prueba con su matriz de
confusión. Por defecto usa una muestra estratificada de 50.000 registros (`N_MUESTRAS` en
`main.py`); con `None` usa el dataset completo.

## Semillas

Todas las semillas son 42: muestreo, partición 70/30, validación cruzada de 5 pliegues
(`shuffle=True`), SMOTE y árboles. En el Random Forest propio, las semillas de cada árbol se
derivan de la semilla general.
