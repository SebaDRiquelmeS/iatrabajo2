# Detección de fraude con aprendizaje supervisado

Proyecto de INFO1185 para comparar k-NN, Árbol de Decisión, SVM y un Random
Forest implementado por el equipo, con y sin SMOTE. La prioridad de selección
es la sensibilidad de la clase fraude; la precisión se usa para desempatar.

## Requisitos y versiones

Versiones con las que se ejecutó el proyecto:

- Python 3.15.0
- numpy 2.5.4
- pandas 3.1.0
- scikit-learn 1.9.1
- imbalanced-learn 0.14.2

Instalar dependencias desde esta carpeta:

```powershell
python -m pip install -r requirements.txt
```

## Dataset y ejecución

Descargar `creditcard.csv` del conjunto **Credit Card Fraud Detection** de
Kaggle y dejarlo en esta misma carpeta, junto a `main.py`. No se debe incluir
el dataset en la entrega.

Ejecutar desde PowerShell:

```powershell
cd "C:\Users\sebapc\Desktop\Nueva carpeta"
python main.py
```

Si se utiliza un entorno virtual y PowerShell impide activar scripts, no es
necesario activarlo: se puede invocar directamente su intérprete:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe main.py
```

`main.py` usa una muestra estratificada de 50.000 registros (`N_MUESTRAS`),
elimina duplicados y luego separa entrenamiento/prueba en proporción 70/30.
Para usar todo el dataset, cambiar `N_MUESTRAS` a `None` en `main.py`.
En la ejecución realizada se encontraron 1.081 filas duplicadas y ningún valor
faltante. Se eliminaron las filas idénticas para evitar contar repetidamente
transacciones iguales; la exploración y esta decisión también se imprimen en
consola.

## Procedimiento y reproducibilidad

- Semilla general de muestreo y partición: `42`.
- Validación: `StratifiedKFold` de 5 particiones, con `shuffle=True` y semilla `42`.
- El escalado y `SelectKBest(f_classif)` viven dentro del pipeline y se ajustan
  solo con el entrenamiento de cada pliegue.
- Se usa `StandardScaler` porque k-NN y SVM son sensibles a la escala de las
  variables; en árboles el escalado no suele ser necesario, pero mantenerlo en
  el pipeline permite comparar bajo el mismo preprocesamiento. `f_classif` es
  un filtro estadístico univariado que puntúa la relación de cada variable con
  `Class`; Grid Search compara `k=10` y `k=15` (según el clasificador). Es una
  selección simple y reproducible, aunque no modela interacciones entre variables.
- SMOTE está dentro del pipeline: solo remuestrea el entrenamiento de cada
  pliegue. El conjunto de validación y el de prueba conservan su distribución.
- Parámetros de SMOTE: `sampling_strategy='auto'`, `k_neighbors=5` (valores
  predeterminados de imbalanced-learn) y `random_state=42`. En esta ejecución,
  el entrenamiento externo tuvo 34.942 ejemplos legítimos y 58 fraudes antes
  de SMOTE; al reajustar el pipeline con todo ese entrenamiento, quedan
  34.942 ejemplos de cada clase. En cada pliegue de CV los conteos cambian
  ligeramente, pero SMOTE balancea solo el subconjunto de entrenamiento de ese
  pliegue.
- El mejor pipeline se vuelve a ajustar con todo el conjunto de entrenamiento.
  El programa imprime sus características seleccionadas y las seis métricas
  medias +/- desviación estándar en CV. Las métricas de prueba son una única
  evaluación final y no se usan para seleccionar modelos ni hiperparámetros.
- El Random Forest propio genera bootstrap del tamaño del entrenamiento,
  entrena árboles con semillas distintas y usa `max_features='sqrt'` en cada
  división; combina las predicciones por mayoría.

## Guía breve para analizar los resultados

En la ejecución con 50.000 registros, la prevalencia observada fue 0,166 % y
el conjunto de prueba tuvo 25 fraudes. Un clasificador que siempre predijera
la clase legítima tendría aproximadamente 99,834 % de exactitud, pero 0 % de
sensibilidad y 100 % de tasa de falsos negativos. Por eso la exactitud sola
no es adecuada para comparar estos modelos.

En los resultados de prueba obtenidos, KNN sin balancear detectó 18 de 25
fraudes con 5 falsas alarmas. El árbol balanceado detectó 21 de 25, pero tuvo
547 falsas alarmas. Esto ilustra el intercambio entre detectar más fraudes y
generar más alertas incorrectas. Predecir todo como fraude lograría 100 % de
sensibilidad, pero marcaría casi todas las transacciones legítimas y tendría
una precisión cercana a la prevalencia.

Estas métricas se basan en solo 25 fraudes en prueba, así que la sensibilidad
puede cambiar bastante con otra partición. Presentar por separado las métricas
de validación cruzada y las del conjunto de prueba, y no ajustar el modelo a
partir de los resultados de prueba.
