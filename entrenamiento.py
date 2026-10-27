# entrenamiento.py

import numpy as np
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
)

# Modelos
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC, LinearSVC
from mi_random_forest import MiRandomForest  # Importamos tu clase propia

# Usar el Pipeline de imblearn es OBLIGATORIO para evitar data leakage al balancear
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE


def tasa_falsos_negativos(y_true, y_pred):
    """Calcula la proporción de fraudes clasificados como legítimos."""
    return 1.0 - recall_score(y_true, y_pred, pos_label=1, zero_division=0)


def definir_modelos():
    # Espacio de búsqueda acotado de cada clasificador.
    # Las claves 'selector__k' y 'clasificador__...' apuntan a los pasos del pipeline.
    return {
        'KNN': {
            # 'brute' calcula todas las distancias de una vez (muy eficiente con NumPy):
            # mismas predicciones que el algoritmo automático, pero bastante más rápido con 10-15 variables
            'modelo': KNeighborsClassifier(algorithm='brute'),
            'params': {'clasificador__n_neighbors': [3, 5], 'selector__k': [10, 15]}
        },
        'Arbol_Decision': {
            'modelo': DecisionTreeClassifier(random_state=42),
            'params': {'clasificador__max_depth': [5, 10, None], 'selector__k': [10, 15]}
        },
        # SVM lineal con LinearSVC (liblinear): resuelve el mismo problema que SVC(kernel='linear')
        # pero su tiempo crece linealmente con las filas, mientras que SVC crece de forma cuadrática.
        # Con datos balanceados con SMOTE, SVC lineal era el paso más lento de todo el proyecto.
        # El kernel rbf sí necesita SVC.
        'SVM': {
            'modelo': LinearSVC(random_state=42),
            'params': [
                {'clasificador': [LinearSVC(random_state=42)],
                 'clasificador__C': [0.1, 1], 'selector__k': [10]},
                {'clasificador': [SVC(kernel='rbf', random_state=42)],
                 'clasificador__C': [0.1, 1], 'selector__k': [10]},
            ]
        },
        'Mi_Random_Forest': {
            'modelo': MiRandomForest(random_state=42),
            'params': {'clasificador__n_estimators': [11, 21], 'clasificador__max_depth': [5, 10], 'selector__k': [10, 15]}
        }
    }


def elegir_mejor_configuracion(cv_results):
    #Criterio de selección: mayor sensibilidad media (fraude).
    #    En caso de empate, mayor precisión media (fraude).
    sensibilidad = np.round(cv_results['mean_test_sensibilidad'], 6)
    precision = np.round(cv_results['mean_test_precision'], 6)
    posicion = np.arange(len(sensibilidad))
    # Ordena por sensibilidad, luego por precisión y, si siguen empatadas, gana la primera combinación
    orden = np.lexsort((-posicion, precision, sensibilidad))
    return orden[-1]


def ejecutar_experimentos(X_train, y_train):
    # Métrica prioritaria: sensibilidad de la clase 1 (fraudes). La precisión solo sirve para desempatar.
    scorer = {
        'sensibilidad': make_scorer(recall_score, pos_label=1),
        'precision': make_scorer(precision_score, pos_label=1, zero_division=0),
        'especificidad': make_scorer(recall_score, pos_label=0, zero_division=0),
        'f1': make_scorer(f1_score, pos_label=1, zero_division=0),
        'exactitud': make_scorer(accuracy_score),
        'fnr': make_scorer(tasa_falsos_negativos),
    }

    # Validación cruzada estratificada de 5 particiones (las mismas para todos los modelos)
    cv_strat = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    modelos = definir_modelos()
    modelos_entrenados = {}

    # ntrenar los 4 clasificadores con y sin compensación del desbalance (8 experimentos)
    for nombre, config in modelos.items():
        for balancear in [False, True]:
            nombre_exp = f"{nombre}_{'Balanceado' if balancear else 'Original'}"
            print(f"\n--- Iniciando GridSearch para: {nombre_exp} ---")

            # Construir los pasos del pipeline
            pasos = [
                ('scaler', StandardScaler()),
                ('selector', SelectKBest(score_func=f_classif))  # Selección de características
            ]

            if balancear:
                # El SMOTE solo se aplicará al subset de entrenamiento en cada pliegue de la validación cruzada
                pasos.append(('sampler', SMOTE(random_state=42)))

            pasos.append(('clasificador', config['modelo']))

            # Crear el pipeline con imblearn
            pipeline = ImbPipeline(pasos)

            # Ejecutar GridSearchCV
            grid = GridSearchCV(
                pipeline,
                config['params'],
                cv=cv_strat,
                scoring=scorer,
                refit=elegir_mejor_configuracion,  # elige la mejor y la reentrena con todo el train
                n_jobs=-1  # Usa todos los núcleos del procesador
            )

            grid.fit(X_train, y_train)

            #Registrar hiperparámetros y métricas medias/desviaciones en CV
            resultados = grid.cv_results_
            mejor = grid.best_index_
            print(f"Mejores hiperparámetros: {grid.best_params_}")
            print("Métricas en validación cruzada (media +/- desviación estándar):")
            for metrica, etiqueta in [
                ('sensibilidad', 'Sensibilidad'),
                ('especificidad', 'Especificidad'),
                ('precision', 'Precisión'),
                ('f1', 'F1-score'),
                ('exactitud', 'Exactitud'),
                ('fnr', 'Tasa de falsos negativos'),
            ]:
                media = resultados[f'mean_test_{metrica}'][mejor]
                desviacion = resultados[f'std_test_{metrica}'][mejor]
                print(f"  {etiqueta:<27} {media:.4f} +/- {desviacion:.4f}")

            # GridSearchCV vuelve a ajustar el mejor pipeline con todo X_train.
            # Se registran los nombres de las características elegidas en ese ajuste.
            selector = grid.best_estimator_.named_steps['selector']
            caracteristicas = np.asarray(X_train.columns)[selector.get_support()]
            print(f"Características seleccionadas ({len(caracteristicas)}): "
                  f"{', '.join(caracteristicas)}")

            # Guardar el modelo con mejor desempeño
            modelos_entrenados[nombre_exp] = grid.best_estimator_

    return modelos_entrenados


def evaluar_en_prueba(modelo, X_test, y_test):
    # Predicción final sobre el conjunto de prueba
    y_pred = modelo.predict(X_test)

    # Cálculo de matriz de confusión
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()

    # Cálculo de métricas solicitadas (todas salen de la misma matriz de confusión)
    sensibilidad = tp / (tp + fn) if (tp + fn) > 0 else 0
    especificidad = tn / (tn + fp) if (tn + fp) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1 = 2 * precision * sensibilidad / (precision + sensibilidad) if (precision + sensibilidad) > 0 else 0
    exactitud = (tp + tn) / (tp + tn + fp + fn)
    fnr = fn / (tp + fn) if (tp + fn) > 0 else 0  # Tasa de Falsos Negativos

    print(f"Sensibilidad (Recall): {sensibilidad:.4f}")
    print(f"Especificidad:         {especificidad:.4f}")
    print(f"Precisión:             {precision:.4f}")
    print(f"F1-score:              {f1:.4f}")
    print(f"Exactitud (Accuracy):  {exactitud:.4f}")
    print(f"Tasa Falsos Negativos: {fnr:.4f}  <-- ¡Esta es la clave del proyecto!")
    print(f"Matriz de Confusión:\n [[TN: {tn} | FP: {fp}]\n  [FN: {fn}  | TP: {tp}]]\n")
