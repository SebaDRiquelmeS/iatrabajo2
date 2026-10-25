# mi_random_forest.py

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.tree import DecisionTreeClassifier


class MiRandomForest(ClassifierMixin, BaseEstimator):
    """Random Forest propio: bagging de árboles de decisión con max_features.

    Clase positiva = 1 (fraude). Etiquetas esperadas: 0 y 1.
    """

    def __init__(self, n_estimators=11, max_depth=None, max_features='sqrt', random_state=42):
        # 5. Parámetros configurables solicitados en la pauta
        # (en __init__ solo se guardan parámetros, así scikit-learn puede clonar el modelo)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.random_state = random_state

    def _muestra_bootstrap(self, X, y, rng):
        # 1. Muestreo aleatorio CON reemplazo y del mismo tamaño que el conjunto original.
        #    Por eso una muestra puede repetir registros y dejar otros fuera.
        n_muestras = X.shape[0]
        indices = rng.randint(0, n_muestras, size=n_muestras)
        return X[indices], y[indices]

    def _entrenar_arbol(self, X, y, rng, semilla_arbol):
        # 2. y 3. Árbol independiente; max_features elige un subconjunto aleatorio
        #    de características en CADA división (lo que distingue RF de un bagging simple)
        X_boot, y_boot = self._muestra_bootstrap(X, y, rng)
        arbol = DecisionTreeClassifier(
            max_depth=self.max_depth,
            max_features=self.max_features,
            random_state=semilla_arbol,  # Semilla distinta por árbol
        )
        return arbol.fit(X_boot, y_boot)

    def fit(self, X, y):
        # float32 es el tipo con el que los árboles trabajan por dentro:
        # convertir una sola vez evita repetir la conversión en cada árbol
        X_np = np.asarray(X, dtype=np.float32)
        y_np = np.asarray(y)
        self.classes_ = np.unique(y_np)

        # 6. Un generador local (no toca la semilla global de numpy) y semillas
        #    distintas derivadas de la semilla general.
        rng = np.random.RandomState(self.random_state)
        # El espacio de semillas es deliberadamente amplio y sin reemplazo:
        # así son distintas y el muestreo sigue siendo ligero para n_estimators pequeño.
        semillas_arboles = rng.choice(100000, size=self.n_estimators, replace=False)

        self.trees_ = [self._entrenar_arbol(X_np, y_np, rng, semilla) for semilla in semillas_arboles]
        return self

    def predict(self, X):
        X_np = np.asarray(X, dtype=np.float32)

        # Predicción de cada árbol: matriz de forma (n_estimators, n_muestras) con 0/1
        predicciones = np.array([arbol.predict(X_np) for arbol in self.trees_])

        # 4. Votación mayoritaria: como las clases son 0/1, basta sumar los votos por fraude.
        #    Regla de empate (solo posible con n_estimators par): se clasifica como fraude,
        #    coherente con la prioridad de minimizar falsos negativos.
        votos_fraude = predicciones.sum(axis=0)
        return (2 * votos_fraude >= len(self.trees_)).astype(int)
