# Extraccion de reglas mediante arboles de decision (TDIDT).
# Utiliza DecisionTreeClassifier de scikit-learn para generar
# reglas de clasificacion que describen cada grupo.

import numpy as np

from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_text


class RuleExtractor:
    def __init__(
        self,
        max_depth=None,
        criterion='entropy',
        random_seed=42,
    ):
        # max_depth: profundidad maxima del arbol (None = sin limite)
        # criterion: metrica de separacion ('entropy' usa ganancia de info)
        # random_seed: semilla para reproducibilidad

        self.classifier = DecisionTreeClassifier(
            max_depth=max_depth,
            criterion=criterion,
            random_state=random_seed,
        )

        self.feature_names = None
        self.class_names = None
        self.is_fitted = False

    def fit(self, X, y, feature_names=None):
        # Entrena el arbol de decision con los datos X
        # y las clases objetivo y.
        # feature_names: nombres de los atributos para
        # generar reglas legibles.

        self.feature_names = feature_names

        self.class_names = sorted(
            set(str(c) for c in y)
        )

        self.classifier.fit(X, y)

        self.is_fitted = True

    def get_rules(self):
        # Exporta el arbol como texto con reglas SI/ENTONCES
        # que son legibles por humanos.

        if not self.is_fitted:

            raise ValueError(
                "El modelo no esta entrenado"
            )

        tree_rules = export_text(
            self.classifier,
            feature_names=self.feature_names,
        )

        return tree_rules

    def get_feature_importance(self):
        # Retorna la importancia de cada atributo
        # segun el arbol entrenado.
        # La suma de todas las importancias es 1.0.

        if not self.is_fitted:

            raise ValueError(
                "El modelo no esta entrenado"
            )

        importances = (
            self.classifier.feature_importances_
        )

        result = {}

        if self.feature_names:

            for i, name in enumerate(
                self.feature_names
            ):

                result[name] = float(
                    importances[i]
                )

        return result

    def predict(self, X):
        # Predice la clase para nuevos registros.

        return self.classifier.predict(X)

    def predict_proba(self, X):
        # Retorna la probabilidad de cada clase
        # para cada registro de entrada.

        return self.classifier.predict_proba(X)

    def get_depth(self):
        # Profundidad del arbol generado.

        return self.classifier.get_depth()

    def get_n_leaves(self):
        # Cantidad de hojas (nodos terminales) del arbol.

        return self.classifier.get_n_leaves()

    def __repr__(self):

        depth = (
            self.get_depth()
            if self.is_fitted
            else 0
        )

        return (
            f"RuleExtractor("
            f"depth={depth}, "
            f"fitted={self.is_fitted})"
        )
