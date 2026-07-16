# Ponderacion de reglas mediante Redes Bayesianas.
# Utiliza pgmpy para construir la red y realizar inferencia
# probabilistica que determina la influencia de cada condicion.

import pandas as pd

from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.parameter_estimator import DiscreteMLE
from pgmpy.inference import VariableElimination


class RuleWeighting:
    def __init__(self):

        self.model = None
        self.inference = None

    def build_from_data(
        self, data, edges
    ):
        # Construye la red bayesiana directamente desde datos
        # usando estimacion por maxima verosimilitud.
        # data: DataFrame con columnas como variables
        # edges: lista de tuplas (padre, hijo)

        self.model = DiscreteBayesianNetwork(edges)

        # Estima las CPT (Tablas de Probabilidad Condicional)
        # a partir de la frecuencia observada en los datos.

        self.model.fit(
            data,
            estimator=DiscreteMLE(),
        )

        self.inference = VariableElimination(
            self.model
        )

    def build_from_probs(
        self, edges, cpds
    ):
        # Construye la red bayesiana utilizando CPTs
        # definidas manualmente.
        # edges: estructura del grafo
        # cpds: lista de TabularCPD con las probabilidades

        self.model = DiscreteBayesianNetwork(edges)

        for cpd in cpds:

            self.model.add_cpds(cpd)

        # Verifica que las CPT sean consistentes
        # (las probabilidades de cada nodo suman 1).

        self.model.check_model()

        self.inference = VariableElimination(
            self.model
        )

    def query(self, variable, evidence=None):
        # Realiza una consulta de inferencia sobre
        # la variable dada, opcionalmente con evidencia.
        # Retorna la distribucion de probabilidad
        # posterior de la variable.

        if self.inference is None:

            raise ValueError(
                "La red no ha sido construida"
            )

        result = self.inference.query(
            variables=[variable],
            evidence=evidence,
        )

        return result

    def get_weight(self, variable, state, evidence=None):
        # Obtiene la probabilidad de un estado especifico
        # de una variable, dada la evidencia.

        result = self.query(
            variable, evidence
        )

        return float(result.values[0])

    def get_all_weights(self, data, target_var, rule_vars):
        # Calcula el peso (probabilidad) de cada combinacion
        # de variables de regla, dado el target.
        # Util para ponderar que condiciones son mas
        # influyentes sobre cada grupo.

        weights = {}

        for var in rule_vars:

            if var == target_var:

                continue

            states = data[var].unique()

            for state in states:

                evidence = {target_var: state}

                try:

                    result = self.query(
                        var, evidence
                    )

                    weights[(var, state)] = (
                        result.values[0]
                    )

                except Exception:

                    weights[(var, state)] = 0.0

        return weights

    def get_model(self):

        return self.model

    def __repr__(self):

        if self.model is None:

            return "RuleWeighting(sin modelo)"

        n_nodes = len(self.model.nodes())

        n_edges = len(self.model.edges())

        return (
            f"RuleWeighting("
            f"nodes={n_nodes}, "
            f"edges={n_edges})"
        )
