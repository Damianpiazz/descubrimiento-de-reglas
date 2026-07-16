# Pipeline completo de Descubrimiento de Reglas:
# 1. SOM -> Clustering (descubrir grupos)
# 2. TDIDT -> Reglas (describir cada grupo)
# 3. Red Bayesiana -> Ponderacion (identificar influencia)

import warnings

warnings.filterwarnings(
    'ignore',
    category=FutureWarning,
    message='.*pgmpy.estimators.StructureScore.*',
)

import numpy as np
from rich.console import Console
from rich.table import Table

from src.clustering import SOMClustering
from src.rules import RuleExtractor
from src.bayesian import RuleWeighting
from src.utils import (
    load_csv,
    get_numeric_matrix,
    add_group_column,
    format_number,
)
from src.visualization import (
    plot_som_u_matrix,
    plot_som_hits,
    plot_clusters_scatter,
    plot_feature_importance,
    plot_group_distribution,
    plot_bayesian_network,
    plot_feature_distributions,
)


console = Console()


# Configuracion del dataset.
FEATURES = ['Age', 'Annual_Income', 'Spending_Score']
SOM_GRID = (4, 4)
BAYESIAN_EDGES = [
    ('Age', 'Grupo'),
    ('Annual_Income', 'Grupo'),
    ('Spending_Score', 'Grupo'),
]


def header(text):
    # Imprime un encabezado formateado en consola.

    console.print(f"\n[bold]{text}[/bold]")
    console.print("-" * len(text))


def run_pipeline():
    # Ejecuta el pipeline completo de 3 etapas:
    # SOM -> TDIDT -> Red Bayesiana

    header('MALL CUSTOMER SEGMENTATION')

    # --- Carga de datos ---
    console.print(
        "\nCargando datos..."
    )

    data = load_csv('data/mall_customers.csv')

    console.print(
        f"Registros: {len(data)}"
    )

    console.print(
        f"Atributos: {FEATURES}"
    )

    # Convierte los atributos a matriz numerica
    # para que el SOM pueda procesarlos.

    X = get_numeric_matrix(data, FEATURES)

    # --- Etapa 1: SOM (Clustering) ---
    # El SOM agrupa registros similares sin usar clases.
    # Cada neurona representa un prototipo de grupo.

    console.print(
        "\n--- Etapa 1: SOM (Clustering) ---"
    )

    som_x, som_y = SOM_GRID

    som = SOMClustering(
        x=som_x,
        y=som_y,
        input_len=len(FEATURES),
    )

    som.train(X, num_iterations=100)

    # Asigna cada registro a su neurona ganadora (BMU).

    groups = som.get_groups(X)

    qe = som.get_quantization_error(X)

    console.print(
        f"Error de cuantizacion: "
        f"{format_number(qe)}"
    )

    # Agrega el grupo como columna al DataFrame
    # para usarlo como clase objetivo en la etapa 2.

    data = add_group_column(data, groups)

    console.print(
        f"Grupos encontrados: "
        f"{len(set(groups))}"
    )

    # Genera visualizaciones del SOM.

    console.print(
        "\nGenerando graficos del SOM..."
    )

    plot_som_u_matrix(som.som)

    plot_som_hits(som, X)

    plot_clusters_scatter(data, FEATURES)

    plot_group_distribution(data)

    plot_feature_distributions(data, FEATURES)

    # --- Etapa 2: TDIDT (Reglas) ---
    # Entrena un arbol de decision con todos los grupos
    # como clase objetivo para generar reglas de pertenencia.

    console.print(
        "\n--- Etapa 2: TDIDT (Reglas) ---"
    )

    X_all = get_numeric_matrix(data, FEATURES)

    y_all = data['Grupo'].values

    extractor = RuleExtractor()

    extractor.fit(
        X_all,
        y_all,
        feature_names=FEATURES,
    )

    rules = extractor.get_rules()

    console.print(
        "\n[bold]Reglas de pertenencia:[/bold]"
    )

    console.print(rules)

    # Grafico de importancia de atributos.

    console.print(
        "\nGenerando grafico de importancia..."
    )

    importances = (
        extractor.get_feature_importance()
    )

    plot_feature_importance(importances)

    # --- Etapa 3: Red Bayesiana ---
    # Construye una red bayesiana para determinar
    # que variables tienen mayor influencia sobre
    # la pertenencia a cada grupo.

    console.print(
        "\n--- Etapa 3: Red Bayesiana ---"
    )

    bayesian_data = data.copy()

    # pgmpy requiere que todas las variables sean categoricas
    # (strings), por eso convertimos los numericos.

    for col in FEATURES:

        bayesian_data[col] = (
            bayesian_data[col].astype(str)
        )

    bayesian_data['Grupo'] = (
        bayesian_data['Grupo'].astype(str)
    )

    weighting = RuleWeighting()

    edges_str = [
        (str(e[0]), str(e[1]))
        for e in BAYESIAN_EDGES
    ]

    # Construye la red desde los datos y estima
    # las Tablas de Probabilidad Condicional (CPT).

    weighting.build_from_data(
        bayesian_data, edges_str
    )

    console.print(
        weighting
    )

    # Consulta la probabilidad de cada grupo
    # para identificar cuales son mas frecuentes.

    console.print(
        "\nPonderacion de reglas:"
    )

    result = weighting.query('Grupo')

    table = Table(
        show_header=True,
    )

    table.add_column("GRUPO")
    table.add_column(
        "PROBABILIDAD", justify="right"
    )

    for state, prob in zip(
        result.state_names['Grupo'],
        result.values,
    ):

        table.add_row(
            state,
            format_number(prob),
        )

    console.print(table)

    # Grafo de la red bayesiana.

    console.print(
        "\nGenerando grafo bayesiano..."
    )

    plot_bayesian_network(edges_str)

    return data, weighting


if __name__ == '__main__':

    run_pipeline()
