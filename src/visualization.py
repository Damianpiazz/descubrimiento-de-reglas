# Visualizaciones del pipeline de Descubrimiento de Reglas.
# Utiliza seaborn y matplotlib para generar graficos
# que facilitan la interpretacion de resultados.

import os

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np
import networkx as nx


def ensure_plots_dir():
    # Crea la carpeta plots/ si no existe.

    os.makedirs(
        'plots',
        exist_ok=True,
    )


def plot_som_u_matrix(som):
    # U-Matrix: muestra las distancias entre neuronas vecinas.
    # Valores altos (oscuros) indican fronteras entre clusters.
    # Valores bajos (claros) indican neuronas dentro del mismo cluster.

    ensure_plots_dir()

    weights = som.get_weights()

    x, y, n_features = weights.shape

    u_matrix = np.zeros((x, y))

    for i in range(x):

        for j in range(y):

            neighbors = []

            if i > 0:
                neighbors.append(
                    weights[i-1, j]
                )
            if i < x - 1:
                neighbors.append(
                    weights[i+1, j]
                )
            if j > 0:
                neighbors.append(
                    weights[i, j-1]
                )
            if j < y - 1:
                neighbors.append(
                    weights[i, j+1]
                )

            dists = [
                np.linalg.norm(
                    weights[i, j] - n
                )
                for n in neighbors
            ]

            u_matrix[i, j] = np.mean(dists)

    plt.figure(
        figsize=(8, 6),
    )

    sns.heatmap(
        u_matrix,
        annot=True,
        fmt='.2f',
        cmap='YlOrRd',
        linewidths=0.5,
        cbar_kws={
            'label': 'Distancia promedio',
        },
    )

    plt.title(
        'U-Matrix del SOM',
        fontsize=14,
        fontweight='bold',
    )

    plt.xlabel('Columna')
    plt.ylabel('Fila')

    plt.tight_layout()

    plt.savefig(
        'plots/som_u_matrix.png',
        dpi=150,
    )

    plt.close()


def plot_som_hits(som, data):
    # Hits: cantidad de registros que caen en cada neurona.
    # Muestra la distribucion de registros sobre el mapa.
    # Neuronas vacias indican que esa zona del espacio
    # no tiene registros representativos.

    ensure_plots_dir()

    hits = np.zeros((som.x, som.y))

    for sample in data:

        bmu = som.get_bmu(sample)

        hits[bmu[0], bmu[1]] += 1

    plt.figure(
        figsize=(8, 6),
    )

    sns.heatmap(
        hits,
        annot=True,
        fmt='.0f',
        cmap='Blues',
        linewidths=0.5,
        cbar_kws={
            'label': 'Cantidad de registros',
        },
    )

    plt.title(
        'Distribucion de Hits por Neurona',
        fontsize=14,
        fontweight='bold',
    )

    plt.xlabel('Columna')
    plt.ylabel('Fila')

    plt.tight_layout()

    plt.savefig(
        'plots/som_hits.png',
        dpi=150,
    )

    plt.close()


def plot_clusters_scatter(data, features):
    # Scatter plot de los 2 ultimos atributos
    # coloreado por grupo.
    # Permite ver la separacion visual de los clusters
    # en el espacio de atributos.

    ensure_plots_dir()

    if len(features) < 2:

        return

    fig, ax = plt.subplots(
        figsize=(10, 7),
    )

    sns.scatterplot(
        data=data,
        x=features[-2],
        y=features[-1],
        hue='Grupo',
        palette='Set2',
        s=60,
        edgecolor='black',
        linewidth=0.5,
        alpha=0.8,
        ax=ax,
    )

    plt.title(
        f'Clusters: {features[-2]} vs {features[-1]}',
        fontsize=14,
        fontweight='bold',
    )

    plt.xlabel(
        features[-2],
        fontsize=11,
    )

    plt.ylabel(
        features[-1],
        fontsize=11,
    )

    plt.legend(
        title='Grupo',
        bbox_to_anchor=(1.05, 1),
        loc='upper left',
    )

    plt.tight_layout()

    plt.savefig(
        'plots/clusters_scatter.png',
        dpi=150,
        bbox_inches='tight',
    )

    plt.close()


def plot_feature_importance(importances):
    # Barras de importancia de atributos del arbol.
    # Indica que variables son mas determinantes
    # para separar los grupos.

    ensure_plots_dir()

    if not importances:

        return

    features = list(importances.keys())

    values = list(importances.values())

    fig, ax = plt.subplots(
        figsize=(8, 5),
    )

    bars = ax.barh(
        features,
        values,
        color='#4ECDC4',
        edgecolor='black',
        linewidth=0.5,
    )

    ax.bar_label(
        bars,
        fmt='%.3f',
        padding=3,
        fontsize=10,
    )

    ax.set_xlabel(
        'Importancia',
        fontsize=11,
    )

    ax.set_title(
        'Importancia de Atributos (TDIDT)',
        fontsize=14,
        fontweight='bold',
    )

    ax.set_xlim(0, max(values) * 1.2)

    plt.tight_layout()

    plt.savefig(
        'plots/feature_importance.png',
        dpi=150,
    )

    plt.close()


def plot_group_distribution(data):
    # Distribucion de registros por grupo.
    # Muestra si los clusters estan balanceados
    # o si hay grupos con pocos registros.

    ensure_plots_dir()

    fig, ax = plt.subplots(
        figsize=(10, 5),
    )

    sns.countplot(
        data=data,
        x='Grupo',
        hue='Grupo',
        palette='Set2',
        edgecolor='black',
        linewidth=0.5,
        legend=False,
        ax=ax,
    )

    ax.set_xlabel(
        'Grupo',
        fontsize=11,
    )

    ax.set_ylabel(
        'Cantidad de Registros',
        fontsize=11,
    )

    ax.set_title(
        'Distribucion de Registros por Grupo',
        fontsize=14,
        fontweight='bold',
    )

    plt.tight_layout()

    plt.savefig(
        'plots/group_distribution.png',
        dpi=150,
    )

    plt.close()


def plot_bayesian_network(edges):
    # Grafo dirigido de la red bayesiana.
    # Muestra las relaciones causales entre variables.

    ensure_plots_dir()

    G = nx.DiGraph()

    for parent, child in edges:

        G.add_edge(parent, child)

    fig, ax = plt.subplots(
        figsize=(8, 6),
    )

    pos = nx.spring_layout(
        G,
        seed=42,
        k=2.0,
    )

    nx.draw_networkx_edges(
        G,
        pos,
        edge_color='#555555',
        width=2.0,
        arrows=True,
        arrowsize=20,
        arrowstyle='-|>',
        connectionstyle='arc3,rad=0.1',
        ax=ax,
    )

    nx.draw_networkx_nodes(
        G,
        pos,
        node_color='#4ECDC4',
        node_size=1500,
        edgecolors='black',
        linewidths=2.0,
        ax=ax,
    )

    nx.draw_networkx_labels(
        G,
        pos,
        font_size=10,
        font_weight='bold',
        font_color='white',
        ax=ax,
    )

    ax.set_title(
        'Red Bayesiana - Estructura del Grafo',
        fontsize=14,
        fontweight='bold',
    )

    ax.set_axis_off()

    plt.tight_layout()

    plt.savefig(
        'plots/bayesian_network.png',
        dpi=150,
    )

    plt.close()


def plot_feature_distributions(data, features):
    # Histogramas de distribucion de cada atributo.
    # Permite ver la forma de los datos antes del clustering.

    ensure_plots_dir()

    n = len(features)

    fig, axes = plt.subplots(
        1, n,
        figsize=(n * 5, 4),
    )

    if n == 1:

        axes = [axes]

    for i, feat in enumerate(features):

        sns.histplot(
            data=data,
            x=feat,
            kde=True,
            color='#FF6B6B',
            edgecolor='black',
            linewidth=0.5,
            ax=axes[i],
        )

        axes[i].set_title(
            f'Distribucion de {feat}',
            fontsize=12,
            fontweight='bold',
        )

        axes[i].set_xlabel(feat)
        axes[i].set_ylabel('Frecuencia')

    plt.tight_layout()

    plt.savefig(
        'plots/feature_distributions.png',
        dpi=150,
    )

    plt.close()
