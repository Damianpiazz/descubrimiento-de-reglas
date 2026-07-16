# Funciones utilitarias para carga de datos,
# procesamiento y formateo de resultados.

import csv

import numpy as np
import pandas as pd


def load_csv(path):
    # Carga un archivo CSV usando pandas.
    # Retorna un DataFrame.

    data = pd.read_csv(path)

    return data


def load_csv_raw(path):
    # Carga un archivo CSV como lista de diccionarios.
    # Util para procesamiento manual fila por fila.

    data = []

    with open(
        path,
        newline='',
        encoding='utf-8',
    ) as file:

        reader = csv.DictReader(file)

        for row in reader:

            data.append(row)

    return data


def get_numeric_matrix(data, columns):
    # Extrae las columnas especificadas como matriz
    # numerica de tipo float.
    # Necesario para SOM y arboles de decision.

    matrix = data[columns].values.astype(float)

    return matrix


def get_unique_values(data, column):
    # Retorna los valores unicos de una columna
    # ordenados de menor a mayor.

    return sorted(data[column].unique())


def add_group_column(data, groups):
    # Agrega una columna 'Grupo' al DataFrame
    # con los IDs de grupo asignados por el SOM.

    result = data.copy()

    result['Grupo'] = groups

    return result


def split_by_group(data, group_col='Grupo'):
    # Divide el DataFrame en subconjuntos
    # separados por grupo.
    # Retorna un diccionario {group_id: DataFrame}.

    groups = {}

    for group_id in data[group_col].unique():

        mask = data[group_col] == group_id

        groups[group_id] = data[mask].copy()

    return groups


def format_percentage(value):
    # Formatea un float como porcentaje con 2 decimales.
    # Ejemplo: 0.857 -> '85.70%'

    return f"{value:.2%}"


def format_number(value, decimals=3):
    # Formatea un float con la cantidad de decimales indicada.

    return f"{value:.{decimals}f}"


def print_separator(char='=', length=60):
    # Imprime una linea separadora en consola.

    print(char * length)


def print_header(text):
    # Imprime un encabezado con separadores.

    print_separator()

    print(f"  {text}")

    print_separator()
