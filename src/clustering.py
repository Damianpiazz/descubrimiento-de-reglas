# Clustering mediante Self Organizing Maps (SOM)
# Utiliza la libreria minisom para descubrir grupos
# de registros similares sin necesidad de clases predefinidas.

import numpy as np

from minisom import MiniSom


class SOMClustering:
    def __init__(
        self,
        x,
        y,
        input_len,
        sigma=1.0,
        learning_rate=0.5,
        random_seed=42,
    ):
        # x, y: dimensiones del mapa (grilla de neuronas)
        # input_len: cantidad de atributos de entrada
        # sigma: radio de vecindad inicial
        # learning_rate: tasa de aprendizaje inicial

        self.som = MiniSom(
            x,
            y,
            input_len,
            sigma=sigma,
            learning_rate=learning_rate,
            random_seed=random_seed,
        )

        self.x = x
        self.y = y
        self.input_len = input_len

    def train(self, data, num_iterations=100):
        # Inicializa los pesos aleatoriamente y entrena
        # el mapa con los datos proporcionados.

        self.som.random_weights_init(data)

        self.som.train(
            data,
            num_iterations,
        )

    def get_bmu(self, sample):
        # BMU = Best Matching Unit
        # Retorna la posicion (fila, col) de la neurona
        # mas cercana al vector de entrada.

        return self.som.winner(sample)

    def get_groups(self, data):
        # Asigna cada registro a un grupo calculando
        # la BMU y convirtiendo la posicion a un ID unico.

        groups = []

        for sample in data:

            bmu = self.som.winner(sample)

            group_id = bmu[0] * self.y + bmu[1]

            groups.append(group_id)

        return np.array(groups)

    def get_quantization_error(self, data):
        # Error de cuantizacion: promedio de la distancia
        # entre cada registro y su BMU.
        # Valores bajos indican mejor ajuste.

        return self.som.quantization_error(data)

    def get_weights(self):
        # Retorna los pesos de todas las neuronas del mapa.
        # Forma: (x, y, input_len)

        return self.som.get_weights()

    def __repr__(self):

        return (
            f"SOMClustering("
            f"x={self.x}, "
            f"y={self.y}, "
            f"input_len={self.input_len})"
        )
