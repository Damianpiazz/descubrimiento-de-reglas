import numpy as np

from src.clustering import SOMClustering


def test_som_creation():

    som = SOMClustering(
        x=5,
        y=5,
        input_len=3,
    )

    assert som.x == 5
    assert som.y == 5
    assert som.input_len == 3


def test_som_train():

    data = np.random.rand(100, 3)

    som = SOMClustering(
        x=3,
        y=3,
        input_len=3,
    )

    som.train(data, num_iterations=50)

    weights = som.get_weights()

    assert weights.shape == (3, 3, 3)


def test_som_get_groups():

    data = np.random.rand(50, 3)

    som = SOMClustering(
        x=3,
        y=3,
        input_len=3,
    )

    som.train(data, num_iterations=50)

    groups = som.get_groups(data)

    assert len(groups) == 50

    assert all(g >= 0 for g in groups)


def test_som_get_bmu():

    data = np.random.rand(10, 3)

    som = SOMClustering(
        x=3,
        y=3,
        input_len=3,
    )

    som.train(data, num_iterations=50)

    bmu = som.get_bmu(data[0])

    assert isinstance(bmu, tuple)
    assert len(bmu) == 2


def test_som_quantization_error():

    data = np.random.rand(50, 3)

    som = SOMClustering(
        x=3,
        y=3,
        input_len=3,
    )

    som.train(data, num_iterations=50)

    qe = som.get_quantization_error(data)

    assert qe >= 0.0


def test_som_repr():

    som = SOMClustering(
        x=5,
        y=5,
        input_len=3,
    )

    repr_str = repr(som)

    assert 'SOMClustering' in repr_str
    assert 'x=5' in repr_str


def test_som_different_sizes():

    data = np.random.rand(100, 5)

    som = SOMClustering(
        x=10,
        y=10,
        input_len=5,
    )

    som.train(data, num_iterations=100)

    groups = som.get_groups(data)

    assert len(groups) == 100
