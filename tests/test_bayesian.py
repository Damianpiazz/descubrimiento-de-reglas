import pandas as pd

from pgmpy.factors.discrete import TabularCPD

from src.bayesian import RuleWeighting


def test_rule_weighting_creation():

    weighting = RuleWeighting()

    assert weighting.model is None
    assert weighting.inference is None


def test_rule_weighting_build_from_probs():

    edges = [
        ('Edad', 'Grupo'),
        ('Ingreso', 'Grupo'),
    ]

    cpd_edad = TabularCPD(
        variable='Edad',
        variable_card=2,
        values=[[0.5], [0.5]],
    )

    cpd_ingreso = TabularCPD(
        variable='Ingreso',
        variable_card=2,
        values=[[0.6], [0.4]],
    )

    cpd_grupo = TabularCPD(
        variable='Grupo',
        variable_card=2,
        values=[
            [0.9, 0.6, 0.7, 0.1],
            [0.1, 0.4, 0.3, 0.9],
        ],
        evidence=['Edad', 'Ingreso'],
        evidence_card=[2, 2],
    )

    weighting = RuleWeighting()

    weighting.build_from_probs(
        edges,
        [cpd_edad, cpd_ingreso, cpd_grupo],
    )

    assert weighting.model is not None
    assert weighting.inference is not None


def test_rule_weighting_query():

    edges = [
        ('Edad', 'Grupo'),
        ('Ingreso', 'Grupo'),
    ]

    cpd_edad = TabularCPD(
        variable='Edad',
        variable_card=2,
        values=[[0.5], [0.5]],
    )

    cpd_ingreso = TabularCPD(
        variable='Ingreso',
        variable_card=2,
        values=[[0.6], [0.4]],
    )

    cpd_grupo = TabularCPD(
        variable='Grupo',
        variable_card=2,
        values=[
            [0.9, 0.6, 0.7, 0.1],
            [0.1, 0.4, 0.3, 0.9],
        ],
        evidence=['Edad', 'Ingreso'],
        evidence_card=[2, 2],
    )

    weighting = RuleWeighting()

    weighting.build_from_probs(
        edges,
        [cpd_edad, cpd_ingreso, cpd_grupo],
    )

    result = weighting.query('Grupo')

    total = sum(result.values)

    assert abs(total - 1.0) < 1e-9


def test_rule_weighting_query_with_evidence():

    edges = [
        ('Edad', 'Grupo'),
        ('Ingreso', 'Grupo'),
    ]

    cpd_edad = TabularCPD(
        variable='Edad',
        variable_card=2,
        values=[[0.5], [0.5]],
    )

    cpd_ingreso = TabularCPD(
        variable='Ingreso',
        variable_card=2,
        values=[[0.6], [0.4]],
    )

    cpd_grupo = TabularCPD(
        variable='Grupo',
        variable_card=2,
        values=[
            [0.9, 0.6, 0.7, 0.1],
            [0.1, 0.4, 0.3, 0.9],
        ],
        evidence=['Edad', 'Ingreso'],
        evidence_card=[2, 2],
    )

    weighting = RuleWeighting()

    weighting.build_from_probs(
        edges,
        [cpd_edad, cpd_ingreso, cpd_grupo],
    )

    result = weighting.query(
        'Edad',
        evidence={'Grupo': 0},
    )

    total = sum(result.values)

    assert abs(total - 1.0) < 1e-9


def test_rule_weighting_not_built():

    weighting = RuleWeighting()

    try:

        weighting.query('Grupo')

        assert False

    except ValueError:

        assert True


def test_rule_weighting_repr():

    weighting = RuleWeighting()

    repr_str = repr(weighting)

    assert 'RuleWeighting' in repr_str
    assert 'sin modelo' in repr_str


def test_rule_weighting_repr_with_model():

    edges = [
        ('Edad', 'Grupo'),
        ('Ingreso', 'Grupo'),
    ]

    cpd_edad = TabularCPD(
        variable='Edad',
        variable_card=2,
        values=[[0.5], [0.5]],
    )

    cpd_ingreso = TabularCPD(
        variable='Ingreso',
        variable_card=2,
        values=[[0.6], [0.4]],
    )

    cpd_grupo = TabularCPD(
        variable='Grupo',
        variable_card=2,
        values=[
            [0.9, 0.6, 0.7, 0.1],
            [0.1, 0.4, 0.3, 0.9],
        ],
        evidence=['Edad', 'Ingreso'],
        evidence_card=[2, 2],
    )

    weighting = RuleWeighting()

    weighting.build_from_probs(
        edges,
        [cpd_edad, cpd_ingreso, cpd_grupo],
    )

    repr_str = repr(weighting)

    assert 'nodes=3' in repr_str
    assert 'edges=2' in repr_str
