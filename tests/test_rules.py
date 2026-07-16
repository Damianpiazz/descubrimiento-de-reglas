import numpy as np

from src.rules import RuleExtractor


def test_rule_extractor_creation():

    extractor = RuleExtractor()

    assert extractor.is_fitted is False
    assert extractor.classifier is not None


def test_rule_extractor_fit():

    X = np.array([
        [1, 2],
        [2, 3],
        [3, 4],
        [4, 5],
        [5, 6],
    ])

    y = np.array(['A', 'A', 'B', 'B', 'B'])

    extractor = RuleExtractor()

    extractor.fit(
        X, y,
        feature_names=['edad', 'ingreso'],
    )

    assert extractor.is_fitted is True
    assert 'A' in extractor.class_names
    assert 'B' in extractor.class_names


def test_rule_extractor_get_rules():

    X = np.array([
        [1, 2],
        [2, 3],
        [3, 4],
        [4, 5],
        [5, 6],
        [6, 7],
    ])

    y = np.array(['A', 'A', 'A', 'B', 'B', 'B'])

    extractor = RuleExtractor()

    extractor.fit(
        X, y,
        feature_names=['edad', 'ingreso'],
    )

    rules = extractor.get_rules()

    assert isinstance(rules, str)
    assert len(rules) > 0


def test_rule_extractor_feature_importance():

    X = np.array([
        [1, 2],
        [2, 3],
        [3, 4],
        [4, 5],
        [5, 6],
    ])

    y = np.array(['A', 'A', 'B', 'B', 'B'])

    extractor = RuleExtractor()

    extractor.fit(
        X, y,
        feature_names=['edad', 'ingreso'],
    )

    importance = extractor.get_feature_importance()

    assert 'edad' in importance
    assert 'ingreso' in importance

    total = sum(importance.values())

    assert abs(total - 1.0) < 1e-9


def test_rule_extractor_predict():

    X_train = np.array([
        [1, 2],
        [2, 3],
        [3, 4],
        [4, 5],
        [5, 6],
    ])

    y_train = np.array(['A', 'A', 'B', 'B', 'B'])

    extractor = RuleExtractor()

    extractor.fit(X_train, y_train)

    X_test = np.array([[2, 3], [5, 6]])

    predictions = extractor.predict(X_test)

    assert len(predictions) == 2
    assert predictions[0] == 'A'
    assert predictions[1] == 'B'


def test_rule_extractor_predict_proba():

    X_train = np.array([
        [1, 2],
        [2, 3],
        [3, 4],
        [4, 5],
        [5, 6],
    ])

    y_train = np.array(['A', 'A', 'B', 'B', 'B'])

    extractor = RuleExtractor()

    extractor.fit(X_train, y_train)

    X_test = np.array([[3, 4]])

    proba = extractor.predict_proba(X_test)

    assert proba.shape[1] == 2

    assert abs(sum(proba[0]) - 1.0) < 1e-9


def test_rule_extractor_not_fitted():

    extractor = RuleExtractor()

    try:

        extractor.get_rules()

        assert False

    except ValueError:

        assert True


def test_rule_extractor_depth():

    X = np.array([
        [1, 2],
        [2, 3],
        [3, 4],
        [4, 5],
        [5, 6],
    ])

    y = np.array(['A', 'A', 'B', 'B', 'B'])

    extractor = RuleExtractor(max_depth=2)

    extractor.fit(X, y)

    depth = extractor.get_depth()

    assert depth <= 2


def test_rule_extractor_repr():

    extractor = RuleExtractor()

    repr_str = repr(extractor)

    assert 'RuleExtractor' in repr_str
    assert 'fitted=False' in repr_str
