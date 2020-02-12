import numpy as np
import pytest
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

from core.data import Data
from core.evaluation import (LinRegression, LogRegression, XGBoost, normalize,
                             split_train_test)
from core.node import (PrimaryNode, NodeFactory, SecondaryNode)


@pytest.fixture()
def data_setup():
    predictors, response = load_iris(return_X_y=True)
    np.random.seed(1)
    np.random.shuffle(predictors)
    np.random.shuffle(response)
    response = response[:100]
    predictors = predictors[:100]
    train_data_x, test_data_x = split_train_test(predictors)
    train_data_y, test_data_y = split_train_test(response)
    train_data = Data(features=normalize(train_data_x), target=train_data_y, idx=None)
    test_data = Data(features=normalize(test_data_x), target=test_data_y, idx=None)
    data_stream = Data(features=predictors, target=response,
                       idx=np.array([i for i in range(len(response))]))
    return train_data, test_data, data_stream


def get_model_metrics_info(class_name, y_true, y_pred):
    print('\n', f'#test_eval_strategy_{class_name}')
    print(classification_report(y_true, y_pred))
    print('Test model accuracy: ', accuracy_score(y_true, y_pred))


def test_node_log_reg():
    test_node = NodeFactory().log_reg()
    assert test_node.eval_strategy.__class__ == LogRegression


def test_node_lin_log():
    test_node = NodeFactory().lin_reg()
    assert test_node.eval_strategy.__class__ == LinRegression
    assert test_node.__class__ == SecondaryNode


def test_node_xgboost():
    test_node = NodeFactory().default_xgb()
    assert test_node.eval_strategy.__class__ == XGBoost


def test_eval_strategy_logreg(data_setup):
    print_metrics = False
    train_data, test_data, data_stream = data_setup

    test_skl_model = LogisticRegression(C=10., random_state=1, solver='liblinear',
                                        max_iter=10000, verbose=0)
    test_skl_model.fit(train_data.features, train_data.target)
    expected_result = test_skl_model.predict(test_data.features)

    test_model_node = PrimaryNode(nodes_from=None, nodes_to=None, data_stream=data_stream,
                                  eval_strategy=LogRegression(seed=1))
    actual_result = test_model_node.apply()
    if print_metrics:
        get_model_metrics_info(test_skl_model.__class__.__name__, test_data.target,
                               actual_result)

    assert actual_result.all() == expected_result.all()
