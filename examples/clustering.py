import random
from copy import deepcopy
from datetime import timedelta

import numpy as np
from sklearn.datasets import load_iris, make_blobs
from sklearn.metrics import adjusted_rand_score

from fedot.core.chains.chain import Chain
from fedot.core.chains.node import PrimaryNode
from fedot.core.composer.gp_composer.gp_composer import \
    GPComposerBuilder, GPComposerRequirements
from fedot.core.composer.visualisation import ComposerVisualiser
from fedot.core.data.data import InputData
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.model_types_repository import ModelTypesRepository
from fedot.core.repository.quality_metrics_repository import ClusteringMetricsEnum, MetricsRepository
from fedot.core.repository.tasks import Task, TaskTypesEnum

random.seed(1)
np.random.seed(1)


def create_simple_clustering_example(data_size=200):
    num_features = 2
    num_clusters = 3

    predictors, response = make_blobs(data_size, num_features, centers=num_clusters, random_state=1)

    return InputData(features=predictors, target=response, idx=np.arange(0, len(predictors)),
                     task=Task(TaskTypesEnum.clustering),
                     data_type=DataTypesEnum.table)


def create_iris_clustering_example(data_size=-1):
    predictors, response = load_iris(return_X_y=True)
    if data_size < 0:
        data_size = predictors.shape[0]
    return InputData(features=predictors[:data_size, :], target=response[:data_size],
                     idx=np.arange(0, data_size),
                     task=Task(TaskTypesEnum.clustering),
                     data_type=DataTypesEnum.table)


def get_atomic_clustering_model(train_data: InputData):
    chain = Chain(PrimaryNode('kmeans'))
    chain.fit(input_data=train_data)

    return chain


def get_composite_clustering_model(train_data: InputData,
                                   max_time: int = 60):
    task = Task(task_type=TaskTypesEnum.clustering)
    dataset_to_compose = train_data
    pop_size = 20
    if max_time <= 1:
        pop_size = 2

    models_repo = ModelTypesRepository()
    available_model_types, _ = models_repo.suitable_model(task_type=task.task_type,
                                                          forbidden_tags=['ensembler'])
    secondary_model_types, _ = models_repo.models_with_tag(['ensembler'])

    metric_function = MetricsRepository(). \
        metric_by_id(ClusteringMetricsEnum.silhouette)

    composer_requirements = GPComposerRequirements(
        primary=available_model_types, secondary=secondary_model_types,
        max_lead_time=timedelta(seconds=max_time), min_arity=3, max_arity=4, pop_size=pop_size)

    # run the search of best suitable model
    chain_evo_composed = GPComposerBuilder(task=task). \
        with_requirements(composer_requirements). \
        with_metrics(metric_function).build().compose_chain(data=dataset_to_compose)
    chain_evo_composed.fit(input_data=dataset_to_compose)

    return chain_evo_composed


def validate_model_quality(model: Chain, dataset_to_validate: InputData):
    predicted_labels = model.predict(dataset_to_validate).predict

    prediction_valid = round(adjusted_rand_score(labels_true=dataset_to_validate.target,
                                                 labels_pred=predicted_labels), 6)

    return prediction_valid, predicted_labels


def run_clustering_example(is_fast=False):
    opt_time_sec = 60
    tune_iters = 30
    simple_data_size = 200
    iris_size = -1

    if is_fast:
        opt_time_sec = 1
        tune_iters = 1
        simple_data_size = 10
        iris_size = 10

    # ensemble clustering example
    data = create_iris_clustering_example(iris_size)
    data_train = deepcopy(data)
    data_train.target = None

    fitted_model = get_atomic_clustering_model(data_train)
    prediction_basic, _ = validate_model_quality(fitted_model, data)

    composite_model = get_composite_clustering_model(data_train, opt_time_sec)

    if not is_fast:
        ComposerVisualiser.visualise(composite_model)

    prediction_composite, _ = validate_model_quality(composite_model, data)

    print(f'adjusted_rand_score for basic model {prediction_basic} with iris')
    print(f'adjusted_rand_score for composite model {prediction_composite} with iris')

    # clustering params tuning example

    data = create_simple_clustering_example(simple_data_size)
    data_train = deepcopy(data)
    data_train.target = None

    fitted_model = get_atomic_clustering_model(data_train)
    prediction_basic, _ = validate_model_quality(fitted_model, data)

    fitted_model.fine_tune_all_nodes(data_train, iterations=tune_iters)

    prediction_tuned, predicted_labels = validate_model_quality(fitted_model, data)

    print(f'adjusted_rand_score for basic model {prediction_basic} with simple data')
    print(f'adjusted_rand_score for tuned model {prediction_tuned} with simple data')

    print(f'Real clusters number is {len(set(data.target))}, '
          f'predicted number is {len(set(predicted_labels))}')

    return prediction_basic, prediction_tuned, prediction_composite


if __name__ == '__main__':
    run_clustering_example()