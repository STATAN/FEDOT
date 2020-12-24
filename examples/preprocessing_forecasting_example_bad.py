import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_absolute_error
from fedot.core.data.preprocessing import preprocessing_func_for_data, PreprocessingStrategy, \
    Scaling, Normalization, ImputationStrategy, EmptyStrategy, TsScalingStrategy
from fedot.core.chains.node import PrimaryNode, SecondaryNode
from fedot.core.chains.ts_chain import TsForecastingChain
from fedot.core.data.data import InputData, OutputData
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams
from examples.time_series_gapfilling_example import generate_synthetic_data

np.random.seed(2020)

# Do we need to process data with single-model chain
single = True

# Generate data and train test split
time_series = generate_synthetic_data()
len_forecast = 150
train_part = time_series[:-len_forecast]
test_part = time_series[-len_forecast:]
if __name__ == '__main__':
    # Regression chain with single model init
    for preproc_func in [EmptyStrategy, Scaling, Normalization,
                         ImputationStrategy, TsScalingStrategy]:
        print(str(preproc_func))

        if single == True:
            print('Single')
            chain = TsForecastingChain(PrimaryNode('gbr', manual_preprocessing_func=preproc_func))
        else:
            print('Multi')
            node_1lv_1 = PrimaryNode('ridge', manual_preprocessing_func=preproc_func)
            node_1lv_2 = PrimaryNode('ridge',manual_preprocessing_func=preproc_func)

            node_2lv_1 = SecondaryNode('linear',manual_preprocessing_func=preproc_func,
                                       nodes_from=[node_1lv_1])
            node_2lv_2 = SecondaryNode('xgbreg', manual_preprocessing_func=preproc_func,
                                       nodes_from=[node_1lv_2])

            # Root node - make final prediction
            node_final = SecondaryNode('adareg', manual_preprocessing_func=preproc_func,
                                       nodes_from=[node_2lv_1,
                                                   node_2lv_2])
            chain = TsForecastingChain(node_final)

        task = Task(TaskTypesEnum.ts_forecasting,
                    TsForecastingParams(forecast_length=len_forecast,
                                        max_window_size=50,
                                        return_all_steps=False,
                                        make_future_prediction=True))

        train_data = InputData(idx=np.arange(0, len(train_part)),
                               features=None,
                               target=train_part,
                               task=task,
                               data_type=DataTypesEnum.ts)

        # Making predictions for the missing part in the time series
        chain.fit_from_scratch(train_data)

        # "Test data" for making prediction for a specific length
        test_data = InputData(idx=np.arange(0, len_forecast),
                              features=None,
                              target=None,
                              task=task,
                              data_type=DataTypesEnum.ts)

        predicted_values = chain.forecast(initial_data=train_data,
                                          supplementary_data=test_data).predict

        # Print MAE metric
        print(f'Предсказанные значения: {predicted_values[:5]}')
        print(f'Действительные значения: {test_part[:5]}')
        print(f'Mean absolute error: {mean_absolute_error(test_part, predicted_values):.3f}\n')