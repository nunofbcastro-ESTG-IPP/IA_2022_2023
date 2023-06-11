import os
import uuid
import h2o
import pandas as pd
import psutil
from h2o.estimators import (H2ODeepLearningEstimator,
                            H2OGradientBoostingEstimator,
                            H2ORandomForestEstimator)

from Config import *
from Metrics import *
from Solution import *


class MachineLearning:
    
    def __init__(self, file_path: str, data_percentage_use: float, columnNamesFirst: int, columnNamesLast: int, columnResult: str, train_ratio: float, test_ratio: float, seed: int = None):
        self.h2o_init()
        
        self.id =  uuid.uuid4()
        self.data = self.load_data(file_path,data_percentage_use)
        self.columnNames = self.data.columns[columnNamesFirst:columnNamesLast]
        self.columnResult = columnResult
        self.train, self.validation, self.test = self.split_data(train_ratio, test_ratio, seed)

    def h2o_init(self):
        h2o_mem_size = int((psutil.virtual_memory().available / (1024 ** 3)) * Config.get_ram_percentage())
        num_threads = int(os.cpu_count() * Config.get_n_threads_percentage())
        h2o.init(nthreads=num_threads, max_mem_size=f"{h2o_mem_size}G")

    def h2o_remove_all(self):
        modelos = h2o.ls()
        for model in modelos['key']:
            if not model.endswith(".hex") and not model.startswith("py_"):
                h2o.remove(model)

    def h2o_close(self):
        h2o.cluster().shutdown()

    def load_data(self, file_path: str, data_percentage_use: float):
        data = h2o.import_file(file_path)
        if data_percentage_use < 1:
            data, _ = data.split_frame(ratios=[data_percentage_use])
        return data

    def split_data(self, train_ratio: float, test_ratio: float, seed: int = None):
        if train_ratio != 1:
            train, test, valid = self.data.split_frame(ratios=[train_ratio, test_ratio], seed=seed)
        else:
            train = h2o.deep_copy(self.data, 'new_df')
            valid = self.data.head(1)
            test = self.data.head(1)
            
        train[self.columnResult] = train[self.columnResult].asfactor()
        valid[self.columnResult] = valid[self.columnResult].asfactor()
        test[self.columnResult] = test[self.columnResult].asfactor()
        
        return train, valid, test

    def get_model_performance(self, model):
        predictions = model.predict(self.test).as_data_frame()

        res = self.test['label'].as_data_frame().values.flatten() == predictions['predict']
        correct = [x for x in res if x]
        accuracy = len(correct) / len(res)
        
        performance = model.model_performance(test_data=self.test)

        metrics = Metrics(
            mse=performance.mse(),
            rmse=performance.rmse(),
            logloss=performance.logloss(),
            mean_per_class_error=performance.mean_per_class_error(),
            r2=performance.r2(),
            accuracy=accuracy
        )

        return metrics

    def train_model(self, model):
        model.train(
            x=self.columnNames,
            y=self.columnResult,
            training_frame=self.train,
            validation_frame=self.train
        )            

    def get_model_deep_learning(self, solution: Solution):
        model = H2ODeepLearningEstimator(
            #model_id='model_DL',
            hidden=solution.item.hidden,
            epochs=solution.item.epochs,
            mini_batch_size=solution.item.mini_batch_size,
            adaptive_rate=solution.item.adaptive_rate,
            rate=solution.item.rate,
            activation=solution.item.activation,
            stopping_rounds = solution.item.stopping_rounds,
            stopping_tolerance = solution.item.stopping_tolerance,
            seed=solution.item.seed,
        )
        self.train_model(model)
        return model

    def get_model_random_forest(self, solution: Solution):
        model = H2ORandomForestEstimator(
            #model_id='model_RF',
            ntrees = solution.item.ntrees,
            max_depth = solution.item.max_depth,
            min_rows = solution.item.min_rows,
            nfolds = solution.item.nfolds,
            sample_rate = solution.item.sample_rate,
            stopping_rounds = solution.item.stopping_rounds,
            stopping_tolerance = solution.item.stopping_tolerance,
            seed=solution.item.seed,
        )
        self.train_model(model)
        return model

    def get_model_gradient_boosting_estimator(self, solution: Solution):
        model = H2OGradientBoostingEstimator(      
            #model_id='model_GBM',  
            ntrees = solution.item.ntrees,
            max_depth = solution.item.max_depth,
            min_rows = solution.item.min_rows,
            nfolds = solution.item.nfolds,
            sample_rate = solution.item.sample_rate,
            col_sample_rate = solution.item.col_sample_rate,
            learn_rate=solution.item.learn_rate,
            stopping_rounds = solution.item.stopping_rounds,
            stopping_tolerance = solution.item.stopping_tolerance,
            seed=solution.item.seed,
        )
        self.train_model(model)
        return model

    def test_model(self, model, test_path: str, sample_path: str, save_path: str):
        test_data = h2o.import_file(test_path)

        preds = model.predict(test_data)
        preds['predict'].as_data_frame().values.shape

        sample_submission = pd.read_csv(sample_path)
        sample_submission.shape
        sample_submission['label'] = preds['predict'].as_data_frame()
        sample_submission.to_csv(save_path, index=False)
        sample_submission.head()
        
