from fraud_detection.utils import read_data, split_train_test
from fraud_detection.logger import logging
from fraud_detection.preprocess import sample_the_dataset
from fraud_detection.model import ModelContext, SVMModel
from fraud_detection import SEED
import pandas as pd
from fraud_detection.evaluate import evaluate
from fraud_detection.preprocess import PreprocessMethod
import random
import numpy as np
from fraud_detection.training import train_model
from fraud_detection.prediction import prediction
from fraud_detection.utils import config
from fraud_detection import COLUMNS, INSTANCE
if __name__ == "__main__":
    # random.seed(SEED)
    # np.random.seed(SEED)
    # #Load the config file
    # config = OmegaConf.load(CONFIG_PATH)
    # df = read_data(config.data.raw_data_path)
    
    # df, labels = sample_the_dataset(df)
    
    # #Split the dataframe
    # X_train, X_test, X_val, y_train, y_test, y_val = split_train_test(df, labels)
    
    # # Columns 'Time' & 'Amount' are not standardized 
    # train_preprocess = PreprocessMethod(X_train[['Time', 'Amount']])
    # X_train[['Time', 'Amount']] = train_preprocess.transform(X_train[['Time', 'Amount']])
    # # print("X_train: ", X_train)
    # logging.info(f"y_train: {y_train.value_counts()}, \ny_test: {y_test.value_counts()}, \ny_val: {y_val.value_counts()}")
    # model_context = ModelContext(SVMModel())
    # model_context.model_fit(X_train, y_train)
    # model_context.save_model(model_context.model_strategy.model, config.model.model_path)
    
    # X_test[['Time', 'Amount']] = train_preprocess.transform(X_test[['Time', 'Amount']])
    # # print("X_test: ", X_test)
    # y_pred = model_context.model_prediction(x_test=X_test)
    # evaluate(y_true=y_test, y_pred=y_pred)

    #Load the config file
    train_model(config=config)
    instance = np.array(INSTANCE).reshape(1,-1)
    df = pd.DataFrame(instance, columns=COLUMNS)
    print(df)
    # instance = pd.Series(data=instance, index=columns)
    pred, prob = prediction(config=config, instance=df)
    print(pred, prob)
    
    
    
    
    
