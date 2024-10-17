import pandas as pd
import pytest
from fraud_detection.utils import config
from fraud_detection.preprocess import sample_the_dataset
from fraud_detection.utils import split_train_test, load_artifact
from fraud_detection.preprocess import PreprocessMethod
from fraud_detection.model import ModelContext, SVMModel
import numpy as np

@pytest.fixture
def read_data():
    df =  pd.read_csv(config.data.raw_data_path)
    df, labels = sample_the_dataset(df)
    X_train, X_test, X_val, y_train, y_test, y_val = split_train_test(df, labels)
    train_preprocess = PreprocessMethod(X_train[['Time', 'Amount']])
    X_train[['Time', 'Amount']] = train_preprocess.transform(X_train[['Time', 'Amount']])
    X_val[['Time', 'Amount']] = train_preprocess.transform(X_val[['Time', 'Amount']])
    X_test[['Time', 'Amount']] = train_preprocess.transform(X_test[['Time', 'Amount']])
    return X_train, X_test, X_val, y_train, y_test, y_val
    

@pytest.fixture
def load_model():
    model_context = load_artifact(config.model.model_path)
    return model_context

def test_model_inference_types(load_model, read_data):
    model_context = load_model
    X_test = read_data[1]
    # print(X_test)
    y_pred, y_proba = model_context.model_prediction(x_test=X_test)
    assert isinstance(y_pred, np.ndarray)
    assert isinstance(X_test, pd.DataFrame)
