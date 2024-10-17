from omegaconf import OmegaConf
from fraud_detection import CONFIG_PATH
from fraud_detection.utils import read_data, split_train_test
from fraud_detection.logger import logging
from fraud_detection.preprocess import sample_the_dataset
from fraud_detection.model import ModelContext, SVMModel
from fraud_detection import SEED
import pandas as pd
from fraud_detection.evaluate import evaluate
from fraud_detection.preprocess import PreprocessMethod
import random
from fraud_detection.utils import save_artifact, load_artifact
import numpy as np
from fraud_detection.aws_service import AWSService
def train_model(config):
    random.seed(SEED)
    np.random.seed(SEED)
    aws_service = AWSService()
    aws_service.download_file_from_s3(config.aws.ingestion_path, config.aws.bucket_name, config.aws.key_filename)
    df = read_data(config.data.raw_data_path)
    df, labels = sample_the_dataset(df)
    #Split the dataframe
    X_train, X_test, X_val, y_train, y_test, y_val = split_train_test(df, labels)
    
    # Columns 'Time' & 'Amount' are not standardized 
    train_preprocess = PreprocessMethod(X_train[['Time', 'Amount']])
    X_train[['Time', 'Amount']] = train_preprocess.transform(X_train[['Time', 'Amount']])
    # print("X_train: ", X_train)
    logging.info(f"y_train: {y_train.value_counts()}, \ny_test: {y_test.value_counts()}, \ny_val: {y_val.value_counts()}")
    model_context = ModelContext(SVMModel())
    model_context.model_fit(X_train, y_train)
    model_context.save_model(model_context, config.model.model_path)
    aws_service.upload_file_to_s3(config.model.model_path, config.aws.bucket_name)
    
    X_test[['Time', 'Amount']] = train_preprocess.transform(X_test[['Time', 'Amount']])
    save_artifact(X_train, config.data.train_data_path)
    aws_service.upload_file_to_s3(config.data.train_data_path, config.aws.bucket_name)
    save_artifact(X_test, config.data.test_data_path)
    aws_service.upload_file_to_s3(config.data.test_data_path, config.aws.bucket_name)
    save_artifact(X_val, config.data.val_data_path)
    aws_service.upload_file_to_s3(config.data.val_data_path, config.aws.bucket_name)
    save_artifact(y_train, config.data.train_label_data_path)
    aws_service.upload_file_to_s3(config.data.train_label_data_path, config.aws.bucket_name)
    save_artifact(y_test, config.data.test_label_data_path)
    aws_service.upload_file_to_s3(config.data.test_label_data_path, config.aws.bucket_name)
    save_artifact(y_val, config.data.val_label_data_path)
    aws_service.upload_file_to_s3(config.data.val_label_data_path, config.aws.bucket_name)
    save_artifact(train_preprocess, config.data.standardize_obj_path)
    aws_service.upload_file_to_s3(config.data.standardize_obj_path, config.aws.bucket_name)
    # print("X_test: ", X_test)
    model_context = load_artifact(config.model.model_path)
    y_pred, y_proba = model_context.model_prediction(x_test=X_test)
    evaluate(y_true=y_test, y_pred=y_pred)

class TrainingInitiator:
    def __init__(self):
        self.config = OmegaConf.load(CONFIG_PATH)
    def start_model_training(self):
        train_model(config=self.config)