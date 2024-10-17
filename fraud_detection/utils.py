import pandas as pd
from sklearn.model_selection import train_test_split
from fraud_detection import SEED
from fraud_detection.logger import logging
import os
import joblib
from omegaconf import OmegaConf
from fraud_detection import CONFIG_PATH

# load dataset
def read_data(file_path):
    return pd.read_csv(file_path)

def split_train_test(features, labels):
    logging.info("Splitting the ds into train test split")
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size = 0.30, stratify = labels, random_state = SEED)
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size = 0.50, stratify = y_test, random_state = SEED)
    logging.info(f"Shapes: X_train: {X_train.shape}, X_test: {X_test.shape}, X_val: {X_val.shape}")
    return X_train, X_test, X_val, y_train, y_test, y_val

def save_artifact(df, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(df, path)

def load_artifact(path):
    return joblib.load(path)

config = OmegaConf.load(CONFIG_PATH)
