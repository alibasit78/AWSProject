from abc import ABC, abstractmethod
from sklearn.svm import SVC
from fraud_detection.logger import logging
import joblib
import os
from fraud_detection import LABEL_NAMES


class ModelStrategy(ABC):
    @abstractmethod
    def predict(self, x_test):
        pass
    
    @abstractmethod
    def fit(self, X_train, y_train):
        pass

    @abstractmethod
    def save_model(self, model, model_path):
        pass
    
    @abstractmethod
    def load_model(self, model_path):
        pass
        
class SVMModel(ModelStrategy):
    def __init__(self):
        super().__init__()
        self.model = SVC(C = 1.0, kernel='rbf', class_weight='balanced', probability=True)
    def fit(self, x_train, y_train):
        logging.info("fit SVM model")
        self.model.fit(x_train, y_train)
    def predict(self, x_test):
        logging.info("predict svm model")
        return self.model.predict(x_test), self.model.predict_proba(x_test)
    def save_model(self, model, model_path):
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(model, filename=model_path)
    
    def load_model(self, model_path):
        return joblib.load(model_path)

# class KNNModel(ModelStrategy):
#     def __init__(self):
#         super().__init__()
#         self.model = 

class ModelContext:
    def __init__(self, model_strategy: ModelStrategy):
        self.model_strategy = model_strategy
    def set_model_strategy(self, model_strategy:ModelStrategy):
        self.model_strategy = model_strategy
    def model_fit(self, x_train, y_train):
        self.model_strategy.fit(x_train, y_train)
    def model_prediction(self, x_test):
        return self.model_strategy.predict(x_test)
    def save_model(self, model, model_path):
        self.model_strategy.save_model(model, model_path)
    def load_model(self, model_path):
        return self.model_strategy.load_model(model_path)