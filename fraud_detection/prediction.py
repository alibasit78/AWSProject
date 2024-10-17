from fraud_detection.utils import load_artifact
from fraud_detection.utils import config

def prediction(config, instance):
    model_context = load_artifact(config.model.model_path)
    data_standardizer = load_artifact(config.data.standardize_obj_path)
    instance[['Time', 'Amount']] = data_standardizer.transform(instance[['Time', 'Amount']])
    y_pred, y_prob = model_context.model_prediction(x_test=instance)
    return y_pred, y_prob

class ModelPrediction:
    def __init__(self):
        self.config = config
    def prediction(self, instance):
        return prediction(self.config, instance=instance)