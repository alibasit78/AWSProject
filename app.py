from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel
from uvicorn import run as app_run
from fraud_detection.training import TrainingInitiator
from fraud_detection.prediction import ModelPrediction
from typing import List
from fraud_detection import APP_HOST, APP_PORT
from fraud_detection import LABEL_NAMES, COLUMNS
import pandas as pd
import numpy as np

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ResponseModel(BaseModel):
    label_name: str
    prob_score: float


@app.get("/train")
def training():
    try:
        training_init = TrainingInitiator()
        training_init.start_model_training()

        return Response("Training successful !!")

    except Exception as e:
        raise Response(f"Error Occurred! {e}")


@app.post("/predict", response_model=ResponseModel)
async def predict_route(instance: List[float]):
    try:
        model_prediction = ModelPrediction()
        instance = np.array(instance).reshape(1, -1)
        instance_df = pd.DataFrame(instance, columns=COLUMNS)
        label, probs = model_prediction.prediction(instance=instance_df)
        label_name = LABEL_NAMES[label[0]]
        prob_score = probs[0][label[0]]
        return {
            "label_name": label_name,
            "prob_score": prob_score
        }

    except Exception as e:
        return Response(f"Error Occurred! {e}")


if __name__ == "__main__":
    app_run(app, host=APP_HOST, port=APP_PORT)