import joblib
import os

from xgboost import XGBClassifier

MODEL_PATH = "../app/models/churn_model_v1.pkl"
ENCODER_PATH = "../data/processed/label_encoders.pkl"


def load_model() -> XGBClassifier:
    model = joblib.load(MODEL_PATH)

    return model

def load_encoders():
    encoders = joblib.load(ENCODER_PATH)

    return encoders
