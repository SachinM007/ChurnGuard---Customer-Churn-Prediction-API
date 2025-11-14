from fastapi import FastAPI, HTTPException
from app.core.schemas import CustomerFeatures, PredictionResponse
from app.core.model_loader import load_model, load_encoders
from app.core.predictor import preprocess_input
from app.core.explainer import init_explainer

import pandas as pd

app = FastAPI(
    title="ChurnGuard",
    description="FastAPI service for Customer churn prediction and model explanations.",
    version="0.0.1"
    )

model = load_model()
encoders = load_encoders()
explainer = init_explainer(model)

@app.get("/")
def root():
    return {"Welcome to ChurnGuard API"}

@app.post("/predict",response_model=PredictionResponse)
def predict(data: CustomerFeatures):
    """
    Accepts customer data and returns churn probability
    """
    try:
        #preprocess input
        processed_data = preprocess_input(data.model_dump(), encoders)
        
        #predict churn probability
        churn_prob = model.predict_proba(processed_data)[:,1][0]
        # determine churn decision
        churn_decision = "Yes" if churn_prob >= 0.5 else "No"

        #compute shap values
        shap_values = explainer.shap_values(processed_data)
        feature_order = processed_data.columns.tolist()
        # Convert numpy types to native Python floats
        shap_dict = {feature: float(value) for feature, value in zip(feature_order, shap_values[0])}
        
        return {
            "churn_probability": float(churn_prob),
            "prediction": churn_decision,
            "explanation": shap_dict,                # <--- matches schema
            "interpretation": (
                "SHAP values indicate feature impact: "
                "positive values increase churn probability, "
                "negative values decrease it. Features are shown in the order you provided."
            )
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


