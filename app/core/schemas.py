from pydantic import BaseModel

class CustomerFeatures(BaseModel):
    CreditScore: int
    Geography: str
    Gender: str
    Age: int
    Tenure: int
    Balance: float
    NumOfProducts: int
    HasCrCard: int
    IsActiveMember: int
    EstimatedSalary: float

class PredictionResponse(BaseModel):
    churn_probability: float
    prediction: str
    explanation: dict #SHAP values explanation (feature impacts)
    interpretation: str