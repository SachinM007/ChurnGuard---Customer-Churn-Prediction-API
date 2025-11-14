from pydantic import BaseModel

class CustomerFeatures(BaseModel):
    CreditScore: int
    Geography: str
    Gender: str
    Age: int
    Tenure: int
    Balance: float
    NumofProducts: int
    HasCrCard: int
    IsActiveMember: int
    EstimatedSalary: float

class PredictionResponse(BaseModel):
    churn_probability: float
    explanation: dict #SHAP values explanation (feature impacts)

