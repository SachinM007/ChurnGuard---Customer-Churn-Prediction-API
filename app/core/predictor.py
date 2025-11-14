import numpy as np
import pandas as pd

def preprocess_input(data: dict, encoders: dict) -> pd.DataFrame:
    """
    data: dict of input features from API
    encoders: dict of fitted label encoders for categorical features
    Returns a processed DataFrame ready for model prediction
    """

    df = pd.DataFrame([data])

    for col in ['Geography', 'Gender']:
        le = encoders.get(col)
        if le:
            df[col] = le.transform(df[col])

    #making sure columns are in the order the model expects
    feature_order = [
        "CreditScore",
        "Geography",
        "Gender",
        "Age",
        "Tenure",
        "Balance",
        "NumOfProducts",
        "HasCrCard",
        "IsActiveMember",
        "EstimatedSalary",
    ]

    df = df[feature_order]

    return df

