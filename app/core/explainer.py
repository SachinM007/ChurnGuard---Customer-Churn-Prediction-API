import shap
import numpy as np

explainer = None

def init_explainer(model, background_data):
    global explainer
    explainer = shap.TreeExplainer(model, data=background_data)

def get_shap_values(processed_df):
    if explainer is None:
        raise ValueError("Explainer is not initialized")
    shap_values = explainer.shap_values(processed_df)
    #return shap values as a dictionary feature -> value
    return dict(zip(processed_df.columns, shap_values[0]))