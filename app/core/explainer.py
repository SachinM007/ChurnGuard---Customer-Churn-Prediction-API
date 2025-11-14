import shap
import numpy as np

explainer = None

def init_explainer(model):
    explainer = shap.TreeExplainer(model)
    return explainer
