import shap

def explain_model(model, X_sample):
    explainer = shap.Explainer(model, X_sample)
    shap_values = explainer(X_sample)
    shap.plots.waterfall(shap_values[0])
