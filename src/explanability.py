import shap
import numpy as np
from src.train import load_model
from src.load import load
from src.preprocess import preprocess

# Load Model
model = load_model("models/xgb.pkl")
df = load("data/train.csv")
X, y = preprocess(df)
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)


def dependence_plot(feature_name:str, interaction_feature_name:str=None, **kwargs):
    """
    Create a SHAP dependence plot to visualize the relationship between a feature and a model's output.

    Parameters:
    model: The machine learning model for which the SHAP values will be computed.
    feature_name (str): The name of the feature to plot.
    label (int): The index of the output class or label to explain.
    interaction_feature_name (str, optional): The name of an interaction feature for interaction plots. 
        If provided, it will show the relationship between the feature and the interaction feature.

    Returns:
    None

    The function generates a SHAP dependence plot to visualize how a single feature affects model output.
    If an interaction feature is specified, it will also show the interaction effect.

    Example:
    dep_plot(model, 'feature1', 0, interaction_feature_name='feature2')
    """
    feature_idx = np.where(X.columns == feature_name)[0][0]
    if interaction_feature_name:
        interaction_feature_idx = np.where(X.columns == interaction_feature_name)[0][0]
        shap.dependence_plot(
            feature_idx,
            shap_values,
            X,
            interaction_index=interaction_feature_idx,  # Set to None to show the main effect
            show=True,
            **kwargs)  # Set to True to display the plot
    else:
        shap.dependence_plot(
            feature_idx,
            shap_values,
            X,
            show=True,
            **kwargs)  # Set to True to display the plot
    return None