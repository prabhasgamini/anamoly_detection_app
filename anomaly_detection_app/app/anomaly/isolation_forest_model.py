from sklearn.ensemble import IsolationForest
import numpy as np

def train_isolation_forest(sensor_data):
    """sensor_data: 2D array (n_samples, n_features)"""
    model = IsolationForest(contamination=0.05, random_state=42)
    model.fit(sensor_data)
    return model

def score_isolation_forest(model, row_values):
    """row_values: 1D array of features"""
    return model.decision_function(row_values.reshape(1, -1))[0]