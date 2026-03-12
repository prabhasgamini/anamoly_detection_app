from sklearn.ensemble import IsolationForest
import numpy as np

def train_isolation_forest(sensor_data, contamination='auto'):
    """
    Train Isolation Forest with automatic contamination detection.
    'auto' lets the algorithm determine the threshold naturally.
    """
    model = IsolationForest(
        contamination=contamination,  # 'auto' means no fixed threshold
        random_state=42,
        n_estimators=100,
        max_samples='auto',
        bootstrap=False
    )
    model.fit(sensor_data)
    return model

def score_isolation_forest(model, row_values):
    """Return raw decision function (negative = anomaly)"""
    return model.decision_function(row_values.reshape(1, -1))[0]

def get_isolation_forest_scores(model, data):
    """Get all decision function scores for training data"""
    return model.decision_function(data)

def is_anomaly_isolation_forest(score):
    """
    Determine if a point is anomaly based on Isolation Forest score.
    Negative score indicates anomaly in Isolation Forest.
    """
    return score < 0