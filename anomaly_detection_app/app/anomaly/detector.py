import pickle
import os
import numpy as np
import pandas as pd
from .isolation_forest_model import train_isolation_forest, get_isolation_forest_scores
from .lstm_model import train_lstm, LSTMDetector
from .hybrid_anomaly import HybridDetector
from tensorflow.keras.models import load_model
from flask import current_app

def get_model_paths(machine_id):
    base = current_app.config['MODEL_DIR']
    iso_path = os.path.join(base, f'machine_{machine_id}_iso.pkl')
    lstm_path = os.path.join(base, f'machine_{machine_id}_lstm.pkl')
    lstm_h5 = os.path.join(base, f'machine_{machine_id}_lstm.h5')
    errors_path = os.path.join(base, f'machine_{machine_id}_errors.npy')
    iso_scores_path = os.path.join(base, f'machine_{machine_id}_iso_scores.npy')
    return iso_path, lstm_path, lstm_h5, errors_path, iso_scores_path

def train_models_for_machine(machine, df):
    """Train models using pure pattern learning approach"""
    sensor_cols = machine.feature_names
    X = df[sensor_cols].values

    # Train Isolation Forest with auto contamination
    iso_model = train_isolation_forest(X, contamination='auto')
    iso_path, lstm_path, lstm_h5, errors_path, iso_scores_path = get_model_paths(machine.id)
    
    with open(iso_path, 'wb') as f:
        pickle.dump(iso_model, f)
    
    # Get Isolation Forest scores for training data
    iso_scores = get_isolation_forest_scores(iso_model, X)
    np.save(iso_scores_path, iso_scores)

    # Train LSTM and get training errors
    lstm_detector, train_errors = train_lstm(df, sensor_cols, return_errors=True)
    lstm_detector.model.save(lstm_h5)
    
    # Save training errors
    np.save(errors_path, train_errors)
    
    detector_config = {
        'seq_len': lstm_detector.seq_len,
        'sensor_cols': lstm_detector.sensor_cols
    }
    with open(lstm_path, 'wb') as f:
        pickle.dump(detector_config, f)

    from app import db
    db.session.commit()
    
    return iso_model, lstm_detector, train_errors, iso_scores

def load_detector_for_machine(machine):
    """Load models and return pure pattern learning detector"""
    iso_path, lstm_path, lstm_h5, errors_path, iso_scores_path = get_model_paths(machine.id)
    
    with open(iso_path, 'rb') as f:
        iso_model = pickle.load(f)
    
    with open(lstm_path, 'rb') as f:
        lstm_config = pickle.load(f)
    
    lstm_model = load_model(lstm_h5)
    train_errors = np.load(errors_path)
    iso_scores = np.load(iso_scores_path)
    
    lstm_detector = LSTMDetector(
        model=lstm_model,
        seq_len=lstm_config['seq_len'],
        sensor_cols=lstm_config['sensor_cols']
    )
    
    # Create pure pattern learning hybrid detector
    hybrid = HybridDetector(
        iso_model=iso_model,
        lstm_detector=lstm_detector,
        lstm_errors_history=train_errors,
        iso_scores_history=iso_scores
    )
    
    return iso_model, lstm_detector, hybrid