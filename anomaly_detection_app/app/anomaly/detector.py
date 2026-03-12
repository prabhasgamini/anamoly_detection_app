import pickle
import os
import numpy as np
import pandas as pd
from .isolation_forest_model import train_isolation_forest
from .lstm_model import train_lstm, LSTMDetector
from .hybrid_anomaly import HybridDetector
from tensorflow.keras.models import load_model

MODEL_DIR = 'models_storage'

def get_model_paths(machine_id):
    iso_path = os.path.join(MODEL_DIR, f'machine_{machine_id}_iso.pkl')
    lstm_path = os.path.join(MODEL_DIR, f'machine_{machine_id}_lstm.pkl')
    lstm_h5 = os.path.join(MODEL_DIR, f'machine_{machine_id}_lstm.h5')
    return iso_path, lstm_path, lstm_h5

def train_models_for_machine(machine, df):
    """df must contain 'timestamp' column and all feature columns."""
    sensor_cols = machine.feature_names
    X = df[sensor_cols].values

    # Train Isolation Forest
    iso_model = train_isolation_forest(X)
    iso_path, lstm_path, lstm_h5 = get_model_paths(machine.id)
    with open(iso_path, 'wb') as f:
        pickle.dump(iso_model, f)

    iso_scores = iso_model.decision_function(X)
    iso_threshold = iso_model.threshold_
    iso_min_score = np.min(iso_scores)

    # Train LSTM
    lstm_detector = train_lstm(df, sensor_cols)
    lstm_detector.model.save(lstm_h5)
    detector_config = {
        'threshold': lstm_detector.threshold,
        'max_error': lstm_detector.max_error,
        'seq_len': lstm_detector.seq_len,
        'sensor_cols': lstm_detector.sensor_cols
    }
    with open(lstm_path, 'wb') as f:
        pickle.dump(detector_config, f)

    # Update machine record
    machine.iso_model_path = iso_path
    machine.lstm_model_path = lstm_path
    machine.iso_threshold = float(iso_threshold)
    machine.iso_min_score = float(iso_min_score)
    machine.lstm_threshold = float(lstm_detector.threshold)
    machine.lstm_max_error = float(lstm_detector.max_error)

    from app import db
    db.session.commit()

def load_detector_for_machine(machine):
    iso_path, lstm_path, lstm_h5 = get_model_paths(machine.id)
    with open(iso_path, 'rb') as f:
        iso_model = pickle.load(f)
    with open(lstm_path, 'rb') as f:
        lstm_config = pickle.load(f)
    lstm_model = load_model(lstm_h5)
    lstm_detector = LSTMDetector(
        model=lstm_model,
        threshold=lstm_config['threshold'],
        max_error=lstm_config['max_error'],
        seq_len=lstm_config['seq_len'],
        sensor_cols=lstm_config['sensor_cols']
    )
    hybrid = HybridDetector(
        iso_threshold=machine.iso_threshold,
        iso_min_score=machine.iso_min_score,
        lstm_threshold=machine.lstm_threshold,
        lstm_max_error=machine.lstm_max_error,
        weight_iso=0.5,
        weight_lstm=0.5,
        hybrid_threshold=0.5
    )
    return iso_model, lstm_detector, hybrid