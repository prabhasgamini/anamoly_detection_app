import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

class LSTMDetector:
    def __init__(self, model, seq_len, sensor_cols):
        self.model = model
        self.seq_len = seq_len
        self.sensor_cols = sensor_cols
        self.buffer = []
        self.prediction_history = []

    def update(self, row_series):
        """Process a new row, return prediction error"""
        values = row_series[self.sensor_cols].values
        self.buffer.append(values)
        
        if len(self.buffer) < self.seq_len + 1:
            return 0.0

        if len(self.buffer) > self.seq_len + 1:
            self.buffer.pop(0)

        X = np.array(self.buffer[:-1]).reshape(1, self.seq_len, -1)
        y_true = self.buffer[-1]
        y_pred = self.model.predict(X, verbose=0)[0]
        error = np.mean(np.abs(y_pred - y_true))
        
        self.prediction_history.append(error)
        if len(self.prediction_history) > 100:
            self.prediction_history.pop(0)
            
        return error

def train_lstm(df, sensor_cols, seq_len=10, return_errors=False):
    """Train LSTM and optionally return training errors"""
    data = df[sensor_cols].values
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len])
    
    X = np.array(X)
    y = np.array(y)

    model = Sequential()
    model.add(LSTM(32, input_shape=(seq_len, len(sensor_cols))))
    model.add(Dense(len(sensor_cols)))
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=5, batch_size=16, verbose=0)

    detector = LSTMDetector(model, seq_len, sensor_cols)
    
    if return_errors:
        # Calculate training errors for adaptive threshold
        preds = model.predict(X, verbose=0)
        train_errors = np.mean(np.abs(preds - y), axis=1)
        return detector, train_errors
    
    return detector