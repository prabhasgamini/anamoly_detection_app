import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

class LSTMDetector:
    def __init__(self, model, threshold, max_error, seq_len, sensor_cols):
        self.model = model
        self.threshold = threshold
        self.max_error = max_error
        self.seq_len = seq_len
        self.sensor_cols = sensor_cols
        self.buffer = []

    def update(self, row_series):
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
        return error

def train_lstm(df, sensor_cols, seq_len=10):
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

    preds = model.predict(X, verbose=0)
    errors = np.mean(np.abs(preds - y), axis=1)
    threshold = np.percentile(errors, 95)
    max_error = np.max(errors)

    detector = LSTMDetector(model, threshold, max_error, seq_len, sensor_cols)
    return detector