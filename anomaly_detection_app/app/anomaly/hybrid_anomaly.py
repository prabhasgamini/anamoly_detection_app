import numpy as np

class HybridDetector:
    """
    Pure pattern learning anomaly detector with no fixed thresholds.
    Anomaly = deviation from learned pattern, measured by:
    - Isolation Forest: negative decision function = anomaly
    - LSTM: prediction error > 2 standard deviations from mean error
    """
    def __init__(self, iso_model, lstm_detector, lstm_errors_history):
        self.iso_model = iso_model
        self.lstm_detector = lstm_detector
        # Learn error distribution from training
        self.lstm_error_mean = np.mean(lstm_errors_history)
        self.lstm_error_std = np.std(lstm_errors_history)
        # Store recent errors for adaptive threshold
        self.recent_errors = []
        self.max_history = 100
        
    def detect(self, iso_raw, lstm_error):
        """
        Detect anomaly based on deviation from learned patterns.
        Returns: (is_anomaly, confidence_score, reason)
        """
        reasons = []
        confidence = 0
        
        # Isolation Forest: negative score = anomaly
        if iso_raw < 0:
            confidence += 0.5
            reasons.append("Isolation Forest detected unusual pattern")
        
        # LSTM: error > 2 std deviations from training mean = anomaly
        if lstm_error > self.lstm_error_mean + 2 * self.lstm_error_std:
            confidence += 0.5
            reasons.append("LSTM prediction deviation")
        
        # Update recent errors for adaptive threshold
        self.recent_errors.append(lstm_error)
        if len(self.recent_errors) > self.max_history:
            self.recent_errors.pop(0)
        
        # Adaptive: if error is significantly higher than recent patterns
        if len(self.recent_errors) > 10:
            recent_mean = np.mean(self.recent_errors[-10:])
            if lstm_error > recent_mean * 1.5:
                confidence += 0.3
                reasons.append("Sudden change from recent pattern")
        
        is_anomaly = confidence >= 0.5  # At least one strong signal
        
        return is_anomaly, min(confidence, 1.0), ", ".join(reasons) if reasons else "Normal pattern"