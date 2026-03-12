import numpy as np

class HybridDetector:
    """
    Pure pattern learning anomaly detector with no fixed thresholds.
    Anomaly = deviation from learned pattern.
    """
    def __init__(self, iso_model, lstm_detector, lstm_errors_history, iso_scores_history=None):
        self.iso_model = iso_model
        self.lstm_detector = lstm_detector
        
        # Learn LSTM error distribution from training
        self.lstm_error_mean = np.mean(lstm_errors_history)
        self.lstm_error_std = np.std(lstm_errors_history)
        
        # Learn Isolation Forest score distribution if provided
        if iso_scores_history is not None and len(iso_scores_history) > 0:
            self.iso_score_mean = np.mean(iso_scores_history)
            self.iso_score_std = np.std(iso_scores_history)
        else:
            self.iso_score_mean = 0
            self.iso_score_std = 0.1
        
        # Store recent errors for adaptive threshold
        self.recent_errors = []
        self.recent_iso_scores = []
        self.max_history = 100
        
    def detect(self, iso_raw, lstm_error):
        """
        Detect anomaly based on deviation from learned patterns.
        Returns: (is_anomaly, confidence_score, reason)
        """
        reasons = []
        confidence = 0
        
        # Isolation Forest: negative score indicates anomaly
        if iso_raw < 0:
            confidence += 0.4
            reasons.append("Isolation Forest detected unusual pattern")
        elif iso_raw < self.iso_score_mean - 2 * self.iso_score_std:
            confidence += 0.3
            reasons.append("Pattern deviation detected")
        
        # LSTM: error > 2 std deviations from training mean = anomaly
        if lstm_error > self.lstm_error_mean + 2 * self.lstm_error_std:
            confidence += 0.5
            reasons.append("Prediction deviation")
        
        # Update recent history
        self.recent_errors.append(lstm_error)
        self.recent_iso_scores.append(iso_raw)
        
        if len(self.recent_errors) > self.max_history:
            self.recent_errors.pop(0)
            self.recent_iso_scores.pop(0)
        
        # Adaptive: compare with recent patterns
        if len(self.recent_errors) > 10:
            recent_error_mean = np.mean(self.recent_errors[-10:])
            recent_iso_mean = np.mean(self.recent_iso_scores[-10:])
            
            if lstm_error > recent_error_mean * 1.5:
                confidence += 0.2
                reasons.append("Sudden change from recent pattern")
            
            if iso_raw < recent_iso_mean - 1.5 * np.std(self.recent_iso_scores[-10:]):
                confidence += 0.2
                reasons.append("Unusual compared to recent data")
        
        is_anomaly = confidence >= 0.5
        
        return is_anomaly, min(confidence, 1.0), ", ".join(reasons) if reasons else "Normal pattern"
    
    def get_anomaly_score(self, iso_raw, lstm_error):
        """Get a normalized anomaly score between 0 and 1"""
        iso_norm = max(0, min(1, (self.iso_score_mean - iso_raw) / (4 * self.iso_score_std + 1e-10)))
        lstm_norm = max(0, min(1, (lstm_error - self.lstm_error_mean) / (4 * self.lstm_error_std + 1e-10)))
        return (iso_norm + lstm_norm) / 2