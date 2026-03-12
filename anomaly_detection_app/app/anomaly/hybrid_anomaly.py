class HybridDetector:
    def __init__(self, iso_threshold, iso_min_score, lstm_threshold, lstm_max_error,
                 weight_iso=0.5, weight_lstm=0.5, hybrid_threshold=0.5):
        self.iso_threshold = iso_threshold
        self.iso_min_score = iso_min_score
        self.lstm_threshold = lstm_threshold
        self.lstm_max_error = lstm_max_error
        self.weight_iso = weight_iso
        self.weight_lstm = weight_lstm
        self.hybrid_threshold = hybrid_threshold

    def normalise_iso(self, raw_score):
        if raw_score >= self.iso_threshold:
            return 0.0
        denom = self.iso_threshold - self.iso_min_score
        if denom <= 0:
            return 1.0 if raw_score < self.iso_threshold else 0.0
        norm = (self.iso_threshold - raw_score) / denom
        return min(max(norm, 0.0), 1.0)

    def normalise_lstm(self, error):
        if error <= self.lstm_threshold:
            return 0.0
        norm = (error - self.lstm_threshold) / (self.lstm_max_error - self.lstm_threshold)
        return min(max(norm, 0.0), 1.0)

    def combine(self, iso_raw, lstm_error):
        iso_norm = self.normalise_iso(iso_raw)
        lstm_norm = self.normalise_lstm(lstm_error)
        combined = self.weight_iso * iso_norm + self.weight_lstm * lstm_norm
        anomaly = 1 if combined >= self.hybrid_threshold else 0
        return combined, anomaly