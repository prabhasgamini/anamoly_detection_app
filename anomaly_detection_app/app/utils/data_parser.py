import pandas as pd

def parse_timestamp(ts_str):
    """Convert "MM:SS.s" to total seconds."""
    try:
        parts = ts_str.split(':')
        minutes = int(parts[0])
        seconds = float(parts[1])
        return minutes * 60 + seconds
    except:
        return float('nan')

def parse_csv(filepath, feature_names):
    """Load CSV, parse timestamp, return DataFrame with timestamp and feature columns."""
    df = pd.read_csv(filepath)
    # Assume first column is timestamp
    if 'timestamp' not in df.columns:
        raise ValueError("CSV must contain a 'timestamp' column")
    # Convert timestamp to numeric (seconds) and also keep original for display
    df['timestamp_original'] = df['timestamp']
    df['timestamp'] = df['timestamp'].apply(parse_timestamp).ffill()
    # Check required features
    missing = set(feature_names) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    # Keep only needed columns
    keep = ['timestamp', 'timestamp_original'] + feature_names
    df = df[keep]
    # Convert timestamp to datetime for DB (use a reference date)
    # We'll store as datetime using a fixed reference
    ref_date = pd.Timestamp('2024-01-01')
    df['timestamp'] = ref_date + pd.to_timedelta(df['timestamp'], unit='s')
    return df