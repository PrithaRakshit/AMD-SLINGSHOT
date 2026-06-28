import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


def get_data(n: int = 500_000) -> pd.DataFrame:
    """Generate synthetic power-usage time-series with injected anomalies.

    Pattern: daily sine-wave cycle + Gaussian noise + 500 random spikes.
    """
    time = np.arange(n)

    # Realistic daily power cycle
    power = (
        10
        + 5 * np.sin(2 * np.pi * time / 1_000)
        + np.random.normal(0, 1, n)
    )

    # Inject 500 hard anomalies (grid faults / bursts)
    anomaly_indices = np.random.choice(n, 500, replace=False)
    power[anomaly_indices] += np.random.uniform(20, 50, 500)

    df = pd.DataFrame({"time": time, "power": power})
    df["hour"] = (time // (1_000 / 24)).astype(int) % 24  # synthetic hour-of-day
    return df


def run_detection(df: pd.DataFrame, contamination: float = 0.001) -> pd.DataFrame:
    """Fit IsolationForest and append anomaly labels (-1 = anomaly, 1 = normal)."""
    model = IsolationForest(contamination=contamination, random_state=42, n_jobs=-1)
    df = df.copy()
    df["anomaly"] = model.fit_predict(df[["power"]])
    df["score"] = model.decision_function(df[["power"]])
    return df, model
