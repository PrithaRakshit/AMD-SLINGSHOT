import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

def get_data(
    n: int = 10_000,
    anomaly_rate: float = 0.05,
    random_state: int = 42
) -> pd.DataFrame:
    """Generate reproducible synthetic power-usage data with injected anomalies."""

    rng = np.random.default_rng(random_state)

    time = np.arange(n)

    # Daily power-consumption cycle
    power = (
        10
        + 5 * np.sin(2 * np.pi * time / 1_000)
        + rng.normal(0, 1, n)
    )

    # Inject anomalies according to the selected anomaly rate
    num_anomalies = max(1, int(n * anomaly_rate))
    anomaly_indices = rng.choice(n, num_anomalies, replace=False)

    power[anomaly_indices] += rng.uniform(20, 50, num_anomalies)

    # Ground-truth labels: 0 = normal, 1 = anomaly
    true_labels = np.zeros(n, dtype=int)
    true_labels[anomaly_indices] = 1

    df = pd.DataFrame({
        "time": time,
        "power": power,
        "true_anomaly": true_labels
    })

    df["hour"] = (time // (1_000 / 24)).astype(int) % 24

    return df

def run_detection(df: pd.DataFrame, contamination: float = 0.001) -> tuple[pd.DataFrame, IsolationForest]:
    """Fit IsolationForest and append anomaly labels (-1 = anomaly, 1 = normal)."""
    model = IsolationForest(contamination=contamination, random_state=42, n_jobs=-1)
    df = df.copy()
    df["anomaly"] = model.fit_predict(df[["power"]])
    df["score"] = model.decision_function(df[["power"]])
    return df, model
