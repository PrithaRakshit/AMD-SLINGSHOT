# ⚡ Grid-Guard — Local Power Anomaly Detector
> AMD Edge Edition · Zero cloud dependency · Sub-200ms inference

## Quick Start

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Project Structure

```
grid-guard/
├── requirements.txt   # Minimal deps: streamlit, pandas, numpy, scikit-learn
├── detector.py        # Data generation + IsolationForest brain
├── app.py             # Streamlit dashboard
└── README.md
```

## How It Works

### 1. Data Generation (`detector.py → get_data`)
Generates a realistic power-usage time series:
- **Daily cycle**: `10 + 5·sin(2π·t/1000)` — simulates load patterns
- **Gaussian noise**: `N(0, 1)` — real-world sensor variance
- **500 injected anomalies**: uniform spikes of `+20–50 kW` (pipe bursts, grid faults)

### 2. Anomaly Detection (`detector.py → run_detection`)
- `IsolationForest(contamination=0.001, n_jobs=-1)`
- Marks each point: `1` = normal, `-1` = anomaly
- Also exposes the raw `decision_function` score for severity ranking

### 3. Dashboard (`app.py`)
- Dark industrial UI (AMD/tech aesthetic)
- KPI cards: total points, anomaly count, rate, inference time, grid status
- Line chart of power signal with anomalies highlighted in red
- Top-20 anomalies table sorted by severity
- Hourly anomaly frequency bar chart
- Sidebar controls for dataset size, contamination rate, viz sample size

---

## Architecture Table

| Phase | Task | AMD Edge Benefit |
|---|---|---|
| Data Ingestion | Local CSV / sensor stream processing | Zero egress/ingress cost |
| Processing | Scikit-learn IsolationForest (CPU) | Multi-core Ryzen throughput |
| Visualisation | Streamlit dashboard | Real-time monitoring, no cloud lag |
| Scaling | Multiple edge nodes | Linear scale-out, no cloud bottleneck |
