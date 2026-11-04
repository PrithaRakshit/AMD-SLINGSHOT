"""
Grid-Guard: Local Power Anomaly Detector
Edge Computing Edition — runs fully on-device, zero cloud dependency.
"""

import time
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


from detector import get_data, run_detection

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Grid-Guard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS: dark industrial theme ────────────────────────────────────────
st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Rajdhani:wght@400;600;700&display=swap');

      html, body, [class*="css"] { font-family: 'Rajdhani', sans-serif; }
      h1, h2, h3 { font-family: 'Share Tech Mono', monospace; letter-spacing: 2px; }

      .stApp { background: #0a0e14; color: #c9d1d9; }

      .metric-card {
        background: #111820;
        border: 1px solid #1e3a5f;
        border-radius: 8px;
        padding: 1.2rem 1.5rem;
        text-align: center;
      }
      .metric-value { font-size: 2.4rem; font-weight: 700; color: #58a6ff; }
      .metric-label { font-size: 0.85rem; color: #8b949e; text-transform: uppercase; letter-spacing: 1px; }
      .metric-alert { color: #f85149 !important; }

      .status-ok   { color: #3fb950; font-weight: 700; }
      .status-warn { color: #f85149; font-weight: 700; }

      section[data-testid="stSidebar"] { background: #0d1117; border-right: 1px solid #1e3a5f; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Sidebar controls ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚡ Grid-Guard")
    st.markdown("**Edge Computing Edition**")
    st.markdown("---")

    n_points = st.select_slider(
        "Dataset size",
        options=[10_000, 50_000, 100_000, 250_000, 500_000],
        value=100_000,
    )

    contamination = st.slider(
    "Expected anomaly rate",
    min_value=0.0001,
    max_value=0.10,
    value=0.05,
    step=0.001,
    format="%.4f"
)

    viz_sample = st.slider("Visualization sample (points)", 500, 5_000, 1_000, 500)

    st.markdown("---")
    run_btn = st.button("🔍 Run Detection", use_container_width=True)
    st.markdown("---")
    st.markdown(
        """
        **Architecture**
        - 🖥️ 100% on-device processing
        - 🚫 Zero cloud egress
        - ⚡ IsolationForest (CPU-optimised)
        - 🌱 Low-carbon local inference
        """
    )

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("# ⚡ GRID-GUARD")
st.markdown("##### Real-time power anomaly detection · Local Edge Architecture · Zero cloud dependency")
st.markdown("---")

# ── Session state ─────────────────────────────────────────────────────────────
if "results" not in st.session_state:
    st.session_state.results = None

# ── Run pipeline ──────────────────────────────────────────────────────────────
# ── Run pipeline ──────────────────────────────────────────────────────────────
if run_btn:
    with st.spinner("Generating synthetic grid data…"):
        t0 = time.perf_counter()
        raw_df = get_data(n=n_points, anomaly_rate=contamination)
        gen_ms = (time.perf_counter() - t0) * 1000

    with st.spinner("Fitting IsolationForest on CPU cores…"):
        t1 = time.perf_counter()
        df, model = run_detection(raw_df, contamination=contamination)
        det_ms = (time.perf_counter() - t1) * 1000

    # Calculate evaluation metrics
    y_true = df["true_anomaly"]
    y_pred = (df["anomaly"] == -1).astype(int)

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    # Store results for dashboard display
    st.session_state.results = {
        "df": df,
        "gen_ms": gen_ms,
        "det_ms": det_ms,
        "n_points": n_points,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": cm,
    }
# ── Contamination Sensitivity Experiment ─────────────────────────────────
if st.session_state.results:
    st.markdown("---")
    st.markdown("### 🔬 Contamination Sensitivity Analysis")

    experiment_df = get_data(
        n=100_000,
        anomaly_rate=0.05
    )

    contamination_values = [
        0.001,
        0.01,
        0.025,
        0.05,
        0.075,
        0.10
    ]

    experiment_results = []

    for rate in contamination_values:
        test_df, _ = run_detection(
            experiment_df,
            contamination=rate
        )

        y_true = test_df["true_anomaly"]
        y_pred = (test_df["anomaly"] == -1).astype(int)

        precision = precision_score(
            y_true, y_pred, zero_division=0
        )
        recall = recall_score(
            y_true, y_pred, zero_division=0
        )
        f1 = f1_score(
            y_true, y_pred, zero_division=0
        )

        experiment_results.append({
            "Contamination": rate,
            "Precision": precision,
            "Recall": recall,
            "F1-Score": f1
        })

    sensitivity_df = pd.DataFrame(experiment_results)

    st.dataframe(
        sensitivity_df.style.format({
            "Contamination": "{:.3%}",
            "Precision": "{:.2%}",
            "Recall": "{:.2%}",
            "F1-Score": "{:.2%}"
        }),
        use_container_width=True
    )

    st.markdown("#### F1-Score Comparison")

    st.bar_chart(
        sensitivity_df.set_index("Contamination")[["F1-Score"]]
    )


# ── Display results ───────────────────────────────────────────────────────────
if st.session_state.results:
    res = st.session_state.results
    df: pd.DataFrame = res["df"]

    anomalies = df[df["anomaly"] == -1]
    normal    = df[df["anomaly"] == 1]
    anomaly_pct = len(anomalies) / len(df) * 100

    # ── KPI row ──────────────────────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:
        st.markdown(
            f'<div class="metric-card"><div class="metric-value">{res["n_points"]:,}</div>'
            f'<div class="metric-label">Data Points</div></div>', unsafe_allow_html=True
        )
    with c2:
        st.markdown(
            f'<div class="metric-card"><div class="metric-value metric-alert">{len(anomalies):,}</div>'
            f'<div class="metric-label">Anomalies Found</div></div>', unsafe_allow_html=True
        )
    with c3:
        st.markdown(
            f'<div class="metric-card"><div class="metric-value">{anomaly_pct:.3f}%</div>'
            f'<div class="metric-label">Anomaly Rate</div></div>', unsafe_allow_html=True
        )
    with c4:
        st.markdown(
            f'<div class="metric-card"><div class="metric-value">{res["det_ms"]:.0f}ms</div>'
            f'<div class="metric-label">Detection Time</div></div>', unsafe_allow_html=True
        )
    with c5:
        label = "NOMINAL" if anomaly_pct < 0.5 else "ALERT"
        css   = "status-ok" if anomaly_pct < 0.5 else "status-warn"
        st.markdown(
            f'<div class="metric-card"><div class="metric-value {css}">{label}</div>'
            f'<div class="metric-label">Grid Status</div></div>', unsafe_allow_html=True
        )
        st.markdown("---")

    # ── Model Evaluation ─────────────────────────────────────────────────────
    st.markdown("### 📋 Model Evaluation")

    eval_col1, eval_col2, eval_col3 = st.columns(3)

    with eval_col1:
        st.metric("Precision", f"{res['precision']:.2%}")

    with eval_col2:
        st.metric("Recall", f"{res['recall']:.2%}")

    with eval_col3:
        st.metric("F1-Score", f"{res['f1']:.2%}")

    st.markdown("#### Confusion Matrix")

    cm_df = pd.DataFrame(
        res["confusion_matrix"],
        index=["Actual Normal", "Actual Anomaly"],
        columns=["Predicted Normal", "Predicted Anomaly"],
    )

    st.dataframe(cm_df, use_container_width=True)

    st.caption(
        "Evaluation is based on artificially injected anomalies "
        "in the synthetic dataset."
    )

    st.markdown("---")

    # ── Power signal chart ───────────────────────────────────────────────────
    st.markdown("### 📊 Power Signal (sampled)")

    sample_df = df.iloc[
        ::max(1, len(df) // viz_sample)
    ][["time", "power", "anomaly"]].copy()

    sample_df["normal_power"] = np.where(
        sample_df["anomaly"] == 1,
        sample_df["power"],
        np.nan
    )

    sample_df["anomaly_power"] = np.where(
        sample_df["anomaly"] == -1,
        sample_df["power"],
        np.nan
    )

    chart_data = sample_df.set_index("time")[
        ["normal_power", "anomaly_power"]
    ]

    st.line_chart(chart_data, color=["#58a6ff", "#f85149"])
    # ── Anomaly score distribution ────────────────────────────────────────────
    col_a, col_b = st.columns([2, 1])

    with col_a:
        st.markdown("### 🚨 Top 20 Anomalies by Severity")
        top_anomalies = (
            anomalies.nsmallest(20, "score")[["time", "power", "score", "hour"]]
            .rename(columns={"time": "Timestep", "power": "Power (kW)",
                             "score": "Anomaly Score", "hour": "Hour-of-Day"})
            .reset_index(drop=True)
        )
        top_anomalies.index += 1
        st.dataframe(top_anomalies.style.format({"Power (kW)": "{:.2f}", "Anomaly Score": "{:.4f}"}),
                     use_container_width=True)

    with col_b:
        st.markdown("### 📈 Stats")
        st.metric("Mean normal power",  f"{normal['power'].mean():.2f} kW")
        st.metric("Mean anomaly power", f"{anomalies['power'].mean():.2f} kW",
                  delta=f"+{anomalies['power'].mean() - normal['power'].mean():.2f} kW",
                  delta_color="inverse")
        st.metric("Max anomaly spike",  f"{anomalies['power'].max():.2f} kW")
        st.metric("Data generation",    f"{res['gen_ms']:.0f} ms")
        st.metric("ML inference",       f"{res['det_ms']:.0f} ms")

    st.markdown("---")
    st.markdown("### 🕐 Anomaly Frequency by Hour-of-Day")
    hourly = anomalies.groupby("hour").size().reset_index(name="count")
    st.bar_chart(hourly.set_index("hour"), color="#f85149")
else:
    # ── Empty state ───────────────────────────────────────────────────────────
 st.info("👈 Configure parameters in the sidebar and click **Run Detection** to start.")

st.markdown(
        """
        #### How it works
        1. **Data Ingestion** — Synthetic sine-wave power data is generated locally (simulates CSV ingest from edge sensors).
        2. **ML Detection** — `IsolationForest` isolates anomalous readings using CPU multi-core parallelism (`n_jobs=-1`).
        3. **Visualisation** — Streamlit renders the dashboard in real-time with zero cloud round-trips.

        #### Edge Computing Advantages
        | Phase | Task | Edge Computing Benefit |
        |---|---|---|
        | Ingestion | Local CSV processing | Zero egress/ingress cost |
        | Processing | Scikit-learn (CPU) | Multi-core CPU processing |
        | Visualisation | Streamlit | Real-time, no cloud lag |
        """
    )
