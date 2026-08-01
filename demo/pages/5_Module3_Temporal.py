import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="Module 3 — Activity Pattern Analysis", page_icon="⏱️", layout="wide")

st.title("⏱️ Module 3 — Activity Pattern Analysis")
st.caption("Detecting coordinated posting bursts via per-product Isolation Forest anomaly detection")

# ============================================================
# TODO — replace this example dictionary with real timestamp
# sequences pulled from your actual df_combined_split.csv for
# a genuine product-window and a burst-affected product-window,
# e.g.:
#
# df = pd.read_csv("data/df_combined_split.csv")
# genuine_ts = df[df.product_id == "SOME_GENUINE_PRODUCT"]["timestamp"]
# burst_ts   = df[df.product_id == "SOME_BURST_PRODUCT"]["timestamp"]
# ============================================================

np.random.seed(1)
GENUINE_TIMES = sorted(np.cumsum(np.random.exponential(scale=200000, size=8)))  # spread over ~3 weeks
BURST_TIMES = sorted(np.concatenate([
    np.cumsum(np.random.exponential(scale=150000, size=4)),
    np.cumsum(np.random.uniform(60, 900, size=8)) + 300000,  # tight burst
]))

EXAMPLES = {
    "Genuine product (normal activity)": {
        "timestamps": GENUINE_TIMES, "review_count": 8, "mean_iat_hours": 55.2,
        "burstiness": -0.08, "anomaly_score": 0.62, "is_burst": False,
    },
    "Burst-affected product (coordinated campaign)": {
        "timestamps": BURST_TIMES, "review_count": 12, "mean_iat_hours": 3.1,
        "burstiness": 0.71, "anomaly_score": 0.91, "is_burst": True,
    },
}

THRESHOLD = -0.07  # native IsolationForest decision_function threshold, for reference only in this normalized demo

tab1, tab2 = st.tabs(["🔴 Product Timeline Walkthrough", "📈 Evaluation Results"])

with tab1:
    st.markdown("### Select a product window to inspect")
    choice = st.selectbox("Product:", list(EXAMPLES.keys()))
    data = EXAMPLES[choice]

    ts_hours = np.array(data["timestamps"]) / 3600.0

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ts_hours, y=[1] * len(ts_hours), mode="markers",
        marker=dict(size=14, color="crimson" if data["is_burst"] else "steelblue"),
        name="Reviews",
    ))
    fig.update_layout(
        title="Review posting timeline (hours since window start)",
        yaxis=dict(visible=False, range=[0.5, 1.5]),
        xaxis_title="Hours",
        height=250,
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Review Count", data["review_count"])
    col2.metric("Mean Inter-Arrival (hrs)", f"{data['mean_iat_hours']:.1f}")
    col3.metric("Burstiness Coefficient", f"{data['burstiness']:.2f}")
    col4.metric("Anomaly Score", f"{data['anomaly_score']:.2f}")

    if data["is_burst"]:
        st.error("🚩 FLAGGED — anomalous burst pattern detected relative to this product's own baseline")
    else:
        st.success("✅ CLEAR — activity consistent with genuine posting history")

with tab2:
    st.markdown("### Feature extraction pipeline")
    st.markdown("""
    - 30-day sliding window, 7-day step, per product
    - 5 features per window: review count, mean inter-arrival time,
      standard deviation of inter-arrival time, burstiness coefficient,
      shortest span across 5 reviews
    - Per-product Isolation Forest trained on that product's own genuine history
    """)

    st.markdown("### Validation-locked configuration")
    st.write(f"**Decision threshold (native scale):** {THRESHOLD}")

    st.markdown("### Test-set results")
    results_df = pd.DataFrame({
        "Metric": ["Accuracy", "Precision", "Recall", "F1-Score", "FP Rate"],
        "Value": [0.9650, 0.5810, 0.7030, 0.6360, 0.0230],
    })
    st.table(results_df)

    st.markdown("### Burst-speed diagnostic")
    speed_df = pd.DataFrame({
        "Group": ["Burst-affected products", "Genuine products"],
        "Median hours to reach 5 reviews": [1.5, 1133.0],
    })
    st.bar_chart(speed_df.set_index("Group"))
    st.caption("Burst-affected products reach 5 reviews over 700x faster than genuine products, "
               "confirming this feature separates cleanly.")
