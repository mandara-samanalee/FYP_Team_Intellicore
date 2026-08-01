import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="Score Fusion Layer", page_icon="🔗", layout="wide")

st.title("🔗 Score Fusion Layer")
st.caption("Combining text, network, and temporal signals into one campaign-level verdict")

W1, W2, W3, THRESHOLD = 0.85, 0.10, 0.05, 0.93

CASES = {
    "Ordinary reviewer": {"text": 0.05, "collusion": 0.15, "temporal": 0.55},
    "AI-assisted, but genuine (no coordination)": {"text": 0.95, "collusion": 0.20, "temporal": 0.60},
    "Coordinated campaign (easy, full overlap)": {"text": 0.98, "collusion": 1.00, "temporal": 0.90},
    "Coordinated campaign (subtle, partial overlap)": {"text": 1.00, "collusion": 0.47, "temporal": 0.80},
    "Custom (adjust sliders)": None,
}

tab1, tab2, tab3 = st.tabs(["🔴 Live Fusion Demo", "📊 Module Comparison", "📈 Detection Curve"])

with tab1:
    st.markdown("### Try a case")
    choice = st.selectbox("Select an example, or choose Custom to set your own values:", list(CASES.keys()))

    if CASES[choice] is None:
        c1, c2, c3 = st.columns(3)
        text_score = c1.slider("Text Score (M1)", 0.0, 1.0, 0.5, 0.01)
        collusion_score = c2.slider("Collusion Score (M2)", 0.0, 1.0, 0.3, 0.01)
        temporal_score = c3.slider("Temporal Score (M3)", 0.0, 1.0, 0.5, 0.01)
    else:
        vals = CASES[choice]
        text_score, collusion_score, temporal_score = vals["text"], vals["collusion"], vals["temporal"]
        c1, c2, c3 = st.columns(3)
        c1.metric("Text Score (M1)", f"{text_score:.2f}")
        c2.metric("Collusion Score (M2)", f"{collusion_score:.2f}")
        c3.metric("Temporal Score (M3)", f"{temporal_score:.2f}")

    fused = W1 * text_score + W2 * collusion_score + W3 * temporal_score
    flagged = fused > THRESHOLD

    st.markdown("---")
    st.markdown(f"**Fusion formula:** `{W1} × text + {W2} × collusion + {W3} × temporal`")

    fig = go.Figure(go.Bar(
        x=[text_score * W1, collusion_score * W2, temporal_score * W3],
        y=["Text contribution", "Network contribution", "Temporal contribution"],
        orientation="h", marker_color=["#2a78d6", "#6fa8dc", "#a4c2f4"],
    ))
    fig.add_vline(x=THRESHOLD, line_dash="dash", line_color="red",
                    annotation_text=f"Threshold = {THRESHOLD}")
    fig.update_layout(height=250, xaxis_range=[0, 1.05], title="Weighted contribution to fused score")
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns([1, 2])
    with col1:
        st.metric("Fused Score", f"{fused:.3f}")
    with col2:
        if flagged:
            st.error(f"🚩 FLAGGED — {fused:.3f} exceeds threshold ({THRESHOLD})")
        else:
            st.success(f"✅ CLEAR — {fused:.3f} below threshold ({THRESHOLD})")

with tab2:
    st.markdown("### Each module alone vs. the fused system")
    st.caption("Each module's own, fairly-measured native evaluation — not a distorted "
               "recomputation through the fusion layer's structure.")

    comp_df = pd.DataFrame({
        "Detector": ["M1 (text)\nalone", "M2 (network)\nalone", "M3 (temporal)\nalone", "Fused\nSystem"],
        "F1-Score": [0.9985, 0.8667, 0.6360, 0.9784],
    })
    fig = go.Figure(go.Bar(
        x=comp_df["Detector"], y=comp_df["F1-Score"],
        marker_color=["#b5d4f4", "#b5d4f4", "#b5d4f4", "#2a78d6"],
        text=[f"{v:.2%}" for v in comp_df["F1-Score"]], textposition="outside",
    ))
    fig.update_layout(yaxis_range=[0, 1.1], yaxis_title="F1-Score", height=400)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Final fusion test-set result")
    result_df = pd.DataFrame({
        "Metric": ["Precision", "Recall", "F1-Score", "AUC-ROC"],
        "Value": ["95.77%", "100.00%", "97.84%", "1.00"],
    })
    st.table(result_df)
    st.caption("Estimated ~3 false positives out of 37,077 real test reviewers (~0.008%).")

with tab3:
    st.markdown("### Weight tuning methodology")
    st.markdown("""
    A joint grid search over every (w1, w2, w3) combination summing to 1,
    crossed with a range of decision thresholds, was run on the validation
    split only. The combination maximising F1 was locked and applied to
    the held-out test split exactly once.
    """)
    st.write(f"**Locked weights:** Text = {W1}, Network = {W2}, Temporal = {W3}")
    st.write(f"**Locked threshold:** {THRESHOLD}")

    st.markdown("### Minimum-evidence gate investigation")
    gate_df = pd.DataFrame({
        "Collusion min.": [0.00, 0.30, 0.35, 0.40],
        "Temporal min.": [0.00, 0.60, 0.70, 0.75],
        "Validation F1": [0.9895, 0.9895, 0.9895, 0.9565],
        "Campaigns Caught": ["47/47", "47/47", "47/47", "44/47"],
    })
    st.table(gate_df)
    st.caption("No gate configuration removed the residual false positives without also "
               "losing genuine detections — confirmed as irreducible signal overlap, "
               "not a correctable structural gap. The un-gated configuration was retained.")
