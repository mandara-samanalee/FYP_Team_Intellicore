import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path

st.set_page_config(page_title="Score Fusion Layer", page_icon="🔗", layout="wide")

st.title("🔗 Score Fusion Layer")
st.caption("Combining text, network, and temporal signals into one campaign-level verdict")

DATA_DIR = Path(__file__).parent.parent / "data"
SCRIPTS_DIR = Path(__file__).parent.parent / "scripts"

W1, W2, W3, THRESHOLD = 0.85, 0.10, 0.05, 0.93


# ── Cached data load — runs ONCE, not on every button click ──
@st.cache_data
def load_sample():
    path = SCRIPTS_DIR / "df_fusion_scored_sample.csv"
    return pd.read_csv(path)


try:
    sample_df = load_sample()
    sample_df["fused_score"] = (
        W1 * sample_df["text_score"] + W2 * sample_df["collusion_score"]
        + W3 * sample_df["temporal_score"]
    )
    sample_df["verdict"] = sample_df["fused_score"].apply(
        lambda s: "🚩 FLAGGED" if s > THRESHOLD else "✅ CLEAR"
    )
    DATA_LOADED = True
except FileNotFoundError:
    DATA_LOADED = False

tab1, tab2, tab3, tab4 = st.tabs(
    ["📋 Sample Overview", "🔍 Case Walkthrough", "📊 Module Comparison", "📈 Tuning & Gate Analysis"]
)

# ============================================================
# TAB 1 — Summary table of the full curated sample
# ============================================================
with tab1:
    if not DATA_LOADED:
        st.error("df_fusion_scored_sample.csv not found in demo/scripts/. "
                 "Run the sample-selection code in your fusion notebook first.")
    else:
        st.markdown("### Curated evaluation sample")
        st.caption(
            f"A stratified sample of {len(sample_df)} rows drawn directly from "
            f"df_fusion_scored.csv — several genuine reviews spanning low-to-moderate "
            f"signal, plus multiple examples from each of the four campaign types "
            f"(A, B, C, D), covering the full documented difficulty range."
        )

        col1, col2, col3 = st.columns(3)
        col1.metric("Total rows", len(sample_df))
        col2.metric("Flagged", (sample_df["verdict"] == "🚩 FLAGGED").sum())
        col3.metric("Clear", (sample_df["verdict"] == "✅ CLEAR").sum())

        display_cols = ["reviewer_id", "product_id", "campaign_type",
                         "text_score", "collusion_score", "temporal_score",
                         "fused_score", "verdict"]
        st.dataframe(
            sample_df[display_cols].style.format({
                "text_score": "{:.3f}", "collusion_score": "{:.3f}",
                "temporal_score": "{:.3f}", "fused_score": "{:.3f}",
            }),
            use_container_width=True, height=400,
        )

        if "is_generated" in sample_df.columns:
            tp = ((sample_df["verdict"] == "🚩 FLAGGED") & (sample_df["is_generated"] == 1)).sum()
            fp = ((sample_df["verdict"] == "🚩 FLAGGED") & (sample_df["is_generated"] == 0)).sum()
            fn = ((sample_df["verdict"] == "✅ CLEAR") & (sample_df["is_generated"] == 1)).sum()
            tn = ((sample_df["verdict"] == "✅ CLEAR") & (sample_df["is_generated"] == 0)).sum()
            st.markdown("#### Live-computed result on this sample")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("True Positives", tp)
            c2.metric("True Negatives", tn)
            c3.metric("False Positives", fp)
            c4.metric("False Negatives", fn)

# ============================================================
# TAB 2 — Step-through walkthrough using session_state
# ============================================================
with tab2:
    if not DATA_LOADED:
        st.error("Sample data not loaded — see Tab 1.")
    else:
        st.markdown("### Walk through individual cases")
        st.caption("Use Previous / Next to narrate each case, or jump directly "
                   "using the dropdown.")

        # ── session_state persists the current index across reruns ──
        if "case_index" not in st.session_state:
            st.session_state.case_index = 0

        nav1, nav2, nav3 = st.columns([1, 3, 1])
        with nav1:
            if st.button("⬅ Previous", use_container_width=True):
                st.session_state.case_index = max(0, st.session_state.case_index - 1)
        with nav3:
            if st.button("Next ➡", use_container_width=True):
                st.session_state.case_index = min(
                    len(sample_df) - 1, st.session_state.case_index + 1
                )
        with nav2:
            jump_to = st.selectbox(
                "Or jump to a specific case:",
                options=list(range(len(sample_df))),
                index=st.session_state.case_index,
                format_func=lambda i: f"{i+1}. {sample_df.iloc[i]['reviewer_id']} "
                                        f"({sample_df.iloc[i]['campaign_type'] or 'genuine'})",
                key="jump_selector",
            )
            if jump_to != st.session_state.case_index:
                st.session_state.case_index = jump_to

        idx = st.session_state.case_index
        row = sample_df.iloc[idx]

        st.markdown(f"**Case {idx+1} of {len(sample_df)}**")
        st.progress((idx + 1) / len(sample_df))

        st.markdown("---")
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown(f"**Reviewer:** `{row['reviewer_id']}`   "
                        f"**Product:** `{row['product_id']}`   "
                        f"**Type:** {row['campaign_type'] or 'Genuine review'}")
            st.markdown(f"> {row['text']}")
        with col2:
            if row["verdict"] == "🚩 FLAGGED":
                st.error(f"### {row['verdict']}")
            else:
                st.success(f"### {row['verdict']}")

        fig = go.Figure(go.Bar(
            x=[row["text_score"] * W1, row["collusion_score"] * W2, row["temporal_score"] * W3],
            y=["Text (M1)", "Network (M2)", "Temporal (M3)"],
            orientation="h", marker_color=["#2a78d6", "#6fa8dc", "#a4c2f4"],
            text=[f"{row['text_score']:.2f} × {W1}", f"{row['collusion_score']:.2f} × {W2}",
                  f"{row['temporal_score']:.2f} × {W3}"],
            textposition="outside",
        ))
        fig.add_vline(x=THRESHOLD, line_dash="dash", line_color="red",
                        annotation_text=f"Threshold = {THRESHOLD}")
        fig.update_layout(height=250, xaxis_range=[0, 1.1],
                            title=f"Weighted contribution — fused score = {row['fused_score']:.3f}")
        st.plotly_chart(fig, use_container_width=True)

# ============================================================
# TAB 3 — Module comparison (unchanged from before)
# ============================================================
with tab3:
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

# ============================================================
# TAB 4 — Tuning methodology + gate analysis (unchanged from before)
# ============================================================
with tab4:
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
