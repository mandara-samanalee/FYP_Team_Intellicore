import streamlit as st
import pandas as pd

st.set_page_config(page_title="Overview", page_icon="📊", layout="wide")

st.title("📊 System Overview")

st.markdown("""
### Problem

Existing fake-review detectors typically examine a single dimension —
text, network, or timing — in isolation. This system combines all three
into one fused verdict, since a campaign coordinated in structure and
timing can evade detection even if its text alone passes undetected.
""")

st.markdown("### Architecture")
st.markdown("""
```
Amazon Electronics Reviews (raw dataset)
        |
Common Preprocessing & Train/Val/Test Split
        |
  ┌─────┴─────┬─────────────┐
Module 1   Module 2      Module 3
(Text)    (Network)    (Temporal)
  |           |             |
Text score  Collusion    Anomaly
per review  score/       score per
            reviewer     window
  └─────┬─────┴─────────────┘
        |
  Score Fusion Layer
  (weighted combination, validation-tuned)
        |
  Campaign-Level Detection Verdict
```
""")

st.markdown("### Dataset Summary")

col1, col2 = st.columns(2)
with col1:
    st.markdown("**Real data**")
    real_df = pd.DataFrame({
        "Metric": ["Total reviews", "Train split", "Validation split", "Test split"],
        "Value": ["463,495", "324,446", "69,524", "69,525"],
    })
    st.table(real_df)

with col2:
    st.markdown("**Synthetic campaign data**")
    synth_df = pd.DataFrame({
        "Metric": ["Campaigns", "Reviewers", "Rows", "Campaign types"],
        "Value": ["115", "922", "8,000", "A, B, C, D"],
    })
    st.table(synth_df)

st.markdown("### Final Combined Dataset")
st.code("df_combined_split.csv — 471,495 rows (463,495 real + 8,000 synthetic)\n"
        "Verified: 0 missing split labels, 0 synthetic rows in train")

st.markdown("### Final System Result")
result_df = pd.DataFrame({
    "Detector": ["Textual (M1) alone", "Network (M2) alone", "Temporal (M3) alone", "**Fused System**"],
    "Precision": ["99.70%", "100.00%", "58.10%", "**95.77%**"],
    "Recall": ["100.00%", "76.47%", "70.30%", "**100.00%**"],
    "F1-Score": ["99.85%", "86.67%", "63.60%", "**97.84%**"],
})
st.table(result_df)
