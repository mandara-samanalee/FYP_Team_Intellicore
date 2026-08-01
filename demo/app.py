import streamlit as st

st.set_page_config(
    page_title="ReviewRadar — Demo",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Simple, professional styling ─────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background-color: #f5f5f5;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
    h1, h2, h3 { font-family: 'Georgia', serif; }
</style>
""", unsafe_allow_html=True)

st.title("🛡️ ReviewRadar")
st.subheader("Multi-Dimensional Detection of Coordinated LLM-Generated Review Campaigns")
st.caption("Team Intellicore — Final Evaluation Demo")

st.markdown("---")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Fused System F1", "97.84%")
with col2:
    st.metric("Precision", "95.77%")
with col3:
    st.metric("Recall", "100.00%")
with col4:
    st.metric("AUC-ROC", "1.00")

st.markdown("---")

st.markdown("""
### Navigation

Use the sidebar to explore each component of the system:

- **📊 Overview** — system architecture and dataset summary
- **📝 Module 1 — Textual Analysis** — live text scoring demo
- **🕸️ Module 2 — Network Collusion Detection** — reviewer graph walkthrough
- **⏱️ Module 3 — Activity Pattern Analysis** — temporal burst walkthrough
- **🔗 Score Fusion Layer** — combined system demo and comparison

This app presents both the final evaluation results and interactive
walkthroughs of each detection module's pipeline.
""")

st.info(
    "💡 For a live text-scoring demo, go to **Module 1** in the sidebar "
    "and type or paste any review text."
)
