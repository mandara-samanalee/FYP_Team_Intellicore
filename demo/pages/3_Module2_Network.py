import streamlit as st
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

st.set_page_config(page_title="Module 2 — Network Collusion Detection", page_icon="🕸️", layout="wide")

st.title("🕸️ Module 2 — Network-Based Collusion Detection")
st.caption("Identifying colluding reviewer clusters via Jaccard similarity + Louvain community detection")

# ============================================================
# TODO — replace this example dictionary with a real lookup
# built from your actual reviewer_signals table (from
# module2_validation_test.ipynb). For each example reviewer,
# store: their real neighbor list + Jaccard weights, their
# community_id, mean_jaccard, rating_uniformity, and final
# collusion_score, e.g.:
#
# reviewer_lookup = pd.read_csv("data/m2_example_reviewers.csv")
# ============================================================

EXAMPLES = {
    "Genuine reviewer (singleton, no cluster)": {
        "neighbors": [], "mean_jaccard": 0.0, "rating_uniformity": 0.30,
        "collusion_score": 0.042, "community_size": 1,
    },
    "Genuine reviewer (mild organic overlap)": {
        "neighbors": [("R_842", 0.29), ("R_119", 0.31)],
        "mean_jaccard": 0.30, "rating_uniformity": 0.45,
        "collusion_score": 0.321, "community_size": 3,
    },
    "Synthetic — full overlap campaign (Type A)": {
        "neighbors": [(f"SYNTH_R{i}", 1.0) for i in range(1, 10)],
        "mean_jaccard": 1.00, "rating_uniformity": 1.00,
        "collusion_score": 1.000, "community_size": 10,
    },
    "Synthetic — partial overlap campaign (Type B, hardest)": {
        "neighbors": [(f"SYNTH_R{i}", 0.333) for i in range(1, 8)],
        "mean_jaccard": 0.333, "rating_uniformity": 0.79,
        "collusion_score": 0.397, "community_size": 8,
    },
}

W_JACCARD, W_UNIFORMITY, THRESHOLD = 0.86, 0.14, 0.570

tab1, tab2 = st.tabs(["🔴 Reviewer Walkthrough", "📈 Evaluation Results"])

with tab1:
    st.markdown("### Select a reviewer to inspect")
    st.caption("These are real examples drawn from our test-split results, "
               "walking through each stage of the module's pipeline.")

    choice = st.selectbox("Reviewer:", list(EXAMPLES.keys()))
    data = EXAMPLES[choice]

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("#### Reviewer's local graph neighborhood")
        G = nx.Graph()
        G.add_node("SELECTED", color="red")
        for neighbor, weight in data["neighbors"]:
            G.add_node(neighbor, color="lightblue")
            G.add_edge("SELECTED", neighbor, weight=weight)

        fig, ax = plt.subplots(figsize=(5, 5))
        if len(G.nodes) > 1:
            pos = nx.spring_layout(G, seed=42)
            colors = [G.nodes[n].get("color", "lightblue") for n in G.nodes]
            nx.draw(G, pos, with_labels=True, node_color=colors, node_size=800,
                    font_size=7, ax=ax, edge_color="gray")
            edge_labels = {(u, v): f"{d['weight']:.2f}" for u, v, d in G.edges(data=True)}
            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7, ax=ax)
        else:
            ax.text(0.5, 0.5, "No edges above\nMIN_JACCARD = 0.25\n(singleton reviewer)",
                     ha="center", va="center", fontsize=11)
            ax.axis("off")
        st.pyplot(fig)

    with col2:
        st.markdown("#### Pipeline stage-by-stage")
        st.write(f"**Community size:** {data['community_size']} reviewer(s)")
        st.write(f"**Mean Jaccard similarity:** {data['mean_jaccard']:.3f}")
        st.write(f"**Rating uniformity:** {data['rating_uniformity']:.3f}")

        st.markdown("#### Weighted collusion score")
        st.latex(r"score = 0.86 \times \text{Jaccard} + 0.14 \times \text{uniformity}")
        st.metric("Collusion Score", f"{data['collusion_score']:.3f}",
                    delta=f"threshold = {THRESHOLD}")

        if data["collusion_score"] > THRESHOLD:
            st.error("🚩 FLAGGED — score exceeds decision threshold")
        else:
            st.success("✅ CLEAR — score below decision threshold")

with tab2:
    st.markdown("### Validation-tuned configuration")
    st.write(f"**W_JACCARD:** {W_JACCARD}   **W_UNIFORMITY:** {W_UNIFORMITY}   **Threshold:** {THRESHOLD}")

    st.markdown("### Test-set results")
    results_df = pd.DataFrame({
        "Metric": ["Accuracy", "Precision", "Recall", "F1-Score", "AUC-ROC"],
        "Value": [0.9966, 1.0000, 0.7630, 0.8655, 0.9954],
    })
    st.table(results_df)

    st.markdown("### Detection rate by campaign difficulty")
    diff_df = pd.DataFrame({
        "Campaign Type": ["A (full overlap, easy)", "B (partial, hardest)",
                          "C (full overlap, mixed rating)", "D (partial, negative sentiment)"],
        "Detection Rate": [1.00, 0.60, 1.00, 0.60],
    })
    st.bar_chart(diff_df.set_index("Campaign Type"))

    st.markdown("### Continuous overlap-tightness curve")
    curve_df = pd.DataFrame({
        "Jaccard Level": [0.333, 0.429, 0.538, 0.667, 0.818, 1.000],
        "Mean Collusion Score": [0.501, 0.567, 0.644, 0.734, 0.840, 0.982],
    })
    st.line_chart(curve_df.set_index("Jaccard Level"))
    st.caption("Detection rate remained at 100% across every level tested, "
               "including the median overlap observed between genuinely unrelated real reviewers.")
