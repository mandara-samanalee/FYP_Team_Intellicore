import streamlit as st
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path

st.set_page_config(page_title="Module 2 — Network Collusion Detection", page_icon="🕸️", layout="wide")

st.title("🕸️ Module 2 — Network-Based Collusion Detection")
st.caption("Identifying colluding reviewer clusters via Jaccard similarity + Louvain community detection")

DATA_DIR = Path(__file__).parent.parent / "data"

# ── Locked configuration from module2_validation_test.ipynb ──
W_JACCARD, W_UNIFORMITY, THRESHOLD = 0.86, 0.14, 0.570


@st.cache_data
def load_data():
    reviewers = pd.read_csv(DATA_DIR / "m2_reviewer_details.csv")
    edges = pd.read_csv(DATA_DIR / "m2_graph_edges.csv")
    sizes = pd.read_csv(DATA_DIR / "m2_community_sizes.csv")
    return reviewers, edges, sizes


try:
    reviewers_df, edges_df, sizes_df = load_data()
    DATA_LOADED = True
except FileNotFoundError:
    DATA_LOADED = False

tab1, tab2 = st.tabs(["🔴 Reviewer Walkthrough", "📈 Evaluation Results"])

with tab1:
    if not DATA_LOADED:
        st.error(
            "Data files not found. Please export m2_reviewer_details.csv, "
            "m2_graph_edges.csv, and m2_community_sizes.csv from your notebook "
            "into demo/data/ (see the export cell provided)."
        )
    else:
        st.markdown("### Select a real reviewer to inspect")

        col_filter1, col_filter2 = st.columns(2)
        with col_filter1:
            group_filter = st.radio(
                "Show:", ["All", "Real reviewers only", "Synthetic (campaign) reviewers only"],
                horizontal=True,
            )
        with col_filter2:
            search = st.text_input("Search reviewer ID (optional):", "")

        filtered = reviewers_df.copy()
        if group_filter == "Real reviewers only":
            filtered = filtered[filtered["is_generated"] == 0]
        elif group_filter == "Synthetic (campaign) reviewers only":
            filtered = filtered[filtered["is_generated"] == 1]
        if search:
            filtered = filtered[filtered["reviewer_id"].str.contains(search, case=False, na=False)]

        if len(filtered) == 0:
            st.warning("No reviewers match this filter/search.")
        else:
            selected_id = st.selectbox(
                f"Reviewer ({len(filtered):,} matching):", filtered["reviewer_id"].tolist()
            )
            row = reviewers_df[reviewers_df["reviewer_id"] == selected_id].iloc[0]

            # find this reviewer's real edges from the actual graph export
            neighbor_edges = edges_df[
                (edges_df["reviewer_1"] == selected_id) | (edges_df["reviewer_2"] == selected_id)
            ]

            col1, col2 = st.columns([1, 1])

            with col1:
                st.markdown("#### Real graph neighborhood")
                G = nx.Graph()
                G.add_node(selected_id, color="red")
                for _, e in neighbor_edges.iterrows():
                    other = e["reviewer_2"] if e["reviewer_1"] == selected_id else e["reviewer_1"]
                    G.add_node(other, color="lightblue")
                    G.add_edge(selected_id, other, weight=e["jaccard_weight"])

                fig, ax = plt.subplots(figsize=(5, 5))
                if len(G.nodes) > 1:
                    pos = nx.spring_layout(G, seed=42)
                    colors = [G.nodes[n].get("color", "lightblue") for n in G.nodes]
                    nx.draw(G, pos, with_labels=True, node_color=colors, node_size=700,
                            font_size=6, ax=ax, edge_color="gray")
                    edge_labels = {(u, v): f"{d['weight']:.2f}" for u, v, d in G.edges(data=True)}
                    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=6, ax=ax)
                else:
                    ax.text(0.5, 0.5, "No edges above\nMIN_JACCARD = 0.25\n(singleton reviewer)",
                             ha="center", va="center", fontsize=11)
                    ax.axis("off")
                st.pyplot(fig)

            with col2:
                st.markdown("#### Pipeline stage-by-stage (real values)")
                community_size = sizes_df[
                    sizes_df["community_id"] == row["community_id"]
                ]["community_size"].values
                community_size = int(community_size[0]) if len(community_size) else 1

                st.write(f"**Reviewer type:** "
                         f"{'Synthetic campaign member' if row['is_generated']==1 else 'Real reviewer'}")
                st.write(f"**Community size:** {community_size} reviewer(s)")
                st.write(f"**Mean Jaccard similarity:** {row['mean_jaccard']:.3f}")
                st.write(f"**Rating uniformity:** {row['rating_uniformity']:.3f}")

                st.markdown("#### Weighted collusion score")
                st.latex(rf"score = {W_JACCARD} \times \text{{Jaccard}} + {W_UNIFORMITY} \times \text{{uniformity}}")
                st.metric("Collusion Score", f"{row['collusion_score']:.3f}",
                            delta=f"threshold = {THRESHOLD}")

                if row["collusion_score"] > THRESHOLD:
                    st.error("🚩 FLAGGED — score exceeds decision threshold")
                else:
                    st.success("✅ CLEAR — score below decision threshold")

with tab2:
    st.markdown("### Validation-tuned configuration")
    st.write(f"**W_JACCARD:** {W_JACCARD}   **W_UNIFORMITY:** {W_UNIFORMITY}   **Threshold:** {THRESHOLD}")

    results_path = DATA_DIR / "m2_final_validation_test_results.csv"
    if results_path.exists():
        results_df = pd.read_csv(results_path)
        st.markdown("### Test-set results (loaded from your saved evaluation)")
        st.table(results_df.T.reset_index().rename(columns={"index": "Metric", 0: "Value"}))
    else:
        st.info("m2_final_validation_test_results.csv not found in demo/data/ — "
                "showing last-known reported values instead.")
        results_df = pd.DataFrame({
            "Metric": ["Accuracy", "Precision", "Recall", "F1-Score", "AUC-ROC"],
            "Value": [0.9966, 1.0000, 0.7630, 0.8655, 0.9954],
        })
        st.table(results_df)

    if DATA_LOADED:
        st.markdown("### Live distribution — real vs synthetic collusion scores")
        st.caption("Computed live from your actual exported reviewer data.")
        import plotly.graph_objects as go
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=reviewers_df[reviewers_df["is_generated"] == 0]["collusion_score"],
            name="Real reviewers", opacity=0.6, nbinsx=30,
        ))
        fig.add_trace(go.Histogram(
            x=reviewers_df[reviewers_df["is_generated"] == 1]["collusion_score"],
            name="Synthetic reviewers", opacity=0.6, nbinsx=30,
        ))
        fig.add_vline(x=THRESHOLD, line_dash="dash", line_color="red",
                        annotation_text=f"Threshold = {THRESHOLD}")
        fig.update_layout(barmode="overlay", xaxis_title="Collusion Score", height=350)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.markdown("### Continuous detection curve — overlap tightness (0.333 to 1.0)")
        st.caption("Detection strength as network overlap between colluding reviewers "
                   "becomes progressively more realistic and harder to distinguish from "
                   "coincidental organic overlap.")

        synth = reviewers_df[reviewers_df["is_generated"] == 1].dropna(subset=["target_jaccard"])
        if len(synth) > 0:
            curve = synth.groupby("target_jaccard")["collusion_score"].agg(
                mean_score="mean", min_score="min", n="count"
            ).reset_index()
            curve["detection_rate"] = synth.groupby("target_jaccard")["collusion_score"].apply(
                lambda s: (s > THRESHOLD).mean()
            ).values

            c1, c2 = st.columns(2)
            with c1:
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(x=curve["target_jaccard"], y=curve["mean_score"],
                                             mode="lines+markers", name="Mean score", line=dict(color="#2a78d6")))
                fig2.add_trace(go.Scatter(x=curve["target_jaccard"], y=curve["min_score"],
                                             mode="lines+markers", name="Minimum score",
                                             line=dict(color="orange", dash="dash")))
                fig2.add_hline(y=THRESHOLD, line_dash="dot", line_color="red",
                                 annotation_text=f"Threshold = {THRESHOLD}")
                fig2.update_layout(title="Collusion score vs Jaccard overlap",
                                     xaxis_title="Jaccard (overlap tightness)",
                                     yaxis_title="Collusion score", height=350)
                st.plotly_chart(fig2, use_container_width=True)
            with c2:
                fig3 = go.Figure(go.Scatter(x=curve["target_jaccard"], y=curve["detection_rate"],
                                               mode="lines+markers", line=dict(color="#1baf7a")))
                fig3.update_layout(title="Detection rate vs Jaccard overlap",
                                     xaxis_title="Jaccard (overlap tightness)",
                                     yaxis_title="Detection rate", yaxis_range=[0, 1.05], height=350)
                st.plotly_chart(fig3, use_container_width=True)

            st.caption(
                "The lowest tested level (0.333) is approximately the MEDIAN Jaccard "
                "weight observed between genuinely unrelated real reviewers — the "
                "hardest, most realistic boundary case, not an easy strawman."
            )
        else:
            st.info("No target_jaccard data found — re-export from your notebook "
                    "with the extended export cell to enable this chart.")

        st.markdown("---")
        st.markdown("### Detection rate by campaign type")
        if "campaign_type" in synth.columns and synth["campaign_type"].notna().any():
            type_detect = synth.dropna(subset=["campaign_type"]).groupby("campaign_type")["collusion_score"].apply(
                lambda s: (s > THRESHOLD).mean()
            ).reset_index()
            type_detect.columns = ["Campaign Type", "Detection Rate"]
            fig4 = go.Figure(go.Bar(x=type_detect["Campaign Type"], y=type_detect["Detection Rate"],
                                       marker_color="#2a78d6",
                                       text=[f"{v:.0%}" for v in type_detect["Detection Rate"]],
                                       textposition="outside"))
            fig4.update_layout(yaxis_range=[0, 1.15], yaxis_title="Detection Rate", height=300)
            st.plotly_chart(fig4, use_container_width=True)

        st.markdown("---")
        st.markdown("### Community cohesion verification")
        st.caption("Does Louvain correctly group every campaign's reviewers into ONE "
                   "community, even at the lowest, hardest overlap levels?")
        cohesion_path = DATA_DIR / "m2_cohesion_check.csv"
        if cohesion_path.exists():
            cohesion_df = pd.read_csv(cohesion_path)
            cohesion_by_level = cohesion_df.groupby("target_jaccard").agg(
                n_campaigns=("campaign_id", "count"),
                pct_cohesive=("cohesive", lambda x: x.mean() * 100),
            ).reset_index()
            st.table(cohesion_by_level)
            overall = cohesion_df["cohesive"].mean() * 100
            fragmented = (~cohesion_df["cohesive"]).sum()
            if overall == 100:
                st.success(f"✅ 100% cohesion across all {len(cohesion_df)} campaigns checked "
                           f"— 0 fragmented campaigns at any overlap level.")
            else:
                st.warning(f"{overall:.1f}% cohesive — {fragmented} fragmented campaign(s) found.")
        else:
            st.info("m2_cohesion_check.csv not found — run the cohesion export cell to enable this.")
