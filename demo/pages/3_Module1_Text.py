import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Module 1 — Textual Analysis", page_icon="📝", layout="wide")

st.title("📝 Module 1 — Textual Analysis")
st.caption("Detecting LLM-generated review text via a fine-tuned DeBERTa-v3 classifier")

# ============================================================
# TODO — replace this mock scorer with your actual trained model.
# Load your DeBERTa-v3 checkpoint ONCE, cached, so it doesn't
# reload on every interaction:
#
# import torch
# from transformers import AutoTokenizer, AutoModelForSequenceClassification
#
# @st.cache_resource
# def load_model():
#     tokenizer = AutoTokenizer.from_pretrained("path/to/your/checkpoint")
#     model = AutoModelForSequenceClassification.from_pretrained("path/to/your/checkpoint")
#     model.eval()
#     return tokenizer, model
#
# tokenizer, model = load_model()
#
# def score_text(text):
#     inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
#     with torch.no_grad():
#         logits = model(**inputs).logits
#     prob = torch.softmax(logits, dim=1)[0][1].item()  # adjust index to your label mapping
#     return prob
# ============================================================

def score_text(text):
    """MOCK scorer — replace with real model call above before Monday."""
    # crude placeholder heuristic just so the UI is testable end-to-end
    ai_markers = ["overall", "furthermore", "in conclusion", "highly recommend",
                  "exceeded my expectations", "seamless", "state-of-the-art"]
    score = sum(marker in text.lower() for marker in ai_markers) / len(ai_markers)
    score = min(1.0, score + np.random.uniform(0.0, 0.15))
    return score

tab1, tab2 = st.tabs(["🔴 Live Demo", "📈 Evaluation Results"])

with tab1:
    st.markdown("### Try it yourself")
    st.markdown("Type or paste any electronics product review below to see a live text score.")

    example_choice = st.selectbox(
        "Or load an example:",
        ["(write your own)", "Genuine review example", "Synthetic (LLM-generated) example"],
    )

    examples = {
        "Genuine review example": "Bought this for my home office setup. Works fine, "
            "battery could be better but does the job. Had to return one because "
            "the connector was slightly loose.",
        "Synthetic (LLM-generated) example": "This product has truly exceeded my "
            "expectations in every way. The build quality is exceptional, and the "
            "performance is nothing short of outstanding. I highly recommend this "
            "to anyone looking for a seamless, state-of-the-art experience.",
    }

    default_text = examples.get(example_choice, "")
    review_text = st.text_area("Review text:", value=default_text, height=120)

    if st.button("Score this review", type="primary"):
        if review_text.strip():
            with st.spinner("Scoring..."):
                score = score_text(review_text)
            st.markdown("#### Result")
            c1, c2 = st.columns([1, 3])
            with c1:
                st.metric("Text Score", f"{score:.3f}")
            with c2:
                st.progress(score)
                if score > 0.5:
                    st.warning("⚠️ Likely LLM-generated")
                else:
                    st.success("✅ Likely genuine")
        else:
            st.error("Please enter some review text first.")

with tab2:
    st.markdown("### Validation-based model selection")
    st.markdown("""
    DeBERTa-v3-base and Binoculars were compared exclusively on the
    validation split; the higher-F1 configuration was selected as the
    module's sole production scorer and evaluated exactly once on test.
    """)

    results_df = pd.DataFrame({
        "Metric": ["Accuracy", "Precision", "Recall", "F1-Score", "AUC-ROC"],
        "DeBERTa-v3 (test)": [0.9985, 0.9970, 1.0000, 0.9985, 1.0000],
        "Binoculars (best variant)": [None, None, None, 0.6992, 0.7265],
    })
    st.table(results_df)

    st.markdown("### System-level separation (full 471,495-row dataset)")
    sep_df = pd.DataFrame({
        "Group": ["Genuine reviews", "Synthetic reviews (val)", "Synthetic reviews (test)"],
        "Mean text_score": [0.03, 0.92, 0.91],
    })
    st.bar_chart(sep_df.set_index("Group"))
