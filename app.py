import sys
import os

# Make sure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
from pipeline import analyze_review

# ── Page config ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="Review Analyzer",
    page_icon="🔍",
    layout="centered"
)

# ── Header ────────────────────────────────────────────────────────────
st.title("Aspect-Based Review Analyzer")
st.markdown(
    "Paste any product review. See the **overall sentiment** and what "
    "the reviewer thinks about **specific product features**."
)
st.markdown("---")

# ── Example reviews the user can try ─────────────────────────────────
st.markdown("**Try an example:**")
col1, col2, col3 = st.columns(3)

example1 = "The battery life is terrible — dies in 2 hours. But the screen looks absolutely amazing. The price is reasonable for what you get."
example2 = "Absolutely love the camera quality! Photos are stunning. However the delivery took 2 weeks and packaging was terrible."
example3 = "Screen brightness is perfect outdoors. Battery charges quickly too. Great value for the price."

if col1.button("Mixed review"):
    st.session_state.review_text = example1
if col2.button("Camera + delivery"):
    st.session_state.review_text = example2
if col3.button("All positive"):
    st.session_state.review_text = example3

# ── Text input ────────────────────────────────────────────────────────
review = st.text_area(
    "Paste a product review here",
    value=st.session_state.get("review_text", ""),
    placeholder="e.g. The battery life is terrible but the screen looks amazing...",
    height=150
)

# ── Model selector ────────────────────────────────────────────────────
model_choice = st.radio(
    "Model",
    ["Upgrade : Sentence-Transformers + XGBoost",
     "Baseline : TF-IDF + Logistic Regression"],
    horizontal=True
)
use_upgrade = model_choice.startswith("Upgrade")

# ── Analyze button ────────────────────────────────────────────────────
if st.button("Analyze", type="primary", use_container_width=True):

    if not review.strip():
        st.warning("Please paste a review first.")

    else:
        with st.spinner("Analyzing..."):
            result = analyze_review(review, use_upgrade=use_upgrade)

        st.markdown("---")

        # ── Overall sentiment ─────────────────────────────────────────
        st.subheader("Overall Sentiment")

        sentiment = result['overall']['sentiment']
        confidence = result['overall']['confidence']

        color_map = {
            "positive": "#d4edda",
            "neutral":  "#e2e3e5",
            "negative": "#f8d7da"
        }
        text_color_map = {
            "positive": "#155724",
            "neutral":  "#383d41",
            "negative": "#721c24"
        }
        icon_map = {
            "positive": "✅ Positive",
            "neutral":  "➖ Neutral",
            "negative": "❌ Negative"
        }

        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown(
                f"""
                <div style="
                    background-color: {color_map[sentiment]};
                    color: {text_color_map[sentiment]};
                    padding: 16px;
                    border-radius: 10px;
                    text-align: center;
                    font-size: 20px;
                    font-weight: bold;
                ">
                    {icon_map[sentiment]}
                </div>
                """,
                unsafe_allow_html=True
            )

        with col_b:
            st.markdown(
                f"""
                <div style="
                    background-color: #f0f0f0;
                    color: #333;
                    padding: 16px;
                    border-radius: 10px;
                    text-align: center;
                    font-size: 20px;
                    font-weight: bold;
                ">
                    {confidence:.0%} confidence
                </div>
                """,
                unsafe_allow_html=True
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Aspect breakdown ──────────────────────────────────────────
        st.subheader("Aspect Breakdown")

        aspects = result['aspects']

        if not aspects:
            st.info(
                "No specific product aspects (battery, screen, camera, "
                "price, delivery) detected in this review."
            )
        else:
            # One badge per aspect
            cols = st.columns(len(aspects))

            for i, (aspect, data) in enumerate(aspects.items()):
                sent  = data['sentiment']
                conf  = data['confidence']
                bg    = color_map[sent]
                tc    = text_color_map[sent]
                icon  = icon_map[sent].split()[0]  # just the emoji

                with cols[i]:
                    st.markdown(
                        f"""
                        <div style="
                            background-color: {bg};
                            color: {tc};
                            border-radius: 10px;
                            padding: 14px 8px;
                            text-align: center;
                        ">
                            <div style="font-size:22px">{icon}</div>
                            <div style="font-weight:bold;font-size:15px;
                                        margin-top:4px">
                                {aspect.capitalize()}
                            </div>
                            <div style="font-size:12px;margin-top:2px">
                                {sent.capitalize()}
                            </div>
                            <div style="font-size:11px;opacity:0.8">
                                {conf:.0%} confidence
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

            # ── Evidence expanders ────────────────────────────────────
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("**Why these results?**")

            for aspect, data in aspects.items():
                with st.expander(
                    f"{aspect.capitalize()} — "
                    f"{data['sentiment']} ({data['confidence']:.0%})"
                ):
                    st.markdown(
                        f"The model classified this sentence as "
                        f"**{data['sentiment']}** for **{aspect}**:"
                    )
                    st.info(f'"{data["sentence"]}"')

        # ── Key insight callout ───────────────────────────────────────
        if aspects:
            st.markdown("---")
            st.markdown(
                "> **The key insight:** Overall sentiment can hide "
                "the real story. A 3-star review might contain a strong "
                "positive about the screen and a strong negative about "
                "the battery. Aspect level analysis reveals both."
            )