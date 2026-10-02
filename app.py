
import streamlit as st
import joblib

st.set_page_config(page_title="IMDb Sentiment Classifier", page_icon="🎬")

model = joblib.load("models/sentiment_model.pkl")
tfidf = joblib.load("models/tfidf_vectorizer.pkl")

st.markdown("""
<style>
.main-title {font-size: 2.8rem; font-weight: 700;}
.subtitle {color: #777; font-size: 1.1rem;}
.result {padding: 18px; border-radius: 10px; margin-top: 15px;}
.footer {text-align:center; color:#888; margin-top:40px; padding:20px;
         border-top:1px solid #ddd;}
</style>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("📌 About")
st.sidebar.markdown("""
**IMDb Movie Review Sentiment Classifier**

**Features:** TF-IDF  
**Dataset:** IMDb Movie Reviews  
**Task:** Sentiment Classification  
**Accuracy:** 89.4%

[View source on GitHub](https://github.com/saikiran-shriram/imdb-sentiment-classifier)
""")

# Header
st.markdown(
    '<div class="main-title">🎬 Movie Review Sentiment Classifier</div>',
    unsafe_allow_html=True
)
st.markdown(
    '<div class="subtitle">Analyze a movie review and predict whether its sentiment is positive or negative.</div>',
    unsafe_allow_html=True
)
st.info("📊 Model accuracy: **89.4%** on the IMDb test set")

# Examples
st.subheader("💡 Try These Examples")

examples = {
    "😊 Positive Example":
        "This movie was absolutely brilliant. The acting was amazing and I loved every minute of it.",
    "👎 Negative Example":
        "This was one of the worst movies I have ever watched. The story was boring and disappointing.",
    "🎭 Mixed Example":
        "The acting was good and the visuals were impressive, but the story was predictable."
}

for name, text in examples.items():
    if st.button(name, use_container_width=True):
        st.session_state.review = text

# Review
st.subheader("🎥 Enter Your Movie Review")

review = st.text_area(
    "Write your review below:",
    height=160,
    placeholder="Example: This movie was absolutely amazing!",
    key="review"
)

# Prediction
if st.button("🔍 Analyze Sentiment", use_container_width=True):

    if not review.strip():
        st.warning("⚠️ Please enter a movie review first.")
    else:
        features = tfidf.transform([review])
        prediction = model.predict(features)[0]

        probability = None
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(features)[0]
            probability = max(probabilities) * 100

        if prediction == "positive":
            st.markdown(
                '<div class="result" style="background:#dff5e8; border-left:6px solid #22a06b;">'
                '<b>🟢 POSITIVE SENTIMENT</b><br>'
                '🎬 The model predicts a positive sentiment.'
                '</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                '<div class="result" style="background:#fde3e3; border-left:6px solid #e5484d;">'
                '<b>🔴 NEGATIVE SENTIMENT</b><br>'
                '🎬 The model predicts a negative sentiment.'
                '</div>',
                unsafe_allow_html=True
            )

        if probability is not None:
            st.metric("Prediction Probability", f"{probability:.2f}%")

# How it works
st.markdown("---")
st.subheader("🧠 How It Works")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 1️⃣ Review\nEnter a movie review.")

with col2:
    st.markdown("### 2️⃣ TF-IDF\nConvert the review into numerical features.")

with col3:
    st.markdown("### 3️⃣ Prediction\nThe ML model predicts the sentiment.")

# Footer
st.markdown(
    '<div class="footer">🎬 IMDb Movie Review Sentiment Classifier<br>'
    'Developed & Designed by <b>Sai Shriram</b></div>',
    unsafe_allow_html=True
)
