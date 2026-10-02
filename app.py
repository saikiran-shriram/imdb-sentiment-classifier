import streamlit as st
import joblib

# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="IMDb Sentiment Classifier",
    page_icon="🎬",
    layout="wide"
)

# --------------------------------------------------
# Load model and TF-IDF vectorizer
# --------------------------------------------------

model = joblib.load("models/sentiment_model.pkl")
tfidf = joblib.load("models/tfidf_vectorizer.pkl")

# --------------------------------------------------
# Custom CSS
# --------------------------------------------------

st.markdown("""
<style>

    /* Main page */
    .main {
        padding-top: 2rem;
    }

    /* Title */
    .main-title {
        font-size: 3rem;
        font-weight: 700;
        margin-bottom: 0.3rem;
    }

    .subtitle {
        font-size: 1.1rem;
        color: #777;
        margin-bottom: 1.5rem;
    }

    /* Section headings */
    .section-title {
        font-size: 1.5rem;
        font-weight: 600;
        margin-top: 1.5rem;
        margin-bottom: 0.8rem;
    }

    /* Result boxes */
    .positive-box {
        padding: 1.2rem;
        border-radius: 10px;
        background-color: #dff5e8;
        border-left: 6px solid #22a06b;
        margin-top: 1rem;
    }

    .negative-box {
        padding: 1.2rem;
        border-radius: 10px;
        background-color: #fde3e3;
        border-left: 6px solid #e5484d;
        margin-top: 1rem;
    }

    .result-title {
        font-size: 1.4rem;
        font-weight: 700;
    }

    .probability {
        font-size: 1.1rem;
        margin-top: 0.5rem;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #888;
        margin-top: 3rem;
        padding: 1.5rem;
        border-top: 1px solid #ddd;
    }

</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# Sidebar
# --------------------------------------------------

st.sidebar.title("📌 About")

st.sidebar.markdown("""
**IMDb Movie Review Sentiment Classifier**

**Features:** TF-IDF  
**Dataset:** IMDb Movie Reviews  
**Task:** Sentiment Classification  
**Accuracy:** 89.4%

---

### Model

The application converts the review into numerical TF-IDF features and passes them to the trained machine-learning model.

---

[View source on GitHub](https://github.com/saikiran-shriram/imdb-sentiment-classifier)
""")


# --------------------------------------------------
# Main Header
# --------------------------------------------------

st.markdown(
    '<div class="main-title">🎬 Movie Review Sentiment Classifier</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Analyze a movie review and predict whether its sentiment is positive or negative.'
    '</div>',
    unsafe_allow_html=True
)

st.info("📊 Model accuracy: **89.4%** on the IMDb test set")


# --------------------------------------------------
# Example Reviews
# --------------------------------------------------

st.markdown(
    '<div class="section-title">💡 Try These Examples</div>',
    unsafe_allow_html=True
)

example1 = st.button(
    "😊 Positive Example",
    use_container_width=True
)

example2 = st.button(
    "👎 Negative Example",
    use_container_width=True
)

example3 = st.button(
    "🎭 Mixed Example",
    use_container_width=True
)

if example1:
    st.session_state.review = (
        "This movie was absolutely brilliant. "
        "The acting was amazing and I loved every minute of it."
    )

if example2:
    st.session_state.review = (
        "This was one of the worst movies I have ever watched. "
        "The story was boring and the acting was disappointing."
    )

if example3:
    st.session_state.review = (
        "The acting was good and the visuals were impressive, "
        "but the story was predictable and the ending was disappointing."
    )


# --------------------------------------------------
# Review Input
# --------------------------------------------------

st.markdown(
    '<div class="section-title">🎥 Enter Your Movie Review</div>',
    unsafe_allow_html=True
)

if "review" not in st.session_state:
    st.session_state.review = ""

review = st.text_area(
    "Write your review below:",
    height=180,
    placeholder=(
        "Example: This movie was absolutely amazing. "
        "The story and acting were fantastic!"
    ),
    key="review"
)


# --------------------------------------------------
# Prediction
# --------------------------------------------------

if st.button("🔍 Analyze Sentiment", use_container_width=True):

    if review.strip() == "":
        st.warning("⚠️ Please enter a movie review first.")

    else:
        # Convert review into TF-IDF features
        review_tfidf = tfidf.transform([review])

        # Make prediction
        prediction = model.predict(review_tfidf)[0]

        # --------------------------------------------------
        # Positive prediction
        # --------------------------------------------------

        if prediction == "positive":

            probability = None

            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(review_tfidf)[0]
                classes = list(model.classes_)

                if "positive" in classes:
                    positive_index = classes.index("positive")
                    probability = probabilities[positive_index] * 100

            st.markdown(
                """
                <div class="positive-box">
                    <div class="result-title">
                        🟢 POSITIVE SENTIMENT
                    </div>
                    <div class="probability">
                        🎬 The model predicts that this review has a positive sentiment.
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            if probability is not None:
                st.metric(
                    "Positive Probability",
                    f"{probability:.2f}%"
                )

        # --------------------------------------------------
        # Negative prediction
        # --------------------------------------------------

        else:

            probability = None

            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(review_tfidf)[0]
                classes = list(model.classes_)

                if "negative" in classes:
                    negative_index = classes.index("negative")
                    probability = probabilities[negative_index] * 100

            st.markdown(
                """
                <div class="negative-box">
                    <div class="result-title">
                        🔴 NEGATIVE SENTIMENT
                    </div>
                    <div class="probability">
                        🎬 The model predicts that this review has a negative sentiment.
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            if probability is not None:
                st.metric(
                    "Negative Probability",
                    f"{probability:.2f}%"
                )


# --------------------------------------------------
# How It Works
# --------------------------------------------------

st.markdown("---")

st.markdown(
    '<div class="section-title">🧠 How It Works</div>',
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    ### 1️⃣ Review
    Enter a movie review into the application.
    """)

with col2:
    st.markdown("""
    ### 2️⃣ TF-IDF
    The review is converted into numerical TF-IDF features.
    """)

with col3:
    st.markdown("""
    ### 3️⃣ Prediction
    The trained ML model predicts positive or negative sentiment.
    """)


# --------------------------------------------------
# Footer
# --------------------------------------------------

st.markdown(
    """
    <div class="footer">
        🎬 IMDb Movie Review Sentiment Classifier<br>
        Developed & Designed by <b>Sai Shriram</b>
    </div>
    """,
    unsafe_allow_html=True
)
