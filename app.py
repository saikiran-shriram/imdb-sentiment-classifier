import streamlit as st
import joblib

# Load the saved model and vectorizer
model = joblib.load('models/sentiment_model.pkl')
tfidf = joblib.load('models/tfidf_vectorizer.pkl')

st.title("Movie Review Sentiment Classifier")
st.write("Enter a movie review below and the model will predict if it's positive or negative.")

review = st.text_area("Your review:", height=150)

if st.button("Predict Sentiment"):
    if review.strip() == "":
        st.warning("Please enter a review first.")
    else:
        review_tfidf = tfidf.transform([review])
        prediction = model.predict(review_tfidf)[0]
        
        if prediction == "positive":
            st.success(f"Prediction: **{prediction.upper()}** 🎬👍")
        else:
            st.error(f"Prediction: **{prediction.upper()}** 🎬👎")
