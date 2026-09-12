# IMDB Movie Review Sentiment Classifier

A machine learning model that classifies movie reviews as positive or negative, built end-to-end from raw data to a deployed, interactive web app.

## 🎬 Live Demo
[Add your Streamlit Cloud link here once deployed]

## 📊 Project Overview
This project uses the IMDB Dataset of 50,000 movie reviews to train a sentiment classifier using classical NLP techniques — TF-IDF vectorization combined with Logistic Regression.

**Model Accuracy: 89.4%**

## 🔍 What This Project Covers
- **Exploratory Data Analysis (EDA):** dataset shape, class balance check (25,000 positive / 25,000 negative), review length distribution, null value check
- **Text Cleaning:** HTML tag removal (`<br />` artifacts common in this dataset), lowercasing
- **Feature Engineering:** TF-IDF vectorization (max 5,000 features)
- **Modeling:** Logistic Regression classifier
- **Evaluation:** Train/test split (80/20), accuracy scoring
- **Deployment:** Interactive Streamlit web app for real-time predictions

## 🛠️ Tech Stack
- Python
- Pandas, NumPy
- Scikit-learn (TF-IDF, Logistic Regression)
- Streamlit (web app)
- Joblib (model persistence)

## 📁 Project Structure
