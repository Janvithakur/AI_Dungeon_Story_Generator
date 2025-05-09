import streamlit as st
import pickle

# Load model and vectorizer
model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

# Streamlit app UI
st.title("📰 Fake News Detector")
st.write("Enter a news article text and check if it's fake or real.")

user_input = st.text_area("News Text")

if st.button("Classify"):
    if user_input:
        transformed_text = vectorizer.transform([user_input])
        prediction = model.predict(transformed_text)
        st.subheader(f"This news is **{prediction[0]}**.")
    else:
        st.warning("Please enter some news text to classify.")
