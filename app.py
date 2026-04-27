import streamlit as st
import joblib
import string
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))

model = joblib.load('model.pkl')
tfidf = joblib.load('tfidf.pkl')

def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return ' '.join(words)

st.set_page_config(page_title="News Credibility Analyzer", page_icon="🔍")
st.title("🔍 News Credibility Analyzer")
st.write("Paste a news article below to check its credibility.")

text_input = st.text_area("📰 Article Text", height=250, placeholder="Paste news article text here...")

if st.button("Analyze", use_container_width=True):
    if not text_input.strip():
        st.warning("Please enter some text.")
    else:
        cleaned = clean_text(text_input)
        features = tfidf.transform([cleaned])
        pred = model.predict(features)[0]
        prob = model.predict_proba(features)[0][pred] * 100

        if pred == 0:
            st.success(f"✅ Credible News — {prob:.1f}% confidence")
        else:
            st.error(f"⚠️ Potentially Fake News — {prob:.1f}% confidence")