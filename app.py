import streamlit as st
import fitz  # PyMuPDF
import pickle
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

# Load saved models
with open('job_role_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('tfidf_vectorizer.pkl', 'rb') as f:
    tfidf = pickle.load(f)

with open('label_encoder.pkl', 'rb') as f:
    le = pickle.load(f)

# Text cleaning function
stop_words = set(stopwords.words('english'))

def clean_resume(text):
    text = re.sub(r'[^\w\s]', '', text)  # remove punctuation
    text = text.lower()
    text = re.sub(r'\d+', '', text)  # remove digits
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

# PDF text extractor
def extract_text_from_pdf(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# Streamlit UI
st.title("📄 Resume Job Role Predictor")

uploaded_file = st.file_uploader("Upload your resume (PDF)", type=["pdf"])

if uploaded_file is not None:
    resume_text = extract_text_from_pdf(uploaded_file)
    cleaned_text = clean_resume(resume_text)
    vectorized = tfidf.transform([cleaned_text]).toarray()
    prediction = model.predict(vectorized)
    predicted_role = le.inverse_transform(prediction)[0]
    
    st.success(f"Predicted Job Role: **{predicted_role}**")
