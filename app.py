
import streamlit as st
import fitz  # PyMuPDF
import pickle
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

# ========================== STYLE SECTION ==========================
st.set_page_config(page_title="Resume Job Role Predictor", layout="centered")

st.markdown("""
    <style>
    .stApp {
        background-image: url("https://raw.githubusercontent.com/Saranya-Susarla/resume-predictor/main/kiwihug-3gifzboyZk0-unsplash.jpg");
        background-size: cover;
        background-attachment: fixed;
        background-repeat: no-repeat;
    }

    .main-title {
        color: #ffffff;
        text-align: center;
        font-size: 42px;
        font-weight: bold;
        margin-top: 20px;
        text-shadow: 2px 2px 6px #000000;
    }

    .subtext {
        color: #eeeeee;
        text-align: center;
        font-size: 18px;
        margin-bottom: 30px;
    }

    .stButton > button {
        background-color: #FF4B4B;
        color: white;
        padding: 10px 20px;
        border-radius: 10px;
        font-size: 16px;
        border: none;
    }

    .result-box {
        background-color: rgba(255, 255, 255, 0.1);
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        color: #ffffff;
        font-size: 22px;
        font-weight: bold;
        margin-top: 20px;
        box-shadow: 0 0 10px #00000066;
    }

    #MainMenu, footer, header {
        visibility: hidden;
    }
    </style>
""", unsafe_allow_html=True)

# ========================== TITLE ==========================
st.markdown('<div class="main-title">🎯 Resume Job Role Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="subtext">Upload your resume (PDF) and get the predicted job role</div>', unsafe_allow_html=True)

# ========================== MODEL LOADING ==========================
with open('job_role_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('tfidf_vectorizer.pkl', 'rb') as f:
    tfidf = pickle.load(f)

with open('label_encoder.pkl', 'rb') as f:
    le = pickle.load(f)

# ========================== HELPER FUNCTIONS ==========================
stop_words = set(stopwords.words('english'))

def clean_resume(text):
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = text.lower()
    text = re.sub(r'\d+', '', text)      # Remove numbers
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

def extract_text_from_pdf(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# ========================== FILE UPLOAD ==========================
uploaded_file = st.file_uploader("📄 Upload Resume (PDF only)", type=["pdf"])

if uploaded_file is not None:
    resume_text = extract_text_from_pdf(uploaded_file)
    cleaned_text = clean_resume(resume_text)
    vectorized = tfidf.transform([cleaned_text]).toarray()
    prediction = model.predict(vectorized)
    predicted_role = le.inverse_transform(prediction)[0]

    st.markdown(f'<div class="result-box">🔍 Predicted Job Role: <br> <span style="color:#00FFAA;">{predicted_role}</span></div>', unsafe_allow_html=True)
