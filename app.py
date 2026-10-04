import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import TextVectorization, Dense, Dropout
from tensorflow.keras.models import Sequential

# 1. Page Configuration
st.set_page_config(
    page_title="SMS & Phishing Spam Detector",
    page_icon="🛡️",
    layout="centered"
)

st.title("🛡️ SMS / Email Spam & Phishing Detector")
st.markdown("""
This Deep Learning NLP application uses a **Neural Network (Sequential Dense architecture)** 
with text vectorization to detect spam, lottery scams, and phishing attempts in real time.
""")

# 2. Build and Train NLP Neural Network (Cached in memory)
@st.cache_resource
def build_and_train_nlp_model():
    # Representative SMS/Phishing dataset
    corpus = [
        "Congratulations! You won a $1,000 Walmart gift card. Click here to claim your prize now.",
        "URGENT: Your bank account is locked due to suspicious activity. Verify credentials at link.",
        "Win a brand new luxury car this weekend! Call this number immediately.",
        "FREE entry into our weekly draw. Text WIN to 80085 now to get cash instantly.",
        "Dear customer, your KYC verification is pending. Click the link to prevent account suspension.",
        "Exclusive offer: Get 90% discount on credit card interest rates today only.",
        "You have inherited 5 million dollars from an overseas estate. Reply with your passport details.",
        "Claim your prize money of 50000 rupees by entering your UPI pin.",
        "Hey, are we still meeting for lunch tomorrow at 1 PM?",
        "Can you send me the lecture notes for today's machine learning class?",
        "Please find attached the quarterly project review slides for our team meeting.",
        "Mom said dinner is ready, come down whenever you are done studying.",
        "Let's finalize the group assignment code tonight on GitHub.",
        "Your package with tracking number 89234 has been delivered to your front door.",
        "Hey, let me know when you reach home safely.",
        "Are you attending the seminar on deep learning this Friday afternoon?"
    ]
    # 1 = Spam / Phishing, 0 = Legitimate (Ham)
    labels = np.array([1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0])

    # Text Vectorization layer (Vocabulary size = 500 words, Sequence length = 30)
    vectorizer = TextVectorization(max_tokens=500, output_sequence_length=30)
    vectorizer.adapt(corpus)

    # Neural Network Architecture
    model = Sequential([
        vectorizer,
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(np.array(corpus), labels, epochs=80, verbose=0)
    return model

with st.spinner("Training Deep Learning NLP Model..."):
    nlp_model = build_and_train_nlp_model()

# 3. Input Text Box
st.subheader("Message Scanner")
default_text = "Congratulations! You have been selected to win a cash prize. Click the link to claim."
user_input = st.text_area("Paste an email snippet or SMS text below:", value=default_text, height=130)

# Quick sample buttons
col1, col2 = st.columns(2)
with col1:
    if st.button("Load Legitimate Sample"):
        st.session_state["sample"] = "Hey, are you free this evening to work on our presentation?"
with col2:
    if st.button("Load Phishing Scam Sample"):
        st.session_state["sample"] = "URGENT: Your bank account is locked. Click here to verify credentials now."

if "sample" in st.session_state:
    user_input = st.session_state["sample"]

# 4. Classification & Output
if st.button("Analyze Text with Deep Learning", type="primary"):
    if user_input.strip() == "":
        st.warning("Please enter some text to analyze.")
    else:
        with st.spinner("Running forward pass through neural network..."):
            prob = nlp_model.predict(np.array([user_input]))[0][0]
            confidence = prob * 100

        st.subheader("Analysis Verdict")
        if prob >= 0.50:
            st.error(f"🚨 **SPAM / PHISHING DETECTED** (Risk Score: `{confidence:.1f}%`)")
            st.progress(float(prob))
            st.write("⚠️ **Warning:** This message shows structural patterns common in credential harvesting, unsolicited marketing, or lottery scams.")
        else:
            st.success(f"✅ **LEGITIMATE (HAM)** (Spam Probability: `{confidence:.1f}%`)")
            st.progress(float(prob))
            st.write("✔️️ **Safe:** This message matches everyday interpersonal or official conversational patterns.")
