import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.feature_extraction.text import TfidfVectorizer

# 1. Page Configuration
st.set_page_config(
    page_title="SMS & Phishing Spam Detector",
    page_icon="🛡️",
    layout="centered"
)

st.title("🛡️ SMS / Email Spam & Phishing Detector")
st.markdown("""
This Deep Learning NLP application uses a **Multi-Layer Perceptron (Neural Network)** 
to detect spam, lottery scams, and phishing attempts in real time.
""")

# 2. Build and Train Deep Learning NLP Model
@st.cache_resource
def build_and_train_nlp_model():
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
    labels = np.array([1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)

    # Preprocess text into TF-IDF representation
    vectorizer = TfidfVectorizer(max_features=250, lowercase=True)
    X_train = vectorizer.fit_transform(corpus).toarray()

    # Deep Neural Network Architecture
    model = Sequential([
        Dense(32, input_shape=(X_train.shape[1],), activation='relu'),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train, labels, epochs=80, verbose=0)
    
    return model, vectorizer

with st.spinner("Training Deep Learning NLP Model..."):
    nlp_model, vectorizer = build_and_train_nlp_model()

# 3. Text Input Interface
st.subheader("Message Scanner")

if "user_text" not in st.session_state:
    st.session_state["user_text"] = "Congratulations! You have been selected to win a cash prize. Click the link to claim."

col1, col2 = st.columns(2)
with col1:
    if st.button("Load Legitimate Sample"):
        st.session_state["user_text"] = "Hey, are you free this evening to work on our presentation?"
with col2:
    if st.button("Load Phishing Scam Sample"):
        st.session_state["user_text"] = "URGENT: Your bank account is locked. Click here to verify credentials now."

user_input = st.text_area("Paste an email snippet or SMS text below:", value=st.session_state["user_text"], height=120)

# 4. Classification
if st.button("Analyze Text with Deep Learning", type="primary"):
    if user_input.strip() == "":
        st.warning("Please enter some text to analyze.")
    else:
        # Vectorize incoming raw text and perform model inference
        sample_vec = vectorizer.transform([user_input]).toarray()
        prob = float(nlp_model.predict(sample_vec)[0][0])
        confidence = prob * 100

        st.subheader("Analysis Verdict")
        if prob >= 0.50:
            st.error(f"🚨 **SPAM / PHISHING DETECTED** (Risk Score: `{confidence:.1f}%`)")
            st.progress(prob)
            st.write("⚠️ **Warning:** This message contains patterns common in credential harvesting or lottery scams.")
        else:
            st.success(f"✅ **LEGITIMATE (HAM)** (Spam Probability: `{confidence:.1f}%`)")
            st.progress(prob)
            st.write("✔ **Safe:** This message matches legitimate interpersonal conversational patterns.")
