import streamlit as st
import joblib
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np

# ----------------------------
# Load models
# ----------------------------
@st.cache_resource
def load_baseline():
    return joblib.load("baseline_tfidf_logreg.joblib")

@st.cache_resource
def load_distilbert():
    model_path = "distilbert_bbc_saved"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    return tokenizer, model

baseline_model = load_baseline()
tokenizer, distilbert_model = load_distilbert()

# ----------------------------
# Label mapping
# ----------------------------
label_list = ['business', 'entertainment', 'politics', 'sport', 'tech']
id_to_label = {i: label for i, label in enumerate(label_list)}

# ----------------------------
# Prediction functions
# ----------------------------
def predict_baseline(text):
    pred = baseline_model.predict([text])[0]
    proba = baseline_model.predict_proba([text])[0]
    return pred, max(proba)

def predict_distilbert(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=256)
    with torch.no_grad():
        outputs = distilbert_model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1).cpu().numpy()[0]
    pred_id = int(np.argmax(probs))
    return id_to_label[pred_id], float(probs[pred_id])

# ----------------------------
# Streamlit UI
# ----------------------------
st.title("BBC News Category Classifier")
st.write("Classify BBC news articles into Business, Entertainment, Politics, Sport, or Tech.")

model_choice = st.radio("Choose model:", ("Baseline TF-IDF + LogisticRegression", "DistilBERT"))

user_input = st.text_area("Enter your news text here:")

if st.button("Predict"):
    if not user_input.strip():
        st.warning("Please enter some text.")
    else:
        if model_choice == "Baseline TF-IDF + LogisticRegression":
            pred_label, confidence = predict_baseline(user_input)
        else:
            pred_label, confidence = predict_distilbert(user_input)

        st.markdown(f"**Predicted category:** {pred_label}")
        st.markdown(f"**Confidence:** {confidence:.2%}")
