import streamlit as st
import numpy as np
import pickle
import json
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ---------------------------
# Page Config
# ---------------------------
st.set_page_config(
    page_title="Next Word Predictor",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 Next Word Prediction App")
st.markdown("Predict the next word using LSTM model")

# ---------------------------
# Load Model & Tokenizer
# ---------------------------

@st.cache_resource
def load_all():
    model = load_model("next_word_model.keras")

    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)

    with open("config.json", "r") as f:
        config = json.load(f)

    return model, tokenizer, config["max_seq_len"]

model, tokenizer, max_seq_len = load_all()

# ---------------------------
# Prediction Function
# ---------------------------

def predict_next_word(text):
    token_list = tokenizer.texts_to_sequences([text])[0]

    token_list = pad_sequences(
        [token_list],
        maxlen=max_seq_len - 1,
        padding='pre'
    )

    predicted = model.predict(token_list, verbose=0)
    predicted_index = np.argmax(predicted)

    for word, index in tokenizer.word_index.items():
        if index == predicted_index:
            return word

    return "Word not found"

# ---------------------------
# UI Input
# ---------------------------

user_input = st.text_input("Enter your sentence")

if st.button("Predict Next Word"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        next_word = predict_next_word(user_input)
        st.success(f"Predicted Next Word: **{next_word}**")