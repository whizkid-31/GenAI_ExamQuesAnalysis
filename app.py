import gradio as gr
import numpy as np
from scipy.sparse import hstack
import joblib
import re
import string

# -------- Load Saved Components --------

log_reg_text = joblib.load("log_reg_text.pkl")
log_reg_combined = joblib.load("log_reg_combined.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")
le = joblib.load("label_encoder.pkl")

# -------- Text Cleaning --------

def clean_text(text):
    text = text.lower()
    text = re.sub(f"[{string.punctuation}]", "", text)
    text = re.sub(r"\d+", "", text)
    return text

# -------- Prediction Functions --------

def predict_text_only(question):
    cleaned = clean_text(question)
    text_vector = tfidf.transform([cleaned])
    prediction = log_reg_text.predict(text_vector)
    return le.inverse_transform(prediction)[0]

def predict_combined(question, mean_score, pass_rate, std_dev, discrimination_index, question_length):
    cleaned = clean_text(question)
    text_vector = tfidf.transform([cleaned])
    stats_array = np.array([[mean_score, pass_rate, std_dev, discrimination_index, question_length]])
    combined_features = hstack([text_vector, stats_array])
    prediction = log_reg_combined.predict(combined_features)
    return le.inverse_transform(prediction)[0]

# -------- UI --------

with gr.Blocks() as demo:

    gr.Markdown("# AI-Based Question Difficulty Predictor")
    gr.Markdown("Choose prediction mode below.")

    with gr.Tabs():

        with gr.Tab("Advanced Mode (High Accuracy)"):
            question = gr.Textbox(label="Enter Question Text")
            mean_score = gr.Number(label="Mean Score")
            pass_rate = gr.Number(label="Pass Rate")
            std_dev = gr.Number(label="Standard Deviation")
            discrimination_index = gr.Number(label="Discrimination Index")
            question_length = gr.Number(label="Question Length")

            output_combined = gr.Textbox(label="Predicted Difficulty")
            btn2 = gr.Button("Predict Difficulty")

            btn2.click(
                predict_combined,
                inputs=[question, mean_score, pass_rate, std_dev, discrimination_index, question_length],
                outputs=output_combined
            )

        with gr.Tab("Quick Mode (Text Only)"):
            question_text = gr.Textbox(label="Enter Question Text")
            output_text = gr.Textbox(label="Predicted Difficulty")

            btn1 = gr.Button("Predict Difficulty")

            btn1.click(
                predict_text_only,
                inputs=question_text,
                outputs=output_text
            )

demo.launch()