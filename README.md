AI-Driven Educational Analytics System  
Milestone 1 – Question Difficulty Prediction using ML & NLP  

🔗 Live Hosted Application:
<https://huggingface.co/spaces/GeraAaryan/genai-edu-analytics>

---

📌 Project Overview

This project implements an AI-driven educational analytics system that predicts exam question difficulty using classical Machine Learning and Natural Language Processing techniques.

The system analyzes:

- Question text (semantic complexity)
- Student performance statistics (assessment analytics)

and classifies questions into:

- Easy  
- Medium  
- Hard  

This milestone focuses on predictive modeling, evaluation, and deployment.

---

🎯 Objective

To design and implement a machine learning pipeline that:

- Accepts exam question text and performance metrics
- Applies NLP preprocessing and feature extraction
- Predicts question difficulty level
- Provides an interactive user interface
- Demonstrates comparative model analysis

---

🧠 Models Implemented

We implemented and compared multiple models:

1. **Logistic Regression (Text Only)**
2. **Logistic Regression (Statistical Features Only)**
3. **Logistic Regression (Combined Features)**
4. **Decision Tree (Combined Features)**

---

📊 Model Performance Comparison

| Model | Accuracy |
|--------|----------|
| Logistic Regression (Text Only) | 61.9% |
| Logistic Regression (Stats Only) | 67% |
| Logistic Regression (Combined) | 93.5% |
| Decision Tree (Combined) | 92% |

### Key Insight:
The hybrid model (text + statistical features) significantly outperformed individual feature models, demonstrating the importance of integrating NLP with educational performance analytics.

---

🔍 Feature Engineering

Text Features
- Lowercasing
- Punctuation removal
- TF-IDF vectorization

Statistical Features
- Mean Score
- Pass Rate
- Standard Deviation
- Discrimination Index
- Question Length

Hybrid feature stacking was performed using sparse matrix combination.

---

📈 Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix
- Bias-Variance Discussion

---

🖥 User Interface

The deployed system supports two modes:

🔹 Advanced Mode (High Accuracy)
Uses both text and statistical features for optimal prediction performance.

🔹 Quick Mode (Text Only)
Uses semantic features only for accessibility when statistics are unavailable.

A flagging mechanism is included to support human-in-the-loop review and iterative improvement.

---

🛠 Tech Stack

- Python
- Scikit-learn
- TF-IDF (NLP)
- Gradio
- Hugging Face Spaces
- Git & GitHub

---

🚀 Deployment

The system is permanently hosted using Hugging Face Spaces with Gradio SDK.

It loads pre-trained models (`.pkl` files) for efficient runtime prediction.

---

🔮 Future Scope (Milestone 2)

In the next phase, the system will evolve into an agent-based AI assistant capable of:

- Autonomous assessment quality analysis
- Retrieval of pedagogical best practices
- Structured question improvement suggestions
- Intelligent reasoning about learning gaps

This transforms the system from a predictive model into an intelligent assessment design assistant.

---

👨‍💻 Authors:

- Aaryan Gera (Team Lead)
- Chaitanya Kumar
- Pranav Sehgal

B.Tech – CS & AI
Mid-Sem Capstone Project
