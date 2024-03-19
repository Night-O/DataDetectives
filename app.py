from flask import Flask, render_template, request, jsonify, redirect, url_for
import os
import fitz  # PyMuPDF
from transformers import BartForConditionalGeneration, BartTokenizer
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC

app = Flask(__name__)

os.environ['http_proxy'] = 'http://edcguest:edcguest@172.31.100.25:3128'
os.environ['https_proxy'] = 'http://edcguest:edcguest@172.31.100.25:3128'

model_name = "facebook/bart-large-cnn"
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)

svm_classifier = joblib.load('sentiment_classifier_model.pkl')
tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')

def summarize_health_report(report_text):
    inputs = tokenizer([report_text], max_length=1024, return_tensors='pt', truncation=True)
    summary_ids = model.generate(inputs.input_ids, num_beams=4, max_length=150, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

def extract_text_from_pdf(pdf_file_path):
    text = ""
    with fitz.open(pdf_file_path) as doc:
        for page in doc:
            text += page.get_text()
    return text

def perform_sentiment_analysis(text):
    text_tfidf = tfidf_vectorizer.transform([text])
    sentiment = svm_classifier.predict(text_tfidf)[0]
    return sentiment

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload-summary')
def upload_summary():
    return render_template('upload_summary.html')

@app.route('/provide-feedback')
def provide_feedback():
    return render_template('provide_feedback.html')

@app.route('/analyze-summary', methods=['POST'])
def analyze_summary():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    file_path = 'uploads/' + file.filename
    file.save(file_path)

    report_text = extract_text_from_pdf(file_path)
    summary = summarize_health_report(report_text)
    sentiment = perform_sentiment_analysis(summary)

    return render_template('summary_result.html', summary=summary, sentiment=sentiment)

@app.route('/submit-feedback', methods=['POST'])
def submit_feedback():
    feedback = request.form.get('feedback')
    # Perform processing for feedback
    # For demonstration, let's assume we analyze the sentiment of the feedback
    sentiment = perform_sentiment_analysis(feedback)
    return render_template('feedback_result.html', feedback=feedback, sentiment=sentiment)

if __name__ == '__main__':
    app.run(debug=True, port=5500)
