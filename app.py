import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import pandas as pd
from transformers import pipeline
from flask import Flask, jsonify
from flask_cors import CORS


app = Flask(__name__)
CORS(app)

# Asegúrate de que los recursos estén descargados
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('vader_lexicon')

# Función para extracción de entidades clave
def extract_entities(text):
    tokens = nltk.word_tokenize(text)
    tags = nltk.pos_tag(tokens)
    entities = nltk.chunk.ne_chunk(tags)
    return entities

# Función para análisis de sentimiento
def analyze_sentiment(text):
    sia = SentimentIntensityAnalyzer()
    sentiment = sia.polarity_scores(text)
    return sentiment

# Función para generar un resumen automático usando sumy
def generate_summary(text, max_length=100, min_length=30):
    summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
    summary = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
    return summary[0]['summary_text']

@app.route('/analyze', methods=['GET'])
def analyze_texts():
    # Lee el dataset
    df = pd.read_csv('./Reviews.csv')
    
    # Selecciona los primeros 5 textos del DataFrame
    texts = df['Text'][:5]

    results = []

    for text in texts:
        entities = extract_entities(text)
        sentiment = analyze_sentiment(text)
        summary = generate_summary(text)
        
        results.append({
            "original": text,
            "entities": str(entities),
            "sentiment": sentiment,
            "summary": summary
        })

    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True)