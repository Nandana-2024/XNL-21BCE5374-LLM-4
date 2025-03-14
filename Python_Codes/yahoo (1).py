from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os
import logging
import pandas as pd
import faiss
import yfinance as yf
import ccxt
import spacy
import nltk
import pytesseract
from sentence_transformers import SentenceTransformer
from pdf2image import convert_from_path
from pyspark.sql import SparkSession
from openai import OpenAI

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Set up logging
logging.basicConfig(level=logging.INFO)

# Load NLP models
nlp = spacy.load("en_core_web_sm")
nltk.download("punkt")

# Initialize PySpark
spark = SparkSession.builder.appName("FinancialDataProcessor").getOrCreate()

# CSV File Path
CSV_FILE_PATH = "data/uploads"

# Load embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# FAISS index
index = None
qa_data = []
document_initialized = False

def process_csv():
    """Finds the latest CSV file and processes it."""
    global index, qa_data, document_initialized
    try:
        if not os.path.exists(CSV_FILE_PATH):
            logging.error("Uploads directory not found!")
            return

        # Find latest CSV file
        csv_files = [f for f in os.listdir(CSV_FILE_PATH) if f.endswith('.csv')]
        if not csv_files:
            logging.error("No CSV files found in data/uploads!")
            return

        latest_csv = max(csv_files, key=lambda f: os.path.getctime(os.path.join(CSV_FILE_PATH, f)))
        csv_path = os.path.join(CSV_FILE_PATH, latest_csv)

        df = pd.read_csv(csv_path)
        df = df.dropna()
        qa_data = df.to_dict(orient='records')
        questions = [row['Question'] for row in qa_data]
        embeddings = embedding_model.encode(questions)

        # Create FAISS index
        d = embeddings.shape[1]
        index = faiss.IndexFlatL2(d)
        index.add(embeddings)
        document_initialized = True

        logging.info(f"CSV processed: {latest_csv}, FAISS index created!")
    except Exception as e:
        logging.error(f"Error processing CSV: {e}")

process_csv()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    """Handles user queries with FAISS."""
    if not document_initialized:
        return jsonify({"error": "No document uploaded or processed yet."}), 400

    query = request.json.get('query')
    if not query:
        return jsonify({"error": "No query provided."}), 400

    query_embedding = embedding_model.encode([query])
    _, closest_idx = index.search(query_embedding, 1)
    retrieved_idx = closest_idx[0][0]

    if retrieved_idx >= 0 and retrieved_idx < len(qa_data):
        retrieved_answer = qa_data[retrieved_idx]['Answer']
        return jsonify({"query": query, "answer": retrieved_answer}), 200
    else:
        return jsonify({"query": query, "answer": "Could you please rephrase that?"}), 200

@app.route('/stock/<ticker>', methods=['GET'])
def get_stock_price(ticker):
    """Fetches real-time stock price from Yahoo Finance."""
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1d")
        if hist is not None and not hist.empty:
            price = hist['Close'].iloc[-1]
            return jsonify({"ticker": ticker.upper(), "price": round(price, 2)}), 200
        else:
            return jsonify({"error": f"Stock data for {ticker.upper()} not available."}), 400
    except Exception as e:
        logging.error(f"Error fetching stock price for {ticker.upper()}: {e}")
        return jsonify({"error": "Could not fetch stock price."}), 500

@app.route('/crypto/<symbol>', methods=['GET'])
def get_crypto_price(symbol):
    """Fetches real-time cryptocurrency price using Binance API."""
    try:
        exchange = ccxt.binance()
        ticker = exchange.fetch_ticker(f"{symbol.upper()}/USDT")
        return jsonify({"crypto": symbol.upper(), "price": round(ticker['last'], 4)}), 200
    except Exception as e:
        logging.error(f"Error fetching crypto price for {symbol.upper()}: {e}")
        return jsonify({"error": f"Could not fetch price for {symbol.upper()}"}), 500

@app.route('/ocr', methods=['POST'])
def process_image():
    """Extracts text from an uploaded image (financial reports, graphs)."""
    file = request.files.get('image')
    if not file:
        return jsonify({"error": "No file provided."}), 400
    
    text = pytesseract.image_to_string(file)
    return jsonify({"extracted_text": text}), 200

@app.route('/synthetic-data', methods=['POST'])
def generate_synthetic_data():
    """Generates synthetic financial data using GPT-3."""
    prompt = request.json.get("prompt")
    if not prompt:
        return jsonify({"error": "No prompt provided."}), 400

    response = OpenAI().completion.create(
        model="gpt-3.5-turbo",
        prompt=prompt,
        max_tokens=100
    )

    return jsonify({"generated_data": response['choices'][0]['text']}), 200

if __name__ == "__main__":
    app.run(debug=True, port=5001)
