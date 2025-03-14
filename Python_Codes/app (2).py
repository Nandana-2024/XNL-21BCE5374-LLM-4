from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import logging
import pandas as pd
import faiss
import yfinance as yf
import ccxt
from sentence_transformers import SentenceTransformer

# Initialize Flask app
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:8080"}})  # Vue runs on port 8080

# Set up logging
logging.basicConfig(level=logging.INFO)

# CSV File Path
CSV_FILE_PATH = "data/uploads"

# Load embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# FAISS index
index = None
qa_data = []
document_initialized = False

def process_csv():
    """Finds the latest CSV file in data/uploads and processes it."""
    global index, qa_data, document_initialized
    try:
        if not os.path.exists(CSV_FILE_PATH):
            logging.error("Uploads directory not found!")
            return

        # Find the most recent CSV file
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
        d = embeddings.shape[1]  # Dimensionality
        index = faiss.IndexFlatL2(d)
        index.add(embeddings)
        document_initialized = True

        logging.info(f"CSV processed: {latest_csv}, FAISS index created successfully!")
    except Exception as e:
        logging.error(f"Error processing CSV: {e}")

# Process CSV on startup
process_csv()

@app.route('/ask', methods=['POST'])
def ask():
    """Handles user queries using FAISS."""
    if not document_initialized:
        return jsonify({"error": "No document uploaded or processed yet."}), 400

    query = request.json.get('query')
    if not query:
        return jsonify({"error": "No query provided."}), 400

    query_embedding = embedding_model.encode([query])
    _, closest_idx = index.search(query_embedding, 1)
    retrieved_idx = closest_idx[0][0]

    if 0 <= retrieved_idx < len(qa_data):
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
    """Fetches real-time cryptocurrency price using Binance API via CCXT."""
    try:
        exchange = ccxt.binance()
        ticker = exchange.fetch_ticker(f"{symbol.upper()}/USDT")
        return jsonify({"crypto": symbol.upper(), "price": round(ticker['last'], 4)}), 200
    except Exception as e:
        logging.error(f"Error fetching crypto price for {symbol.upper()}: {e}")
        return jsonify({"error": f"Could not fetch price for {symbol.upper()}"}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5001)

