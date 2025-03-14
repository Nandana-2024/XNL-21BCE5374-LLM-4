from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os
import logging
import pandas as pd
import faiss
import yfinance as yf
import ccxt
import torch
from sentence_transformers import SentenceTransformer
from transformers import CLIPProcessor, CLIPModel, BertForQuestionAnswering, BertTokenizer
import networkx as nx

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Set up logging
logging.basicConfig(level=logging.INFO)

# CSV File Path
CSV_FILE_PATH = "data/uploads"

# Load models
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
bert_model = BertForQuestionAnswering.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
bert_tokenizer = BertTokenizer.from_pretrained("bert-large-uncased")

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

@app.route('/')
def home():
    return render_template('index.html')

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

    if retrieved_idx >= 0 and retrieved_idx < len(qa_data):
        retrieved_answer = qa_data[retrieved_idx]['Answer']
        return jsonify({"query": query, "answer": retrieved_answer}), 200
    else:
        return jsonify({"query": query, "answer": "Could you please rephrase that?"}), 200

@app.route('/retrieve_image', methods=['POST'])
def retrieve_image():
    """Retrieve images based on text queries using CLIP."""
    data = request.json
    text_query = data.get('text')
    image_paths = data.get('image_paths')  # List of image file paths
    
    if not text_query or not image_paths:
        return jsonify({"error": "Invalid input."}), 400
    
    text_inputs = clip_processor(text=[text_query], return_tensors="pt", padding=True)
    text_features = clip_model.get_text_features(**text_inputs)
    
    image_scores = []
    for img_path in image_paths:
        image = clip_processor(images=[img_path], return_tensors="pt")
        image_features = clip_model.get_image_features(**image)
        score = torch.nn.functional.cosine_similarity(text_features, image_features)
        image_scores.append((img_path, score.item()))
    
    best_match = max(image_scores, key=lambda x: x[1])
    return jsonify({"best_match": best_match[0], "score": best_match[1]})

@app.route('/retrieve_graph', methods=['POST'])
def retrieve_graph():
    """Retrieve financial graph insights using network analysis."""
    data = request.json
    node = data.get('node')
    
    if not node:
        return jsonify({"error": "No node provided."}), 400
    
    # Example: Create a sample financial graph
    G = nx.Graph()
    G.add_edges_from([("StockA", "StockB"), ("StockB", "StockC"), ("StockA", "StockC"), ("StockC", "StockD")])
    
    if node not in G:
        return jsonify({"error": "Node not found in graph."}), 404
    
    neighbors = list(G.neighbors(node))
    return jsonify({"node": node, "connections": neighbors})

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
