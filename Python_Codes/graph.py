import os
import logging
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import numpy as np

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Set up logging
logging.basicConfig(level=logging.INFO)

# CSV File Path
CSV_FILE_PATH = "data/uploads"

@app.route('/')
def home():
    return render_template('index2.html')

@app.route('/csv-to-graph', methods=['POST'])
def csv_to_graph():
    """Handles the uploaded CSV and generates a graph."""
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file and file.filename.endswith('.csv'):
        try:
            # Read the CSV file into a pandas DataFrame
            df = pd.read_csv(file)
            df = df.dropna()

            # Check if required columns exist
            if 'CreditScore' not in df.columns or 'Age' not in df.columns:
                return jsonify({"error": "CSV must contain 'CreditScore' and 'Age' columns"}), 400

            # Plot the graph
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.scatter(df['Age'], df['CreditScore'], color='b', alpha=0.5)
            ax.set_title('Credit Score vs Age')
            ax.set_xlabel('Age')
            ax.set_ylabel('Credit Score')

            # Convert the plot to a PNG image and encode it in base64
            img = io.BytesIO()
            plt.savefig(img, format='png')
            img.seek(0)
            img_base64 = base64.b64encode(img.getvalue()).decode('utf-8')
            plt.close(fig)

            # Return the image as base64
            return jsonify({"image": img_base64}), 200

        except Exception as e:
            logging.error(f"Error processing the CSV file: {e}")
            return jsonify({"error": "Failed to process the file"}), 500
    else:
        return jsonify({"error": "Invalid file type. Only CSV files are allowed."}), 400

if __name__ == "__main__":
    app.run(debug=True, port=5001)

