from flask import Flask, request, jsonify, render_template
import os
import pandas as pd
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from werkzeug.utils import secure_filename
from haystack.document_stores import FAISSDocumentStore
from haystack.nodes import EmbeddingRetriever
from haystack.pipelines import ExtractiveQAPipeline
from haystack.nodes import FARMReader
from transformers import pipeline

app = Flask(__name__)

# Load models
csv_model = SentenceTransformer('all-MiniLM-L6-v2')
paraphrase_model = pipeline("text2text-generation", model="facebook/bart-large-cnn")

# Paths for uploads and FAISS indices
UPLOAD_FOLDER = "uploads"
FAISS_CSV_INDEX = "csv_index"
FAISS_PDF_INDEX = "pdf_index"

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(FAISS_CSV_INDEX, exist_ok=True)
os.makedirs(FAISS_PDF_INDEX, exist_ok=True)

# Initialize FAISS for CSV
csv_dim = 384  # Dimension of embeddings from MiniLM
csv_index = faiss.IndexFlatL2(csv_dim)
csv_data = []

# Initialize FAISS for PDF
pdf_store = FAISSDocumentStore(faiss_index_path=os.path.join(FAISS_PDF_INDEX, "faiss_index"))
pdf_retriever = EmbeddingRetriever(document_store=pdf_store, embedding_model="sentence-transformers/all-MiniLM-L6-v2")
pdf_reader = FARMReader("deepset/roberta-base-squad2")
pdf_pipeline = ExtractiveQAPipeline(reader=pdf_reader, retriever=pdf_retriever)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files['file']
    if not file:
        return jsonify({"error": "No file provided"})

    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    if filename.endswith('.csv'):
        process_csv(filepath)
    elif filename.endswith('.pdf'):
        process_pdf(filepath)
    else:
        return jsonify({"error": "Unsupported file type"})

    return jsonify({"message": "File uploaded and processed successfully"})


def process_csv(filepath):
    global csv_index, csv_data
    df = pd.read_csv(filepath)
    if 'question' not in df.columns or 'answer' not in df.columns:
        return jsonify({"error": "CSV must contain 'question' and 'answer' columns"})

    questions = df['question'].tolist()
    answers = df['answer'].tolist()
    embeddings = csv_model.encode(questions)

    for i, emb in enumerate(embeddings):
        csv_index.add(np.array([emb], dtype=np.float32))
        csv_data.append({"question": questions[i], "answer": answers[i]})


def process_pdf(filepath):
    from haystack.utils import convert_files_to_docs
    docs = convert_files_to_docs(dir_path=os.path.dirname(filepath))
    pdf_store.write_documents(docs)
    pdf_store.update_embeddings(pdf_retriever)

@app.route('/ask', methods=['POST'])
def ask_question():
    data = request.json
    query = data.get("question")
    source = data.get("source", "csv")

    if source == "csv":
        return jsonify(handle_csv_query(query))
    elif source == "pdf":
        return jsonify(handle_pdf_query(query))
    else:
        return jsonify({"error": "Invalid source. Use 'csv' or 'pdf'."})


def handle_csv_query(query):
    query_embedding = csv_model.encode([query])
    _, indices = csv_index.search(np.array(query_embedding, dtype=np.float32), 1)
    best_match = csv_data[indices[0][0]]
    paraphrased_response = paraphrase_model(best_match['answer'], max_length=50, do_sample=False)[0]['generated_text']
    return {"question": best_match['question'], "answer": paraphrased_response}


def handle_pdf_query(query):
    prediction = pdf_pipeline.run(query=query, params={"Retriever": {"top_k": 1}, "Reader": {"top_k": 1}})
    answer = prediction['answers'][0].answer if prediction['answers'] else "No answer found."
    paraphrased_response = paraphrase_model(answer, max_length=50, do_sample=False)[0]['generated_text']
    return {"answer": paraphrased_response}

if __name__ == '__main__':
    app.run(debug=True)