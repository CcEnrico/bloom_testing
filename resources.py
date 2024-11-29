import argparse
import faiss
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
import torch
import os

# File paths for saving/loading
INDEX_PATH = "document_index.faiss"
DOC_PATH = "documents.npy"
MODEL_PATH = "bloom_model"
TOKENIZER_PATH = "bloom_tokenizer"
EMBEDDING_MODEL_PATH = "sentence_transformer_model"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# Global variables to store loaded resources
tokenizer = None
model = None
device = None
index = None
documents = None
embedding_model = None


def prepare_resources(documents):
    """
    Prepares and saves resources for later use.
    
    Args:
        documents (list): A list of documents to index.
    """
    print("Preparing resources...")

    # Load and save BLOOM model and tokenizer
    print("Downloading BLOOM model and tokenizer...")
    model_name = "bigscience/bloom-560m"
    # model_name = "bigscience/bloom-1b1"
    # model_name = "bigscience/bloom-1b7"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(TOKENIZER_PATH)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
    model.save_pretrained(MODEL_PATH)

    # Initialize SentenceTransformer embedding model
    print("Downloading SentenceTransformer embedding model...")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    embedding_model.save(EMBEDDING_MODEL_PATH)

    # Generate embeddings for the documents
    print("Generating embeddings for documents...")
    document_embeddings = embedding_model.encode(documents, convert_to_numpy=True)

    # Initialize FAISS index and add document embeddings
    print("Loading FAISS index and adding embeddings...")
    embedding_dimension = document_embeddings.shape[1]  # e.g., 384 for "all-MiniLM-L6-v2"
    index = faiss.IndexFlatL2(embedding_dimension)
    index.add(document_embeddings)

    # Save FAISS index and documents
    faiss.write_index(index, INDEX_PATH)
    np.save(DOC_PATH, documents)

    print("Resources prepared successfully.")



def load_resources():
    global tokenizer, model, device, index, documents, embedding_model

    print("Loading resources...")

    # Load BLOOM model and tokenizer
    print("Loading BLOOM model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.float16)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    # Load FAISS index and documents
    print("Loading FAISS index and documents...")
    index = faiss.read_index(INDEX_PATH)
    documents = np.load(DOC_PATH, allow_pickle=True)

    # Load SentenceTransformer model
    print("Loading SentenceTransformer embedding model...")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_PATH)

    
    print("Resources loaded successfully.")
    print(f"Tokenizer: {tokenizer}")
    print(f"Model: {model}")
    print(f"Device: {device}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Device Name: {torch.cuda.get_device_name(0)}")
    

    return tokenizer, model, device, index, documents, embedding_model


def load_documents_from_txt(folder_path):
    """
    Loads text documents from a specified folder.

    Args:
        folder_path (str): Path to the folder containing .txt files.

    Returns:
        list: A list of document contents as strings.
    """
    documents = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".txt"):
            file_path = os.path.join(folder_path, file_name)
            with open(file_path, 'r', encoding='utf-8') as file:
                documents.append(file.read())
    return documents

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare or load resources.")
    parser.add_argument("--prepare", action="store_true", help="Prepare resources for Docker build.")
    args = parser.parse_args()

    if args.prepare:
        folder_path = "./docs"  
        # Load documents from .txt files
        print("Loading documents from .txt files...")
        documents = load_documents_from_txt(folder_path)
        if documents:
            print(f"Loaded {len(documents)} documents. ")
            # Print a preview of the first and last few characters of the first document
            preview_length = 100
            first_document = documents[0]
            preview_start = first_document[:preview_length]
            preview_end = first_document[-preview_length:]
            print(f"First document ({os.listdir(folder_path)[0]}) preview: ")
            print(f"{preview_start} ... {preview_end}")
            prepare_resources(documents)
        else:
            print("No .txt files found in the specified folder.")
    else:
        load_resources()
