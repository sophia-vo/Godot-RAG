# build_db.py
import os
import shutil
import bs4
from bs4 import SoupStrainer

# --- LangChain specific imports ---
from langchain_community.document_loaders import DirectoryLoader, BSHTMLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# --- Configuration ---
# List of all directories to load documentation from
data_paths = ["classes/", "tutorials/", "getting_started/"]
# Path to the FAISS vector database
db_faiss_path = "vectorstore/"
# The embedding model to use
embedding_model_name = "sentence-transformers/all-mpnet-base-v2"


def build_vector_db():
    """
    Creates a new FAISS vector database from HTML documents.
    It will first remove any existing database to ensure a fresh build.
    """
    # Remove the old database to force a rebuild
    if os.path.exists(db_faiss_path):
        print(f"Removing existing database at '{db_faiss_path}'...")
        shutil.rmtree(db_faiss_path)
    
    print("Creating new vector database...")
    all_documents = []
    # We only want the main content of the Godot docs HTML files
    strainer = bs4.SoupStrainer('div', class_='document')

    # Loop through all specified data paths
    for path in data_paths:
        if not os.path.exists(path):
            print(f"Warning: Data path '{path}' not found. Skipping.")
            continue
        print(f"Loading documents from '{path}'...")
        loader = DirectoryLoader(
            path,
            glob="**/*.html",
            loader_cls=BSHTMLLoader,
            show_progress=True,
            use_multithreading=True,
            loader_kwargs={'bs_kwargs': {'parse_only': strainer}, 'get_text_separator': ' '}
        )
        documents = loader.load()
        all_documents.extend(documents)

    if not all_documents:
        print("Error: No documents found. Please add HTML files under the configured paths.")
        exit(1)

    print(f"Loaded a total of {len(all_documents)} documents from all sources.")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(all_documents)
    print(f"Split documents into {len(texts)} chunks.")

    print("Creating embeddings... (This may take a while)")
    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model_name, 
        model_kwargs={'device': 'cpu'}
    )

    db = FAISS.from_documents(texts, embeddings)
    db.save_local(db_faiss_path)
    print(f"SUCCESS: Vector database created and saved to '{db_faiss_path}'.")


if __name__ == "__main__":
    build_vector_db()