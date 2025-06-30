import os
import shutil
from pathlib import Path
import bs4
from bs4 import SoupStrainer

# --- LangChain specific imports ---
from langchain_community.document_loaders import DirectoryLoader, BSHTMLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.llms import CTransformers

# --- Configuration ---
# List of all directories to load documentation from
data_paths = ["classes/", "tutorials/"]
# Path to the FAISS vector database
db_faiss_path = "vectorstore/"
# Path to your local LLM model
model_file = "models/DeepSeek-R1-0528-UD-TQ1_0.gguf"
# The embedding model to use
embedding_model_name = "sentence-transformers/all-mpnet-base-v2"


# --- Core RAG Functions ---

def create_vector_db():
    """
    Creates or loads the FAISS vector database from HTML documents
    in all specified data_paths.
    """
    # Check if the vector store already exists
    if os.path.exists(db_faiss_path) and os.listdir(db_faiss_path):
        print(f"Loading existing vector database from {db_faiss_path}...")
        embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model_name, 
            model_kwargs={'device': 'cpu'}
        )
        db = FAISS.load_local(db_faiss_path, embeddings, allow_dangerous_deserialization=True)
        print("Vector database loaded successfully.")
        return db

    # If not, create it
    print("Creating new vector database...")
    all_documents = []
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
        print("No documents found. Please add HTML files under the configured paths.")
        exit(1)

    print(f"Loaded a total of {len(all_documents)} documents from all sources.")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(all_documents)
    print(f"Split documents into {len(texts)} chunks.")

    print("Creating embeddings... (This may take a while on the first run)")
    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model_name, 
        model_kwargs={'device': 'cpu'}
    )

    db = FAISS.from_documents(texts, embeddings)
    db.save_local(db_faiss_path)
    print(f"Vector database created and saved to {db_faiss_path}.")
    return db


def create_rag_chain():
    """
    Creates the complete RAG chain with the LLM, retriever, and prompt.
    """
    db = create_vector_db()
    retriever = db.as_retriever(search_kwargs={'k': 3})

    prompt_template = """
    You are a helpful assistant **trained on Godot 4.4.1-stable documentation**.
    Use the following pieces of context—*all of which come from the Godot 4.4.1-stable manual*—to answer the question at the end.
    If you don’t know the answer, say you don’t know; don’t hallucinate.

    Context: {context}

    Question: {question}
    Helpful Answer:
    """

    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain_type_kwargs = {"prompt": PROMPT}

    print("Loading local LLM...")
    if not os.path.exists(model_file):
        print(f"Model file not found at {model_file}. Please download it and place it in the 'models' directory.")
        exit(1)

    llm = CTransformers(
        model=model_file,
        model_type='llama',
        config={'max_new_tokens': 1024, 'temperature': 0.1, 'context_length': 4096}
    )
    print("LLM loaded successfully.")

    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs=chain_type_kwargs,
        return_source_documents=True
    )

    return rag_chain


def main():
    """
    Main function to run the chatbot.
    """
    rag_chain = create_rag_chain()
    print("\n--- Godot Documentation Chatbot ---")
    print("Ask a question about the documentation. Type 'exit' or 'quit' to end.")

    while True:
        query = input("\n> ")
        if query.lower() in ['exit', 'quit']:
            print("Goodbye!")
            break

        print("Thinking...")
        result = rag_chain.invoke(query)

        print("\nAnswer:")
        print(result['result'].strip())

        print("\n--- Sources ---")
        sources = set(os.path.basename(doc.metadata.get('source', 'Unknown')) for doc in result['source_documents'])
        for i, name in enumerate(sorted(sources), 1):
            print(f"Source {i}: {name}")
        print("---------------")


if __name__ == "__main__":
    # Since you're changing the embedding model, the old vector database is incompatible.
    # This script will automatically remove the old one to force a rebuild.
    if os.path.exists(db_faiss_path):
        print(f"Found existing database. Removing '{db_faiss_path}' for a fresh build with the new embedding model.")
        shutil.rmtree(db_faiss_path)
    main()