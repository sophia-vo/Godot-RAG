import os
import shutil
from pathlib import Path

# --- LangChain specific imports ---
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import BSHTMLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.llms import CTransformers

# --- Configuration ---
DB_FAISS_PATH = "vectorstore/"
MODEL_FILE = "models/llama-2-7b-chat.Q4_K_M.gguf"
DATA_PATH = "classes/"

# --- Helper Function to Create Dummy Files (for out-of-the-box runnability) ---
def setup_dummy_files():
    """
    Creates dummy HTML files and directories if they don't exist,
    so the script can be run without manual setup.
    """
    print("--- Running initial setup ---")
    
    # Create models directory if it doesn't exist
    Path("models").mkdir(exist_ok=True)
    
    # Create data directory if it doesn't exist
    Path(DATA_PATH).mkdir(exist_ok=True)
    
    # Define file contents
    files_to_create = {
        "class_animationnodeblend2.html": """
        <!DOCTYPE html>
        <html><head><title>(DEV) AnimationNodeBlend2</title></head>
        <body>
        <h1>AnimationNodeBlend2</h1>
        <p><strong>Inherits:</strong> AnimationNodeSync < AnimationNode < Resource < RefCounted < Object</p>
        <p>Blends two animations linearly inside of an AnimationNodeBlendTree.</p>
        <h2>Description</h2>
        <p>A resource to add to an AnimationNodeBlendTree. Blends two animations linearly based on the amount value.</p>
        <p>In general, the blend value should be in the <code>[0.0, 1.0]</code> range. Values outside of this range can blend amplified or inverted animations, however, AnimationNodeAdd2 works better for this purpose.</p>
        </body></html>
        """,
        "class_animationnodeblend3.html": """
        <!DOCTYPE html>
        <html><head><title>(DEV) AnimationNodeBlend3</title></head>
        <body>
        <h1>AnimationNodeBlend3</h1>
        <p><strong>Inherits:</strong> AnimationNodeSync < AnimationNode < Resource < RefCounted < Object</p>
        <p>Blends two of three animations linearly inside of an AnimationNodeBlendTree.</p>
        <h2>Description</h2>
        <p>A resource to add to an AnimationNodeBlendTree. Blends two animations out of three linearly out of three based on the amount value.</p>
        <p>This animation node has three inputs:</p>
        <ul>
        <li>The base animation to blend with</li>
        <li>A "-blend" animation to blend with when the blend amount is negative value</li>
        <li>A "+blend" animation to blend with when the blend amount is positive value</li>
        </ul>
        <p>In general, the blend value should be in the <code>[-1.0, 1.0]</code> range. Values outside of this range can blend amplified animations, however, AnimationNodeAdd3 works better for this purpose.</p>
        </body></html>
        """
    }
    
    # Create files if they don't exist
    for filename, content in files_to_create.items():
        filepath = Path(DATA_PATH) / filename
        if not filepath.exists():
            print(f"Creating dummy file: {filepath}")
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(content)
        else:
            print(f"File already exists: {filepath}")

    print("--- Initial setup complete ---")


# --- Core RAG Functions ---

def create_vector_db():
    """
    Creates or loads the FAISS vector database from HTML documents.
    """
    # Check if the vector store already exists
    if os.path.exists(DB_FAISS_PATH) and os.listdir(DB_FAISS_PATH):
        print(f"Loading existing vector database from {DB_FAISS_PATH}...")
        embeddings = HuggingFaceEmbeddings(
            model_name='sentence-transformers/all-MiniLM-L6-v2',
            model_kwargs={'device': 'cpu'}
        )
        db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
        print("Vector database loaded successfully.")
        return db

    # If not, create it
    print(f"Creating new vector database from documents in {DATA_PATH}...")
    
    # Use BSHTMLLoader for parsing HTML
    loader = DirectoryLoader(
        DATA_PATH, 
        glob="**/*.html", 
        loader_cls=BSHTMLLoader, 
        show_progress=True,
        use_multithreading=True,
        loader_kwargs={'get_text_separator': ' '}
    )
    
    documents = loader.load()
    if not documents:
        print("No documents found. Please place your HTML files in the 'classes' directory.")
        exit()
        
    print(f"Loaded {len(documents)} documents.")
    
    # Split the documents into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200
    )
    texts = text_splitter.split_documents(documents)
    print(f"Split documents into {len(texts)} chunks.")

    # Create embeddings
    print("Creating embeddings... (This may take a while on the first run)")
    embeddings = HuggingFaceEmbeddings(
        model_name='sentence-transformers/all-MiniLM-L6-v2',
        model_kwargs={'device': 'cpu'}
    )

    # Create the FAISS vector store
    db = FAISS.from_documents(texts, embeddings)
    
    # Save the vector store to disk
    db.save_local(DB_FAISS_PATH)
    print(f"Vector database created and saved to {DB_FAISS_PATH}.")
    
    return db

def create_rag_chain():
    """
    Creates the complete RAG chain with the LLM, retriever, and prompt.
    """
    # Load the vector database
    db = create_vector_db()
    
    # Create a retriever
    retriever = db.as_retriever(search_kwargs={'k': 3}) # Retrieve top 3 most similar chunks

    # Define the prompt template
    prompt_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}

Question: {question}
Helpful Answer:"""
    
    PROMPT = PromptTemplate(
        template=prompt_template, 
        input_variables=["context", "question"]
    )
    
    chain_type_kwargs = {"prompt": PROMPT}

    # Load the local LLM
    print("Loading local LLM...")
    if not os.path.exists(MODEL_FILE):
        print(f"Model file not found at {MODEL_FILE}")
        exit()

    llm = CTransformers(
        model=MODEL_FILE,
        model_type='llama',
        config={'max_new_tokens': 512, 'temperature': 0.1, 'context_length': 4096}
    )
    print("LLM loaded successfully.")

    # Create the RetrievalQA chain
    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs=chain_type_kwargs,
        return_source_documents=True
    )

    return rag_chain

# --- Main Execution ---
def main():
    """
    Main function to run the chatbot.
    """
    # Run setup to create dummy files if needed
    setup_dummy_files()

    # Create the RAG chain
    rag_chain = create_rag_chain()
    
    # Start the interactive chat loop
    print("\n--- Godot Documentation Chatbot ---")
    print("Ask a question about the documentation. Type 'exit' or 'quit' to end.")
    
    while True:
        query = input("\n> ")
        if query.lower() in ['exit', 'quit']:
            print("Goodbye!")
            break
            
        print("Thinking...")
        
        # Get the answer from the chain
        result = rag_chain.invoke(query)
        
        # Print the answer
        print("\nAnswer:")
        print(result['result'].strip())
        
        # Print the sources used
        print("\n--- Sources ---")
        if result['source_documents']:
            for i, doc in enumerate(result['source_documents'], 1):
                source_name = doc.metadata.get('source', 'Unknown')
                print(f"Source {i}: {os.path.basename(source_name)}")
        else:
            print("No relevant sources found in the documentation.")
        print("---------------")

if __name__ == "__main__":
    # Clean up old database for a fresh start if desired.
    # For this script, we'll keep it to show caching works.
    # If you want to force a rebuild, uncomment the line below.
    if os.path.exists(DB_FAISS_PATH):
        print(f"Removing old database at {DB_FAISS_PATH}")
        shutil.rmtree(DB_FAISS_PATH)
        
    main()