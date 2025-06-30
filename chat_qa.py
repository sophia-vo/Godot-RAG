# chat.py
import os

# --- LangChain specific imports ---
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
# --- CHANGE HERE: Import LlamaCpp instead of CTransformers ---
from langchain_community.llms import LlamaCpp

# --- Configuration ---
# Path to the FAISS vector database
db_faiss_path = "vectorstore/"
# Path to your local LLM model
model_file = "models/deepseek-llm-7b-chat.Q8_0.gguf"
# The embedding model to use (MUST MATCH the one used in build_db.py)
embedding_model_name = "sentence-transformers/all-mpnet-base-v2"


def create_rag_chain():
    """
    Creates the RAG chain by loading the existing vector store and LLM.
    """
    # --- 1. Load the vector store ---
    print("Loading existing vector database...")
    if not os.path.exists(db_faiss_path):
        print(f"Error: Vector database not found at '{db_faiss_path}'.")
        print("Please run 'python build_db.py' first to create it.")
        exit(1)
        
    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model_name,
        model_kwargs={'device': 'cpu'}
    )
    db = FAISS.load_local(db_faiss_path, embeddings, allow_dangerous_deserialization=True)
    retriever = db.as_retriever(search_kwargs={'k': 3})
    print("Vector database loaded successfully.")

    # --- 2. Create the Prompt Template ---
    # prompt_template = """
    # You are a helpful assistant **trained on Godot 4.4.1-stable documentation**.
    # Use the following pieces of context—*all of which come from the Godot 4.4.1-stable manual*—to answer the question at the end.
    # If you don’t know the answer, say you don’t know; don’t hallucinate.

    # Context: {context}

    # Question: {question}
    # Helpful Answer:
    # """

    prompt_template = """
    ### System Role
    You are an expert assistant for the Godot 4 game engine. Your knowledge base is strictly limited to the documentation provided in the context.

    ### Task
    A user has asked a question. I have performed a search on the official Godot 4.4.1-stable documentation and retrieved the following text snippets. Your task is to synthesize a comprehensive and helpful answer based *exclusively* on these snippets.

    ### Rules
    1.  **Synthesize, Don't Assume:** Combine information from the provided snippets to form a coherent answer. Do not add any information that is not present in the context.
    2.  **Acknowledge Limitations:** The provided context consists of excerpts and may not be complete. Some snippets might be more relevant than others. You must critically evaluate their relevance to the question.
    3.  **No Answer is an Answer:** If the context does not contain the information needed to answer the question, you MUST state that the provided documentation does not contain the answer. Do not try to guess or use outside knowledge.
    4.  **Be Direct:** Directly answer the user's question. If the context includes relevant code examples, incorporate them into your answer using Markdown for formatting.

    ### Context
    {context}

    ### Question
    {question}

    ### Helpful Answer
    """

    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain_type_kwargs = {"prompt": PROMPT}

    # --- 3. Load the LLM ---
    print("Loading local LLM...")
    if not os.path.exists(model_file):
        print(f"Error: Model file not found at '{model_file}'.")
        print("Please download it and place it in the 'models' directory.")
        exit(1)

    # --- CHANGE HERE: Use LlamaCpp instead of CTransformers ---
    llm = LlamaCpp(
        model_path=model_file,
        max_tokens=1024,      # Corresponds to max_new_tokens
        n_ctx=4096,           # Corresponds to context_length
        temperature=0.1,
        n_gpu_layers=0,       # Set to 0 for CPU-only. Set to a positive number (e.g., 40) or -1 to offload to GPU if you installed with GPU support.
        verbose=False,        # Set to True to see llama.cpp logs
    )
    print("LLM loaded successfully.")

    # --- 4. Create the RAG chain ---
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
    main()