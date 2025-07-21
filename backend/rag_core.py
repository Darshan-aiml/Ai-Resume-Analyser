# backend/rag_core.py
# This file contains the core RAG logic using LangChain, Google, and Pinecone.

import os
from dotenv import load_dotenv

# LangChain, Google, and Pinecone components
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_pinecone import Pinecone as PineconeLangChain
from langchain.chains import RetrievalQA
# We need the main pinecone library to manage the index
import pinecone
from pinecone.core.client.exceptions import NotFoundException

# --- Configuration ---
load_dotenv()

# Check for API Keys
if not os.getenv("GOOGLE_API_KEY"):
    raise EnvironmentError("GOOGLE_API_KEY not found in .env file.")
if not os.getenv("PINECONE_API_KEY"):
    raise EnvironmentError("PINECONE_API_KEY not found in .env file.")

PINECONE_INDEX_NAME = "resume-analyser"
QA_CHAIN = None

def clear_index():
    """
    Connects to the Pinecone index and deletes all vectors, effectively clearing the knowledge base.
    """
    print(f"Connecting to Pinecone index '{PINECONE_INDEX_NAME}' to clear all data...")
    pc = pinecone.Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    
    if PINECONE_INDEX_NAME in pc.list_indexes().names():
        index = pc.Index(PINECONE_INDEX_NAME)
        print("Clearing all records from the index...")
        # UPDATED: Wrap the delete call in a try/except block to handle empty indexes gracefully.
        try:
            index.delete(delete_all=True)
            print("Index cleared successfully.")
        except NotFoundException:
            print("Index was already empty. Nothing to clear.")
        except Exception as e:
            print(f"An unexpected error occurred while clearing the index: {e}")

    else:
        print(f"Index '{PINECONE_INDEX_NAME}' not found. Nothing to clear.")

    # Reset the global QA chain to force re-initialization on next query
    global QA_CHAIN
    QA_CHAIN = None

def create_vector_store(resumes_path: str):
    """
    Clears the old data, loads new resumes, splits them, creates embeddings, 
    and upserts them to the Pinecone index.
    """
    # Use the dedicated clear function to ensure a fresh start
    clear_index()

    print(f"Loading resumes from '{resumes_path}'...")
    loader = DirectoryLoader(
        resumes_path,
        glob="*.pdf",
        loader_cls=PyPDFLoader,
        show_progress=True,
        use_multithreading=True
    )
    documents = loader.load()

    if not documents:
        raise ValueError(f"No PDF files found in '{resumes_path}'.")

    print(f"Splitting {len(documents)} document(s) into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)

    print("Generating embeddings with Google's 'text-embedding-004' model...")
    embeddings = GoogleGenerativeAIEmbeddings(model="text-embedding-004")
    
    print(f"Upserting new documents to Pinecone index '{PINECONE_INDEX_NAME}'...")
    PineconeLangChain.from_documents(texts, embeddings, index_name=PINECONE_INDEX_NAME)
    
    print("Documents successfully added to Pinecone.")

def initialize_qa_chain():
    """
    Connects to the Pinecone index and initializes the RetrievalQA chain.
    """
    global QA_CHAIN
    if QA_CHAIN is not None:
        return

    print("Initializing QA chain from Pinecone with Google models...")
    embeddings = GoogleGenerativeAIEmbeddings(model="text-embedding-004")
    
    vector_store = PineconeLangChain.from_existing_index(PINECONE_INDEX_NAME, embeddings)
    
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.2, convert_system_message_to_human=True)
    
    QA_CHAIN = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    print("QA chain is ready.")

def get_answer(query: str) -> dict:
    """
    Takes a user query, runs it through the QA chain, and returns the result.
    """
    if QA_CHAIN is None:
        initialize_qa_chain()
        
    print(f"Running query: '{query}'")
    result = QA_CHAIN.invoke({"query": query})
    
    if 'source_documents' in result:
        for doc in result['source_documents']:
            doc.metadata['source'] = os.path.basename(doc.metadata.get('source', 'Unknown'))
            
    return result
```

I've wrapped the `index.delete()` command in a `try...except` block. This will now gracefully handle cases where the index is already empty, preventing the crash and allowing your application to proceed correctly.

To apply this fix, please update your `backend/rag_core.py` file with this new code and push the change to your GitHub repository. This will trigger a redeployment on Render and resolve the err
