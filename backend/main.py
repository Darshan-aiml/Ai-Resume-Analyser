# backend/main.py
# This file creates the FastAPI application and its API endpoints.

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import rag_core
import os
from typing import List

# Initialize the FastAPI app
app = FastAPI(
    title="AI Resume Analyzer API",
    description="An API for processing resumes and answering questions using a RAG pipeline.",
)

# Configure CORS (Cross-Origin Resource Sharing)
# This allows the frontend (running on a different port) to communicate with this backend.
origins = [
    "http://localhost",
    "http://localhost:8080",
    "http://127.0.0.1",
    "http://127.0.0.1:8080",
    # Add other origins if your frontend is served from a different address
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- API Endpoints ---

@app.get("/", summary="Root endpoint to check server status")
def read_root():
    """A simple endpoint to confirm the server is running."""
    return {"status": "AI Resume Analyzer API is running."}


@app.post("/process-resumes/", summary="Upload and process resumes")
async def process_resumes(files: List[UploadFile] = File(...)):
    """
    Endpoint to upload multiple PDF resumes.
    The files are saved and then processed into a FAISS vector store.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files were uploaded.")

    # Create a directory to store resumes if it doesn't exist
    resumes_dir = "resumes"
    os.makedirs(resumes_dir, exist_ok=True)
    
    # Save uploaded files to the directory
    saved_files = []
    for file in files:
        file_path = os.path.join(resumes_dir, file.filename)
        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())
        saved_files.append(file_path)

    try:
        # Call the RAG core function to create the vector store
        rag_core.create_vector_store(resumes_dir)
        return {"message": f"Successfully processed {len(saved_files)} resumes."}
    except Exception as e:
        # Handle potential errors during processing
        raise HTTPException(status_code=500, detail=f"Failed to process resumes: {str(e)}")


# Pydantic model for the question request body
class Question(BaseModel):
    query: str

@app.post("/ask/", summary="Ask a question about the resumes")
async def ask_question(question: Question):
    """
    Endpoint to ask a question. It uses the existing vector store
    to find relevant context and generate an answer.
    """
    try:
        # Get the answer and source documents from the RAG core
        answer_data = rag_core.get_answer(question.query)
        return {
            "answer": answer_data.get("result", "No answer found."),
            "sources": answer_data.get("source_documents", [])
        }
    except Exception as e:
        # Handle errors, e.g., if the vector store doesn't exist yet
        raise HTTPException(status_code=500, detail=str(e))

