# backend/main.py
# This file creates the FastAPI application and its API endpoints.

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import rag_core
import os
from typing import List
import re

# Initialize the FastAPI app
app = FastAPI(
    title="AI Resume Analyzer API",
    description="An API for processing resumes and answering questions using a RAG pipeline.",
)

# Configure CORS (Cross-Origin Resource Sharing)
# UPDATED: This now uses a regular expression to allow any Vercel deployment URL
# for your project, which is a more robust solution for preview and production URLs.
origins = [
    "http://localhost",
    "http://localhost:8080",
    "http://127.0.0.1",
    "http://127.0.0.1:8080",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_origin_regex=r"https://ai-resume-analyser.*\.vercel\.app", # Allows all your Vercel subdomains
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
    Endpoint to upload PDF resumes. The files are saved and then
    processed into a vector store, clearing any old data.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files were uploaded.")

    resumes_dir = "/tmp/resumes"
    os.makedirs(resumes_dir, exist_ok=True)
    
    for file in files:
        file_path = os.path.join(resumes_dir, file.filename)
        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())

    try:
        rag_core.create_vector_store(resumes_dir)
        return {"message": f"Successfully processed {len(files)} resume(s)."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process resumes: {str(e)}")


# NEW: Endpoint to reset the session
@app.post("/reset/", summary="Reset the session")
async def reset_session():
    """
    Endpoint to clear all data from the Pinecone index.
    This effectively resets the knowledge base for a new user session.
    """
    try:
        rag_core.clear_index()
        return {"message": "Session reset successfully. The knowledge base is now empty."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to reset session: {str(e)}")


class Question(BaseModel):
    query: str

@app.post("/ask/", summary="Ask a question about the resumes")
async def ask_question(question: Question):
    """
    Endpoint to ask a question. It uses the existing vector store
    to find relevant context and generate an answer.
    """
    try:
        answer_data = rag_core.get_answer(question.query)
        return {
            "answer": answer_data.get("result", "No answer found."),
            "sources": answer_data.get("source_documents", [])
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
