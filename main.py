# main.py

import os
from fastapi import FastAPI, Header, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List
from dotenv import load_dotenv

# Import the core processing logic
from rag_pipeline2 import process_query

# Load environment variables from .env file
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="Intelligent Query-Retrieval System",
    description="An LLM-powered system to process documents and answer questions based on the provided context.",
    version="1.0.0"
)

# --- CORS Configuration ---
origins = [
    "http://localhost",
    "http://localhost:8080",
    "null"  # Allow requests from local files (origin: null)
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pydantic Models for Request and Response ---
class QueryRequest(BaseModel):
    documents: str = Field(..., description="URL to the document (PDF or DOCX) to be processed.")
    questions: List[str] = Field(..., description="A list of questions to be answered based on the document.")

class QueryResponse(BaseModel):
    answers: List[str] = Field(..., description="A list of answers corresponding to the questions asked.")

# --- Authentication ---
EXPECTED_TOKEN = os.getenv("BEARER_TOKEN")
if not EXPECTED_TOKEN:
    EXPECTED_TOKEN = "83dcb9bcb612e5fe9a628e89abafec1cdab3c42235f1d44dbb2d17dfd96e6b0c"

# --- Optional Root Endpoint for Status Check ---
@app.get("/", tags=["Status"])
def read_root():
    """Provides a simple status message to show that the API is running."""
    return {"status": "ok", "message": "Intelligent Query-Retrieval System is running."}

# --- Main API Endpoint ---
@app.post("/hackrx/run", response_model=QueryResponse, tags=["Submission"])
async def run_submission(
    request: QueryRequest = Body(...),
    authorization: str = Header(None, description="Authentication token (e.g., 'Bearer your_token_here')")
):
    """
    This endpoint processes a document from a URL and answers questions about it.
    """
    # 1. Authenticate the request
    if not authorization:
        raise HTTPException(status_code=401, detail="Authorization header is missing")
    
    try:
        auth_type, token = authorization.split()
        if auth_type.lower() != "bearer" or token != EXPECTED_TOKEN:
            raise HTTPException(status_code=401, detail="Invalid authentication token")
    except ValueError:
        raise HTTPException(status_code=401, detail="Invalid authorization header format. Use 'Bearer <token>'.")

    # 2. Process the request using the core RAG pipeline
    try:
        answers = await process_query(url=request.documents, questions=request.questions)
        return QueryResponse(answers=answers)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred during processing: {str(e)}")
