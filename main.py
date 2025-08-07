# main.py

import os
import uvicorn
from fastapi import FastAPI, Header, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse 
from pydantic import BaseModel, Field
from typing import List
from dotenv import load_dotenv

# Import the core processing logic from your RAG pipeline file
from rag_pipeline2 import process_query

load_dotenv()

app = FastAPI(
    title="Intelligent Query-Retrieval System",
    description="An LLM-powered system to process documents and answer questions.",
    version="1.0.0"
)

# --- CORS Configuration ---
# Allows your index.html to communicate with the backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pydantic Models for Request and Response ---
class QueryRequest(BaseModel):
    documents: str = Field(...)
    questions: List[str] = Field(...)

class QueryResponse(BaseModel):
    answers: List[str]

# --- Authentication ---
EXPECTED_TOKEN = os.getenv("BEARER_TOKEN", "83dcb9bcb612e5fe9a628e89abafec1cdab3c42235f1d44dbb2d17dfd96e6b0c")

# --- Main API Endpoint ---
@app.post("/hackrx/run", response_model=QueryResponse, tags=["Submission"])
async def run_submission(
    request: QueryRequest = Body(...),
    authorization: str = Header(None)
):
    if not authorization:
        raise HTTPException(status_code=401, detail="Authorization header is missing")
    
    try:
        auth_type, token = authorization.split()
        if auth_type.lower() != "bearer" or token != EXPECTED_TOKEN:
            raise HTTPException(status_code=401, detail="Invalid authentication token")
    except ValueError:
        raise HTTPException(status_code=401, detail="Invalid authorization header format.")

    try:
        answers = await process_query(url=request.documents, questions=request.questions)
        return QueryResponse(answers=answers)
    except Exception as e:
        # This will now catch the error from the pipeline and report it
        print(f"Error during processing: {e}") # Print error to server console
        raise HTTPException(status_code=500, detail=f"An error occurred during processing: {str(e)}")

# --- Root Endpoint to Serve the HTML Frontend ---
@app.get("/", include_in_schema=False)
async def read_index():
    # This serves your HTML file when someone visits the main URL
    return FileResponse('index.html')

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)