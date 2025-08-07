# rag_pipeline.py

import os
import httpx # Asynchronous HTTP client
import tempfile
from typing import List, Dict
import fitz  # PyMuPDF
import pdfplumber
from docx import Document
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import google.generativeai as genai
from dotenv import load_dotenv
from urllib.parse import urlparse
import asyncio # For running tasks in parallel

# --- Caching Mechanism ---
db_cache: Dict[str, 'VectorDatabase'] = {}

# --- Helper Classes (Upgraded with pdfplumber) ---

class DocumentExtractor:
    """Extracts text and tables from documents at a given URL."""
    @staticmethod
    async def extract_from_url(url: str) -> str: # Made this function asynchronous
        try:
            parsed_url = urlparse(url)
            file_path = parsed_url.path

            # Use httpx for non-blocking download
            async with httpx.AsyncClient() as client:
                response = await client.get(url, follow_redirects=True, timeout=30.0)
                response.raise_for_status()

            suffix = ".pdf" if file_path.lower().endswith('.pdf') else ".docx"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
                temp_file.write(response.content)
                temp_filepath = temp_file.name

            if suffix == '.pdf':
                text = DocumentExtractor._extract_from_pdf_with_tables(temp_filepath)
            else: # .docx
                text = DocumentExtractor._extract_from_docx(temp_filepath)
            
            os.unlink(temp_filepath)
            return text
            
        except Exception as e:
            raise IOError(f"Failed to process document from {url}: {e}")

    @staticmethod
    def _extract_from_pdf_with_tables(filepath: str) -> str:
        """Extracts both plain text and tables (as Markdown) using pdfplumber."""
        full_text = ""
        with pdfplumber.open(filepath) as pdf:
            for i, page in enumerate(pdf.pages):
                page_text = page.extract_text(x_tolerance=2, y_tolerance=2)
                if page_text:
                    full_text += page_text + "\n"

                tables = page.extract_tables()
                if tables:
                    full_text += f"\n--- Tables on Page {i+1} ---\n"
                    for table_data in tables:
                        if not table_data or not table_data[0]: continue
                        
                        header = " | ".join(str(cell) if cell is not None else '' for cell in table_data[0])
                        separator = " | ".join(['---'] * len(table_data[0]))
                        
                        body_rows = []
                        for row in table_data[1:]:
                            if row:
                                body_rows.append(" | ".join(str(cell) if cell is not None else '' for cell in row))
                        
                        body = "\n".join(f"| {row} |" for row in body_rows if row)
                        
                        markdown_table = f"| {header} |\n| {separator} |\n{body}"
                        full_text += markdown_table + "\n\n"
        return full_text

    @staticmethod
    def _extract_from_docx(filepath: str) -> str:
        text = ""
        doc = Document(filepath)
        for para in doc.paragraphs:
            text += para.text + "\n"
        return text

class TextChunker:
    """Splits text into smaller, semantically aware chunks."""
    @staticmethod
    def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 150) -> List[str]:
        if not text: return []
        splits = text.split("\n\n")
        chunks = []
        for split in splits:
            if len(split) <= chunk_size:
                chunks.append(split)
            else:
                import re
                sentences = re.split(r'(?<=[.!?])\s+', split)
                current_chunk = ""
                for sentence in sentences:
                    if len(current_chunk) + len(sentence) <= chunk_size:
                        current_chunk += sentence + " "
                    else:
                        chunks.append(current_chunk.strip())
                        current_chunk = sentence + " "
                if current_chunk:
                    chunks.append(current_chunk.strip())
        return chunks

class VectorDatabase:
    """Manages document embeddings and semantic search using FAISS."""
    def __init__(self):
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.dimension = self.embedding_model.get_sentence_embedding_dimension()
        self.index = faiss.IndexFlatL2(self.dimension)
        self.documents = []

    def add_documents(self, chunks: List[str]):
        if not chunks: return
        embeddings = self.embedding_model.encode(chunks, convert_to_tensor=False, show_progress_bar=False)
        self.index.add(np.array(embeddings, dtype=np.float32))
        self.documents.extend(chunks)

    def search(self, query: str, k: int = 5) -> List[str]:
        if self.index.ntotal == 0: return []
        query_embedding = self.embedding_model.encode([query])
        _, indices = self.index.search(np.array(query_embedding, dtype=np.float32), k)
        return [self.documents[i] for i in indices[0] if i < len(self.documents)]

class AnswerGenerator:
    """Generates answers using an LLM based on a query and context."""
    def __init__(self):
        load_dotenv()
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables.")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.5-flash-preview-05-20')

    async def generate(self, query: str, context_chunks: List[str]) -> str:
        context = "\n\n".join(context_chunks)
        prompt = f"""
        You are an expert assistant for analyzing policy documents. Based *only* on the context provided below, answer the user's question. The context may contain both plain text and tables formatted in Markdown. Do not use any external knowledge. If the answer is not in the context, state that clearly.
        CONTEXT:
        ---
        {context}
        ---
        QUESTION: {query}
        ANSWER:
        """
        try:
            # Run the synchronous SDK call in a separate thread
            response = await asyncio.to_thread(self.model.generate_content, prompt)
            return response.text.strip()
        except Exception as e:
            return f"Error generating answer from LLM: {str(e)}"

# --- Main Processing Function (Optimized for Speed) ---

async def answer_single_question(question: str, vector_db: VectorDatabase, answer_generator: AnswerGenerator) -> str:
    """Handles the search and generation for one question."""
    print(f"Answering question: {question}")
    # Run synchronous search in a thread
    relevant_chunks = await asyncio.to_thread(vector_db.search, question)
    # Generate answer asynchronously
    answer = await answer_generator.generate(question, relevant_chunks)
    return answer

async def process_query(url: str, questions: List[str]) -> List[str]:
    """Orchestrates the end-to-end RAG pipeline, using a cache and parallel processing."""
    
    if url in db_cache:
        print(f"Using cached vector database for: {url}")
        vector_db = db_cache[url]
    else:
        print(f"Processing and caching new document from: {url}")
        # 1. Asynchronously extract text
        # --- FIX IS HERE ---
        # The 'questions' argument has been removed from the function call below
        document_text = await DocumentExtractor.extract_from_url(url)
        # --- END OF FIX ---
        
        if not document_text:
            raise ValueError("Failed to extract any text from the document.")

        # The following steps are CPU-bound, so they remain synchronous
        text_chunks = TextChunker.chunk_text(document_text)
        if not text_chunks:
            raise ValueError("Failed to create text chunks from the document.")
        
        vector_db = VectorDatabase()
        vector_db.add_documents(text_chunks)
        db_cache[url] = vector_db
    
    answer_generator = AnswerGenerator()
    
    # --- OPTIMIZATION: Process all questions in parallel ---
    tasks = [answer_single_question(q, vector_db, answer_generator) for q in questions]
    answers = await asyncio.gather(*tasks)
    
    return answers
