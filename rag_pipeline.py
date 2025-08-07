import os
import httpx
import tempfile
from typing import List, Dict, Optional
import fitz  # PyMuPDF
from docx import Document
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import google.generativeai as genai
from dotenv import load_dotenv
from urllib.parse import urlparse
import asyncio
from concurrent.futures import ThreadPoolExecutor
import threading
import hashlib

# Pre-compiled regex for faster sentence splitting (if needed)
import re
_sentence_pattern = re.compile(r'(?<=[.!?])\s+')

# --- Global optimizations ---
# Pre-load embedding model globally to avoid reloading
_embedding_model = None
_model_lock = threading.Lock()

def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        with _model_lock:
            if _embedding_model is None:
                _embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    return _embedding_model

# Thread pool for CPU-bound operations
_thread_pool = ThreadPoolExecutor(max_workers=4)

# --- Caching Mechanism ---
db_cache: Dict[str, 'VectorDatabase'] = {}
text_cache: Dict[str, str] = {}  # Cache extracted text

def get_url_hash(url: str) -> str:
    """Generate a hash for URL to use as cache key."""
    return hashlib.md5(url.encode()).hexdigest()

class DocumentExtractor:
    """Optimized document extraction with minimal processing."""
    
    @staticmethod
    async def extract_from_url(url: str) -> str:
        url_hash = get_url_hash(url)
        
        # Check text cache first
        if url_hash in text_cache:
            print(f"Using cached text for: {url}")
            return text_cache[url_hash]
        
        try:
            parsed_url = urlparse(url)
            file_path = parsed_url.path

            # Optimized download with streaming
            async with httpx.AsyncClient() as client:
                async with client.stream('GET', url, follow_redirects=True, timeout=30.0) as response:
                    response.raise_for_status()
                    content = b""
                    async for chunk in response.aiter_bytes():
                        content += chunk

            suffix = ".pdf" if file_path.lower().endswith('.pdf') else ".docx"
            
            # Use asyncio to run extraction in thread pool
            if suffix == '.pdf':
                text = await asyncio.get_event_loop().run_in_executor(
                    _thread_pool, DocumentExtractor._extract_from_pdf_fast, content
                )
            else:
                text = await asyncio.get_event_loop().run_in_executor(
                    _thread_pool, DocumentExtractor._extract_from_docx_fast, content
                )
            
            # Cache the extracted text
            text_cache[url_hash] = text
            return text
            
        except Exception as e:
            raise IOError(f"Failed to process document from {url}: {e}")

    @staticmethod
    def _extract_from_pdf_fast(content: bytes) -> str:
        """Fast PDF extraction - text only, no table processing."""
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
            temp_file.write(content)
            temp_filepath = temp_file.name
        
        try:
            full_text = ""
            doc = fitz.open(temp_filepath)
            
            # Extract text from all pages in batch
            for page in doc:
                page_text = page.get_text("text")
                if page_text.strip():  # Only add non-empty pages
                    full_text += page_text + "\n"
            
            doc.close()
            return full_text
        finally:
            os.unlink(temp_filepath)

    @staticmethod
    def _extract_from_docx_fast(content: bytes) -> str:
        """Fast DOCX extraction."""
        with tempfile.NamedTemporaryFile(suffix='.docx', delete=False) as temp_file:
            temp_file.write(content)
            temp_filepath = temp_file.name
        
        try:
            text_parts = []
            doc = Document(temp_filepath)
            
            # Batch process paragraphs
            for para in doc.paragraphs:
                if para.text.strip():  # Skip empty paragraphs
                    text_parts.append(para.text)
            
            return "\n".join(text_parts)
        finally:
            os.unlink(temp_filepath)

class UltraFastChunker:
    """Ultra-optimized chunking with multiple strategies for maximum speed."""
    
    @staticmethod
    def chunk_text_ultra_fast(text: str, chunk_size: int = 1000, overlap: int = 100) -> List[str]:
        """Ultra-fast chunking using the most efficient method based on text characteristics."""
        if not text:
            return []
        
        text_len = len(text)
        if text_len <= chunk_size:
            return [text.strip()] if text.strip() else []
        
        # Choose strategy based on text characteristics for maximum speed
        if '\n\n' in text and text.count('\n\n') > 10:
            # For well-structured documents with paragraphs
            return UltraFastChunker._paragraph_based_chunking(text, chunk_size, overlap)
        elif text.count('\n') > text_len * 0.05:  # Many line breaks
            # For documents with many line breaks
            return UltraFastChunker._line_based_chunking(text, chunk_size, overlap)
        else:
            # For dense text, use fixed-size chunking (fastest)
            return UltraFastChunker._fixed_size_chunking(text, chunk_size, overlap)
    
    @staticmethod
    def _fixed_size_chunking(text: str, chunk_size: int, overlap: int) -> List[str]:
        """Fastest chunking method - fixed size with minimal boundary checking."""
        chunks = []
        start = 0
        text_len = len(text)
        
        while start < text_len:
            end = min(start + chunk_size, text_len)
            
            # Quick word boundary check only for non-final chunks
            if end < text_len and end - start > 500:  # Only check if chunk is substantial
                # Quick search for space in last 50 chars (much smaller window)
                space_pos = text.rfind(' ', max(end - 50, start), end)
                if space_pos > start + 200:  # Ensure we don't make chunks too small
                    end = space_pos
            
            chunk = text[start:end]
            if len(chunk.strip()) > 50:  # Skip very small chunks
                chunks.append(chunk.strip())
            
            start = end - overlap if end < text_len else text_len
        
        return chunks
    
    @staticmethod
    def _paragraph_based_chunking(text: str, chunk_size: int, overlap: int) -> List[str]:
        """Fast paragraph-based chunking for well-structured documents."""
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        if not paragraphs:
            return UltraFastChunker._fixed_size_chunking(text, chunk_size, overlap)
        
        chunks = []
        current_chunk = []
        current_size = 0
        
        i = 0
        while i < len(paragraphs):
            para = paragraphs[i]
            para_len = len(para)
            
            # If single paragraph is too large, split it
            if para_len > chunk_size:
                if current_chunk:  # Save current chunk first
                    chunks.append('\n\n'.join(current_chunk))
                    current_chunk = []
                    current_size = 0
                # Split large paragraph using fixed chunking
                para_chunks = UltraFastChunker._fixed_size_chunking(para, chunk_size, overlap)
                chunks.extend(para_chunks)
            elif current_size + para_len <= chunk_size:
                current_chunk.append(para)
                current_size += para_len + 2  # +2 for \n\n
            else:
                # Save current chunk and start new one
                if current_chunk:
                    chunks.append('\n\n'.join(current_chunk))
                current_chunk = [para]
                current_size = para_len
            
            i += 1
        
        # Add final chunk
        if current_chunk:
            chunks.append('\n\n'.join(current_chunk))
        
        return [c for c in chunks if len(c.strip()) > 50]
    
    @staticmethod
    def _line_based_chunking(text: str, chunk_size: int, overlap: int) -> List[str]:
        """Fast line-based chunking for documents with many line breaks."""
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        if not lines:
            return []
        
        chunks = []
        current_chunk = []
        current_size = 0
        
        for line in lines:
            line_len = len(line)
            
            if line_len > chunk_size:
                # Save current chunk if exists
                if current_chunk:
                    chunks.append('\n'.join(current_chunk))
                    current_chunk = []
                    current_size = 0
                # Split long line
                line_chunks = UltraFastChunker._fixed_size_chunking(line, chunk_size, overlap)
                chunks.extend(line_chunks)
            elif current_size + line_len <= chunk_size:
                current_chunk.append(line)
                current_size += line_len + 1  # +1 for \n
            else:
                # Save current chunk and start new one
                if current_chunk:
                    chunks.append('\n'.join(current_chunk))
                current_chunk = [line]
                current_size = line_len
        
        # Add final chunk
        if current_chunk:
            chunks.append('\n'.join(current_chunk))
        
        return [c for c in chunks if len(c.strip()) > 50]

class VectorDatabase:
    """Optimized vector database with batch operations."""
    
    def __init__(self):
        self.embedding_model = get_embedding_model()  # Use global model
        self.dimension = self.embedding_model.get_sentence_embedding_dimension()
        self.index = faiss.IndexFlatL2(self.dimension)
        self.documents = []

    def add_documents_batch(self, chunks: List[str], batch_size: int = 32):
        """Add documents in batches for better performance."""
        if not chunks:
            return
        
        all_embeddings = []
        
        # Process in batches to manage memory
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            batch_embeddings = self.embedding_model.encode(
                batch, 
                convert_to_tensor=False, 
                show_progress_bar=False,
                batch_size=batch_size
            )
            all_embeddings.append(batch_embeddings)
        
        # Combine all embeddings
        embeddings = np.vstack(all_embeddings) if len(all_embeddings) > 1 else all_embeddings[0]
        
        # Add to FAISS index
        self.index.add(embeddings.astype(np.float32))
        self.documents.extend(chunks)

    def search_fast(self, query: str, k: int = 5) -> List[str]:
        if self.index.ntotal == 0:
            return []
        
        query_embedding = self.embedding_model.encode([query], show_progress_bar=False)
        _, indices = self.index.search(query_embedding.astype(np.float32), min(k, len(self.documents)))
        
        return [self.documents[i] for i in indices[0] if 0 <= i < len(self.documents)]

class AnswerGenerator:
    """Optimized answer generation."""
    
    def __init__(self):
        load_dotenv()
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables.")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.5-flash-preview-05-20')

    async def generate(self, query: str, context_chunks: List[str]) -> str:
        # Limit context to avoid token limits and improve speed
        context = "\n\n".join(context_chunks[:3])  # Use top 3 chunks only
        
        prompt = f"""Answer based on the context below. Be concise.

CONTEXT:
{context}

QUESTION: {query}
ANSWER:"""
        
        try:
            response = await asyncio.to_thread(self.model.generate_content, prompt)
            return response.text.strip()
        except Exception as e:
            return f"Error: {str(e)}"

# --- Optimized Main Functions ---

async def answer_single_question(question: str, vector_db: VectorDatabase, answer_generator: AnswerGenerator) -> str:
    """Fast single question processing."""
    # Run search in thread pool to avoid blocking
    relevant_chunks = await asyncio.get_event_loop().run_in_executor(
        _thread_pool, vector_db.search_fast, question, 3  # Reduced from 5 to 3 for speed
    )
    answer = await answer_generator.generate(question, relevant_chunks)
    return answer

async def process_query_optimized(url: str, questions: List[str]) -> List[str]:
    """Highly optimized RAG pipeline with minimal preprocessing."""
    
    url_hash = get_url_hash(url)
    
    # Check if vector DB is cached
    if url_hash in db_cache:
        print(f"Using cached vector database for: {url}")
        vector_db = db_cache[url_hash]
    else:
        print(f"Fast processing document from: {url}")
        
        # Step 1: Extract text (cached if previously processed)
        document_text = await DocumentExtractor.extract_from_url(url)
        
        if not document_text.strip():
            raise ValueError("No text extracted from document.")

        # Step 2: Ultra-fast chunking with optimized strategy selection
        text_chunks = await asyncio.get_event_loop().run_in_executor(
            _thread_pool, UltraFastChunker.chunk_text_ultra_fast, document_text, 1000, 50  # Reduced overlap from 100 to 50
        )
        
        if not text_chunks:
            raise ValueError("No chunks created from document.")
        
        # Step 3: Create vector DB and add embeddings
        vector_db = VectorDatabase()
        await asyncio.get_event_loop().run_in_executor(
            _thread_pool, vector_db.add_documents_batch, text_chunks
        )
        
        # Cache the vector DB
        db_cache[url_hash] = vector_db
        print(f"Cached vector database with {len(text_chunks)} chunks")
    
    # Step 4: Process all questions in parallel
    answer_generator = AnswerGenerator()
    tasks = [answer_single_question(q, vector_db, answer_generator) for q in questions]
    answers = await asyncio.gather(*tasks)
    
    return answers

# Convenience function for backwards compatibility
async def process_query(url: str, questions: List[str]) -> List[str]:
    return await process_query_optimized(url, questions)