# rag_pipeline.py - Complete optimized replacement for your original file

import os
import httpx
import tempfile
import asyncio
import hashlib
import numpy as np
from typing import List, Dict, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor
import threading
import time
from urllib.parse import urlparse

# Core imports
import fitz  # PyMuPDF
from docx import Document
from sentence_transformers import SentenceTransformer
import faiss
import google.generativeai as genai
from dotenv import load_dotenv

# Thread pool for CPU-bound operations
_thread_pool = ThreadPoolExecutor(max_workers=os.cpu_count() or 4)

# === MULTI-TIER MODEL STRATEGY ===
class ModelManager:
    """Manages multiple embedding models for different use cases"""
    
    def __init__(self):
        self.models = {}
        self.model_lock = threading.Lock()
    
    def get_model(self, model_type: str = "fast"):
        """Get appropriate model based on speed/accuracy requirements"""
        if model_type not in self.models:
            with self.model_lock:
                if model_type not in self.models:
                    # For a hackathon, we prioritize speed.
                    # This model is ultra-fast with decent accuracy.
                    # IMPORTANT: Change to 'cuda' if you have a GPU for a massive speedup.
                    device = 'cuda' if 'CUDA_VISIBLE_DEVICES' in os.environ else 'cpu'
                    self.models[model_type] = SentenceTransformer(
                        'all-MiniLM-L6-v2',
                        device=device 
                    )
        return self.models[model_type]

# Global model manager
model_manager = ModelManager()

# === DOCUMENT EXTRACTION (INTEGRATED) ===
class DocumentExtractor:
    """Optimized document extraction"""
    
    @staticmethod
    async def extract_from_url(url: str) -> str:
        try:
            parsed_url = urlparse(url)
            file_path = parsed_url.path

            async with httpx.AsyncClient() as client:
                async with client.stream('GET', url, follow_redirects=True, timeout=30.0) as response:
                    response.raise_for_status()
                    content = await response.aread()

            suffix = ".pdf" if file_path.lower().endswith('.pdf') else ".docx"
            
            loop = asyncio.get_event_loop()
            if suffix == '.pdf':
                text = await loop.run_in_executor(
                    _thread_pool, DocumentExtractor._extract_from_pdf_fast, content
                )
            else:
                text = await loop.run_in_executor(
                    _thread_pool, DocumentExtractor._extract_from_docx_fast, content
                )
            
            return text
            
        except Exception as e:
            raise IOError(f"Failed to process document from {url}: {e}")

    @staticmethod
    def _extract_from_pdf_fast(content: bytes) -> str:
        """Fast PDF extraction using PyMuPDF"""
        with fitz.open(stream=content, filetype="pdf") as doc:
            return "\n".join(page.get_text("text") for page in doc)

    @staticmethod
    def _extract_from_docx_fast(content: bytes) -> str:
        """Fast DOCX extraction"""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as temp_file:
            temp_file.write(content)
            temp_filepath = temp_file.name
        
        try:
            doc = Document(temp_filepath)
            return "\n".join(para.text for para in doc.paragraphs if para.text.strip())
        finally:
            os.unlink(temp_filepath)

# === NEW FASTER CHUNKING STRATEGY ===
class SmartChunker:
    """
    Optimized for speed by creating fewer, larger chunks.
    This is the key optimization to reduce embedding time.
    """
    @staticmethod
    def smart_chunk(text: str, chunk_size: int = 2000, overlap: int = 200) -> List[Dict]:
        if not text: return []
        
        # First, split the text into sections using double newlines
        splits = text.split('\n\n')
        
        chunks = []
        position = 0
        current_chunk = ""

        # Now, group these sections into larger chunks of the desired size
        for split in splits:
            if not split.strip():
                continue
            
            # If adding the next split exceeds chunk_size, finalize the current chunk
            if len(current_chunk) + len(split) + 1 > chunk_size:
                if current_chunk.strip():
                    chunks.append({"text": current_chunk.strip(), "position": position})
                    position += 1
                
                # Start a new chunk, adding overlap from the end of the previous one
                overlap_text = current_chunk[-overlap:]
                current_chunk = overlap_text + " " + split
            else:
                current_chunk += "\n\n" + split
        
        # Add the last remaining chunk
        if current_chunk.strip():
            chunks.append({"text": current_chunk.strip(), "position": position})
            
        return chunks

# === OPTIMIZED VECTOR DATABASE ===
class OptimizedVectorDB:
    """High-performance vector DB with smart retrieval"""
    
    def __init__(self, model_type: str = "fast"):
        self.model = model_manager.get_model(model_type)
        self.dimension = self.model.get_sentence_embedding_dimension()
        self.index = faiss.IndexFlatL2(self.dimension)
        self.chunks = []

    def add_chunks_optimized(self, chunks: List[Dict], batch_size: int = 64):
        if not chunks: return
        
        texts = [chunk["text"] for chunk in chunks]
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True
        )
        
        self.index.add(embeddings.astype(np.float32))
        self.chunks.extend(chunks)
    
    def search(self, query: str, k: int = 5) -> List[Dict]:
        if self.index.ntotal == 0: return []
        
        query_embedding = self.model.encode([query])
        distances, indices = self.index.search(query_embedding.astype(np.float32), k)
        
        return [self.chunks[idx] for idx in indices[0] if 0 <= idx < len(self.chunks)]

# === TOKEN-EFFICIENT ANSWER GENERATION ===
class TokenOptimizedGenerator:
    """Optimized answer generation"""
    
    def __init__(self):
        load_dotenv()
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key: raise ValueError("GEMINI_API_KEY not found")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash-latest') # Updated to latest flash model
    
    async def generate_efficient_answer(self, query: str, context_chunks: List[Dict]) -> str:
        if not context_chunks:
            return "No relevant information found in the document."
        
        context = "\n\n".join(chunk['text'] for chunk in context_chunks)
        
        prompt = f"""Based on the context below, answer the question accurately.

CONTEXT:
{context}

QUESTION: {query}

ANSWER:"""
        
        try:
            safety_settings = [
                {"category": c, "threshold": "BLOCK_NONE"}
                for c in ["HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH", 
                          "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT"]
            ]
            
            response = await asyncio.to_thread(
                self.model.generate_content,
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.1, max_output_tokens=300
                ),
                safety_settings=safety_settings
            )
            
            return response.text.strip() if response.parts else "Response blocked by safety filter."
        except Exception as e:
            return f"Error generating answer: {str(e)}"

# === CACHING LAYER ===
class CacheManager:
    """Multi-level caching for speed"""
    
    def __init__(self):
        self.vector_db_cache: Dict[str, OptimizedVectorDB] = {}
        self.answer_cache: Dict[str, str] = {}
    
    def get_cache_key(self, url: str, additional: str = "") -> str:
        return hashlib.md5(f"{url}_{additional}".encode()).hexdigest()

    def get_cached_vector_db(self, url: str) -> Optional[OptimizedVectorDB]:
        return self.vector_db_cache.get(self.get_cache_key(url, "vectordb"))

    def cache_vector_db(self, url: str, vector_db: OptimizedVectorDB):
        self.vector_db_cache[self.get_cache_key(url, "vectordb")] = vector_db

    def get_cached_answer(self, url: str, question: str) -> Optional[str]:
        return self.answer_cache.get(self.get_cache_key(url, question))

    def cache_answer(self, url: str, question: str, answer: str):
        self.answer_cache[self.get_cache_key(url, question)] = answer

cache_manager = CacheManager()

# === MAIN OPTIMIZED PIPELINE ===
async def process_query(url: str, questions: List[str]) -> List[str]:
    start_time = time.time()
    
    vector_db = cache_manager.get_cached_vector_db(url)
    
    if vector_db is None:
        print(f"Processing document: {url}")
        document_text = await DocumentExtractor.extract_from_url(url)
        
        # Use the new, faster chunker
        chunks = SmartChunker.smart_chunk(document_text)
        
        print(f"Created {len(chunks)} smart chunks (Optimized for speed)")
        
        vector_db = OptimizedVectorDB()
        # No need to run add_chunks in executor, SentenceTransformer handles its own parallelism
        vector_db.add_chunks_optimized(chunks)
        
        cache_manager.cache_vector_db(url, vector_db)
        print(f"🚀 Document processed and indexed in {time.time() - start_time:.2f}s")
    
    answer_generator = TokenOptimizedGenerator()
    
    async def answer_single_question(question: str) -> str:
        cached_answer = cache_manager.get_cached_answer(url, question)
        if cached_answer: return cached_answer

        # Search can be quick, so running in executor is optional but safe
        loop = asyncio.get_event_loop()
        relevant_chunks = await loop.run_in_executor(_thread_pool, vector_db.search, question, 4)
        answer = await answer_generator.generate_efficient_answer(question, relevant_chunks)
        
        cache_manager.cache_answer(url, question, answer)
        return answer
    
    tasks = [answer_single_question(q) for q in questions]
    answers = await asyncio.gather(*tasks)
    
    print(f"Total processing time: {time.time() - start_time:.2f}s")
    return answers