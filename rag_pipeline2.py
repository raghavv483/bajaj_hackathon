# rag_pipeline2.py - FINAL CORRECTED VERSION

import os
import httpx
import tempfile
import asyncio
import hashlib
import numpy as np
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor
import threading
import time
from urllib.parse import urlparse

# Core imports
import fitz  # PyMuPDF
from docx import Document
from sentence_transformers import SentenceTransformer
import faiss
from dotenv import load_dotenv
from openai import OpenAI

# NOTE: The primary error was a recurring typo in the class constructors.
# They were named _init_ (single underscore) instead of the correct __init__ (double underscore).
# This has been corrected in all classes below.

_thread_pool = ThreadPoolExecutor(max_workers=os.cpu_count() or 4)

class ModelManager:
    # FIXED: Corrected constructor to __init__
    def __init__(self):
        self.models = {}
        self.model_lock = threading.Lock()

    def get_model(self, model_type: str = "fast"):
        if model_type not in self.models:
            with self.model_lock:
                if model_type not in self.models:
                    device = 'cuda' if 'CUDA_VISIBLE_DEVICES' in os.environ else 'cpu'
                    self.models[model_type] = SentenceTransformer('all-MiniLM-L6-v2', device=device)
        return self.models[model_type]

model_manager = ModelManager()

class DocumentExtractor:
    # DocumentExtractor class is correct and unchanged
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
                return await loop.run_in_executor(_thread_pool, DocumentExtractor._extract_from_pdf_fast, content)
            else:
                return await loop.run_in_executor(_thread_pool, DocumentExtractor._extract_from_docx_fast, content)
        except Exception as e:
            raise IOError(f"Failed to process document from {url}: {e}")
    @staticmethod
    def _extract_from_pdf_fast(content: bytes) -> str:
        with fitz.open(stream=content, filetype="pdf") as doc:
            return "\n".join(page.get_text("text") for page in doc)
    @staticmethod
    def _extract_from_docx_fast(content: bytes) -> str:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as temp_file:
            temp_file.write(content)
            temp_filepath = temp_file.name
        try:
            doc = Document(temp_filepath)
            return "\n".join(para.text for para in doc.paragraphs if para.text.strip())
        finally:
            os.unlink(temp_filepath)

class SmartChunker:
    # SmartChunker class is correct and unchanged
    @staticmethod
    def smart_chunk(text: str, chunk_size: int = 1500, overlap: int = 100) -> List[Dict]:
        if not text: return []
        splits = text.split('\n\n')
        chunks, position, current_chunk = [], 0, ""
        for split in splits:
            if not split.strip(): continue
            if len(current_chunk) + len(split) + 1 > chunk_size:
                if current_chunk.strip():
                    chunks.append({"text": current_chunk.strip(), "position": position})
                    position += 1
                current_chunk = current_chunk[-overlap:] + " " + split
            else:
                current_chunk += "\n\n" + split
        if current_chunk.strip():
            chunks.append({"text": current_chunk.strip(), "position": position})
        return chunks

class OptimizedVectorDB:
    # FIXED: Corrected constructor to __init__
    def __init__(self, model_type: str = "fast"):
        self.model = model_manager.get_model(model_type)
        self.dimension = self.model.get_sentence_embedding_dimension()
        self.index = faiss.IndexFlatL2(self.dimension)
        self.chunks = []

    def add_chunks_optimized(self, chunks: List[Dict], batch_size: int = 64):
        if not chunks: return
        texts = [chunk["text"] for chunk in chunks]
        embeddings = self.model.encode(
            texts, batch_size=batch_size, show_progress_bar=False, convert_to_numpy=True
        )
        self.index.add(embeddings.astype(np.float32))
        self.chunks.extend(chunks)

    def search(self, query: str, k: int = 5) -> List[Dict]:
        if self.index.ntotal == 0: return []
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        distances, indices = self.index.search(query_embedding.astype(np.float32), k)
        return [self.chunks[idx] for idx in indices[0] if 0 <= idx < len(self.chunks)]

class TokenOptimizedGenerator:
    # FIXED: Corrected constructor to __init__
    def __init__(self):
        load_dotenv()
        self.api_key = os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY not found in .env file.")
        self.client = OpenAI(api_key=self.api_key, base_url="https://api.groq.com/openai/v1")
        self.model_name = "llama3-8b-8192"

    async def generate_efficient_answer(self, query: str, context_chunks: List[Dict]) -> str:
        if not context_chunks:
            return "No relevant information found in the document."
        
        context = "\n\n".join(chunk['text'] for chunk in context_chunks)
        
        prompt = f"Based on the context below, answer the question accurately and concisely.\n\nCONTEXT:\n{context}\n\nQUESTION: {query}\n\nANSWER:"
        
        def get_completion():
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that provides concise answers based on the provided context."},
                    {"role": "user", "content": prompt},
                ],
                model=self.model_name, temperature=0.1, max_tokens=300,
            )
            return chat_completion.choices[0].message.content
        
        try:
            response_text = await asyncio.to_thread(get_completion)
            return response_text.strip()
        except Exception as e:
            return f"Error generating answer from Groq: {str(e)}"

class CacheManager:
    # FIXED: Corrected constructor to __init__
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

async def process_query(url: str, questions: List[str]) -> List[str]:
    start_time = time.time()
    vector_db = cache_manager.get_cached_vector_db(url)
    if vector_db is None:
        print(f"Processing document: {url}")
        document_text = await DocumentExtractor.extract_from_url(url)
        chunks = await asyncio.to_thread(SmartChunker.smart_chunk, document_text)
        vector_db = OptimizedVectorDB()
        await asyncio.to_thread(vector_db.add_chunks_optimized, chunks)
        cache_manager.cache_vector_db(url, vector_db)
        print(f"🚀 Document processed and indexed in {time.time() - start_time:.2f}s")

    answer_generator = TokenOptimizedGenerator()

    async def answer_single_question(question: str) -> str:
        cached_answer = cache_manager.get_cached_answer(url, question)
        if cached_answer:
            return cached_answer
        
        relevant_chunks = await asyncio.to_thread(vector_db.search, question, 4)
        answer = await answer_generator.generate_efficient_answer(question, relevant_chunks)
        cache_manager.cache_answer(url, question, answer)
        return answer

    tasks = [answer_single_question(q) for q in questions]
    answers = await asyncio.gather(*tasks)

    print(f"Total processing time: {time.time() - start_time:.2f}s")
    return answers