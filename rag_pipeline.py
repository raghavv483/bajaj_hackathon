# rag_pipeline.py

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
_thread_pool = ThreadPoolExecutor(max_workers=4)

# === MULTI-TIER MODEL STRATEGY ===
class ModelManager:
    """Manages multiple embedding models for different use cases"""
    
    def __init__(self):
        self.models = {}
        self.model_lock = threading.Lock()
    
    def get_model(self, model_type: str = "balanced"):
        """Get appropriate model based on speed/accuracy requirements"""
        if model_type not in self.models:
            with self.model_lock:
                if model_type not in self.models:
                    if model_type == "fast":
                        self.models[model_type] = SentenceTransformer(
                            'all-MiniLM-L6-v2',
                            device='cpu',
                            model_kwargs={'torch_dtype': 'float16'}
                        )
                    elif model_type == "accurate":
                        self.models[model_type] = SentenceTransformer('all-mpnet-base-v2')
                    else:  # balanced (default)
                        self.models[model_type] = SentenceTransformer(
                            'all-MiniLM-L6-v2',
                            device='cpu'
                        )
        return self.models[model_type]

# Global model manager
model_manager = ModelManager()

# === DOCUMENT EXTRACTION (INTEGRATED) ===
class DocumentExtractor:
    """Optimized document extraction with minimal processing"""
    
    @staticmethod
    async def extract_from_url(url: str) -> str:
        """Extract text from PDF or DOCX URL"""
        try:
            parsed_url = urlparse(url)
            file_path = parsed_url.path

            async with httpx.AsyncClient() as client:
                async with client.stream('GET', url, follow_redirects=True, timeout=30.0) as response:
                    response.raise_for_status()
                    content = b""
                    async for chunk in response.aiter_bytes():
                        content += chunk

            suffix = ".pdf" if file_path.lower().endswith('.pdf') else ".docx"
            
            if suffix == '.pdf':
                text = await asyncio.get_event_loop().run_in_executor(
                    _thread_pool, DocumentExtractor._extract_from_pdf_fast, content
                )
            else:
                text = await asyncio.get_event_loop().run_in_executor(
                    _thread_pool, DocumentExtractor._extract_from_docx_fast, content
                )
            
            return text
            
        except Exception as e:
            raise IOError(f"Failed to process document from {url}: {e}")

    @staticmethod
    def _extract_from_pdf_fast(content: bytes) -> str:
        """Fast PDF extraction - text only, no table processing"""
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
            temp_file.write(content)
            temp_filepath = temp_file.name
        
        try:
            full_text = ""
            doc = fitz.open(temp_filepath)
            
            for page in doc:
                page_text = page.get_text("text")
                if page_text.strip():
                    full_text += page_text + "\n"
            
            doc.close()
            return full_text
        finally:
            os.unlink(temp_filepath)

    @staticmethod
    def _extract_from_docx_fast(content: bytes) -> str:
        """Fast DOCX extraction"""
        with tempfile.NamedTemporaryFile(suffix='.docx', delete=False) as temp_file:
            temp_file.write(content)
            temp_filepath = temp_file.name
        
        try:
            text_parts = []
            doc = Document(temp_filepath)
            
            for para in doc.paragraphs:
                if para.text.strip():
                    text_parts.append(para.text)
            
            return "\n".join(text_parts)
        finally:
            os.unlink(temp_filepath)

# === INTELLIGENT CHUNKING STRATEGY ===
class SmartChunker:
    """Intelligent chunking that preserves context while optimizing for speed"""
    
    @staticmethod
    def smart_chunk(text: str, chunk_size: int = 900, overlap: int = 100) -> List[Dict]:
        if not text:
            return []
        
        text = text.replace('\n\n\n+', '\n\n')
        text_len = len(text)
        
        if text_len <= chunk_size:
            return [{"text": text.strip(), "position": 0, "importance": 1.0}]
        
        chunks = []
        sentences = SmartChunker._fast_sentence_split(text)
        
        current_chunk = []
        current_size = 0
        position = 0
        
        for i, sentence in enumerate(sentences):
            sentence_len = len(sentence)
            
            if sentence_len > chunk_size:
                if current_chunk:
                    chunk_text = ' '.join(current_chunk).strip()
                    chunks.append({
                        "text": chunk_text,
                        "position": position,
                        "importance": SmartChunker._calculate_importance(chunk_text)
                    })
                    current_chunk = []
                    current_size = 0
                
                sub_chunks = SmartChunker._split_long_text(sentence, chunk_size)
                for sub_chunk in sub_chunks:
                    chunks.append({
                        "text": sub_chunk,
                        "position": position,
                        "importance": SmartChunker._calculate_importance(sub_chunk)
                    })
                    position += 1
            
            elif current_size + sentence_len <= chunk_size:
                current_chunk.append(sentence)
                current_size += sentence_len + 1
            else:
                if current_chunk:
                    chunk_text = ' '.join(current_chunk).strip()
                    chunks.append({
                        "text": chunk_text,
                        "position": position,
                        "importance": SmartChunker._calculate_importance(chunk_text)
                    })
                    position += 1
                
                overlap_sentences = current_chunk[-2:] if len(current_chunk) > 1 else current_chunk
                current_chunk = overlap_sentences + [sentence]
                current_size = sum(len(s) for s in current_chunk) + len(current_chunk)
        
        if current_chunk:
            chunk_text = ' '.join(current_chunk).strip()
            chunks.append({
                "text": chunk_text,
                "position": position,
                "importance": SmartChunker._calculate_importance(chunk_text)
            })
        
        return [c for c in chunks if len(c["text"]) > 50]
    
    @staticmethod
    def _fast_sentence_split(text: str) -> List[str]:
        import re
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    @staticmethod
    def _split_long_text(text: str, max_size: int) -> List[str]:
        words = text.split()
        chunks = []
        current_chunk = []
        current_size = 0
        
        for word in words:
            word_len = len(word) + 1
            if current_size + word_len > max_size and current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = [word]
                current_size = len(word)
            else:
                current_chunk.append(word)
                current_size += word_len
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    @staticmethod
    def _calculate_importance(text: str) -> float:
        importance = 1.0
        length = len(text)
        if 200 <= length <= 800:
            importance += 0.2
        if any(word in text.lower() for word in ['what', 'how', 'why', 'when', 'where', 'which']):
            importance += 0.3
        if any(phrase in text.lower() for phrase in ['is defined as', 'refers to', 'means', 'is the']):
            importance += 0.2
        if any(char.isdigit() for char in text):
            importance += 0.1
        return min(importance, 2.0)

# === OPTIMIZED VECTOR DATABASE ===
class OptimizedVectorDB:
    """High-performance vector DB with smart retrieval"""
    
    def __init__(self, model_type: str = "balanced"):
        self.model = model_manager.get_model(model_type)
        self.dimension = self.model.get_sentence_embedding_dimension()
        self.index = faiss.IndexFlatL2(self.dimension)
        self.chunks = []
        self.embeddings_cache = {}
        
    def add_chunks_optimized(self, chunks: List[Dict], batch_size: int = 16):
        if not chunks:
            return
        
        texts = [chunk["text"] for chunk in chunks]
        
        cache_key = hashlib.md5(''.join(texts).encode()).hexdigest()
        if cache_key in self.embeddings_cache:
            embeddings = self.embeddings_cache[cache_key]
        else:
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=False,
                convert_to_tensor=False,
                normalize_embeddings=True
            )
            self.embeddings_cache[cache_key] = embeddings
        
        self.index.add(embeddings.astype(np.float32))
        self.chunks.extend(chunks)
    
    def search_with_reranking(self, query: str, k: int = 5) -> List[Dict]:
        if self.index.ntotal == 0:
            return []
        
        initial_k = min(k * 3, len(self.chunks), 15)
        
        query_embedding = self.model.encode([query], normalize_embeddings=True)
        distances, indices = self.index.search(
            query_embedding.astype(np.float32), 
            initial_k
        )
        
        candidates = []
        for i, idx in enumerate(indices[0]):
            if 0 <= idx < len(self.chunks):
                chunk = self.chunks[idx]
                similarity_score = 1 / (1 + distances[0][i])
                final_score = similarity_score * chunk.get("importance", 1.0)
                
                candidates.append({
                    **chunk,
                    "similarity_score": similarity_score,
                    "final_score": final_score
                })
        
        candidates.sort(key=lambda x: x["final_score"], reverse=True)
        return candidates[:k]

# === TOKEN-EFFICIENT ANSWER GENERATION ===
class TokenOptimizedGenerator:
    """Optimized answer generation focusing on token efficiency"""
    
    def __init__(self):
        load_dotenv()
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.5-flash-preview-05-20')
    
    async def generate_efficient_answer(self, query: str, context_chunks: List[Dict]) -> str:
        if not context_chunks:
            return "No relevant information found in the document."
        
        context = self._build_optimal_context(context_chunks, max_tokens=1500)
        
        prompt = f"""Based on the provided context, answer the question accurately and concisely.

CONTEXT:
{context}

QUESTION: {query}

INSTRUCTIONS:
- Provide a direct, accurate answer based solely on the context
- Be concise but complete
- If information is not in the context, state this clearly
- Use specific details from the context when available

ANSWER:"""
        
        try:
            # --- SAFETY SETTINGS ADDED HERE ---
            safety_settings = [
                {
                    "category": "HARM_CATEGORY_HARASSMENT",
                    "threshold": "BLOCK_NONE",
                },
                {
                    "category": "HARM_CATEGORY_HATE_SPEECH",
                    "threshold": "BLOCK_NONE",
                },
                {
                    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "threshold": "BLOCK_NONE",
                },
                {
                    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                    "threshold": "BLOCK_NONE",
                },
            ]

            response = await asyncio.to_thread(
                self.model.generate_content,
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.1,
                    max_output_tokens=300,
                    top_p=0.8
                ),
                safety_settings=safety_settings # Pass the safety settings
            )
            # --- END OF FIX ---
            
            # Check for valid response before accessing .text
            if response.parts:
                return response.text.strip()
            else:
                # Handle cases where the response was blocked
                return "The response was blocked by the safety filter. Please try rephrasing your question."

        except Exception as e:
            return f"Error generating answer: {str(e)}"
    
    def _build_optimal_context(self, chunks: List[Dict], max_tokens: int = 1500) -> str:
        current_tokens = 0
        selected_chunks = []
        
        sorted_chunks = sorted(chunks, key=lambda x: x.get("final_score", 0), reverse=True)
        
        for chunk in sorted_chunks:
            chunk_text = chunk["text"]
            chunk_tokens = len(chunk_text) // 4
            
            if current_tokens + chunk_tokens <= max_tokens:
                selected_chunks.append(chunk_text)
                current_tokens += chunk_tokens
            else:
                remaining_tokens = max_tokens - current_tokens
                if remaining_tokens > 50:
                    partial_text = chunk_text[:remaining_tokens * 4]
                    last_period = partial_text.rfind('.')
                    if last_period > len(partial_text) * 0.7:
                        partial_text = partial_text[:last_period + 1]
                    selected_chunks.append(partial_text)
                break
        
        return "\n\n".join(selected_chunks)

# === CACHING LAYER ===
class CacheManager:
    """Multi-level caching for maximum speed"""
    
    def __init__(self):
        self.text_cache: Dict[str, str] = {}
        self.chunk_cache: Dict[str, List[Dict]] = {}
        self.embedding_cache: Dict[str, 'OptimizedVectorDB'] = {}
        self.answer_cache: Dict[str, str] = {}
    
    def get_cache_key(self, url: str, additional: str = "") -> str:
        return hashlib.md5(f"{url}_{additional}".encode()).hexdigest()
    
    def cache_text(self, url: str, text: str):
        key = self.get_cache_key(url, "text")
        self.text_cache[key] = text
    
    def get_cached_text(self, url: str) -> Optional[str]:
        key = self.get_cache_key(url, "text")
        return self.text_cache.get(key)
    
    def cache_vector_db(self, url: str, vector_db: OptimizedVectorDB):
        key = self.get_cache_key(url, "vectordb")
        self.embedding_cache[key] = vector_db
    
    def get_cached_vector_db(self, url: str) -> Optional[OptimizedVectorDB]:
        key = self.get_cache_key(url, "vectordb")
        return self.embedding_cache.get(key)
    
    def cache_answer(self, url: str, question: str, answer: str):
        key = self.get_cache_key(url, question)
        self.answer_cache[key] = answer
    
    def get_cached_answer(self, url: str, question: str) -> Optional[str]:
        key = self.get_cache_key(url, question)
        return self.answer_cache.get(key)

# Global cache manager
cache_manager = CacheManager()

# === MAIN OPTIMIZED PIPELINE ===
async def process_query_balanced(url: str, questions: List[str]) -> List[str]:
    """
    Balanced optimization for accuracy, token efficiency, and latency
    Target: 5-10 seconds with high accuracy
    """
    start_time = time.time()
    
    cached_answers = []
    uncached_questions = []
    question_indices = []
    
    for i, question in enumerate(questions):
        cached_answer = cache_manager.get_cached_answer(url, question)
        if cached_answer:
            cached_answers.append((i, cached_answer))
        else:
            uncached_questions.append(question)
            question_indices.append(i)
    
    if not uncached_questions:
        result = [''] * len(questions)
        for idx, answer in cached_answers:
            result[idx] = answer
        return result
    
    vector_db = cache_manager.get_cached_vector_db(url)
    
    if vector_db is None:
        print(f"Processing document: {url}")
        
        cached_text = cache_manager.get_cached_text(url)
        if cached_text:
            document_text = cached_text
        else:
            document_text = await DocumentExtractor.extract_from_url(url)
            cache_manager.cache_text(url, document_text)
        
        chunks = await asyncio.get_event_loop().run_in_executor(
            None, SmartChunker.smart_chunk, document_text
        )
        
        print(f"Created {len(chunks)} smart chunks")
        
        vector_db = OptimizedVectorDB(model_type="balanced")
        await asyncio.get_event_loop().run_in_executor(
            None, vector_db.add_chunks_optimized, chunks
        )
        
        cache_manager.cache_vector_db(url, vector_db)
        
        processing_time = time.time() - start_time
        print(f"Document processed in {processing_time:.2f}s")
    
    answer_generator = TokenOptimizedGenerator()
    
    async def answer_single_question(question: str, question_idx: int) -> Tuple[int, str]:
        relevant_chunks = await asyncio.get_event_loop().run_in_executor(
            None, vector_db.search_with_reranking, question, 4
        )
        
        answer = await answer_generator.generate_efficient_answer(question, relevant_chunks)
        
        cache_manager.cache_answer(url, question, answer)
        
        return question_idx, answer
    
    tasks = [
        answer_single_question(q, question_indices[i]) 
        for i, q in enumerate(uncached_questions)
    ]
    
    new_answers = await asyncio.gather(*tasks)
    
    result = [''] * len(questions)
    
    for idx, answer in cached_answers:
        result[idx] = answer
    
    for idx, answer in new_answers:
        result[idx] = answer
    
    total_time = time.time() - start_time
    print(f"Total processing time: {total_time:.2f}s")
    
    return result

# === BACKWARDS COMPATIBILITY ===
# Export the main function (backwards compatibility with your existing main.py)
async def process_query(url: str, questions: List[str]) -> List[str]:
    """Main entry point - maintains API compatibility"""
    return await process_query_balanced(url, questions)

# Also export the optimized version for direct access
async def process_query_optimized(url: str, questions: List[str]) -> List[str]:
    """Alias for the optimized function"""
    return await process_query_balanced(url, questions)
