import PyPDF2
import re
import nltk
import time
import uuid
import streamlit as st
from init_pinecone import pc, index
from sentence_transformers import SentenceTransformer
from fpdf import FPDF
from datetime import datetime
import io
import requests
from typing import List, Tuple
import threading
from collections import deque

# Free embedding model - works on Streamlit Cloud, no API key needed
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"  # Free, fast, 384 dimensions

# API Configuration
HF_API_TOKEN = st.secrets.get("HF_API_TOKEN", "")
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", "")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

# Rate limiting configuration
MAX_REQUESTS_PER_MINUTE = 25  # Conservative limit (Groq free tier = 30 RPM)
MIN_REQUEST_INTERVAL = 60 / MAX_REQUESTS_PER_MINUTE  # Seconds between requests

class RateLimiter:
    """Thread-safe rate limiter to prevent API quota exhaustion"""
    
    def __init__(self, max_calls_per_minute=25):
        self.max_calls = max_calls_per_minute
        self.interval = 60.0 / max_calls_per_minute
        self.timestamps = deque()
        self.lock = threading.Lock()
    
    def wait_if_needed(self):
        """Wait if necessary to respect rate limits"""
        with self.lock:
            now = time.time()
            
            # Remove timestamps older than 1 minute
            while self.timestamps and self.timestamps[0] < now - 60:
                self.timestamps.popleft()
            
            # If at limit, wait until oldest request expires
            if len(self.timestamps) >= self.max_calls:
                sleep_time = 60 - (now - self.timestamps[0]) + 0.1
                if sleep_time > 0:
                    time.sleep(sleep_time)
                    now = time.time()
                    # Clean up again after sleeping
                    while self.timestamps and self.timestamps[0] < now - 60:
                        self.timestamps.popleft()
            
            # Add current timestamp
            self.timestamps.append(time.time())
            
            # Always enforce minimum interval between requests
            if len(self.timestamps) > 1:
                time_since_last = now - self.timestamps[-2]
                if time_since_last < self.interval:
                    time.sleep(self.interval - time_since_last)

# Global rate limiter instance
rate_limiter = RateLimiter(max_calls_per_minute=MAX_REQUESTS_PER_MINUTE)

def validate_api_keys():
    """Validate that required API keys are present"""
    missing_keys = []
    
    if not GROQ_API_KEY:
        missing_keys.append("GROQ_API_KEY")
    
    if missing_keys:
        st.error(f"⚠️ Missing API keys: {', '.join(missing_keys)}")
        st.info("Please add these keys to your `.streamlit/secrets.toml` file")
        st.code("""
# .streamlit/secrets.toml
GROQ_API_KEY = "your_key_here"
HF_API_TOKEN = "your_key_here"  # Optional, for future features
        """)
        st.stop()

@st.cache_resource
def load_embedding_model():
    """Load the sentence transformer model for embeddings"""
    try:
        return SentenceTransformer(EMBEDDING_MODEL_NAME)
    except Exception as e:
        st.error(f"Failed to load embedding model: {str(e)}")
        st.stop()

@st.cache_resource
def download_nltk_resources():
    """Download required NLTK resources"""
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        try:
            nltk.download('punkt', quiet=True)
        except Exception as e:
            st.warning(f"Could not download NLTK resources: {str(e)}")

@st.cache_data
def extract_text_from_pdfs(pdf_bytes: bytes):
    """Extract text from PDF bytes"""
    text = ""
    try:
        reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
        return text
    except Exception as e:
        raise Exception(f"Error extracting text from PDF: {str(e)}")

@st.cache_data
def preprocess_text(text):
    """Clean and normalize text"""
    if not text:
        return ""
    
    # Remove multiple newlines and replace with single space
    text = re.sub(r'\n+', ' ', text)
    # Remove multiple spaces and replace with single space
    text = re.sub(r'\s+', ' ', text)
    # Remove common form feed characters
    text = text.replace('\x0c', '')
    # Trim leading/trailing whitespace
    text = text.strip()
    
    return text

@st.cache_data
def chunk_text_by_token(text, _tokenizer, max_len, reserve_space=128, overlap_tokens=40):
    """Split text into chunks based on token count"""
    if not text:
        return []
    
    words = text.split()
    chunks = []
    i = 0
    
    while i < len(words):
        j = i
        curr_token = 0
        
        # Find the maximum number of words that fit within token limit
        while j < len(words):
            candidate = " ".join(words[i:j+1])
            token_count = len(_tokenizer.encode(candidate, add_special_tokens=False))
            if token_count + reserve_space > max_len:
                break
            curr_token = token_count
            j += 1
        
        # Handle edge case where even one word is too long
        if j == i:
            chunk_text = " ".join(words[i:i+50])  # Take 50 words max
            i += 50
        else:
            chunk_text = " ".join(words[i:j])
            # Better overlap calculation
            chunk_size = j - i
            overlap_words = min(overlap_tokens // 4, chunk_size // 2)
            i = j - overlap_words
        
        chunks.append(chunk_text)
        
        # Safety check to prevent infinite loops
        if i >= len(words):
            break
    
    return chunks

@st.cache_resource
def get_tokenizer_and_max_len():
    """Get a simple word-based tokenizer for chunking"""
    class SimpleTokenizer:
        def encode(self, text, add_special_tokens=False):
            # Simple word-based tokenization (~4 chars per token average)
            return text.split()
        
        def __getattr__(self, name):
            if name == "model_max_length":
                return 1024
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
    
    tokenizer = SimpleTokenizer()
    max_len = 1024
    return tokenizer, max_len

def query_pinecone_for_context(user_question, top_k=10):
    """Query Pinecone for relevant context chunks"""
    try:
        # Use sentence-transformers for embeddings
        embedding_model = load_embedding_model()
        question_vector = embedding_model.encode(user_question).tolist()
        pdf_id = st.session_state.get("pdf_id", None)
        # Query Pinecone for similar chunks
        query_response = index.query(
            vector=question_vector,
            top_k=top_k,
            include_metadata=True,
            namespace=pdf_id
        )
        # Extract and clean the chunk text from metadata
        context_chunks = [match['metadata']['text'] for match in query_response['matches']]
        
        # Clean the context: remove URLs, extra whitespace, and formatting
        cleaned_chunks = []
        for chunk in context_chunks:
            # Remove URLs
            chunk = re.sub(r'https?://\S+', '', chunk)
            chunk = re.sub(r'www\.\S+', '', chunk)
            # Remove multiple spaces
            chunk = re.sub(r'\s+', ' ', chunk).strip()
            if chunk:
                cleaned_chunks.append(chunk)
        
        return " ".join(cleaned_chunks)
    except Exception as e:
        st.error(f"Error querying Pinecone: {str(e)}")
        return ""

def call_groq_api_with_retry(payload, max_retries=3):
    """
    Call Groq API with automatic rate limiting and retry logic
    
    Args:
        payload: The request payload
        max_retries: Maximum number of retry attempts
    
    Returns:
        API response text or None on failure
    """
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    
    for attempt in range(max_retries):
        try:
            # Wait if needed to respect rate limits
            rate_limiter.wait_if_needed()
            
            response = requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=60)
            
            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"].strip()
            
            elif response.status_code == 429:
                # Rate limit hit - wait longer
                wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
                st.warning(f"⏳ Rate limit reached. Waiting {wait_time}s before retry...")
                time.sleep(wait_time)
                continue
            
            elif response.status_code >= 500:
                # Server error - retry
                if attempt < max_retries - 1:
                    time.sleep(1)
                    continue
                else:
                    return None
            
            else:
                # Other errors - don't retry
                st.error(f"API Error: {response.status_code}")
                return None
                
        except requests.exceptions.Timeout:
            if attempt < max_retries - 1:
                st.warning("⏳ Request timed out, retrying...")
                time.sleep(1)
                continue
            else:
                return None
        
        except Exception as e:
            st.error(f"Error calling API: {str(e)}")
            return None
    
    return None

def get_expanded_answer(question, relevant_context):
    """Get answer from Groq API using context"""
    # Limit context length - match what we store in Pinecone (10000 chars)
    context_words = relevant_context.split()
    if len(context_words) > 2000:  # ~10000 chars at ~5 chars/word average
        relevant_context = " ".join(context_words[:2000])

    payload = {
        "model": "llama-3.1-8b-instant",
        "messages": [{
            "role": "user",
            "content": f"""You are a helpful AI assistant analyzing a PDF document.
                CONTEXT FROM THE DOCUMENT:
                {relevant_context}
                USER QUESTION:
                {question}
                INSTRUCTIONS:
                - Answer based ONLY on the context provided
                - Be specific and cite relevant details
                - If context is insufficient, say so clearly
                - Keep answer focused (2-4 sentences)
                ANSWER:"""

        }],
        "max_tokens": 500
    }
    
    answer = call_groq_api_with_retry(payload)
    
    if answer is None:
        answer = "Sorry, I couldn't generate an answer. Please try again."
    
    return answer, 1.0

def get_summary_for_text(text, min_length=50, max_length=150):
    """Generate summary using Groq API with rate limiting"""
    payload = {
        "model": "llama-3.1-8b-instant",
        "messages": [{
            "role": "user",
            "content": f"Summarize the following text in {min_length}-{max_length} words. Only return the summary, nothing else.\n\nText: {text}"
        }],
        "max_tokens": max_length * 2
    }
    
    summary = call_groq_api_with_retry(payload)
    
    if summary is None:
        return f"Summary unavailable. Preview: {text[:100]}..."
    
    return summary

def batch_summarize_chunks(text_chunks: List[str], batch_size: int = 3) -> List[str]:
    """
    Summarize multiple chunks in a single API call for efficiency
    
    Args:
        text_chunks: List of text chunks to summarize
        batch_size: Number of chunks to combine per API call
    
    Returns:
        List of summaries
    """
    all_summaries = []
    
    for i in range(0, len(text_chunks), batch_size):
        batch = text_chunks[i:i+batch_size]
        
        # Combine chunks with clear separators
        combined_text = "\n\n---CHUNK SEPARATOR---\n\n".join(
            [f"CHUNK {j+1}:\n{chunk}" for j, chunk in enumerate(batch)]
        )
        
        payload = {
            "model": "llama-3.1-8b-instant",
            "messages": [{
                "role": "user",
                "content": f"""Summarize each chunk separately. Return summaries in the format:
                    SUMMARY 1: [summary here]
                    SUMMARY 2: [summary here]
                    etc.
                    {combined_text}"""
            }],
            "max_tokens": 500 * len(batch)
        }
        
        result = call_groq_api_with_retry(payload)
        
        if result:
            # Parse individual summaries
            summaries = re.findall(r'SUMMARY \d+: (.+?)(?=SUMMARY \d+:|$)', result, re.DOTALL)
            all_summaries.extend([s.strip() for s in summaries])
        else:
            # Fallback: use chunk previews
            all_summaries.extend([f"Preview: {chunk[:100]}..." for chunk in batch])
    
    return all_summaries

def get_summaries_for_chunks(text_chunks, min_summary_length=50, max_summary_length=150):
    """
    Generate summaries for all text chunks with smart rate limiting
    
    Options:
    1. Sequential (safe but slow): One API call per chunk
    2. Batched (faster): Multiple chunks per API call
    """
    summaries = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Load embedding model once
    embedding_model = load_embedding_model()
    pdf_id = f"pdf-{st.session_state.pdf_filename}-{uuid.uuid4()}"
    st.session_state.pdf_id = pdf_id  # Store in session state for later use
    # Determine strategy based on chunk count
    num_chunks = len(text_chunks)
    
    if num_chunks > 30:
        st.warning(f"⚠️ Large document detected ({num_chunks} chunks). Using batched processing to avoid rate limits...")
        use_batching = True
        batch_size = 3  # Summarize 3 chunks per API call
    else:
        use_batching = False
    
    if use_batching:
        # BATCHED APPROACH - Faster, fewer API calls
        status_text.text("Using batched summarization for efficiency...")
        summaries = batch_summarize_chunks(text_chunks, batch_size=3)
        
        # Still need to embed chunks individually
        for i, chunk in enumerate(text_chunks):
            progress_percent = (i + 1) / len(text_chunks)
            status_text.text(f"Embedding chunk {i+1}/{len(text_chunks)} to Pinecone...")
            progress_bar.progress(progress_percent)
            
            try:
                vector = embedding_model.encode(chunk).tolist()
                index.upsert(vectors=[{
                    "id": f"chunk-{i}",
                    "values": vector,
                    "metadata": {"text": chunk[:10000]}
                }],namespace=pdf_id)
            except Exception as e:
                st.warning(f"Error embedding chunk {i+1}: {str(e)}")
    
    else:
        # SEQUENTIAL APPROACH - Safer, better quality
        for i, chunk in enumerate(text_chunks):
            progress_percent = (i + 1) / len(text_chunks)
            status_text.text(f"Processing chunk {i+1}/{len(text_chunks)}...")
            progress_bar.progress(progress_percent)
            
            try:
                # Embed to Pinecone
                vector = embedding_model.encode(chunk).tolist()
                index.upsert(vectors=[{
                    "id": f"chunk-{i}",
                    "values": vector,
                    "metadata": {"text": chunk[:10000]}
                }],namespace=pdf_id)
                
                # Calculate appropriate lengths
                chunk_length = len(chunk.split())
                adjusted_max_length = min(max_summary_length, max(10, chunk_length // 2))
                adjusted_min_length = min(min_summary_length, max(5, adjusted_max_length // 2))
                
                # Skip very short chunks
                if chunk_length < 20:
                    summaries.append(f"Chunk {i+1}: {chunk[:100]}..." if len(chunk) > 100 else chunk)
                    continue
                
                # Generate summary with rate limiting
                summary = get_summary_for_text(
                    chunk,
                    min_length=adjusted_min_length,
                    max_length=adjusted_max_length
                )
                
                summaries.append(summary)
                
            except Exception as e:
                st.warning(f"Error processing chunk {i+1}: {str(e)}")
                fallback_summary = chunk[:200] + "..." if len(chunk) > 200 else chunk
                summaries.append(f"Summary unavailable. Preview: {fallback_summary}")
    
    progress_bar.empty()
    status_text.empty()
    st.success("✅ All chunks processed and embedded to Pinecone!")
    
    return summaries

def generate_summary_pdf(chunk_summaries, final_summary, pdf_filename="summary"):
    """
    Generate a clean PDF with section summaries and final summary.
    Returns PDF as bytes for Streamlit download.
    """
    class SummaryPDF(FPDF):
        def header(self):
            # Simple header bar
            self.set_fill_color(30, 90, 180)
            self.rect(0, 0, 210, 18, "F")
            self.set_font("Helvetica", "B", 13)
            self.set_text_color(255, 255, 255)
            self.set_xy(10, 5)
            self.cell(0, 8, "PDF Explainer AI - Summary Report", align="L")
            self.set_text_color(0, 0, 0)
            self.ln(18)

        def footer(self):
            self.set_y(-10)
            self.set_font("Helvetica", "I", 8)
            self.set_text_color(128, 128, 128)
            self.cell(0, 10, f"Page {self.page_no()}", align="C")

    pdf = SummaryPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_margins(12, 22, 12)

    # Helper to safely encode text for latin-1
    def safe(text):
        return text.encode("latin-1", errors="replace").decode("latin-1")

    # Document title and source
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(30, 90, 180)
    name = pdf_filename.replace(".pdf", "") if pdf_filename else "Document"
    pdf.cell(0, 8, safe(f"Source: {name}"), ln=True)
    pdf.ln(3)

    # Section Summaries
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(255, 255, 255)
    pdf.set_fill_color(30, 90, 180)
    pdf.cell(0, 7, "  SECTION SUMMARIES", fill=True, ln=True)
    pdf.ln(4)

    for i, summary in enumerate(chunk_summaries):
        # Section number
        pdf.set_font("Helvetica", "B", 10)
        pdf.set_text_color(30, 90, 180)
        pdf.cell(0, 6, f"Section {i + 1}", ln=True)
        
        # Summary text
        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(0, 0, 0)
        pdf.multi_cell(0, 5, safe(summary), ln=True)
        pdf.ln(2)

    # Final Summary
    pdf.ln(2)
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(255, 255, 255)
    pdf.set_fill_color(30, 90, 180)
    pdf.cell(0, 7, "  FINAL SUMMARY", fill=True, ln=True)
    pdf.ln(4)

    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(0, 0, 0)
    pdf.multi_cell(0, 5, safe(final_summary), ln=True)

    # Return PDF as bytes
    pdf_bytes = bytes(pdf.output())
    return pdf_bytes