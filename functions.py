import PyPDF2
import re
import nltk
from transformers import pipeline
from transformers import AutoTokenizer
import streamlit as st
from init_pinecone import pc, index
from sentence_transformers import SentenceTransformer
from fpdf import FPDF
from datetime import datetime
import io

# Free embedding model - works on Streamlit Cloud, no API key needed
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"  # Free, fast, 384 dimensions, deployable

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL_NAME)

model_name = "facebook/bart-large-cnn"  # 📈 Better quality, 3x larger
# model_name = "sshleifer/distilbart-cnn-12-6"  # 🏃‍♂️ Current (fast)
# model_name = "google/pegasus-xsum"  # 📰 Great for short summaries
# model_name = "microsoft/DialoGPT-medium"  # 💬 Conversational summaries
qa_model_name = "google/flan-t5-large"  # Larger model for better, longer answers
# qa_model_name = "google/flan-t5-large"  # 📈 Better quality, slower
# qa_model_name = "deepset/roberta-base-squad2"  # 🏃 Fast but short span answers only

@st.cache_resource
def load_qa_model():
    return pipeline("text2text-generation", model=qa_model_name, max_new_tokens=200)

@st.cache_resource
def load_summarizer_model():
    summarizer = pipeline("summarization", model=model_name)
    return summarizer

@st.cache_resource
def download_nltk_resources():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
# Remove the immediate call - will be called from app.py after set_page_config


@st.cache_data
def extract_text_from_pdfs(pdf_file):
    text=""
    try:
        reader = PyPDF2.PdfReader(pdf_file)
        for page_num in range(len(reader.pages)):
            page_text = reader.pages[page_num].extract_text()
            if page_text:
                text += page_text
        return text
    except Exception as e:
        print(f"Error extracting text from {pdf_file}: {e}")
        return None
    

@st.cache_data
def preprocess_text(text):
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
            # Better overlap calculation: move forward by chunk size minus overlap
            chunk_size = j - i
            overlap_words = min(overlap_tokens // 4, chunk_size // 2)  # Convert tokens to approximate words
            i = j - overlap_words
            
        chunks.append(chunk_text)
        
        # Safety check to prevent infinite loops
        if i >= len(words):
            break
    
    return chunks

@st.cache_resource
def get_tokenizer_and_max_len():                                                                                                                                                                            
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    max_len = getattr(tokenizer, "model_max_length", 1024)
    if max_len < 1 or max_len > 1_000_000:
        max_len = 1024
    return tokenizer, max_len

def query_pinecone_for_context(user_qus, top_k=3):
    # Use free sentence-transformers instead of OpenAI
    embedding_model = load_embedding_model()
    question_vector = embedding_model.encode(user_qus).tolist()

    # Query Pinecone for similar chunks
    query_response = index.query(
        vector=question_vector,
        top_k=top_k,
        include_metadata=True
    )

    # Extract the chunk text from metadata and clean it
    context_chunks = [match['metadata']['text'] for match in query_response['matches']]
    
    # Clean the context: remove URLs, extra whitespace, and formatting
    cleaned_chunks = []
    for chunk in context_chunks:
        # Remove URLs
        chunk = re.sub(r'https?://\S+', '', chunk)
        chunk = re.sub(r'www\.\S+', '', chunk)
        # Remove multiple spaces
        chunk = re.sub(r'\s+', ' ', chunk).strip()
        if chunk:  # Only add non-empty chunks
            cleaned_chunks.append(chunk)
    
    return " ".join(cleaned_chunks)

def get_expanded_answer(question, relevant_context):
    """
    Uses flan-t5 to generate a detailed, expanded answer from the relevant context.
    """
    qa_pipeline = load_qa_model()

    # Provide more context for better answers
    context_words = relevant_context.split()
    if len(context_words) > 500:  # Increased from 300 to 500
        relevant_context = " ".join(context_words[:500])

    # Clear and direct prompt - less hallucination
    prompt = f"Answer the question using only the context below. Be thorough and detailed.\n\nContext: {relevant_context}\n\nQuestion: {question}\n\nAnswer:"

    try:
        result = qa_pipeline(
            prompt,
            max_new_tokens=300,       # Increased for much longer answers
            min_new_tokens=50,        # Force minimum length
            do_sample=False,
            no_repeat_ngram_size=3,   # Prevent repeating 3-gram phrases
            repetition_penalty=2.0,   # Penalize repetition
            early_stopping=True,
            num_beams=4               # Beam search for better quality
        )
        answer = result[0]["generated_text"].strip()

        # Remove the prompt if it's included in the output
        if "Answer:" in answer:
            answer = answer.split("Answer:")[-1].strip()

        # Clean URLs and metadata artifacts from the answer
        answer = re.sub(r'https?://\S+', '', answer)
        answer = re.sub(r'www\.\S+', '', answer)
        answer = re.sub(r'\(\s*Methodological Reference\s*\)', '', answer)
        answer = re.sub(r'\s+', ' ', answer).strip()

        if not answer or len(answer.split()) < 3:
            answer = "I could not find a relevant answer in the document for this question."

    except Exception as e:
        answer = f"Error generating answer: {str(e)}"

    return answer, 1.0

def get_summaries_for_chunks(summarizer, text_chunks,  min_summary_length=50, max_summary_length=150):
    summaries = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i,chunk in enumerate(text_chunks):
        progress_percent = (i + 1) / len(text_chunks)
        status_text.text(f"Summarizing chunk {i+1}/{len(text_chunks)}...")
        progress_bar.progress(progress_percent)
        try:
            # Free embedding using sentence-transformers
            embedding_model = load_embedding_model()
            vector = embedding_model.encode(chunk).tolist()
            index.upsert(vectors=[{
                "id": f"chunk-{i}",
                "values": vector,
                "metadata": {"text": chunk[:500]}  # Store more context
            }])
            # Calculate appropriate max_length based on chunk length
            chunk_length = len(chunk.split())
            adjusted_max_length = min(max_summary_length, max(10, chunk_length // 2))
            adjusted_min_length = min(min_summary_length, max(5, adjusted_max_length // 2))
            
            # Skip very short chunks that can't be meaningfully summarized
            if chunk_length < 20:
                summaries.append(f"Chunk {i+1}: {chunk[:100]}..." if len(chunk) > 100 else chunk)
                continue
                
            summary_result = summarizer(chunk, 
                                      min_length=adjusted_min_length, 
                                      max_length=adjusted_max_length, 
                                      do_sample=False,
                                      truncation=True)
            
            # Handle different possible return formats
            if isinstance(summary_result, list) and len(summary_result) > 0:
                if isinstance(summary_result[0], dict) and 'summary_text' in summary_result[0]:
                    summaries.append(summary_result[0]['summary_text'])
                else:
                    summaries.append(str(summary_result[0]))
            else:
                summaries.append(f"Could not generate summary for chunk {i+1}")
                
        except Exception as e:
            st.warning(f"Error summarizing chunk {i+1}: {e}")
            # Fallback: use first 200 characters as summary
            fallback_summary = chunk[:200] + "..." if len(chunk) > 200 else chunk
            summaries.append(f"Summary unavailable. Preview: {fallback_summary}")
    st.success("All chunks embedded into Pinecone!")
    progress_bar.empty()
    status_text.empty()
    status_text.text("Summarization complete.")
    return summaries


def generate_summary_pdf(chunk_summaries, final_summary, pdf_filename="summary"):
    """
    Generates a clean, simplified PDF with section summaries first, then final summary.
    Returns the PDF as bytes for Streamlit download.
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

    # ---- Section Summaries ----
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

    # ---- Final Summary ----
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
