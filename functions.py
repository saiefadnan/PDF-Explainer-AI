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
qa_model_name = "google/flan-t5-base"  # 🎯 Generates full explained answers, free & deployable
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

    # Extract the chunk text from metadata
    context_chunks = [match['metadata']['text'] for match in query_response['matches']]
    return " ".join(context_chunks)

def get_expanded_answer(question, relevant_context):
    """
    Uses flan-t5 to generate a clean answer from the relevant context.
    """
    qa_pipeline = load_qa_model()

    # Truncate context to avoid repetition issues
    context_words = relevant_context.split()
    if len(context_words) > 300:
        relevant_context = " ".join(context_words[:300])

    # Clear and direct prompt - less hallucination
    prompt = f"Answer the question using only the context below. Be concise and direct.\n\nContext: {relevant_context}\n\nQuestion: {question}\n\nAnswer:"

    try:
        result = qa_pipeline(
            prompt,
            max_new_tokens=120,       # Limit length to prevent repetition
            do_sample=False,
            no_repeat_ngram_size=4,   # Prevent repeating 4-gram phrases
            repetition_penalty=2.5,   # Penalize repetition heavily
            early_stopping=True,
            num_beams=4               # Beam search for better quality
        )
        answer = result[0]["generated_text"].strip()

        # Remove the prompt if it's included in the output
        if "Answer:" in answer:
            answer = answer.split("Answer:")[-1].strip()

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
    Generates a styled PDF containing chunk summaries and the final document summary.
    Returns the PDF as bytes for Streamlit download.
    """

    class SummaryPDF(FPDF):
        def header(self):
            # Background header bar
            self.set_fill_color(30, 90, 180)          # deep blue
            self.rect(0, 0, 210, 22, "F")
            self.set_font("Helvetica", "B", 14)
            self.set_text_color(255, 255, 255)
            self.set_xy(10, 5)
            self.cell(0, 12, "PDF Explainer AI  -  Summary Report", align="L")
            # Date on the right
            self.set_font("Helvetica", "", 9)
            date_str = datetime.now().strftime("%B %d, %Y")
            self.set_xy(0, 8)
            self.cell(200, 8, date_str, align="R")
            self.set_text_color(0, 0, 0)
            self.ln(20)

        def footer(self):
            self.set_y(-14)
            self.set_fill_color(30, 90, 180)
            self.rect(0, self.get_y(), 210, 14, "F")
            self.set_font("Helvetica", "I", 8)
            self.set_text_color(200, 220, 255)
            self.cell(0, 14, f"Page {self.page_no()}  |  Generated by PDF Explainer AI", align="C")

    pdf = SummaryPDF()
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()
    pdf.set_margins(15, 28, 15)

    # ── Helper to safely encode text for latin-1 ──────────────────────────────
    def safe(text):
        return text.encode("latin-1", errors="replace").decode("latin-1")

    # ── Source file name banner ───────────────────────────────────────────────
    name = pdf_filename if pdf_filename else "Uploaded Document"
    pdf.set_fill_color(235, 242, 255)
    pdf.set_draw_color(30, 90, 180)
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_text_color(30, 90, 180)
    pdf.cell(0, 9, safe(f"  Source: {name}"), border="B", fill=True, ln=True)
    pdf.ln(4)

    # ── Final Document Summary ────────────────────────────────────────────────
    pdf.set_fill_color(30, 90, 180)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 9, "  Overall Document Summary", fill=True, ln=True)
    pdf.ln(2)

    pdf.set_fill_color(240, 245, 255)
    pdf.set_draw_color(30, 90, 180)
    pdf.set_text_color(20, 20, 20)
    pdf.set_font("Helvetica", "", 10)
    pdf.set_line_width(0.5)
    # Draw left accent bar
    x, y = pdf.get_x(), pdf.get_y()
    pdf.set_fill_color(240, 245, 255)
    pdf.multi_cell(0, 6, safe(final_summary), border=0, fill=True)
    # Draw left blue accent line
    pdf.set_draw_color(30, 90, 180)
    pdf.set_line_width(1.2)
    pdf.line(15, y, 15, pdf.get_y())
    pdf.set_line_width(0.2)
    pdf.ln(5)

    # ── Divider ───────────────────────────────────────────────────────────────
    pdf.set_draw_color(180, 180, 180)
    pdf.line(15, pdf.get_y(), 195, pdf.get_y())
    pdf.ln(5)

    # ── Chunk Summaries ───────────────────────────────────────────────────────
    pdf.set_fill_color(30, 90, 180)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 9, f"  Section Summaries  ({len(chunk_summaries)} sections)", fill=True, ln=True)
    pdf.ln(3)

    for i, summary in enumerate(chunk_summaries):
        # Section number badge
        pdf.set_fill_color(30, 90, 180)
        pdf.set_text_color(255, 255, 255)
        pdf.set_font("Helvetica", "B", 9)
        pdf.cell(28, 7, f"  Section {i + 1}", fill=True)

        # Section title line (light blue bg)
        pdf.set_fill_color(210, 225, 255)
        pdf.set_text_color(30, 30, 30)
        pdf.set_font("Helvetica", "I", 9)
        pdf.cell(0, 7, f"  Chunk {i + 1} of {len(chunk_summaries)}", fill=True, ln=True)

        # Summary text box
        pdf.set_fill_color(250, 250, 252)
        pdf.set_draw_color(200, 210, 230)
        pdf.set_text_color(30, 30, 30)
        pdf.set_font("Helvetica", "", 10)
        pdf.set_line_width(0.3)
        pdf.multi_cell(0, 6, safe(summary), border=1, fill=True)
        pdf.ln(4)

    # ── Back-cover footer strip ────────────────────────────────────────────────
    # (handled by the footer() method automatically)

    # ── Return bytes ──────────────────────────────────────────────────────────
    pdf_bytes = bytes(pdf.output())
    return pdf_bytes
