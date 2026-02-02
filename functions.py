import PyPDF2
import re
import nltk
from transformers import pipeline
from transformers import AutoTokenizer
import streamlit as st

model_name = "facebook/bart-large-cnn"  # 📈 Better quality, 3x larger
# model_name = "sshleifer/distilbart-cnn-12-6"  # 🏃‍♂️ Current (fast)
# model_name = "google/pegasus-xsum"  # 📰 Great for short summaries
# model_name = "microsoft/DialoGPT-medium"  # 💬 Conversational summaries
qa_model_name = "deepset/roberta-base-squad2"  # 🎯 Handles unanswerable questions
# qa_model_name = "distilbert-base-cased-distilled-squad"  # 🏃‍♂️ Current (fast)
# qa_model_name = "bert-large-uncased-whole-word-masking-finetuned-squad"  # 📈 Best accuracy
# qa_model_name = "microsoft/DialoGPT-medium"  # 💬 Conversational Q&A

@st.cache_resource
def load_qa_model():
    return pipeline("question-answering", model=qa_model_name)


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
            text += reader.pages[page_num].extract_text()
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
def load_summarizer_model():
    summarizer = pipeline("summarization", model=model_name)
    return summarizer

@st.cache_resource
def get_tokenizer_and_max_len():                                                                                                                                                                            
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    max_len = getattr(tokenizer, "model_max_length", 1024)
    if max_len < 1 or max_len > 1_000_000:
        max_len = 1024
    return tokenizer, max_len

def get_summaries_for_chunks(text_chunks,  min_summary_length=50, max_summary_length=150):
    summarizer = load_summarizer_model()
    summaries = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i,chunk in enumerate(text_chunks):
        progress_percent = (i + 1) / len(text_chunks)
        status_text.text(f"Summarizing chunk {i+1}/{len(text_chunks)}...")
        progress_bar.progress(progress_percent)
        try:
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
    
    status_text.text("Summarization complete.")
    return summaries