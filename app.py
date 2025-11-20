import PyPDF2
import os
import re
import nltk
from nltk.tokenize import sent_tokenize
from transformers import pipeline
from transformers import AutoTokenizer
import streamlit as st

# This must be the first Streamlit command
model_name = "sshleifer/distilbart-cnn-12-6"
st.set_page_config(page_title="PDF Summarizer AI", layout="wide")

@st.cache_resource
def download_nltk_resources():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
download_nltk_resources()


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
    chunks=[]
    i=0
    while i < len(words):
        j=i;
        curr_token=0;
        while j < len(words):
            candidate= " ".join(words[i:j+1])
            token_count = len(_tokenizer.encode(candidate, add_special_tokens=False))
            if token_count + reserve_space > max_len:
                break
            curr_token=token_count
            j+=1
        if j==i:
            chunk_text = " ".join(words[i:i+50])
            i+=50
        else:
            chunk_text = " ".join(words[i:j])
            i=max(i+1, j - overlap_tokens)
        chunks.append(chunk_text)
    return chunks
@st.cache_resource
def load_summarizer_model():
    summarizer = pipeline("summarization", model=model_name)
    return summarizer

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


st.title("PDF Summarizer AI")
st.markdown("Upload a PDF file, and this app will extract the text, chunk it, and generate summaries for each chunk using a pre-trained AI model.")


st.sidebar.header("Configuration")
# desired_chunk_size = st.sidebar.number_input("Chunk Size (characters)", min_value=200, max_value=2000, value=700, step=50)
chunk_overlap_token_size = st.sidebar.number_input("Chunk Overlap Size (tokens)", min_value=40, max_value=300, value=40, step=10)
min_summary_length = st.sidebar.number_input("Minimum Summary Length (words)", min_value=5, max_value=100, value=10, step=5)
max_summary_length = st.sidebar.number_input("Maximum Summary Length (words)", min_value=10, max_value=1000, value=500, step=5)
st.sidebar.markdown("---")
st.sidebar.markdown("Developed with ❤️ using Streamlit & Hugging Face")

uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

if uploaded_file is not None:
    with st.spinner('Extracting text from PDF...'):
        extracted_text = extract_text_from_pdfs(uploaded_file)

    if extracted_text: 
        st.success("Text extraction successful.")
        st.subheader("1. Preprocessing Text...")
        with st.spinner('Preprocessing text...'):
            preprocessed_text = preprocess_text(extracted_text)
        st.write(f"Original text length: {len(extracted_text)} characters.")
        st.write(f"Preprocessed text length: {len(preprocessed_text)} characters.")
        if preprocessed_text:
            st.subheader("2. Chunking Text...")
            with st.spinner('Chunking text...'):
                tokenizer, max_len = get_tokenizer_and_max_len()
                text_chunks = chunk_text_by_token(
                                        text=preprocessed_text, 
                                        _tokenizer=tokenizer,
                                        max_len=max_len,
                                        reserve_space= 128,
                                        overlap_tokens=chunk_overlap_token_size)
        st.write(f"Number of text chunks created: {len(text_chunks)}")

        if text_chunks:
            st.subheader("3. Generating Summaries for Each Chunk...")
            with st.expander("Summarizing..."):
                chunk_summaries = get_summaries_for_chunks(text_chunks,
                                                           
                                                           min_summary_length,
                                                           max_summary_length)
            
            
            st.markdown("---")
            st.subheader("Chunk Summaries")
            for i, summary in enumerate(chunk_summaries):
                st.write(f"**Summary for Chunk {i+1}:**")
                st.info(summary)

            if chunk_summaries:
                st.markdown("---")
                st.subheader("Overall Document Summary...")
                full_doc_summary = " ".join(chunk_summaries)

                if len(full_doc_summary.split()) > max_summary_length * 2:
                    st.subheader("*Generating a more concise overall summary...*")
                    try:
                        # Generate a summary of all the chunk summaries combined
                        with st.spinner('Generating concise overall summary...'):
                            concise_summary_list = get_summaries_for_chunks([full_doc_summary], 
                                                                           min_summary_length, 
                                                                           max_summary_length)
                        # Extract the actual summary text from the list
                        if concise_summary_list and len(concise_summary_list) > 0:
                            final_summary = concise_summary_list[0]
                            # Clean up any "Summary unavailable" prefixes
                            if final_summary.startswith("Summary unavailable. Preview: "):
                                final_summary = final_summary.replace("Summary unavailable. Preview: ", "")
                            st.success("**Final Document Summary:**")
                            st.info(final_summary)
                        else:
                            st.warning("Could not generate a concise summary. Here's the combined summary:")
                            st.write(full_doc_summary)
                    except Exception as e:
                        st.error(f"Error generating concise overall summary: {e}")
                        st.warning("Showing combined chunk summaries instead:")
                        st.write(full_doc_summary)
                else:
                    st.success("**Final Document Summary:**")
                    st.write(full_doc_summary)
            else:
                st.info("No chunk summaries were generated.")
        else:
            st.info("No text chunks were created.")
    else:
        st.info("No text was extracted from the PDF.")
else:
    st.info('Please upload a PDF to get started!')