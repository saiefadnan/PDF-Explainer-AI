import os
from functions import extract_text_from_pdfs, preprocess_text
from functions import chunk_text_by_token, get_tokenizer_and_max_len, query_pinecone_for_context
from functions import download_nltk_resources, get_expanded_answer, generate_summary_pdf, get_summaries_for_chunks
from functions import validate_api_keys
import streamlit as st

st.set_page_config(page_title="PDF Summarizer AI", layout="wide")

# Validate API keys before proceeding
validate_api_keys()

# Download NLTK resources after page config
download_nltk_resources()

# Initialize tokenizer only once
if "tokenizer" not in st.session_state:
    st.session_state.tokenizer, st.session_state.max_len = get_tokenizer_and_max_len()

tokenizer = st.session_state.tokenizer
max_len = st.session_state.max_len

st.title("PDF Summarizer AI")
st.markdown("Upload a PDF file, and this app will extract the text, chunk it, and generate summaries for each chunk using a pre-trained AI model.")

# Initialize session state variables
if "preprocessed_text" not in st.session_state:
    st.session_state.preprocessed_text = ""
if "chunk_summaries" not in st.session_state:
    st.session_state.chunk_summaries = []
if "final_summary" not in st.session_state:
    st.session_state.final_summary = ""
if "pdf_filename" not in st.session_state:
    st.session_state.pdf_filename = ""
if "processing_done" not in st.session_state:
    st.session_state.processing_done = False
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.sidebar.header("Configuration")
chunk_overlap_token_size = st.sidebar.number_input("Chunk Overlap Size (tokens)", min_value=40, max_value=300, value=40, step=10)
min_summary_length = st.sidebar.number_input("Minimum Summary Length (words)", min_value=5, max_value=100, value=10, step=5)
max_summary_length = st.sidebar.number_input("Maximum Summary Length (words)", min_value=10, max_value=1000, value=500, step=5)
st.sidebar.markdown("---")
st.sidebar.markdown("Developed with ❤️ using Streamlit & Hugging Face")

uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

if uploaded_file is not None:
    # Check if this is a new file
    if "current_file_name" not in st.session_state or st.session_state.current_file_name != uploaded_file.name:
        st.session_state.current_file_name = uploaded_file.name
        st.session_state.processing_done = False
        st.session_state.chunk_summaries = []
        st.session_state.final_summary = ""
        st.session_state.preprocessed_text = ""
        st.session_state.chat_history = []
    
    # Only process if not already done
    if not st.session_state.processing_done:
        st.session_state.pdf_filename = uploaded_file.name
        
        with st.spinner('Extracting text from PDF...'):
            try:
                pdf_bytes = uploaded_file.read()
                extracted_text = extract_text_from_pdfs(pdf_bytes)
            except Exception as e:
                st.error(f"Failed to extract text from PDF: {str(e)}")
                st.stop()
        
        if not extracted_text or len(extracted_text.strip()) == 0:
            st.error("No text could be extracted from this PDF. The file might be empty or contain only images.")
            st.stop()
            
        st.success("Text extraction successful.")
        st.subheader("1. Preprocessing Text...")
        
        with st.spinner('Preprocessing text...'):
            preprocessed_text = preprocess_text(extracted_text)
            st.session_state.preprocessed_text = preprocessed_text
        
        st.write(f"Original text length: {len(extracted_text)} characters.")
        st.markdown(f"""
            <div style='background-color: #f8f9fa; border-left: 4px solid #0084ff; padding: 12px; margin-bottom: 12px; border-radius: 4px;'>
                <div style='font-size: 12px; color: #666; font-weight: bold; margin-bottom: 8px;'>
                    PREPROCESSED TEXT
                </div>
                <div style='font-size: 14px; color: #1a1a1a; line-height: 1.6;'>
                    {preprocessed_text[:500]}{'...' if len(preprocessed_text) > 500 else ''}
                </div>
            </div>
        """, unsafe_allow_html=True)
        st.write(f"Preprocessed text length: {len(st.session_state.preprocessed_text)} characters.")
        
        if st.session_state.preprocessed_text:
            st.subheader("2. Chunking Text...")
            with st.spinner('Chunking text...'):
                text_chunks = chunk_text_by_token(
                    text=st.session_state.preprocessed_text, 
                    _tokenizer=tokenizer,
                    max_len=max_len,
                    reserve_space=128,
                    overlap_tokens=chunk_overlap_token_size
                )
            st.write(f"Number of text chunks created: {len(text_chunks)}")

            if text_chunks:
                st.subheader("3. Generating Summaries for Each Chunk...")
                with st.expander("Summarizing..."):
                    try:
                        chunk_summaries = get_summaries_for_chunks(
                            text_chunks,
                            min_summary_length,
                            max_summary_length
                        )
                    except Exception as e:
                        st.error(f"Error generating summaries: {str(e)}")
                        st.stop()
                
                st.session_state.chunk_summaries = chunk_summaries
                st.session_state.processing_done = True
                st.markdown("---")
                st.subheader("📑 Section Summaries")
                
                # Display summaries in a professional report style
                for i, summary in enumerate(chunk_summaries):
                    st.markdown(f"""
                    <div style='background-color: #f8f9fa; border-left: 4px solid #0084ff; padding: 12px; margin-bottom: 12px; border-radius: 4px;'>
                        <div style='font-size: 12px; color: #666; font-weight: bold; margin-bottom: 8px;'>
                            SECTION {i+1} 
                        </div>
                        <div style='font-size: 14px; color: #1a1a1a; line-height: 1.6;'>
                            {summary}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                if chunk_summaries:
                    st.markdown("---")
                    st.subheader("Overall Document Summary...")
                    full_doc_summary = " ".join(chunk_summaries)

                    if len(full_doc_summary.split()) > max_summary_length * 2:
                        st.subheader("*Generating a more concise overall summary...*")
                        try:
                            with st.spinner('Generating concise overall summary...'):
                                concise_summary_list = get_summaries_for_chunks(
                                    [full_doc_summary],
                                    min_summary_length,
                                    max_summary_length
                                )
                            
                            if concise_summary_list and len(concise_summary_list) > 0:
                                final_summary = concise_summary_list[0]
                                if final_summary.startswith("Summary unavailable. Preview: "):
                                    final_summary = final_summary.replace("Summary unavailable. Preview: ", "")
                                st.session_state.final_summary = final_summary
                                st.success("**Final Document Summary:**")
                                st.info(final_summary)
                            else:
                                st.session_state.final_summary = full_doc_summary
                                st.warning("Could not generate a concise summary. Here's the combined summary:")
                                st.write(full_doc_summary)
                        except Exception as e:
                            st.session_state.final_summary = full_doc_summary
                            st.error(f"Error generating concise overall summary: {e}")
                            st.warning("Showing combined chunk summaries instead:")
                            st.write(full_doc_summary)
                    else:
                        st.session_state.final_summary = full_doc_summary
                        st.success("**Final Document Summary:**")
                        st.write(full_doc_summary)

                    # Download Summary PDF Button
                    if st.session_state.final_summary and st.session_state.chunk_summaries:
                        st.markdown("---")
                        st.markdown("### 📥 Download Summary Report")
                        st.markdown(
                            "<p style='color:#555;font-size:14px;margin-top:-8px'>"
                            "Download a beautifully formatted PDF containing all section summaries "
                            "and the final document summary.</p>",
                            unsafe_allow_html=True,
                        )
                        with st.spinner("Preparing PDF..."):
                            try:
                                summary_pdf_bytes = generate_summary_pdf(
                                    chunk_summaries=st.session_state.chunk_summaries,
                                    final_summary=st.session_state.final_summary,
                                    pdf_filename=st.session_state.pdf_filename,
                                )
                                fname = st.session_state.pdf_filename.replace(".pdf", "") or "document"
                                download_name = f"{fname}_summary.pdf"
                                st.download_button(
                                    label="⬇️  Download Summary PDF",
                                    data=summary_pdf_bytes,
                                    file_name=download_name,
                                    mime="application/pdf",
                                    use_container_width=True,
                                    type="primary",
                                )
                            except Exception as e:
                                st.error(f"Error generating PDF: {str(e)}")
                else:
                    st.info("No chunk summaries were generated.")
            else:
                st.info("No text chunks were created.")
        else:
            st.info("No text was extracted from the PDF.")
    
    # Show cached results if already processed
    elif st.session_state.processing_done and st.session_state.chunk_summaries:
        st.markdown("---")
        st.subheader("📑 Section Summaries")
        for i, summary in enumerate(st.session_state.chunk_summaries):
            st.markdown(f"""
            <div style='background-color: #f8f9fa; border-left: 4px solid #0084ff; padding: 12px; margin-bottom: 12px; border-radius: 4px;'>
                <div style='font-size: 12px; color: #666; font-weight: bold; margin-bottom: 8px;'>
                    SECTION {i+1} 
                </div>
                <div style='font-size: 14px; color: #1a1a1a; line-height: 1.6;'>
                    {summary}
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.subheader("Overall Document Summary")
        st.success("**Final Document Summary:**")
        st.info(st.session_state.final_summary)
        
        # Download button
        if st.session_state.final_summary and st.session_state.chunk_summaries:
            st.markdown("---")
            st.markdown("### 📥 Download Summary Report")
            with st.spinner("Preparing PDF..."):
                try:
                    summary_pdf_bytes = generate_summary_pdf(
                        chunk_summaries=st.session_state.chunk_summaries,
                        final_summary=st.session_state.final_summary,
                        pdf_filename=st.session_state.pdf_filename,
                    )
                    fname = st.session_state.pdf_filename.replace(".pdf", "") or "document"
                    download_name = f"{fname}_summary.pdf"
                    st.download_button(
                        label="⬇️  Download Summary PDF",
                        data=summary_pdf_bytes,
                        file_name=download_name,
                        mime="application/pdf",
                        use_container_width=True,
                        type="primary",
                    )
                except Exception as e:
                    st.error(f"Error generating PDF: {str(e)}")
else:
    st.info('Please upload a PDF to get started!')

# Chat Interface
context = st.session_state.preprocessed_text

if context:
    st.markdown("---")
    st.subheader("💬 Chat with Your PDF")
    
    # Display chat history
    if st.session_state.chat_history:
        for i, (question, answer) in enumerate(st.session_state.chat_history):
            # User question (right aligned)
            st.markdown(f"""
            <div style='text-align: right; margin-bottom: 10px;'>
                <div style='background-color: #0084ff; color: white; padding: 10px; border-radius: 15px; display: inline-block; max-width: 70%;'>
                    <strong>You:</strong> {question}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # AI answer (left aligned)
            st.markdown(f"""
            <div style='text-align: left; margin-bottom: 20px;'>
                <div style='background-color: #f1f1f1; color: black; padding: 10px; border-radius: 15px; display: inline-block; max-width: 70%;'>
                    <strong>🤖 AI:</strong> {answer}
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    # Chat input container
    user_input = st.chat_input("💭 Ask me anything about this PDF...")
    
    # Handle suggested questions
    if "pending_question" in st.session_state:
        user_input = st.session_state.pending_question
        del st.session_state.pending_question

    # Process message when user sends
    if user_input and user_input.strip():
        with st.spinner('🤔 Thinking...'):
            try:
                # Get relevant context from Pinecone
                relevant_context = query_pinecone_for_context(user_input, top_k=3)
                # Get expanded answer
                answer, confidence = get_expanded_answer(user_input, relevant_context)
                # Add to chat history
                st.session_state.chat_history.append((user_input, answer))
                st.rerun()
                
            except Exception as e:
                error_msg = f"Sorry, I encountered an error: {str(e)}"
                st.session_state.chat_history.append((user_input, error_msg))
                st.rerun()
    
    # Clear chat button
    if st.session_state.chat_history:
        if st.button("🗑️ Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()
    
    # Chat suggestions
    if not st.session_state.chat_history:
        st.markdown("**💡 Suggested questions:**")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📋 What is this document about?"):
                st.session_state.pending_question = "What is this document about?"
                st.rerun()
        
        with col2:
            if st.button("🔍 Summarize the key points"):
                st.session_state.pending_question = "What are the main key points?"
                st.rerun()
        
        with col3:
            if st.button("❓ Explain technical terms"):
                st.session_state.pending_question = "Explain the technical terms mentioned"
                st.rerun()
else:
    st.markdown("---")
    st.info("📄 Upload and process a PDF to start chatting!")