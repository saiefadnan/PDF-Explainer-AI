from functions import extract_text_from_pdfs, preprocess_text
from functions import chunk_text_by_token, get_tokenizer_and_max_len
from functions import  load_qa_model, get_summaries_for_chunks, download_nltk_resources
from transformers import pipeline
import streamlit as st

st.set_page_config(page_title="PDF Summarizer AI", layout="wide")

# Download NLTK resources after page config
download_nltk_resources()
qa_pipeline = load_qa_model()

st.title("PDF Summarizer AI")
st.markdown("Upload a PDF file, and this app will extract the text, chunk it, and generate summaries for each chunk using a pre-trained AI model.")
if "preprocessed_text" not in st.session_state:
    st.session_state.preprocessed_text = ""

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
            st.session_state.preprocessed_text = preprocess_text(extracted_text)
        st.write(f"Original text length: {len(extracted_text)} characters.")
        st.write(f"Preprocessed text length: {len(st.session_state.preprocessed_text)} characters.")
        if st.session_state.preprocessed_text:
            st.subheader("2. Chunking Text...")
            with st.spinner('Chunking text...'):
                tokenizer, max_len = get_tokenizer_and_max_len()
                text_chunks = chunk_text_by_token(
                                        text=st.session_state.preprocessed_text, 
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

context = st.session_state.preprocessed_text

if context:
    st.markdown("---")
    st.subheader("💬 Chat with Your PDF")
    
    # Initialize chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
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
    
    # Chat input container (ChatGPT-style)
    user_input = st.chat_input("💭 Ask me anything about this PDF...")
    
    # Handle suggested questions
    if "pending_question" in st.session_state:
        user_input = st.session_state.pending_question
        del st.session_state.pending_question

    # Process message when user sends
    if user_input and user_input.strip():
        with st.spinner('🤔 Thinking...'):
            try:
                # Get answer from QA model
                result = qa_pipeline({"question": user_input, "context": context})
                answer = result["answer"]
                
                # Add confidence score if available
                confidence = result.get("score", 0)
                if confidence < 0.3:
                    answer += f" *(Low confidence: {confidence:.2f})*"
                
                # Add to chat history
                st.session_state.chat_history.append((user_input, answer))
                
                # Auto-rerun to show new message
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