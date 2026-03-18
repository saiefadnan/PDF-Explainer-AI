import streamlit as st
import os

# Try to import Pinecone with error handling
try:
    from pinecone import Pinecone, ServerlessSpec
except ImportError as e:
    st.error(f"❌ Pinecone library not installed: {str(e)}")
    st.stop()
except Exception as e:
    st.warning(f"⚠️ Pinecone import warning: {str(e)}")

# Get API keys from Streamlit secrets
PINECONE_API_KEY = st.secrets.get("PINECONE_API_KEY", "")

if not PINECONE_API_KEY:
    st.error("❌ PINECONE_API_KEY not found in secrets!")
    st.stop()

# Initialize Pinecone with v3 API
try:
    pc = Pinecone(api_key=PINECONE_API_KEY)
except Exception as e:
    st.error(f"❌ Failed to initialize Pinecone: {str(e)}")
    st.info("Make sure PINECONE_API_KEY is set in .streamlit/secrets.toml")
    st.stop()

index_name = "pdfexplainer078"
EMBEDDING_DIM = 384  # all-MiniLM-L6-v2 uses 384 dimensions (FREE model)

# Get or create index using Pinecone v3 API
try:
    # Check if index exists
    existing_indexes = pc.list_indexes().names()
    
    if index_name not in existing_indexes:
        print(f"Creating index '{index_name}' with {EMBEDDING_DIM} dimensions...")
        # Create index with ServerlessSpec
        pc.create_index(
            name=index_name,
            dimension=EMBEDDING_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        print(f"Index '{index_name}' created successfully!")
    
    # Get index reference
    index = pc.Index(index_name)
    print(f"✅ Connected to index '{index_name}'")
    
except Exception as e:
    st.error(f"❌ Error setting up Pinecone index: {str(e)}")
    st.stop()