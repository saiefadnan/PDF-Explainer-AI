import streamlit as st
import sys

# Try to import Pinecone with error handling
try:
    from pinecone import Pinecone, ServerlessSpec, PodSpec
except ImportError as e:
    st.error(f"❌ Pinecone library not installed: {str(e)}")
    st.stop()
except Exception as e:
    # Pinecone itself is raising an exception during import
    # This can happen on some environments - we'll defer initialization
    st.warning(f"⚠️ Pinecone import warning: {str(e)}")
    Pinecone = None

# Get API keys from Streamlit secrets (works on Cloud and local with secrets.toml)
PINECONE_API_KEY = st.secrets.get("PINECONE_API_KEY", "")

if not PINECONE_API_KEY:
    st.error("❌ PINECONE_API_KEY not found in secrets!")
    st.stop()

# Create a Pinecone client instance with safer initialization
pc = None
index = None

try:
    if Pinecone is not None:
        pc = Pinecone(api_key=PINECONE_API_KEY)
    else:
        st.error("❌ Pinecone client not available")
        st.stop()
except Exception as e:
    st.error(f"❌ Failed to initialize Pinecone: {str(e)}")
    st.info("Try reinstalling: pip install --upgrade pinecone-client")
    st.stop()

index_name = "pdfexplainer078"
EMBEDDING_DIM = 384  # all-MiniLM-L6-v2 uses 384 dimensions (FREE model)

# Get or create index
try:
    if pc is None:
        st.error("❌ Pinecone client not initialized")
        st.stop()
    
    existing_indexes = pc.list_indexes()
    index_exists = any(idx.name == index_name for idx in existing_indexes)
    
    if not index_exists:
        print(f"Creating index '{index_name}' with {EMBEDDING_DIM} dimensions...")
        try:
            # Try to create with ServerlessSpec (for free tier)
            pc.create_index(
                name=index_name,
                dimension=EMBEDDING_DIM,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
        except Exception as serverless_err:
            # Fallback: create with PodSpec for standard tier
            try:
                pc.create_index(
                    name=index_name,
                    dimension=EMBEDDING_DIM,
                    metric="cosine",
                    spec=PodSpec(environment="us-east-1")
                )
            except Exception as pod_err:
                st.error(f"❌ Could not create index. ServerlessSpec error: {str(serverless_err)}, PodSpec error: {str(pod_err)}")
                st.stop()
        print(f"Index '{index_name}' created successfully!")
    
    # Get index reference
    index = pc.Index(index_name)
    print(f"✅ Connected to index '{index_name}'")
    
except Exception as e:
    st.error(f"❌ Error setting up Pinecone index: {str(e)}")
    st.stop()