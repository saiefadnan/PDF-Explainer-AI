import streamlit as st

try:
    from pinecone import Pinecone
except ImportError:
    st.error("❌ Pinecone not installed. Please run: pip install pinecone-client")
    st.stop()

# Get API keys from Streamlit secrets (works on Cloud and local with secrets.toml)
PINECONE_API_KEY = st.secrets.get("PINECONE_API_KEY", "")

if not PINECONE_API_KEY:
    st.error("❌ PINECONE_API_KEY not found in secrets!")
    st.stop()

# Create a Pinecone client instance with safer initialization
try:
    pc = Pinecone(api_key=PINECONE_API_KEY)
except Exception as e:
    st.error(f"❌ Failed to initialize Pinecone: {str(e)}")
    st.stop()

index_name = "pdfexplainer078"
EMBEDDING_DIM = 384  # all-MiniLM-L6-v2 uses 384 dimensions (FREE model)

# Get or create index
try:
    existing_indexes = pc.list_indexes()
    index_exists = any(idx.name == index_name for idx in existing_indexes)
    
    if not index_exists:
        print(f"Creating index '{index_name}' with {EMBEDDING_DIM} dimensions...")
        try:
            # Try to create with ServerlessSpec (for free tier)
            from pinecone import ServerlessSpec
            pc.create_index(
                name=index_name,
                dimension=EMBEDDING_DIM,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
        except Exception as e:
            # Fallback: create with PodSpec for standard tier
            try:
                from pinecone import PodSpec
                pc.create_index(
                    name=index_name,
                    dimension=EMBEDDING_DIM,
                    metric="cosine",
                    spec=PodSpec(environment="us-east-1")
                )
            except:
                st.error(f"❌ Could not create index with available specs. Error: {str(e)}")
                st.stop()
        print(f"Index '{index_name}' created successfully!")
    
    # Get index reference
    index = pc.Index(index_name)
    print(f"✅ Connected to index '{index_name}'")
    
except Exception as e:
    st.error(f"❌ Error setting up Pinecone index: {str(e)}")
    st.stop()