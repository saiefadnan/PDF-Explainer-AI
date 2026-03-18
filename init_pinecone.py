import streamlit as st
from pinecone import Pinecone, ServerlessSpec

# Get API keys from Streamlit secrets (works on Cloud and local with secrets.toml)
PINECONE_API_KEY = st.secrets.get("PINECONE_API_KEY", "")
PINECONE_ENV = st.secrets.get("PINECONE_ENV", "us-east-1-aws")

if not PINECONE_API_KEY:
    st.error("❌ PINECONE_API_KEY not found in secrets!")
    st.stop()

# Create a Pinecone client instance
pc = Pinecone(api_key=PINECONE_API_KEY)

index_name = "pdfexplainer078"
EMBEDDING_DIM = 384  # all-MiniLM-L6-v2 uses 384 dimensions (FREE model)

# Recreate index if dimension is wrong or doesn't exist
existing_indexes = pc.list_indexes().names()

if index_name in existing_indexes:
    # Check dimension of existing index
    index_info = pc.describe_index(index_name)
    if index_info.dimension != EMBEDDING_DIM:
        print(f"Deleting old index with wrong dimension ({index_info.dimension})...")
        pc.delete_index(index_name)
        existing_indexes = []  # Force recreation

if index_name not in existing_indexes:
    print(f"Creating index '{index_name}' with {EMBEDDING_DIM} dimensions...")
    pc.create_index(
        name=index_name,
        dimension=EMBEDDING_DIM,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
    print(f"Index '{index_name}' created successfully!")

index = pc.Index(index_name)
stats = index.describe_index_stats()
print(f"Index ready! Stats: {stats}")