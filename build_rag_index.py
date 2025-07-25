import os
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core import StorageContext
from llama_index.llms.anthropic import Anthropic  # Changed from OpenAI
from pinecone import Pinecone, ServerlessSpec
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# Initialize Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
pc.create_index(
    name="crypto-edu-index",
    dimension=384,  # Must match HuggingFace embedding model output
    metric="cosine",
    spec=ServerlessSpec(cloud="aws", region="us-east-1")
)
pinecone_index = pc.Index("crypto-edu-index")

# Load docs
documents = SimpleDirectoryReader("crypto_kb/").load_data()

# Use HuggingFace embedding model (384-dim, matches rag_with_learning.py)
embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Create vector store and index
vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(documents, storage_context=storage_context, embed_model=embed_model)

print("Knowledge base indexed!")

# Example query test (updated LLM)
llm = Anthropic(model="claude-3-5-sonnet-20240620")  # Changed from OpenAI
query_engine = index.as_query_engine(llm=llm)
response = query_engine.query("What is a blockchain?")
print(response)  # Retrieves from indexed docs and generates answer