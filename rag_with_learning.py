import os
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core import StorageContext
from llama_index.llms.anthropic import Anthropic  # Changed from OpenAI
from llama_index.core.chat_engine import SimpleChatEngine
from pinecone import Pinecone
import json
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# Load index (from build_rag_index.py)
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
pinecone_index = pc.Index("crypto-edu-index")
vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
index = VectorStoreIndex(
    [],
    storage_context=storage_context,
    embed_model=embed_model
)

llm = Anthropic(model="claude-3-5-sonnet-20240620")  # Changed from OpenAI
chat_engine = index.as_chat_engine(llm=llm, verbose=True)

# Load transcripts as queries
with open("transcripts.json", "r") as f:
    queries = json.load(f)

# Simulate conversation loop
for query in queries:
    response = chat_engine.chat(query)  # Uses history
    print(f"Query: {query}\nResponse: {response}\n")

    # Update index with convo for long-term learning
    from llama_index.core import Document
    convo_doc = Document(text=f"User: {query} | Agent: {response}")
    index.insert(convo_doc)