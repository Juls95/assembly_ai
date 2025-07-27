# AssemblyAI RAG with Learning System

A comprehensive system that combines AssemblyAI's Universal-Streaming technology with Retrieval-Augmented Generation (RAG) using LlamaIndex, Anthropic, and Pinecone. This project enables real-time speech-to-text transcription, semantic search, and conversational AI with long-term learning capabilities.

[![Assembly AI Video](https://img.youtube.com/vi/pV6RO6BN4P4/0.jpg)](https://youtu.be/pV6RO6BN4P4)

## 🏗️ Project Structure

```
assembly_ai/
├── assembly.py                 # Main AssemblyAI streaming implementation
├── build_rag_index.py         # Pinecone index construction
├── rag_with_learning.py       # RAG chat engine with learning
├── transcribe_test_audio.py   # Batch audio transcription
├── crypto_kb/                 # Knowledge base documents
├── test_audio/                # Audio test samples
├── test_cases.json           # Test case definitions
└── .env                      # Environment variables (create this)
```

## 🚀 Features

- **Real-time Speech-to-Text**: Using AssemblyAI's Universal-Streaming API
- **Domain-Specific Word Boost**: Enhanced recognition for crypto/finance terminology
- **RAG Pipeline**: Semantic search with Pinecone and LlamaIndex
- **Long-term Learning**: Conversation history integration
- **Batch Testing**: Audio transcription accuracy evaluation
- **Multi-Modal Input**: Support for both live audio and pre-recorded files

## 📋 Prerequisites

- Python 3.8+
- macOS (for audio capture) or Linux/Windows with appropriate audio drivers
- AssemblyAI API key
- Anthropic API key
- Pinecone API key

## 🛠️ Installation

### 1. Clone and Setup Environment

```bash
git clone https://github.com/Juls95/assembly_ai
cd assembly_ai
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 2. Install Dependencies

```bash
# Core dependencies
pip install assemblyai[extras] pyaudio websockets

# LlamaIndex and RAG components
pip install llama-index-core llama-index-llms-anthropic llama-index-embeddings-huggingface llama-index-vector-stores-pinecone

# Optional: Install OpenAI packages for compatibility (if using agent features)
pip install llama-index-llms-openai
```

### 3. System Dependencies (macOS)

```bash
# Install PortAudio for audio capture
brew install portaudio
```

### 4. Environment Variables

Create a `.env` file in the project root: (or set environment variables in the current shell session using export variable_name ='variable_value')

```bash
# AssemblyAI API Key
ASSEMBLYAI_API_KEY=your_assemblyai_api_key_here

# Anthropic API Key
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Pinecone API Key
PINECONE_API_KEY=your_pinecone_api_key_here
```

## 🎯 Usage Guide

### Step 1: Build the RAG Index

First, prepare your knowledge base documents in the `crypto_kb/` directory, then build the vector index:

```bash
python build_rag_index.py
```

This will:
- Create a Pinecone index with 384-dimensional vectors
- Ingest documents using HuggingFace embeddings
- Test the index with a sample query

### Step 2: Test Real-time Transcription

Start the AssemblyAI streaming client:

```bash
python assembly.py
```

Speak into your microphone to see real-time transcription with word boost for crypto terminology.

### Step 3: Run RAG Chat with Learning

```bash
python rag_with_learning.py
```

This starts a conversational interface that:
- Uses the RAG index for semantic search
- Generates responses using Anthropic's Claude
- Learns from conversations by updating the index

### Step 4: Batch Test Audio Files

```bash
python assembly.py --test
```

This runs batch transcription tests using your `test_cases.json` file.

## 📁 File Descriptions

### Core Files

- **`assembly.py`**: Main streaming implementation using AssemblyAI v3 API
  - Real-time microphone capture
  - Word boost for domain-specific terms
  - Event-driven architecture for session management

- **`build_rag_index.py`**: Pinecone index construction
  - Creates vector store with HuggingFace embeddings
  - Ingest documents from `crypto_kb/` directory
  - Configure 384-dimensional vectors for compatibility

- **`rag_with_learning.py`**: RAG chat engine with learning capabilities
  - Semantic search using the built index
  - Anthropic Claude for response generation
  - Conversation history integration

- **`transcribe_test_audio.py`**: Batch audio processing
  - Transcribe multiple audio files
  - Save results for RAG integration

### Configuration Files

- **`test_cases.json`**: Audio test case definitions
  ```json
  [
    {
      "filename": "test_audio/sample1.mp3",
      "expected_transcript": "Expected transcription text"
    }
  ]
  ```

- **`.env`**: Environment variables for API keys

## 🔧 Troubleshooting

### Audio Issues

**Problem**: "Input overflowed" or microphone not detected
**Solution**: 
```bash
# Check available audio devices and use the index of the correct one
python -c "import pyaudio; p = pyaudio.PyAudio(); [print(f'Index {i}: {p.get_device_info_by_index(i)["name"]}') for i in range(p.get_device_count())]"
```

**Problem**: Microphone permissions denied
**Solution**: 
- macOS: System Settings > Privacy & Security > Microphone
- Enable terminal/IDE access to microphone

### Embedding Model Issues

**Problem**: "OpenAI embedding model could not be loaded"
**Solution**: Ensure you're explicitly setting the embedding model:
```python
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
```

### Pinecone Dimension Mismatch

**Problem**: "Vector dimension 384 does not match the dimension of the index 1536"
**Solution**: Recreate the Pinecone index with the correct dimension:
```python
pc.create_index(
    name="your-index-name",
    dimension=384,  # Must match HuggingFace embedding output
    metric="cosine"
)
```

## 🎨 Customization

### Adding New Word Boost Terms

Edit the `word_boost` list in `assembly.py`:
```python
word_boost = [
    "your", "domain", "specific", "terms", "here"
]
```

### Changing Embedding Model

Update the model in both `build_rag_index.py` and `rag_with_learning.py`:
```python
embed_model = HuggingFaceEmbedding(model_name="your-preferred-model")
```

### Adding New Audio Test Cases

Update `test_cases.json`:
```json
[
  {
    "filename": "path/to/audio.mp3",
    "expected_transcript": "Expected text"
  }
]
```

## 📊 Performance Tips

- Use `sample_rate=8000` for better microphone compatibility
- Ensure Pinecone index dimension matches embedding model output
- Monitor API usage for cost optimization
- Use appropriate device index for multi-microphone setups

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

MIT License. See [LICENSE](https://github.com/Juls95/algoliaMCP/blob/main/LICENSE).

## 🆘 Support

For issues related to:
- **AssemblyAI**: Check their [documentation](https://www.assemblyai.com/docs/)
- **LlamaIndex**: Visit their [docs](https://docs.llamaindex.ai/)
- **Pinecone**: See their [guides](https://docs.pinecone.io/)

---

**Happy coding! 🚀** 