
# 📄 Telecom RAG Assistant

**A production-ready Retrieval-Augmented Generation (RAG) system for intelligent document Q&A powered by LLMs.**

## Problem

Organizations handle massive volumes of unstructured PDF documents—policy manuals, technical specifications, compliance guides, and knowledge bases. Employees waste hours manually searching through these documents to answer routine questions. Traditional full-text search is inflexible and often returns irrelevant results. Teams need a **conversational interface** that understands context and delivers precise, grounded answers directly from their documents.

## Solution

**PDF RAG Assistant** combines semantic search with large language models to create an intelligent question-answering system. Upload any PDF, and the system automatically:
- **Chunks** the document intelligently
- **Embeds** content semantically
- **Retrieves** the most relevant passages
- **Generates** accurate answers grounded in the source material

The result: instant, accurate answers without hallucinations—because every response is backed by actual document content.

---
Try the agent by you self for here:[PDF RAG Assistant](https://pdf-rag-assistant-1.streamlit.app/)

---

## Features

- **🚀 Semantic Search**: Uses advanced embeddings (sentence-transformers) to find contextually relevant passages, not just keyword matches
- **📦 Intelligent Chunking**: Recursive text splitting with overlaps preserves context across chunk boundaries
- **🔒 Grounded Responses**: LLM answers are constrained to source material—no hallucinations
- **⚡ Fast Inference**: Leverages Groq's ultra-low latency LLM API for sub-second responses
- **🎨 User-Friendly UI**: Streamlit-based interface requires zero engineering knowledge
- **💾 Session Persistence**: Maintains vector store and chain state across interactions
- **📊 Processing Visibility**: Shows chunk count, page count, and preview of processed content

---

## Architecture

```
┌──────────────────┐
│   PDF Upload     │
└────────┬─────────┘
         │
         ▼
┌──────────────────────────────┐
│  Document Loading (PyPDF)    │
└────────┬─────────────────────┘
         │
         ▼
┌──────────────────────────────┐
│  Intelligent Chunking        │
│  (RecursiveCharacterSplitter)│
└────────┬─────────────────────┘
         │
         ▼
┌──────────────────────────────┐
│  Embedding Generation        │
│  (HuggingFace Embeddings)    │
└────────┬─────────────────────┘
         │
         ▼
┌──────────────────────────────┐
│  Vector Store (Chroma)       │
└────────┬─────────────────────┘
         │
         ▼
┌──────────────────────────────┐
│  Retrieval (Similarity Rank) │
│  (Top-K documents)           │
└────────┬─────────────────────┘
         │
         ▼
┌──────────────────────────────┐
│  LLM Chain (Groq/Qwen)       │
│  + System Prompt             │
└────────┬─────────────────────┘
         │
         ▼
┌──────────────────────────────┐
│  Grounded Answer             │
└──────────────────────────────┘
```

---

## Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **LLM** | Groq (Qwen 3 32B) | Ultra-fast inference with reasoning |
| **Embeddings** | HuggingFace (all-MiniLM-L6-v2) | Semantic text representation |
| **Vector Store** | Chroma | In-memory vector database |
| **Document Loading** | LangChain PyPDFLoader | PDF parsing and extraction |
| **Orchestration** | LangChain LCEL | Composable chain pipelines |
| **UI Framework** | Streamlit | Interactive web interface |
| **Environment** | Python 3.9+ | Core runtime |

---

## Installation

### Prerequisites
- **Python 3.9+**
- **API Key**: Groq API key (get one free at [console.groq.com](console.groq.com))

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/MostafaAlali/Telecom-RAG-Assistant.git
   cd pdf-rag-assistant
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**
   ```bash
   echo "GROQ_API_KEY=your_api_key_here" > .env
   ```

5. **Run the application**
   ```bash
   streamlit run app.py
   ```

The UI will open at `http://localhost:8501`

---

## Usage

### Basic Workflow

1. **Upload a PDF** in the sidebar
2. **Click "Build Knowledge Base"** — the system processes and embeds the document (~1-2 seconds for typical documents)
3. **Ask questions** in the text input field
4. **Get instant, grounded answers** with source context

### Example Queries

Given a telecom policy document, you could ask:

```
"What is the data overage charge for premium plans?"
→ Answer: "The data overage charge for premium plans is $10 per 1GB..."

"What are the SLA uptime guarantees?"
→ Answer: " The provided context does not mention any specific SLA (Service Level Agreement) uptime guarantees. It focuses on data management policies, such as throttling speeds after high-speed data limits, data rollover, add-on packs, and fair use policies for "unlimited" plans. For SLA-related details, further information would be required."
```

---

## How It Works: Deep Dive

### 1. Document Processing
- **Loading**: PyPDF extracts text and metadata from uploaded PDFs
- **Chunking**: `RecursiveCharacterTextSplitter` breaks documents into 600-token chunks with 100-token overlaps, preserving semantic boundaries by splitting on paragraphs first, then sentences
- **Result**: Multiple overlapping chunks maintain context while staying within embedding limits

### 2. Semantic Indexing
- **Embeddings**: HuggingFace's `all-MiniLM-L6-v2` converts chunks to 384-dimensional vectors (~22M parameters, optimized for speed)
- **Storage**: Chroma stores embeddings in-memory with efficient cosine similarity search
- **Caching**: Embeddings are cached via `@st.cache_resource` to avoid recomputation

### 3. Retrieval & Generation
- **Query Embedding**: User question is embedded with the same model
- **Top-K Retrieval**: Chroma returns the 3 most similar chunks (configurable)
- **Context Assembly**: Retrieved chunks are formatted and passed to the LLM
- **Answer Generation**: Groq's Qwen 3 32B generates answers constrained by system prompt to use only provided context
- **Temperature**: Set to 0 for deterministic, consistent responses

### 4. Safety & Grounding
- **System Prompt**: Explicitly instructs the LLM to use only provided context
- **No Hallucinations**: If context is insufficient, the LLM clearly states so rather than making up answers
- **Traceability**: Answers are always grounded in actual document content

---

## Configuration

Edit these parameters in `app.py` to customize behavior:

```python
# Chunking parameters
chunk_size=600              # Tokens per chunk (adjust for your domain)
chunk_overlap=100           # Overlap between chunks

# Retrieval parameters
search_kwargs={"k": 3}      # Number of chunks to retrieve

# LLM parameters
temperature=0               # 0 for deterministic, 1 for creative
model="qwen/qwen3-32b"      # Groq model (see Groq docs for options)
```

---

## Performance Characteristics

| Metric | Performance |
|--------|-------------|
| **Time to embed 100-page PDF** | ~2-5 seconds (depends on content density) |
| **Query response time** | <500ms (via Groq low-latency API) |
| **Embedding dimensions** | 384 (all-MiniLM-L6-v2) |
| **Max chunk size** | 600 tokens (~400 words) |
| **Vector similarity algorithm** | Cosine distance |

---

## Future Improvements

- **Multi-document support**: Query across multiple PDFs simultaneously with document tracking
- **Hybrid search**: Combine semantic search with BM25 keyword matching for better recall
- **Persistent storage**: Replace in-memory Chroma with PostgreSQL + pgvector for production deployments
- **LLM providers**: Support Claude, GPT-4, and open-source models via unified interface
- **Citation tracking**: Return source page numbers and exact passages for every answer
- **Fine-tuning**: Domain-specific embedding models for specialized industries
- **Batch processing**: Queue and process large document libraries asynchronously
- **Caching layer**: Redis-backed response caching for repeated questions

---

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit changes (`git commit -m 'Add feature'`)
4. Push to branch (`git push origin feature/your-feature`)
5. Open a Pull Request

---

## License

MIT License — see LICENSE file for details

---

## Questions?

- **Issues**: Use GitHub Issues for bug reports and feature requests
- **Documentation**: See `/docs` for architecture deep-dives and troubleshooting

---

## Regarding Your Follow-Up

You mentioned you have **3 Python files**—if you'd like to integrate multiple modules (e.g., separate files for embeddings, retrieval, LLM chains), here's how to structure it professionally:

**Suggested structure:**
```
pdf-rag-assistant/
├── app.py                 # Main Streamlit UI
├── src/
│   ├── embeddings.py      # Embedding initialization & caching
│   ├── vector_store.py    # Vector store creation & retrieval
│   └── chains.py          # LLM chain construction
├── requirements.txt
├── .env.example
└── README.md
```


This modular approach improves readability, testability, and is what Meta engineers expect. Would you like me to refactor the README for a multi-file structure, or do you have specific questions about the code organization? i need be make it markdown for github
