# Economic Survey 2024-25 RAG System

[![RAG Architecture](https://img.shields.io/badge/Architecture-RAG-brightgreen)](https://arxiv.org/abs/2005.11401)

End-to-end Retrieval Augmented Generation (RAG) pipeline for querying India's Economic Survey documents.

## RAG Components
1. **Document Processor**: PDF text extraction and chunking
2. **Embedding Model**: Local `sentence-transformers` (avoid cloud costs)
3. **Vector Database**: Pinecone for similarity search
4. **LLM Interface**: Groq/Gemini for response generation

## Quick Setup

### 1. Clone & Initialize
```bash
git clone https://github.com/chvikas/economic_survey_2024-25.git
cd economic_survey_2024-25
python -m venv .rag_env && source .rag_env/bin/activate  # On Windows use: .rag_env\Scripts\activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure API Keys
Create a `.env` file and add your API keys:
```bash
echo "PINECONE_API_KEY=your_pinecone_key_here" > .env
echo "GROQ_API_KEY=your_groq_key_here" >> .env  # or Gemini API Key
```

### 4. Process Documents
```bash
python pdf_chunk.py  # Extracts text, chunks it, and stores embeddings in Pinecone
```

### 5. Launch RAG Interface
```bash
flask run --host=0.0.0.0 --port=5000
```

## Critical RAG Parameters
| Component       | Configuration              |
|-----------------|---------------------------|
| Chunk Size      | 1024 tokens                |
| Overlap         | 100 tokens                 |
| Embedding Model | all-MiniLM-L6-v2           |
| Index Type      | Pinecone (pod-based)       |

## Query Flow
1. User question → Embedding conversion
2. Pinecone similarity search
3. Context aggregation
4. LLM response generation
5. Confidence scoring

> **Local Embedding Note:**  
> `pdf_chunk.py` uses local Hugging Face models to prevent API overages:
> ```python
> from sentence_transformers import SentenceTransformer
> model = SentenceTransformer('all-MiniLM-L6-v2')
> embeddings = model.encode(chunks)
> ```

---

### ✅ What's Fixed & Improved:
- **Fixed Formatting & Typos** (e.g., `requiements.text` → `requirements.txt`)
- **Added Windows Activation Command**
- **Clarified `.env` Setup**
- **Structured the Query Flow Better**
- **Ensured Code Blocks Are Readable & Accurate**

This version is ready to use. Just copy-paste it into `README.md`! 🚀
