# Chunking Benchmark Pipeline

A comprehensive, industry-standard pipeline for evaluating different text chunking strategies in RAG (Retrieval-Augmented Generation) systems. This project implements **true late chunking** using local models and compares it with traditional chunking methods using real-world evaluation metrics.

## 🌟 Features

### Document Processing
- **Multi-format support**: PDF, DOCX, PPTX, TXT, Markdown, scripts (Python, JavaScript, etc.), and images
- **Unstructured API integration**: Handles tables, images, and complex layouts (with fallback for simple files)
- **Smart extraction**: Preserves document structure and hierarchy
- **Automatic fallback**: Simple text files load without API dependencies

### Chunking Strategies

1. **Fixed-size Chunking**: Simple character-based splitting with overlap
2. **Recursive Chunking**: Respects natural text boundaries (paragraphs, sentences)
3. **Semantic Chunking**: Groups by document structure (sections, elements)
4. **Late Chunking** ⭐: Context-aware embeddings using local Jina model

### Embedding & Storage
- **Local Jina Embeddings v3**: State-of-the-art embeddings running locally (no API calls!)
- **True Late Chunking**: Implements the official late chunking method from [Jina AI's research](https://jina.ai/news/late-chunking-in-long-context-embedding-models)
- **Pinecone Vector Database**: Scalable cloud vector storage with namespace isolation per strategy
- **GPU Support**: Optional CUDA acceleration for faster embeddings

### Evaluation
- **Ragas Framework**: Industry-standard RAG evaluation metrics
  - Context Precision
  - Context Recall
  - Faithfulness
  - Answer Relevancy
- **Azure OpenAI Integration**: Uses GPT-4 for evaluation
- **Comparative Analysis**: Side-by-side comparison of all strategies
- **Automated Benchmarking**: Complete evaluation pipeline with CSV export

## 🏗️ Architecture

```
┌─────────────────┐
│   Documents     │ (PDF, DOCX, PPTX, TXT, Markdown, etc.)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Document Loader │ (Unstructured API + Local Fallback)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Chunking       │ (Fixed / Recursive / Semantic / Late)
│  Strategies     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Local Jina     │ (jinaai/jina-embeddings-v3)
│  Embeddings     │ (True Late Chunking Implementation)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Pinecone       │ (Vector storage, namespaced by strategy)
│  Vector DB      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Ragas          │ (Azure OpenAI GPT-4 Evaluation)
│  Evaluation     │
└─────────────────┘
```

## 📦 Installation

### Prerequisites
- Python 3.13+
- ~2GB disk space for model download
- API keys for:
  - [Pinecone](https://www.pinecone.io/) (required)
  - [Azure OpenAI](https://azure.microsoft.com/en-us/products/ai-services/openai-service) (required for evaluation)
  - [Unstructured](https://unstructured.io/api-key) (optional - for complex PDFs with tables/images)

### Setup

1. **Clone the repository**
```bash
git clone <your-repo-url>
cd chunking-benchmark
```

2. **Install dependencies**
```bash
# Using uv (recommended)
uv sync

# Or using pip
pip install -e .
```

3. **Configure API keys**
```bash
# Copy example environment file
cp .env.example .env

# Edit .env and add your API keys
nano .env
```

4. **Set up your .env file**
```bash
# Required: Pinecone for vector storage
PINECONE_API_KEY=your_key_here

# Required: Azure OpenAI for evaluation
AZURE_OPENAI_API_KEY=your_key_here
AZURE_OPENAI_ENDPOINT=https://your-resource-name.openai.azure.com/

# Optional: Unstructured for complex document parsing
UNSTRUCTURED_API_KEY=your_key_here  # Leave empty to use fallback for simple files

# Optional: Disable tokenizer warnings
TOKENIZERS_PARALLELISM=false
```

5. **Configure Azure OpenAI Deployments**

In `src/config.py`, update the deployment names to match your Azure Portal:
```python
azure_deployment_name: str = "your-gpt-deployment-name"  # e.g., "gpt-4", "gpt-4-1"
azure_embedding_deployment: str = "your-embedding-deployment"  # e.g., "text-embedding-3-large"
```

6. **Test Azure OpenAI Connection** (Optional but recommended)
```bash
uv run test_azure_openai.py
```

## 🚀 Usage

### Quick Start

Run the full pipeline on your documents:

```bash
uv run src/pipeline.py path/to/documents
```

This will:
1. Load all supported documents from the path
2. Process them with all 4 chunking strategies
3. Generate embeddings using local Jina model
4. Store vectors in Pinecone (separate namespace per strategy)
5. Evaluate and compare all strategies using Ragas
6. Save results to `evaluation_results.csv`

### Command-Line Options

```bash
uv run src/pipeline.py <source> [options]

Arguments:
  source                Path to document file or directory

Options:
  --strategies          Chunking strategies to test
                       Choices: fixed, recursive, semantic, late
                       Default: all strategies

  --no-eval            Skip evaluation phase (just ingest documents)

  --clear              Clear existing vectors before ingesting

  --interactive        Start interactive query mode after pipeline
```

### Examples

**Process a single file:**
```bash
uv run src/pipeline.py document.pdf
```

**Process a directory with specific strategies:**
```bash
uv run src/pipeline.py ./documents --strategies semantic late
```

**Ingest without evaluation:**
```bash
uv run src/pipeline.py ./documents --no-eval
```

**Clear and re-index:**
```bash
uv run src/pipeline.py ./documents --clear
```

**Interactive query mode:**
```bash
uv run src/pipeline.py ./documents --interactive
```

### Interactive Query Mode

Test retrieval across all strategies interactively:

```bash
uv run src/pipeline.py ./documents --interactive
```

Then enter queries to see retrieval results:
```
Enter your query: What are the DIB goals?

FIXED Strategy:
--------------------------------------------------------------
1. Score: 0.8542
   Text: The DIB goals include...
   Source: documents/dib-goals.md

LATE Strategy:
--------------------------------------------------------------
1. Score: 0.8892  # Notice: Better score with context!
   Text: The DIB goals include...
   Source: documents/dib-goals.md
```

## 📂 Project Structure

```
chunking-benchmark/
├── src/
│   ├── config.py              # Configuration (model, Pinecone, Azure)
│   ├── document_loader.py     # Multi-format document loading
│   ├── chunking_strategies.py # All 4 chunking implementations
│   ├── embeddings.py          # Local Jina model embeddings
│   ├── vector_store.py        # Pinecone integration
│   ├── evaluation.py          # Ragas evaluation framework
│   ├── pipeline.py            # Main orchestration script
│   └── chunk.py               # Original late chunking reference
├── test_azure_openai.py      # Azure OpenAI connection test
├── delete_records.py          # Pinecone cleanup utility
├── .env                       # API keys (gitignored)
├── .env.example              # Example environment file
├── pyproject.toml            # Project dependencies
└── README.md                 # This file
```

## 🎯 Chunking Strategies Explained

### 1. Fixed-size Chunking
- **Method**: Splits text into fixed character chunks with overlap
- **Pros**: Simple, predictable chunk sizes
- **Cons**: May break semantic boundaries
- **Best for**: Uniform documents, testing baselines

### 2. Recursive Chunking
- **Method**: Splits at natural boundaries (paragraphs → sentences → words)
- **Pros**: Preserves semantic meaning better than fixed-size
- **Cons**: Variable chunk sizes
- **Best for**: General-purpose text processing

### 3. Semantic Chunking
- **Method**: Groups by document structure (sections, elements from Unstructured)
- **Pros**: Maintains document hierarchy and context
- **Cons**: Requires structured input
- **Best for**: Complex documents with clear structure

### 4. Late Chunking ⭐
- **Method**: Embed full document with local model, then pool embeddings by chunks
- **Implementation**: Follows [Jina AI's late chunking paper](https://arxiv.org/abs/2409.04701)
- **Pros**: Maximum context preservation, best retrieval quality
- **How it works**:
  1. Tokenize full document
  2. Get token embeddings from model
  3. Pool embeddings according to sentence boundaries
  4. Each chunk has full document context!
- **Best for**: When retrieval quality is critical

## 📊 Evaluation Metrics

The pipeline uses [Ragas](https://docs.ragas.io/) with Azure OpenAI for comprehensive evaluation:

### Context Metrics
- **Context Precision**: What % of retrieved chunks are relevant?
- **Context Recall**: Did we retrieve all relevant information?

### Answer Metrics
- **Faithfulness**: Is the answer grounded in retrieved context?
- **Answer Relevancy**: How well does the answer address the query?

### Output
Results are saved to `evaluation_results.csv`:
```csv
strategy,context_precision,context_recall,faithfulness,answer_relevancy
fixed,0.75,0.68,0.82,0.79
recursive,0.78,0.71,0.84,0.81
semantic,0.82,0.75,0.87,0.83
late,0.89,0.83,0.91,0.88  ← Best!
```

## ⚙️ Configuration

### Model Configuration

The pipeline uses **jinaai/jina-embeddings-v3** locally. To change models or settings, edit `src/config.py`:

```python
@dataclass
class JinaConfig:
    local_model_name: str = "jinaai/jina-embeddings-v3"  # Change model here
    device: str = "cpu"  # Change to "cuda" for GPU

@dataclass
class PineconeConfig:
    dimension: int = 1024  # Must match model output dimension
    index_name: str = "chunking-benchmark-v3"  # Change index name
```

### Chunking Parameters

Adjust chunking behavior:

```python
@dataclass
class ChunkingConfig:
    fixed_chunk_size: int = 512  # Characters per chunk
    chunk_overlap: int = 50      # Overlap between chunks
    max_chunk_length: int = 1000 # Max tokens for late chunking
```

### Evaluation Queries

Customize test queries in `src/config.py`:

```python
@dataclass
class EvaluationConfig:
    test_queries: List[str] = field(default_factory=lambda: [
        "Your domain-specific question 1?",
        "Your domain-specific question 2?",
    ])
    ground_truths: List[str] = field(default_factory=lambda: [
        "Expected answer 1",
        "Expected answer 2",
    ])
```

## 🐛 Troubleshooting

### Model Loading Issues

**Error: `FileNotFoundError` in transformers cache**
```bash
# Clear the cache and re-download
rm -rf ~/.cache/huggingface/modules/transformers_modules/jinaai/
uv run src/pipeline.py ./documents
```

### Pinecone Dimension Mismatch

**Error: `Vector dimension X does not match the dimension of the index Y`**

Solution: Update dimension in `src/config.py` to match your model:
- jina-embeddings-v2: `dimension: int = 768`
- jina-embeddings-v3: `dimension: int = 1024`

Then either:
1. Delete the index in Pinecone dashboard, or
2. Change `index_name` to create a new index

### Azure OpenAI 404 Error

**Error: `Resource not found`**

1. Run the test: `uv run test_azure_openai.py`
2. Check deployment names in Azure Portal
3. Update `src/config.py` with exact deployment names

### Out of Memory

For large documents or many chunks:
1. Reduce `fixed_chunk_size` in config
2. Process fewer documents at once
3. Use GPU if available: `device: "cuda"` in config

## 🔬 Advanced Usage

### Programmatic API

```python
from src.config import get_config
from src.pipeline import ChunkingPipeline

# Initialize pipeline
config = get_config()
pipeline = ChunkingPipeline(config)

# Load documents
documents = pipeline.load_documents("./documents")

# Process with specific strategy
chunks, embeddings = pipeline.process_with_strategy(documents, "late")

# Store in Pinecone
pipeline.ingest_to_vector_store("late", chunks, embeddings)

# Query
from src.embeddings import JinaEmbedder
embedder = JinaEmbedder(config.jina)
query_embedding = embedder.embed_query("your question")
results = pipeline.vector_store.query_strategy("late", query_embedding, top_k=5)
```

### Using GPU

For faster embeddings, enable CUDA:

```python
# In src/config.py
device: str = "cuda"  # Requires CUDA-compatible GPU
```

### Batch Processing

For large document collections:

```python
import glob

for doc_batch in glob.glob("documents/*.pdf")[:10]:  # Process 10 at a time
    pipeline.run_full_pipeline(
        source=doc_batch,
        evaluate=False  # Evaluate at the end
    )
```

## 📚 References

- [Late Chunking (Jina AI)](https://jina.ai/news/late-chunking-in-long-context-embedding-models)
- [Late Chunking Paper (arXiv)](https://arxiv.org/abs/2409.04701)
- [Ragas Evaluation Framework](https://docs.ragas.io/en/stable/)
- [Jina Embeddings v3](https://huggingface.co/jinaai/jina-embeddings-v3)
- [Pinecone Documentation](https://docs.pinecone.io/)

## 🤝 Contributing

Contributions welcome! Areas for improvement:

- Additional chunking strategies (sliding window, hierarchical)
- More evaluation metrics
- Support for additional embedding models
- Performance optimizations
- Better error handling

## 📄 License

MIT License - feel free to use for your projects!

## 🙏 Acknowledgments

- [Jina AI](https://jina.ai/) for the late chunking technique and models
- [Ragas](https://github.com/explodinggradients/ragas) for the evaluation framework
- [Unstructured](https://unstructured.io/) for document parsing
- [Pinecone](https://www.pinecone.io/) for vector storage

## 📞 Support

For issues and questions:
- Open an issue on GitHub
- Check existing issues for solutions
- Refer to API documentation for provider-specific problems

---

**Ready to benchmark?** Start with: `uv run src/pipeline.py ./documents --interactive` 🚀
