"""Configuration settings for the chunking benchmark pipeline."""

import os
from dataclasses import dataclass, field
from typing import Optional, List
from dotenv import load_dotenv

load_dotenv()


@dataclass
class UnstructuredAPIConfig:
    """Configuration for Unstructured API."""
    api_key: Optional[str] = None
    api_url: str = "https://api.unstructuredapp.io/general/v0/general"  # Official default URL
    strategy: str = "fast"  # "fast", "hi_res", "auto", "ocr_only"
    extract_image_block_types: List[str] = field(default_factory=lambda: ["Image", "Table"])

    def __post_init__(self):
        if self.api_key is None:
            self.api_key = os.getenv("UNSTRUCTURED_API_KEY")
        # Allow override from environment variable
        url_from_env = os.getenv("UNSTRUCTURED_API_URL")
        if url_from_env:
            self.api_url = url_from_env


@dataclass
class JinaConfig:
    """Configuration for Jina Embeddings."""

    # Local model settings
    local_model_name: str = "jinaai/jina-embeddings-v3"
    device: str = "cpu"  # or "cuda" if GPU available


@dataclass
class ChunkingConfig:
    """Configuration for chunking strategies."""
    # Traditional chunking
    fixed_chunk_size: int = 512
    chunk_overlap: int = 50

    # Semantic chunking
    use_semantic_chunking: bool = True

    # Late chunking
    max_chunk_length: int = 1000
    return_tokens: bool = True
    return_chunks: bool = True
    use_jina_api_chunking: bool = False  # API creates too many chunks; use local method for now


@dataclass
class PineconeConfig:
    """Configuration for Pinecone vector database."""
    api_key: Optional[str] = None
    index_name: str = "chunking-benchmark-v2"
    dimension: int = 1024 
    metric: str = "cosine"
    cloud: str = "aws"
    region: str = "us-east-1"

    def __post_init__(self):
        if self.api_key is None:
            self.api_key = os.getenv("PINECONE_API_KEY")


@dataclass
class RagasConfig:
    """Configuration for Ragas evaluation framework using Azure OpenAI."""
    # Azure OpenAI Configuration
    azure_openai_api_key: Optional[str] = None
    azure_openai_endpoint: Optional[str] = None
    azure_deployment_name: str = "gpt-4.1"  # Your GPT-4.1 deployment name
    azure_embedding_deployment: str = "text-embedding-3-large"  # Your embedding deployment name
    azure_api_version: str = "2024-12-01-preview"  # Azure OpenAI API version

    metrics: List[str] = field(default_factory=lambda: [
        "context_precision",
        "context_recall",
        "faithfulness",
        "answer_relevancy",
    ])

    def __post_init__(self):
        if self.azure_openai_api_key is None:
            self.azure_openai_api_key = os.getenv("AZURE_OPENAI_API_KEY")
        if self.azure_openai_endpoint is None:
            self.azure_openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")


@dataclass
class EvaluationConfig:
    """Configuration for evaluation settings."""
    test_queries: List[str] = field(default_factory=lambda: [
        "What is the main innovation introduced in the Transformer architecture?",
        "What does BERT stand for and what is its key training objective?",
        "How many parameters does GPT-3 have and what makes it different from GPT-2?",
        "What is the main contribution of ResNet and what problem does it solve?",
        "What are the two main components of RAG and how do they work together?",
        "What is multi-head attention and why is it used in Transformers?",
        "What is the difference between BERT's pre-training tasks: MLM and NSP?",
        "What is few-shot learning in the context of GPT-3?",
        "What is the skip connection or residual connection in ResNet?",
        "How does RAG combine retrieval and generation for question answering?",
    ])
    ground_truths: List[str] = field(default_factory=lambda: [
        "The main innovation in the Transformer architecture is the self-attention mechanism, which allows the model to process all positions in the input sequence in parallel, eliminating the need for recurrent connections. The Transformer relies entirely on attention mechanisms to draw global dependencies between input and output, making it more efficient and parallelizable than RNNs.",
        "BERT stands for Bidirectional Encoder Representations from Transformers. Its key training objective is masked language modeling (MLM), where random tokens in the input are masked and the model learns to predict them based on bidirectional context. BERT also uses next sentence prediction (NSP) as a secondary objective.",
        "GPT-3 has 175 billion parameters, making it significantly larger than GPT-2 which had 1.5 billion parameters. The key difference is that GPT-3 demonstrates strong few-shot learning capabilities, meaning it can perform many tasks with just a few examples in the prompt, without requiring fine-tuning.",
        "The main contribution of ResNet is the introduction of residual or skip connections, which allow gradients to flow directly through the network. This solves the vanishing gradient problem and enables training of very deep networks (50, 101, or even 152 layers) that would otherwise be difficult or impossible to train.",
        "RAG has two main components: a retriever and a generator. The retriever uses dense passage retrieval to find relevant documents from a knowledge source, and the generator (typically a seq2seq model) uses these retrieved documents as additional context to generate the final answer. They work together end-to-end to combine parametric and non-parametric memory.",
        "Multi-head attention is a mechanism where the attention operation is performed multiple times in parallel with different learned linear projections. Each attention head can focus on different aspects of the input, allowing the model to jointly attend to information from different representation subspaces at different positions. The outputs are concatenated and linearly transformed.",
        "MLM (Masked Language Modeling) involves randomly masking 15% of tokens in the input and training the model to predict them based on bidirectional context. NSP (Next Sentence Prediction) is a binary classification task where the model predicts whether two sentences appear consecutively in the original text. MLM enables bidirectional learning while NSP helps with understanding sentence relationships.",
        "Few-shot learning in GPT-3 refers to the model's ability to perform a new task given only a few examples in the prompt, without any gradient updates or fine-tuning. The model is given a task description and a few input-output examples (typically 10-100), and can then perform the task on new inputs. This is different from zero-shot (no examples) and one-shot (one example) learning.",
        "A skip connection or residual connection in ResNet is a direct pathway that bypasses one or more layers by adding the input of a block directly to its output. Mathematically, instead of learning H(x), the network learns F(x) = H(x) - x, making it easier to learn identity mappings. This allows gradients to flow backward through the network more easily.",
        "RAG combines retrieval and generation by first using a neural retriever (DPR - Dense Passage Retrieval) to find relevant documents from a knowledge base given a question. These retrieved documents are then passed along with the question to a sequence-to-sequence generator model (BART), which uses both the question and the retrieved context to generate the final answer. The retriever and generator are trained end-to-end.",
    ])
    top_k: int = 5
    batch_size: int = 10


@dataclass
class Config:
    """Main configuration class combining all settings."""
    unstructured: UnstructuredAPIConfig = field(default_factory=UnstructuredAPIConfig)
    jina: JinaConfig = field(default_factory=JinaConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    pinecone: PineconeConfig = field(default_factory=PineconeConfig)
    ragas: RagasConfig = field(default_factory=RagasConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)


def get_config() -> Config:
    """Get the default configuration."""
    return Config()
