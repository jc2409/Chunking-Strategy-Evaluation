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
    api_url: str = "https://api.unstructured.io/general/v0/general"
    strategy: str = "hi_res"  # "fast", "hi_res", or "auto"
    extract_image_block_types: List[str] = field(default_factory=lambda: ["Image", "Table"])

    def __post_init__(self):
        if self.api_key is None:
            self.api_key = os.getenv("UNSTRUCTURED_API_KEY")


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
        "What is the capital of Germany?",
        "Tell me about Berlin's economy",
        "What is Berlin's transport infrastructure like?",
        "What are the universities in Berlin?",
    ])
    ground_truths: List[str] = field(default_factory=lambda: [
        "Berlin is the capital and largest city of Germany.",
        "Berlin's economy is primarily based on the service sector, encompassing creative industries, media corporations, and convention venues. It's a hub for technology startups and innovation.",
        "Berlin's transport infrastructure features an extensive public transportation network including U-Bahn (subway), S-Bahn (urban rail), trams, and buses, plus two commercial airports.",
        "Berlin is home to world-renowned universities such as the Humboldt University, the Technical University, and the Free University of Berlin.",
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
