"""Embedding generation using Jina local models."""

import numpy as np
from typing import List, Tuple, Optional
from tqdm import tqdm
import torch
from transformers import AutoModel, AutoTokenizer

from config import JinaConfig
from chunking_strategies import Chunk


class JinaEmbedder:
    """Generate embeddings using Jina local model."""

    def __init__(self, config: JinaConfig):
        """Initialize with Jina configuration."""
        self.config = config

        # Load local model for embeddings
        print(f"Loading local model: {self.config.local_model_name}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.local_model_name,
            trust_remote_code=True
        )
        self.model = AutoModel.from_pretrained(
            self.config.local_model_name,
            trust_remote_code=True
        )
        self.model.to(self.config.device)
        self.model.eval()
        print(f"Model loaded on device: {self.config.device}")


    def embed_chunks_traditional(self, chunks: List[Chunk]) -> List[np.ndarray]:
        """
        Embed chunks using traditional method (chunk-then-embed).

        Args:
            chunks: List of chunks to embed

        Returns:
            List of embeddings (one per chunk)
        """
        return self._embed_chunks_local(chunks)

    def _embed_chunks_local(self, chunks: List[Chunk]) -> List[np.ndarray]:
        """Embed chunks using local model."""

        embeddings = []
        texts = [chunk.text for chunk in chunks]

        with torch.no_grad():
            for text in tqdm(texts, desc="Embedding chunks (local model)"):
                # Use model's encode method if available
                if hasattr(self.model, 'encode'):
                    embedding = self.model.encode([text])[0]
                else:
                    # Manual encoding
                    inputs = self.tokenizer(text, return_tensors='pt', truncation=True, max_length=8192)
                    inputs = {k: v.to(self.config.device) for k, v in inputs.items()}
                    outputs = self.model(**inputs)
                    # Mean pooling
                    embedding = outputs.last_hidden_state.mean(dim=1)[0].cpu().numpy()

                embeddings.append(embedding)

        return embeddings


    def embed_with_late_chunking(
        self,
        full_text: str,
        span_annotations: List[Tuple[int, int]]
    ) -> List[np.ndarray]:
        """
        Embed using late chunking method (embed-then-chunk).

        This is the key method that implements late chunking as described in:
        https://jina.ai/news/late-chunking-in-long-context-embedding-models

        Args:
            full_text: Full document text
            span_annotations: List of (start, end) token positions for chunks

        Returns:
            List of chunk embeddings with full document context
        """
        # Tokenize the full document
        inputs = self.tokenizer(full_text, return_tensors='pt', truncation=True, max_length=8192)
        inputs = {k: v.to(self.config.device) for k, v in inputs.items()}

        # Get model output (all token embeddings)
        with torch.no_grad():
            model_output = self.model(**inputs)

        # Apply late chunking pooling
        embeddings = self._late_chunking_pooling(model_output, [span_annotations])[0]

        return embeddings

    def _late_chunking_pooling(
        self,
        model_output,
        span_annotations: List[List[Tuple[int, int]]],
        max_length: Optional[int] = None
    ) -> List[List[np.ndarray]]:
        """
        Context-sensitive chunked pooling - the core of late chunking.

        This function pools token embeddings according to span annotations,
        preserving the document context in each chunk embedding.

        Args:
            model_output: Model output containing token embeddings
            span_annotations: List of lists of (start, end) token positions
            max_length: Maximum sequence length (for truncation)

        Returns:
            List of lists of pooled chunk embeddings
        """

        token_embeddings = model_output[0]  # Shape: (batch_size, seq_len, hidden_dim)
        outputs = []

        for embeddings, annotations in zip(token_embeddings, span_annotations):
            if max_length is not None:
                # Remove annotations beyond max length
                annotations = [
                    (start, min(end, max_length - 1))
                    for (start, end) in annotations
                    if start < (max_length - 1)
                ]

            # Pool embeddings for each span (mean pooling)
            pooled_embeddings = [
                embeddings[start:end].sum(dim=0) / (end - start)
                for start, end in annotations
                if (end - start) >= 1
            ]

            # Convert to numpy
            pooled_embeddings = [
                embedding.detach().cpu().numpy() for embedding in pooled_embeddings
            ]

            outputs.append(pooled_embeddings)

        return outputs

    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed a query for retrieval.

        Args:
            query: Query text

        Returns:
            Query embedding
        """

        # Use local model
        with torch.no_grad():
            if hasattr(self.model, 'encode'):
                return self.model.encode([query])[0]
            else:
                inputs = self.tokenizer(query, return_tensors='pt', truncation=True, max_length=8192)
                inputs = {k: v.to(self.config.device) for k, v in inputs.items()}
                outputs = self.model(**inputs)
                return outputs.last_hidden_state.mean(dim=1)[0].cpu().numpy()