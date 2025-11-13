"""Embedding generation using Jina local models and API."""

import numpy as np
from typing import List, Tuple, Optional
from tqdm import tqdm
import torch
from transformers import AutoModel, AutoTokenizer
import requests
import os

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


class JinaAPIEmbedder:
    """Generate embeddings using Jina AI API with late chunking support."""

    def __init__(self, config: JinaConfig, api_key: Optional[str] = None):
        """
        Initialize with Jina configuration.

        Args:
            config: JinaConfig instance
            api_key: Jina API key (if not provided, reads from JINA_API_KEY env var)
        """
        self.config = config
        self.api_key = api_key or os.getenv("JINA_API_KEY")

        if not self.api_key:
            raise ValueError(
                "Jina API key not found. Please set JINA_API_KEY environment variable "
                "or pass it to the constructor."
            )

        self.api_url = "https://api.jina.ai/v1/embeddings"
        self.model_name = "jina-embeddings-v3"

        print(f"Initialized Jina API Embedder with model: {self.model_name}")

    def embed_with_late_chunking(
        self,
        chunk_texts: List[str],
        task: str = "retrieval.passage",
        dimensions: int = 1024,
        batch_size: int = 100
    ) -> List[np.ndarray]:
        """
        Embed chunks using Jina API's late chunking feature with batching.

        This method uses the API's built-in late_chunking parameter, which
        concatenates all input texts and embeds them with full document context.

        Note: When late_chunking=True, the API concatenates inputs and total
        tokens across all inputs must be <= 8192. We use smaller batches to
        avoid hitting this limit.

        Args:
            chunk_texts: List of chunk texts to embed
            task: Task type (retrieval.passage, retrieval.query, text-matching, etc.)
            dimensions: Output embedding dimensions (max 1024)
            batch_size: Number of chunks to process per API call (max 512, default 100 for safety)

        Returns:
            List of chunk embeddings with full document context
        """
        # Jina API limit: 512 items per request, but with late_chunking
        # we need to be more conservative due to 8192 token limit
        max_batch_size = min(batch_size, 100)

        all_embeddings = []

        # Process in batches
        print(f"Processing {len(chunk_texts)} chunks in batches of {max_batch_size} (late chunking enabled)")
        for i in tqdm(range(0, len(chunk_texts), max_batch_size), desc="Late chunking batches"):
            batch = chunk_texts[i:i + max_batch_size]

            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }

            payload = {
                "model": self.model_name,
                "task": task,
                "dimensions": dimensions,
                "late_chunking": True,  # Enable late chunking
                "input": batch
            }

            try:
                response = requests.post(self.api_url, json=payload, headers=headers)
                response.raise_for_status()

                result = response.json()

                # Extract embeddings from response
                for item in result.get("data", []):
                    embedding = np.array(item["embedding"], dtype=np.float32)
                    all_embeddings.append(embedding)

            except requests.exceptions.RequestException as e:
                print(f"Error calling Jina API for batch {i//max_batch_size + 1}: {e}")
                if hasattr(e, 'response') and e.response is not None:
                    print(f"Response: {e.response.text}")
                raise

        return all_embeddings

    def embed_chunks_traditional(
        self,
        chunks: List[Chunk],
        task: str = "retrieval.passage",
        dimensions: int = 1024,
        batch_size: int = 500
    ) -> List[np.ndarray]:
        """
        Embed chunks using traditional method (without late chunking) with batching.

        Args:
            chunks: List of chunks to embed
            task: Task type (retrieval.passage, retrieval.query, text-matching, etc.)
            dimensions: Output embedding dimensions (max 1024)
            batch_size: Number of chunks to process per API call (max 512, default 500)

        Returns:
            List of embeddings (one per chunk)
        """
        chunk_texts = [chunk.text for chunk in chunks]

        # Jina API limit: 512 items per request
        max_batch_size = min(batch_size, 512)

        all_embeddings = []

        # Process in batches
        if len(chunk_texts) > max_batch_size:
            print(f"Processing {len(chunk_texts)} chunks in batches of {max_batch_size}")
        for i in tqdm(range(0, len(chunk_texts), max_batch_size), desc="Traditional batches", disable=len(chunk_texts) <= max_batch_size):
            batch = chunk_texts[i:i + max_batch_size]

            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }

            payload = {
                "model": self.model_name,
                "task": task,
                "dimensions": dimensions,
                "late_chunking": False,  # Disable late chunking
                "input": batch
            }

            try:
                response = requests.post(self.api_url, json=payload, headers=headers)
                response.raise_for_status()

                result = response.json()

                # Extract embeddings from response
                for item in result.get("data", []):
                    embedding = np.array(item["embedding"], dtype=np.float32)
                    all_embeddings.append(embedding)

            except requests.exceptions.RequestException as e:
                print(f"Error calling Jina API for batch {i//max_batch_size + 1}: {e}")
                if hasattr(e, 'response') and e.response is not None:
                    print(f"Response: {e.response.text}")
                raise

        return all_embeddings

    def embed_query(
        self,
        query: str,
        task: str = "retrieval.query",
        dimensions: int = 1024
    ) -> np.ndarray:
        """
        Embed a query for retrieval.

        Args:
            query: Query text
            task: Task type (usually retrieval.query for queries)
            dimensions: Output embedding dimensions (max 1024)

        Returns:
            Query embedding
        """
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        payload = {
            "model": self.model_name,
            "task": task,
            "dimensions": dimensions,
            "late_chunking": False,
            "input": [query]
        }

        try:
            response = requests.post(self.api_url, json=payload, headers=headers)
            response.raise_for_status()

            result = response.json()
            embedding = np.array(result["data"][0]["embedding"], dtype=np.float32)

            return embedding

        except requests.exceptions.RequestException as e:
            print(f"Error calling Jina API: {e}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"Response: {e.response.text}")
            raise