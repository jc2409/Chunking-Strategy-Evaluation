"""Vector store operations using Pinecone."""

import time
from typing import List, Dict, Tuple
import numpy as np
from tqdm import tqdm

from pinecone import Pinecone, ServerlessSpec

from config import PineconeConfig
from chunking_strategies import Chunk


class PineconeVectorStore:
    """Manage vector storage and retrieval with Pinecone."""

    def __init__(self, config: PineconeConfig, namespace: str = "default"):
        """
        Initialize Pinecone vector store.

        Args:
            config: Pinecone configuration
            namespace: Namespace for organizing vectors (e.g., strategy name)
        """
        self.config = config
        self.namespace = namespace

        if not self.config.api_key:
            raise ValueError("Pinecone API key is required")

        # Initialize Pinecone client
        self.pc = Pinecone(api_key=self.config.api_key)
        self.index_name = self.config.index_name

        # Create or connect to index
        self._ensure_index_exists()
        self.index = self.pc.Index(self.index_name)

    def _ensure_index_exists(self):
        """Create index if it doesn't exist."""
        existing_indexes = [index.name for index in self.pc.list_indexes()]

        if self.index_name not in existing_indexes:
            print(f"Creating Pinecone index: {self.index_name}")
            self.pc.create_index(
                name=self.index_name,
                dimension=self.config.dimension,
                metric=self.config.metric,
                spec=ServerlessSpec(
                    cloud=self.config.cloud,
                    region=self.config.region
                )
            )
            # Wait for index to be ready
            time.sleep(1)
            print(f"Index {self.index_name} created successfully")
        else:
            print(f"Using existing index: {self.index_name}")

    def upsert_chunks(
        self,
        chunks: List[Chunk],
        embeddings: List[np.ndarray],
        batch_size: int = 100
    ) -> int:
        """
        Upsert chunks with their embeddings to Pinecone.

        Args:
            chunks: List of chunks
            embeddings: List of embeddings (same length as chunks)
            batch_size: Batch size for upsert operations

        Returns:
            Number of vectors upserted
        """
        if len(chunks) != len(embeddings):
            raise ValueError("Number of chunks and embeddings must match")

        vectors = []
        for chunk, embedding in zip(chunks, embeddings):
            vector = {
                'id': chunk.chunk_id,
                'values': embedding.tolist() if isinstance(embedding, np.ndarray) else embedding,
                'metadata': {
                    'text': chunk.text[:1000],  # Limit text size in metadata
                    'strategy': chunk.strategy,
                    'source': chunk.metadata.get('source', ''),
                    'chunk_index': chunk.metadata.get('chunk_index', 0),
                    'file_type': chunk.metadata.get('file_type', ''),
                    'chunk_method': chunk.metadata.get('chunk_method', ''),
                }
            }
            vectors.append(vector)

        # Upsert in batches
        total_upserted = 0
        for i in tqdm(range(0, len(vectors), batch_size), desc=f"Upserting to Pinecone ({self.namespace})"):
            batch = vectors[i:i + batch_size]
            self.index.upsert(vectors=batch, namespace=self.namespace)
            total_upserted += len(batch)

        print(f"Upserted {total_upserted} vectors to namespace '{self.namespace}'")
        return total_upserted

    def query(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        filter_dict: Dict = None
    ) -> List[Dict]:
        """
        Query the vector store for similar chunks.

        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            filter_dict: Optional metadata filter

        Returns:
            List of matching results with scores
        """
        query_vector = query_embedding.tolist() if isinstance(query_embedding, np.ndarray) else query_embedding

        results = self.index.query(
            vector=query_vector,
            top_k=top_k,
            namespace=self.namespace,
            include_metadata=True,
            filter=filter_dict
        )

        matches = []
        for match in results['matches']:
            matches.append({
                'id': match['id'],
                'score': match['score'],
                'text': match['metadata'].get('text', ''),
                'strategy': match['metadata'].get('strategy', ''),
                'source': match['metadata'].get('source', ''),
                'chunk_index': match['metadata'].get('chunk_index', 0),
                'metadata': match['metadata']
            })

        return matches

    def delete_namespace(self):
        """Delete all vectors in the current namespace."""
        self.index.delete(delete_all=True, namespace=self.namespace)
        print(f"Deleted all vectors in namespace '{self.namespace}'")

    def get_stats(self) -> Dict:
        """Get index statistics."""
        stats = self.index.describe_index_stats()
        return stats


class MultiStrategyVectorStore:
    """Manage multiple vector stores for different chunking strategies."""

    def __init__(self, config: PineconeConfig):
        """Initialize with Pinecone configuration."""
        self.config = config
        self.stores = {}

    def get_store(self, strategy: str) -> PineconeVectorStore:
        """
        Get or create a vector store for a specific strategy.

        Args:
            strategy: Strategy name (e.g., "fixed", "recursive", "late")

        Returns:
            PineconeVectorStore instance
        """
        if strategy not in self.stores:
            self.stores[strategy] = PineconeVectorStore(
                self.config,
                namespace=strategy
            )
        return self.stores[strategy]

    def upsert_for_strategy(
        self,
        strategy: str,
        chunks: List[Chunk],
        embeddings: List[np.ndarray]
    ) -> int:
        """
        Upsert chunks for a specific strategy.

        Args:
            strategy: Strategy name
            chunks: List of chunks
            embeddings: List of embeddings

        Returns:
            Number of vectors upserted
        """
        store = self.get_store(strategy)
        return store.upsert_chunks(chunks, embeddings)

    def query_strategy(
        self,
        strategy: str,
        query_embedding: np.ndarray,
        top_k: int = 5
    ) -> List[Dict]:
        """
        Query a specific strategy's vector store.

        Args:
            strategy: Strategy name
            query_embedding: Query embedding
            top_k: Number of results

        Returns:
            List of matches
        """
        store = self.get_store(strategy)
        return store.query(query_embedding, top_k=top_k)

    def query_all_strategies(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5
    ) -> Dict[str, List[Dict]]:
        """
        Query all strategies and return results grouped by strategy.

        Args:
            query_embedding: Query embedding
            top_k: Number of results per strategy

        Returns:
            Dictionary mapping strategy names to their results
        """
        results = {}
        for strategy, store in self.stores.items():
            results[strategy] = store.query(query_embedding, top_k=top_k)
        return results

    def clear_all(self):
        """Delete all vectors from all strategy namespaces."""
        for strategy, store in self.stores.items():
            store.delete_namespace()
            print(f"Cleared strategy: {strategy}")
