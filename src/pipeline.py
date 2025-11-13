"""Main pipeline for chunking benchmark."""

import argparse
from pathlib import Path
from typing import List

from config import get_config
from document_loader import DocumentLoader
from chunking_strategies import TraditionalChunker, LateChunker
from embeddings import JinaEmbedder, JinaAPIEmbedder
from vector_store import MultiStrategyVectorStore
from evaluation import ChunkingEvaluator


class ChunkingPipeline:
    """End-to-end pipeline for document processing and chunking evaluation."""

    def __init__(self, config_override=None):
        """Initialize pipeline with configuration."""
        self.config = config_override or get_config()

        # Initialize components
        self.doc_loader = DocumentLoader(self.config.unstructured)
        self.traditional_chunker = TraditionalChunker(self.config.chunking)
        self.late_chunker = LateChunker(self.config.jina, self.config.chunking)

        # Choose embedder based on configuration
        if self.config.jina.use_api:
            print("Using Jina API for embeddings")
            self.embedder = JinaAPIEmbedder(self.config.jina)
        else:
            print("Using local Jina model for embeddings")
            self.embedder = JinaEmbedder(self.config.jina)

        self.vector_store = MultiStrategyVectorStore(self.config.pinecone)

    def load_documents(self, source: str, recursive: bool = True):
        """
        Load documents from a file or directory.

        Args:
            source: Path to file or directory
            recursive: Whether to search subdirectories

        Returns:
            List of loaded documents
        """
        print(f"\n{'='*60}")
        print(f"Loading documents from: {source}")
        print(f"{'='*60}\n")

        path = Path(source)

        if path.is_file():
            docs = [self.doc_loader.load(source)]
        elif path.is_dir():
            docs = self.doc_loader.load_directory(source, recursive=recursive)
        else:
            raise ValueError(f"Invalid source: {source}")

        print(f"Loaded {len(docs)} document(s)")

        return docs

    def process_with_strategy(self, documents, strategy: str):
        """
        Process documents with a specific chunking strategy.

        Args:
            documents: List of documents
            strategy: Strategy name ("fixed", "recursive", "semantic", "late")

        Returns:
            Tuple of (chunks, embeddings)
        """
        print(f"\n{'='*60}")
        print(f"Processing with strategy: {strategy}")
        print(f"{'='*60}\n")

        all_chunks = []
        all_embeddings = []

        # Late chunking requires special handling: chunk and embed per document
        if strategy == "late":
            for doc in documents:
                print(f"Chunking: {doc.source}")

                # Chunk the document once
                chunks, chunk_texts, span_annotations = self.late_chunker.chunk_with_late_chunking(doc)

                # Generate embeddings with late chunking
                if isinstance(self.embedder, JinaAPIEmbedder):
                    # API-based: pass chunk_texts directly (API handles concatenation)
                    embeddings = self.embedder.embed_with_late_chunking(
                        chunk_texts,
                        task=self.config.jina.api_task,
                        dimensions=self.config.jina.api_dimensions
                    )
                else:
                    # Local model: pass full_text and span_annotations
                    full_text = self.late_chunker._combine_elements(doc.elements)
                    embeddings = self.embedder.embed_with_late_chunking(
                        full_text,
                        span_annotations
                    )

                all_chunks.extend(chunks)
                all_embeddings.extend(embeddings)

            print(f"Created {len(all_chunks)} chunks")
            print("Generating embeddings...")
            print(f"Generated {len(all_embeddings)} embeddings")

            return all_chunks, all_embeddings

        # Traditional chunking strategies
        for doc in documents:
            print(f"Chunking: {doc.source}")

            if strategy == "fixed":
                chunks = self.traditional_chunker.chunk_fixed_size(doc)
            elif strategy == "recursive":
                chunks = self.traditional_chunker.chunk_recursive(doc)
            elif strategy == "semantic":
                chunks = self.traditional_chunker.chunk_semantic(doc)
            else:
                raise ValueError(f"Unknown strategy: {strategy}")

            all_chunks.extend(chunks)

        print(f"Created {len(all_chunks)} chunks")

        # Generate embeddings
        print("Generating embeddings...")
        if isinstance(self.embedder, JinaAPIEmbedder):
            # API-based: pass chunks with task and dimensions
            embeddings = self.embedder.embed_chunks_traditional(
                all_chunks,
                task=self.config.jina.api_task,
                dimensions=self.config.jina.api_dimensions
            )
        else:
            # Local model
            embeddings = self.embedder.embed_chunks_traditional(all_chunks)

        print(f"Generated {len(embeddings)} embeddings")

        return all_chunks, embeddings

    def ingest_to_vector_store(self, strategy: str, chunks, embeddings):
        """
        Ingest chunks and embeddings to vector store.

        Args:
            strategy: Strategy name
            chunks: List of chunks
            embeddings: List of embeddings
        """
        print(f"\n{'='*60}")
        print(f"Ingesting to Pinecone (namespace: {strategy})")
        print(f"{'='*60}\n")

        num_upserted = self.vector_store.upsert_for_strategy(
            strategy,
            chunks,
            embeddings
        )

        print(f"Successfully ingested {num_upserted} vectors")

    def run_full_pipeline(
        self,
        source: str,
        strategies: List[str] = None,
        evaluate: bool = True,
        clear_existing: bool = False
    ):
        """
        Run the complete pipeline: load -> chunk -> embed -> store -> evaluate.

        Args:
            source: Path to documents
            strategies: List of strategies to test (default: all)
            evaluate: Whether to run evaluation
            clear_existing: Whether to clear existing vectors
        """
        if strategies is None:
            strategies = ["fixed", "recursive", "semantic", "late"]

        print("\n" + "="*60)
        print("CHUNKING BENCHMARK PIPELINE")
        print("="*60)
        print(f"Source: {source}")
        print(f"Strategies: {', '.join(strategies)}")
        print("="*60 + "\n")

        # Clear existing vectors if requested
        if clear_existing:
            print("Clearing existing vectors...")
            self.vector_store.clear_all()

        # Load documents once
        documents = self.load_documents(source)

        # Process each strategy
        for strategy in strategies:
            try:
                chunks, embeddings = self.process_with_strategy(documents, strategy)
                self.ingest_to_vector_store(strategy, chunks, embeddings)
            except Exception as e:
                print(f"Error processing strategy '{strategy}': {e}")
                continue

        # Evaluate if requested
        if evaluate:
            print("\n" + "="*60)
            print("EVALUATION PHASE")
            print("="*60 + "\n")

            evaluator = ChunkingEvaluator(self.config)
            results_df = evaluator.evaluate_all_strategies(strategies)

            # Save results
            output_path = "evaluation_results.csv"
            results_df.to_csv(output_path, index=False)
            print(f"\nEvaluation results saved to: {output_path}")

            return results_df

    def query_interactive(self, strategies: List[str] = None):
        """
        Interactive query mode to test retrieval.

        Args:
            strategies: List of strategies to query (default: all available)
        """
        if strategies is None:
            strategies = list(self.vector_store.stores.keys())

        if not strategies:
            print("No strategies available. Please run the pipeline first.")
            return

        print("\n" + "="*60)
        print("INTERACTIVE QUERY MODE")
        print("="*60)
        print("Type 'quit' or 'exit' to stop")
        print("="*60 + "\n")

        while True:
            query = input("\nEnter your query: ").strip()

            if query.lower() in ['quit', 'exit', 'q']:
                print("Exiting...")
                break

            if not query:
                continue

            # Embed query
            if isinstance(self.embedder, JinaAPIEmbedder):
                # API-based: use retrieval.query task
                query_embedding = self.embedder.embed_query(
                    query,
                    task="retrieval.query",
                    dimensions=self.config.jina.api_dimensions
                )
            else:
                # Local model
                query_embedding = self.embedder.embed_query(query)

            # Query all strategies
            print(f"\n{'='*60}")
            print(f"Results for: {query}")
            print(f"{'='*60}\n")

            for strategy in strategies:
                try:
                    results = self.vector_store.query_strategy(
                        strategy,
                        query_embedding,
                        top_k=3
                    )

                    print(f"\n{strategy.upper()} Strategy:")
                    print("-" * 60)

                    for idx, result in enumerate(results, 1):
                        print(f"\n{idx}. Score: {result['score']:.4f}")
                        print(f"   Text: {result['text'][:200]}...")
                        print(f"   Source: {result['source']}")

                except Exception as e:
                    print(f"Error querying {strategy}: {e}")


def main():
    """Main entry point for the pipeline."""
    parser = argparse.ArgumentParser(description="Chunking Benchmark Pipeline")

    parser.add_argument(
        'source',
        type=str,
        help='Path to document file or directory'
    )

    parser.add_argument(
        '--strategies',
        nargs='+',
        choices=['fixed', 'recursive', 'semantic', 'late'],
        default=['fixed', 'recursive', 'semantic', 'late'],
        help='Chunking strategies to test'
    )

    parser.add_argument(
        '--no-eval',
        action='store_true',
        help='Skip evaluation phase'
    )

    parser.add_argument(
        '--clear',
        action='store_true',
        help='Clear existing vectors before ingesting'
    )

    parser.add_argument(
        '--interactive',
        action='store_true',
        help='Start interactive query mode after pipeline'
    )

    args = parser.parse_args()

    # Run pipeline
    pipeline = ChunkingPipeline()
    pipeline.run_full_pipeline(
        source=args.source,
        strategies=args.strategies,
        evaluate=not args.no_eval,
        clear_existing=args.clear
    )

    # Interactive mode
    if args.interactive:
        pipeline.query_interactive(args.strategies)


if __name__ == "__main__":
    main()
