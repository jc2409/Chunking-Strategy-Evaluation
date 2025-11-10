"""Evaluation framework using Ragas for comparing chunking strategies."""

import pandas as pd
import numpy as np
from typing import List, Dict
from datasets import Dataset

from ragas import evaluate
from ragas.metrics import (
    context_precision,
    context_recall,
    faithfulness,
    answer_relevancy,
)
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings

from config import Config
from embeddings import JinaEmbedder
from vector_store import MultiStrategyVectorStore


class ChunkingEvaluator:
    """Evaluate different chunking strategies using Ragas."""

    def __init__(self, config: Config):
        """Initialize evaluator with configuration."""
        self.config = config
        self.embedder = JinaEmbedder(config.jina)
        self.vector_store = MultiStrategyVectorStore(config.pinecone)

        # Initialize LLM for Ragas (using Azure OpenAI)
        self.llm = AzureChatOpenAI(
            azure_deployment=config.ragas.azure_deployment_name,
            azure_endpoint=config.ragas.azure_openai_endpoint,
            api_key=config.ragas.azure_openai_api_key,
            api_version=config.ragas.azure_api_version
        )
        self.azure_embeddings = AzureOpenAIEmbeddings(
            azure_deployment=config.ragas.azure_embedding_deployment,
            azure_endpoint=config.ragas.azure_openai_endpoint,
            api_key=config.ragas.azure_openai_api_key,
            api_version=config.ragas.azure_api_version
        )

    def evaluate_strategy(
        self,
        strategy: str,
        questions: List[str],
        ground_truths: List[str],
        generate_answers: bool = True
    ) -> Dict:
        """
        Evaluate a specific chunking strategy.

        Args:
            strategy: Strategy name (e.g., "fixed", "recursive", "late")
            questions: List of evaluation questions
            ground_truths: List of ground truth answers
            generate_answers: Whether to generate answers using LLM

        Returns:
            Dictionary with evaluation metrics
        """
        print(f"\n{'='*60}")
        print(f"Evaluating strategy: {strategy}")
        print(f"{'='*60}\n")

        # Retrieve contexts for each question
        contexts = []
        answers = []

        for question in questions:
            # Embed query
            query_embedding = self.embedder.embed_query(question)

            # Retrieve top-k chunks
            results = self.vector_store.query_strategy(
                strategy,
                query_embedding,
                top_k=self.config.evaluation.top_k
            )

            # Extract context texts
            context = [result['text'] for result in results]
            contexts.append(context)

            # Generate answer if requested
            if generate_answers:
                answer = self._generate_answer(question, context)
                answers.append(answer)
            else:
                # Use ground truth as answer for context-only metrics
                answers.append(ground_truths[questions.index(question)])

        # Prepare dataset for Ragas
        eval_data = {
            'question': questions,
            'answer': answers,
            'contexts': contexts,
            'ground_truth': ground_truths
        }

        dataset = Dataset.from_dict(eval_data)

        # Evaluate with Ragas
        metrics = [
            context_precision,
            context_recall,
            faithfulness,
            answer_relevancy,
        ]

        try:
            results = evaluate(
                dataset=dataset,
                metrics=metrics,
                llm=self.llm,
                embeddings=self.azure_embeddings
            )

            # Extract mean scores from results (Ragas returns per-question scores)
            scores = {
                'strategy': strategy,
                'context_precision': float(np.mean(results['context_precision']) if isinstance(results['context_precision'], list) else results['context_precision']),
                'context_recall': float(np.mean(results['context_recall']) if isinstance(results['context_recall'], list) else results['context_recall']),
                'faithfulness': float(np.mean(results['faithfulness']) if isinstance(results['faithfulness'], list) else results['faithfulness']),
                'answer_relevancy': float(np.mean(results['answer_relevancy']) if isinstance(results['answer_relevancy'], list) else results['answer_relevancy']),
            }

            print(f"\nResults for {strategy}:")
            for metric, value in scores.items():
                if metric != 'strategy':
                    print(f"  {metric}: {value:.4f}")

            return scores

        except Exception as e:
            print(f"Error evaluating {strategy}: {e}")
            return {
                'strategy': strategy,
                'error': str(e)
            }

    def evaluate_all_strategies(
        self,
        strategies: List[str],
        questions: List[str] = None,
        ground_truths: List[str] = None
    ) -> pd.DataFrame:
        """
        Evaluate all chunking strategies and compare results.

        Args:
            strategies: List of strategy names to evaluate
            questions: Evaluation questions (uses config default if None)
            ground_truths: Ground truth answers (uses config default if None)

        Returns:
            DataFrame with comparison results
        """
        if questions is None:
            questions = self.config.evaluation.test_queries
        if ground_truths is None:
            ground_truths = self.config.evaluation.ground_truths

        if len(questions) != len(ground_truths):
            raise ValueError("Number of questions and ground truths must match")

        results = []
        for strategy in strategies:
            scores = self.evaluate_strategy(strategy, questions, ground_truths)
            results.append(scores)

        # Create comparison DataFrame
        df = pd.DataFrame(results)

        print(f"\n{'='*60}")
        print("COMPARISON SUMMARY")
        print(f"{'='*60}\n")
        print(df.to_string(index=False))

        # Calculate and display best strategy for each metric
        print(f"\n{'='*60}")
        print("BEST STRATEGIES")
        print(f"{'='*60}\n")

        metrics = ['context_precision', 'context_recall', 'faithfulness', 'answer_relevancy']
        for metric in metrics:
            if metric in df.columns:
                best_idx = df[metric].idxmax()
                best_strategy = df.loc[best_idx, 'strategy']
                best_score = df.loc[best_idx, metric]
                print(f"{metric}: {best_strategy} ({best_score:.4f})")

        return df

    def _generate_answer(self, question: str, contexts: List[str]) -> str:
        """
        Generate an answer using LLM and retrieved contexts.

        Args:
            question: Question to answer
            contexts: Retrieved context chunks

        Returns:
            Generated answer
        """
        context_text = "\n\n".join(contexts)

        prompt = f"""Answer the question based on the context below.

Context:
{context_text}

Question: {question}

Answer:"""

        try:
            response = self.llm.invoke(prompt)
            return response.content
        except Exception as e:
            print(f"Error generating answer: {e}")
            return "Error generating answer"

    def compute_additional_metrics(
        self,
        strategy: str,
        questions: List[str]
    ) -> Dict:
        """
        Compute additional custom metrics.

        Args:
            strategy: Strategy name
            questions: List of questions

        Returns:
            Dictionary with additional metrics
        """
        retrieval_times = []
        avg_chunk_sizes = []

        for question in questions:
            import time
            start = time.time()

            query_embedding = self.embedder.embed_query(question)
            results = self.vector_store.query_strategy(
                strategy,
                query_embedding,
                top_k=self.config.evaluation.top_k
            )

            retrieval_time = time.time() - start
            retrieval_times.append(retrieval_time)

            # Calculate average chunk size
            chunk_sizes = [len(result['text']) for result in results]
            avg_chunk_sizes.extend(chunk_sizes)

        metrics = {
            'avg_retrieval_time_ms': np.mean(retrieval_times) * 1000,
            'avg_chunk_size': np.mean(avg_chunk_sizes),
            'median_chunk_size': np.median(avg_chunk_sizes),
        }

        return metrics


def compare_strategies(
    config: Config,
    strategies: List[str],
    output_path: str = None
) -> pd.DataFrame:
    """
    Convenience function to compare multiple chunking strategies.

    Args:
        config: Configuration object
        strategies: List of strategy names
        output_path: Optional path to save results CSV

    Returns:
        DataFrame with comparison results
    """
    evaluator = ChunkingEvaluator(config)
    results_df = evaluator.evaluate_all_strategies(strategies)

    if output_path:
        results_df.to_csv(output_path, index=False)
        print(f"\nResults saved to: {output_path}")

    return results_df
