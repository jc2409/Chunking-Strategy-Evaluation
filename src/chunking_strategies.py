"""Chunking strategies for document processing."""

from typing import List, Dict, Tuple
from dataclasses import dataclass

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
)
import tiktoken

from config import ChunkingConfig, JinaConfig
from document_loader import Document, DocumentElement


@dataclass
class Chunk:
    """Represents a chunk of text."""
    text: str
    chunk_id: str
    metadata: Dict[str, any]
    strategy: str  # "fixed", "recursive", "semantic", "late"
    span_annotation: Tuple[int, int] = None  # For late chunking


class TraditionalChunker:
    """Implements traditional chunking strategies."""

    def __init__(self, config: ChunkingConfig):
        """Initialize with chunking configuration."""
        self.config = config

    def chunk_fixed_size(self, document: Document) -> List[Chunk]:
        """
        Fixed-size chunking with overlap.

        Args:
            document: Document to chunk

        Returns:
            List of chunks
        """
        # Combine all text elements
        full_text = self._combine_elements(document.elements)

        # Use CharacterTextSplitter for fixed-size chunks
        splitter = CharacterTextSplitter(
            chunk_size=self.config.fixed_chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            separator="\n\n",
            length_function=len
        )

        texts = splitter.split_text(full_text)

        chunks = []
        for idx, text in enumerate(texts):
            chunk = Chunk(
                text=text,
                chunk_id=f"{document.source}_fixed_{idx}",
                metadata={
                    **document.metadata,
                    'chunk_index': idx,
                    'chunk_method': 'fixed_size'
                },
                strategy="fixed"
            )
            chunks.append(chunk)

        return chunks

    def chunk_recursive(self, document: Document) -> List[Chunk]:
        """
        Recursive character text splitting - respects natural boundaries.

        Args:
            document: Document to chunk

        Returns:
            List of chunks
        """
        # Combine all text elements
        full_text = self._combine_elements(document.elements)

        # RecursiveCharacterTextSplitter tries to split at natural boundaries
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.fixed_chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len
        )

        texts = splitter.split_text(full_text)

        chunks = []
        for idx, text in enumerate(texts):
            chunk = Chunk(
                text=text,
                chunk_id=f"{document.source}_recursive_{idx}",
                metadata={
                    **document.metadata,
                    'chunk_index': idx,
                    'chunk_method': 'recursive'
                },
                strategy="recursive"
            )
            chunks.append(chunk)

        return chunks

    def chunk_semantic(self, document: Document) -> List[Chunk]:
        """
        Semantic chunking - groups by document structure (paragraphs, sections).

        Args:
            document: Document to chunk

        Returns:
            List of chunks
        """
        chunks = []
        current_chunk_text = []
        current_tokens = 0
        chunk_idx = 0

        encoding = tiktoken.get_encoding("cl100k_base")

        for element in document.elements:
            if not element.text.strip():
                continue

            element_tokens = len(encoding.encode(element.text))

            # If adding this element exceeds max size, save current chunk
            if current_tokens + element_tokens > self.config.fixed_chunk_size and current_chunk_text:
                chunk = Chunk(
                    text="\n\n".join(current_chunk_text),
                    chunk_id=f"{document.source}_semantic_{chunk_idx}",
                    metadata={
                        **document.metadata,
                        'chunk_index': chunk_idx,
                        'chunk_method': 'semantic',
                        'num_elements': len(current_chunk_text)
                    },
                    strategy="semantic"
                )
                chunks.append(chunk)

                # Start new chunk with overlap
                if self.config.chunk_overlap > 0 and len(current_chunk_text) > 0:
                    current_chunk_text = [current_chunk_text[-1]]
                    current_tokens = len(encoding.encode(current_chunk_text[0]))
                else:
                    current_chunk_text = []
                    current_tokens = 0

                chunk_idx += 1

            current_chunk_text.append(element.text)
            current_tokens += element_tokens

        # Add final chunk
        if current_chunk_text:
            chunk = Chunk(
                text="\n\n".join(current_chunk_text),
                chunk_id=f"{document.source}_semantic_{chunk_idx}",
                metadata={
                    **document.metadata,
                    'chunk_index': chunk_idx,
                    'chunk_method': 'semantic',
                    'num_elements': len(current_chunk_text)
                },
                strategy="semantic"
            )
            chunks.append(chunk)

        return chunks

    def _combine_elements(self, elements: List[DocumentElement]) -> str:
        """Combine document elements into a single text."""
        text_parts = []
        for element in elements:
            if element.text.strip():
                # Add element type as context for tables and images
                if element.element_type == "Table":
                    text_parts.append(f"[TABLE]\n{element.text}\n[/TABLE]")
                elif element.element_type in ["Image", "Figure"]:
                    text_parts.append(f"[IMAGE: {element.text}]")
                else:
                    text_parts.append(element.text)

        return "\n\n".join(text_parts)


class LateChunker:
    """Implements late chunking using local tokenizer."""

    def __init__(self, jina_config: JinaConfig, chunking_config: ChunkingConfig):
        """Initialize with configuration and load tokenizer."""
        self.jina_config = jina_config
        self.chunking_config = chunking_config

        # Load tokenizer for chunking
        from transformers import AutoTokenizer
        print(f"Loading tokenizer: {self.jina_config.local_model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.jina_config.local_model_name,
            trust_remote_code=True
        )

    def chunk_with_late_chunking(self, document: Document) -> Tuple[List[Chunk], List[str], List[Tuple[int, int]]]:
        """
        Late chunking using local tokenizer or Jina API.

        This method:
        1. Chunks text by sentences using the tokenizer or Jina API
        2. Returns chunks with token-level span annotations for late pooling

        Args:
            document: Document to chunk

        Returns:
            Tuple of (chunks, chunk_texts, span_annotations)
        """
        # Combine all text elements
        full_text = self._combine_elements(document.elements)

        # Get chunks and span annotations using configured method
        if self.chunking_config.use_jina_api_chunking:
            chunk_texts, span_annotations = self._chunk_by_tokenizer_api(full_text)
        else:
            chunk_texts, span_annotations = self._chunk_by_sentences(full_text)

        # Create Chunk objects
        chunks = []
        for idx, (text, span) in enumerate(zip(chunk_texts, span_annotations)):
            chunk = Chunk(
                text=text,
                chunk_id=f"{document.source}_late_{idx}",
                metadata={
                    **document.metadata,
                    'chunk_index': idx,
                    'chunk_method': 'late_chunking',
                    'span_start': span[0],
                    'span_end': span[1]
                },
                strategy="late",
                span_annotation=span
            )
            chunks.append(chunk)

        return chunks, chunk_texts, span_annotations

    def _chunk_by_sentences(self, input_text: str) -> Tuple[List[str], List[Tuple[int, int]]]:
        """
        Split input text into sentences using the tokenizer.

        This is the same approach as in the late chunking notebook.

        Args:
            input_text: The text snippet to split into sentences

        Returns:
            A tuple containing the list of text chunks and their corresponding token spans
        """
        inputs = self.tokenizer(input_text, return_tensors='pt', return_offsets_mapping=True)
        punctuation_mark_id = self.tokenizer.convert_tokens_to_ids('.')
        sep_id = self.tokenizer.convert_tokens_to_ids('[SEP]')
        token_offsets = inputs['offset_mapping'][0]
        token_ids = inputs['input_ids'][0]

        chunk_positions = [
            (i, int(start + 1))
            for i, (token_id, (start, end)) in enumerate(zip(token_ids, token_offsets))
            if token_id == punctuation_mark_id
            and (
                token_offsets[i + 1][0] - token_offsets[i][1] > 0
                or token_ids[i + 1] == sep_id
            )
        ]

        chunks = [
            input_text[x[1] : y[1]]
            for x, y in zip([(1, 0)] + chunk_positions[:-1], chunk_positions)
        ]

        span_annotations = [
            (x[0], y[0]) for (x, y) in zip([(1, 0)] + chunk_positions[:-1], chunk_positions)
        ]

        return chunks, span_annotations

    def _chunk_by_tokenizer_api(self, input_text: str) -> Tuple[List[str], List[Tuple[int, int]]]:
        """
        Split text using Jina's official tokenization API.
        This is the recommended approach for late chunking with Jina embeddings.

        The API returns character positions, which we convert to token positions
        for compatibility with the late chunking pooling method.

        Args:
            input_text: Text to split

        Returns:
            Tuple of (chunks, span_annotations in token positions)
        """
        import requests

        # Define the API endpoint and payload
        url = 'https://tokenize.jina.ai/'
        payload = {
            "content": input_text,
            "return_chunks": "true",
            "max_chunk_length": str(self.chunking_config.max_chunk_length)
        }

        try:
            # Make the API request
            response = requests.post(url, json=payload, timeout=30)
            response.raise_for_status()
            response_data = response.json()

            # Extract chunks and character positions from the response
            chunks = response_data.get("chunks", [])
            char_positions = response_data.get("chunk_positions", [])

            if not chunks or not char_positions:
                raise ValueError("API returned empty chunks or positions")

            # Convert character positions to token positions
            # We need to tokenize the full text to get the mapping
            inputs = self.tokenizer(input_text, return_tensors='pt', return_offsets_mapping=True)
            token_offsets = inputs['offset_mapping'][0]  # Maps tokens to character positions

            span_annotations = []
            for char_start, char_end in char_positions:
                # Find token indices that correspond to these character positions
                token_start = None
                token_end = None

                for token_idx, (tok_char_start, tok_char_end) in enumerate(token_offsets):
                    # Find the first token that starts at or after char_start
                    if token_start is None and tok_char_start >= char_start:
                        token_start = token_idx

                    # Find the last token that ends at or before char_end
                    if tok_char_end <= char_end:
                        token_end = token_idx + 1

                    if token_start is not None and token_end is not None:
                        break

                # Fallback to reasonable defaults if not found
                if token_start is None:
                    token_start = 0
                if token_end is None:
                    token_end = len(token_offsets)

                span_annotations.append((token_start, token_end))

            print(f"API chunking: {len(chunks)} chunks from Jina API")
            return chunks, span_annotations

        except Exception as e:
            print(f"API chunking failed ({e}), falling back to local sentence splitting")
            return self._chunk_by_sentences(input_text)

    def _combine_elements(self, elements: List[DocumentElement]) -> str:
        """Combine document elements into a single text."""
        text_parts = []
        for element in elements:
            if element.text.strip():
                # Add element type as context for tables and images
                if element.element_type == "Table":
                    text_parts.append(f"[TABLE]\n{element.text}\n[/TABLE]")
                elif element.element_type in ["Image", "Figure"]:
                    text_parts.append(f"[IMAGE: {element.text}]")
                else:
                    text_parts.append(element.text)

        return "\n\n".join(text_parts)
