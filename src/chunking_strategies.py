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
        self.jina_config = jina_config
        self.chunking_config = chunking_config

        from transformers import AutoTokenizer
        print(f"Loading tokenizer: {self.jina_config.local_model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.jina_config.local_model_name,
            trust_remote_code=True
        )

    def chunk_with_late_chunking(
        self, document: Document
    ) -> Tuple[List[Chunk], List[str], List[Tuple[int, int]]]:
        """Late chunking using tokenizer spans."""
        full_text = self._combine_elements(document.elements)

        try:
            chunk_texts, span_annotations = self._chunk_by_sentences(full_text)
        except Exception as e:
            print(f"Tokenizer chunking failed ({e})")

        chunks = []
        for idx, (text, span) in enumerate(zip(chunk_texts, span_annotations)):
            chunks.append(
                Chunk(
                    text=text,
                    chunk_id=f"{document.source}_late_{idx}",
                    metadata={
                        **document.metadata,
                        "chunk_index": idx,
                        "chunk_method": "late_chunking",
                        "span_start": span[0],
                        "span_end": span[1],
                    },
                    strategy="late",
                    span_annotation=span,
                )
            )

        return chunks, chunk_texts, span_annotations

    def _chunk_by_sentences(self, input_text: str) -> Tuple[List[str], List[Tuple[int, int]]]:
        """Split text into sentences using tokenizer offsets."""
        inputs = self.tokenizer(input_text, return_offsets_mapping=True, add_special_tokens=False)
        token_offsets = inputs["offset_mapping"]

        sentence_boundaries = []
        for i, (start, end) in enumerate(token_offsets):
            if input_text[end - 1:end] in {".", "!", "?"}:
                sentence_boundaries.append(i)

        if not sentence_boundaries:
            # fallback to entire text if no punctuation found
            return [input_text], [(0, len(input_text))]

        chunks = []
        spans = []
        prev_end_char = 0
        for boundary_idx in sentence_boundaries:
            end_char = token_offsets[boundary_idx][1]
            chunk_text = input_text[prev_end_char:end_char].strip()
            if chunk_text:
                chunks.append(chunk_text)
                spans.append((prev_end_char, end_char))
            prev_end_char = end_char + 1

        return chunks, spans

    def _combine_elements(self, elements: List[DocumentElement]) -> str:
        """Combine elements into text with type markers."""
        text_parts = []
        for element in elements:
            if not element.text.strip():
                continue
            if element.element_type == "Table":
                text_parts.append(f"[TABLE]\n{element.text}\n[/TABLE]")
            elif element.element_type in ["Image", "Figure"]:
                text_parts.append(f"[IMAGE: {element.text}]")
            else:
                text_parts.append(element.text)
        return "\n\n".join(text_parts)
