"""Document loader using Unstructured API for multiple file formats."""

import os
import requests
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass

from config import UnstructuredAPIConfig


@dataclass
class DocumentElement:
    """Represents a single element from a document."""
    text: str
    element_type: str  # "NarrativeText", "Table", "Image", "Title", etc.
    metadata: Dict[str, any]


@dataclass
class Document:
    """Represents a loaded document with its elements."""
    elements: List[DocumentElement]
    metadata: Dict[str, any]
    source: str


class DocumentLoader:
    """Load documents from various file formats using Unstructured API."""

    SUPPORTED_EXTENSIONS = {
        '.txt', '.md', '.py', '.js', '.java', '.cpp', '.c', '.h',
        '.sh', '.yaml', '.yml', '.json', '.xml', '.html', '.css',
        '.pdf', '.docx', '.pptx', '.doc', '.ppt', '.xlsx', '.xls',
        '.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.heic',
        '.eml', '.msg', '.rtf', '.odt', '.epub'
    }

    def __init__(self, config: UnstructuredAPIConfig):
        """Initialize the document loader with API configuration."""
        self.config = config
        self.use_api = bool(self.config.api_key)

    def load(self, file_path: str) -> Document:
        """
        Load a document from a file path using Unstructured API.

        Args:
            file_path: Path to the file to load

        Returns:
            Document object with structured elements
        """
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        if not path.is_file():
            raise ValueError(f"Path is not a file: {file_path}")

        extension = path.suffix.lower()

        if extension not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported file format: {extension}")

        # Unstructured API
        try:
            elements = self._call_unstructured_api(path)
        except Exception as e:
            raise e

        metadata = {
            'source': str(path),
            'file_name': path.name,
            'file_type': extension,
            'file_size': path.stat().st_size,
            'num_elements': len(elements)
        }

        return Document(elements=elements, metadata=metadata, source=str(path))

    def load_directory(self, directory_path: str, recursive: bool = True) -> List[Document]:
        """
        Load all supported documents from a directory.

        Args:
            directory_path: Path to the directory
            recursive: Whether to search subdirectories

        Returns:
            List of Document objects
        """
        path = Path(directory_path)

        if not path.exists():
            raise FileNotFoundError(f"Directory not found: {directory_path}")

        if not path.is_dir():
            raise ValueError(f"Path is not a directory: {directory_path}")

        documents = []
        pattern = '**/*' if recursive else '*'

        for file_path in path.glob(pattern):
            if file_path.is_file() and file_path.suffix.lower() in self.SUPPORTED_EXTENSIONS:
                try:
                    doc = self.load(str(file_path))
                    if doc:
                        documents.append(doc)
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")

        return documents

    def _load_simple_text(self, path: Path) -> List[DocumentElement]:
        """
        Load simple text files without API.

        Args:
            path: Path to the text file

        Returns:
            List of DocumentElement objects
        """
        encodings = ['utf-8', 'latin-1', 'cp1252']

        text = None
        for encoding in encodings:
            try:
                with open(path, 'r', encoding=encoding) as f:
                    text = f.read()
                break
            except UnicodeDecodeError:
                continue

        if text is None:
            raise ValueError(f"Could not decode file with supported encodings: {path}")

        # Create a single DocumentElement with the full text
        element = DocumentElement(
            text=text,
            element_type="NarrativeText",
            metadata={
                'filename': path.name,
                'filetype': path.suffix
            }
        )

        return [element]

    def _call_unstructured_api(self, file_path: Path) -> List[DocumentElement]:
        """
        Call Unstructured API to process a document.

        Args:
            file_path: Path to the file

        Returns:
            List of DocumentElement objects
        """
        headers = {
            "unstructured-api-key": self.config.api_key,
        }

        # Prepare the file for upload
        with open(file_path, 'rb') as f:
            files = {
                'files': (file_path.name, f, self._get_mime_type(file_path))
            }

            # API parameters
            data = {
                'strategy': self.config.strategy,
                'extract_image_block_types': ','.join(self.config.extract_image_block_types),
                'coordinates': 'true',
                'output_format': 'application/json'
            }

            # Make API request
            response = requests.post(
                self.config.api_url,
                headers=headers,
                files=files,
                data=data
            )

            if response.status_code != 200:
                raise Exception(
                    f"Unstructured API error: {response.status_code} - {response.text}"
                )

            # Parse response
            elements_data = response.json()

        # Convert to DocumentElement objects
        elements = []
        for elem_data in elements_data:
            element = DocumentElement(
                text=elem_data.get('text', ''),
                element_type=elem_data.get('type', 'Unknown'),
                metadata={
                    'element_id': elem_data.get('element_id'),
                    'coordinates': elem_data.get('metadata', {}).get('coordinates'),
                    'page_number': elem_data.get('metadata', {}).get('page_number'),
                    'filename': elem_data.get('metadata', {}).get('filename'),
                    'filetype': elem_data.get('metadata', {}).get('filetype'),
                    **elem_data.get('metadata', {})
                }
            )
            elements.append(element)

        return elements

    def _get_mime_type(self, file_path: Path) -> str:
        """Get MIME type for a file."""
        mime_types = {
            '.pdf': 'application/pdf',
            '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            '.doc': 'application/msword',
            '.pptx': 'application/vnd.openxmlformats-officedocument.presentationml.presentation',
            '.ppt': 'application/vnd.ms-powerpoint',
            '.txt': 'text/plain',
            '.md': 'text/markdown',
            '.html': 'text/html',
            '.json': 'application/json',
            '.xml': 'application/xml',
            '.png': 'image/png',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
        }
        return mime_types.get(file_path.suffix.lower(), 'application/octet-stream')


def load_documents(source: str, config: UnstructuredAPIConfig, recursive: bool = True) -> List[Document]:
    """
    Convenience function to load documents from a file or directory.

    Args:
        source: Path to file or directory
        config: Unstructured API configuration
        recursive: Whether to search subdirectories (for directories)

    Returns:
        List of Document objects
    """
    loader = DocumentLoader(config)
    path = Path(source)

    if path.is_file():
        doc = loader.load(source)
        return [doc] if doc else []
    elif path.is_dir():
        return loader.load_directory(source, recursive=recursive)
    else:
        raise ValueError(f"Invalid source path: {source}")
