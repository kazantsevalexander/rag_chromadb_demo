"""
Пакет loader содержит модули для загрузки и обработки документов.
"""

from .txt_loader import load_txt, clean_text
from .html_loader import load_html, clean_html_text, extract_metadata_from_html
from .chunker import chunk_text, chunk_text_smart, create_chunks_with_metadata

__all__ = [
    'load_txt',
    'clean_text',
    'load_html',
    'clean_html_text',
    'extract_metadata_from_html',
    'chunk_text',
    'chunk_text_smart',
    'create_chunks_with_metadata'
]

