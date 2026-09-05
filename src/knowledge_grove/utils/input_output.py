from pathlib import Path

from sqlalchemy.orm import Session

from knowledge_grove.constants import ContentType
from knowledge_grove.crud import add_raw_document
from knowledge_grove.models import Document

_EXTENSION_CONTENT_TYPES = {
    ".py": ContentType.PYTHON,
    ".sql": ContentType.SQL,
}


def file_to_string(file_path: str) -> str:
    """Read the contents of a file and return it as a string."""
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def detect_content_type(file_path: str) -> str:
    """Guess a file's content_type (for chunking, see add_raw_document) from
    its extension. Defaults to markdown for anything not specifically
    recognized -- also the right choice for plain prose/text files.
    """
    return _EXTENSION_CONTENT_TYPES.get(Path(file_path).suffix.lower(), ContentType.MARKDOWN)


def add_file_as_document(
    session: Session,
    file_path: str,
    owner_agent: str,
    source_url: str | None = None,
    content_type: str | None = None,
) -> list[Document]:
    """Read a file and add its contents as a document, chunking it into smaller pieces.

    `content_type` picks the chunker (see add_raw_document); if omitted, it's
    guessed from the file's extension via detect_content_type.
    """
    document_content = file_to_string(file_path)
    if content_type is None:
        content_type = detect_content_type(file_path)
    return add_raw_document(session, document_content, owner_agent, source_url, content_type=content_type)


def add_files_as_documents(
    session: Session,
    file_paths: list[str],
    owner_agent: str,
    source_urls: list[str | None] | None = None,
    content_types: list[str | None] | None = None,
) -> list[list[Document]]:
    """Read multiple files and add their contents as documents, chunking each into smaller pieces.

    `content_types` applies per file, matched by position to `file_paths`;
    a `None` entry (or omitting the list entirely) falls back to per-file
    extension detection, same as add_file_as_document.
    """
    documents_list = []
    for i, file_path in enumerate(file_paths):
        source_url = source_urls[i] if source_urls is not None else None
        content_type = content_types[i] if content_types is not None else None
        document_chunks = add_file_as_document(session, file_path, owner_agent, source_url, content_type)
        documents_list.append(document_chunks)
    return documents_list
