from sqlalchemy.orm import Session

from knowledge_grove.crud import add_raw_document
from knowledge_grove.models import Document

def file_to_string(file_path: str) -> str:
    """Read the contents of a file and return it as a string."""
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

def add_file_as_document(session: Session, file_path: str, owner_agent: str, source_url: str | None = None) -> list[Document]:
    """Read a file and add its contents as a document, chunking it into smaller pieces."""
    document_content = file_to_string(file_path)
    return add_raw_document(session, document_content, owner_agent, source_url)

def add_files_as_documents(session: Session, file_paths: list[str], owner_agent: str, source_urls: list[str | None] | None = None) -> list[list[Document]]:
    """Read multiple files and add their contents as documents, chunking each into smaller pieces."""
    documents_list = []
    for i, file_path in enumerate(file_paths):
        source_url = source_urls[i] if source_urls is not None else None
        document_chunks = add_file_as_document(session, file_path, owner_agent, source_url)
        documents_list.append(document_chunks)
    return documents_list