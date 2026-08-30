import uuid

from sqlalchemy import select
from sqlalchemy.orm import Session

from knowledge_grove.models import Document, DocumentAccess, DocumentTag, Edge, RetrievalFeedback
from knowledge_grove.utils.embedding import get_embedding_model
from knowledge_grove.constants import EdgeType, PERMISSIONS, SHARED_READER
from knowledge_grove.utils.chunking import chunk_markdown


def add_document(
    session: Session,
    content: str,
    owner_agent: str,
    embedding: list[float] | None = None,
    summary: str | None = None,
    summary_embedding: list[float] | None = None,
    source_url: str | None = None,
    roles: dict[str, list[str]] | None = None,
) -> Document:
    """Insert a new chunk row.

    `embedding` (and `summary_embedding`, when `summary` is given) are computed
    automatically from the text if not supplied — pass one explicitly only if
    you have a reason to override the default model (a different model, or a
    precomputed batch embedding).
    """
    if embedding is None:
        embedding = get_embedding_model().embed_text(content)
    if summary is not None and summary_embedding is None:
        summary_embedding = get_embedding_model().embed_text(summary)
    if roles is None:
        roles = {SHARED_READER: [PERMISSIONS.READ]}

    document = Document(
        content=content,
        embedding=embedding,
        owner_agent=owner_agent,
        summary=summary,
        summary_embedding=summary_embedding,
        source_url=source_url,
    )
    session.add(document)
    session.flush()

    for role in roles or []:
        for permission in roles[role]:
            grant_access(session, document.id, role, permission)

    return document

def add_sequential_documents(
    session: Session,
    contents: list[str],
    owner_agent: str,
    embeddings: list[list[float]] | None = None,
    summaries: list[str] | None = None,
    summary_embeddings: list[list[float]] | None = None,
    source_urls: list[str] | None = None,
    descriptions: list[str] | None = None,
    roles: dict[str, list[str]] | None = None,
) -> list[Document]:
    """Insert a sequence of new chunk rows, linking each to the previous one with a 'follows' edge.

    `embeddings` (and `summary_embeddings`, when `summaries` is given) are computed
    automatically from the text if not supplied — pass them explicitly only if
    you have a reason to override the default model (a different model, or a
    precomputed batch embedding).
    """
    documents = []
    previous_document_id = None

    for i, content in enumerate(contents):
        embedding = get_embedding_model().embed_text(content) if embeddings is None else embeddings[i]
        summary = summaries[i] if summaries is not None else None
        summary_embedding = get_embedding_model().embed_text(summary) if summary is not None and summary_embeddings is None else (summary_embeddings[i] if summary_embeddings is not None else None)
        source_url = source_urls[i] if source_urls is not None else None
        description = descriptions[i] if descriptions is not None else None

        document = add_document(
            session,
            content=content,
            owner_agent=owner_agent,
            embedding=embedding,
            summary=summary,
            summary_embedding=summary_embedding,
            source_url=source_url,
            roles=roles
        )
        documents.append(document)

        # If this is not the first document, create a 'follows' edge from the previous document to this one.
        if previous_document_id is not None:
            add_edge(
                session,
                from_document_id=document.id,
                edge_type=EdgeType.PREV,
                description=description,
                to_document_id=previous_document_id,
            )

        previous_document_id = document.id

    return documents

def add_raw_document(
        session: Session, 
        document: str, 
        owner_agent: str, 
        source_url: str | None = None, 
        roles: dict[str, list[str]] | None = None
        ) -> list[Document]:
    """Insert a raw document string, chunking it into smaller pieces."""
    chunked_documents = chunk_markdown(document)
    return add_sequential_documents(
        session,
        contents=chunked_documents,
        owner_agent=owner_agent,
        source_urls=[source_url] * len(chunked_documents) if source_url is not None else None,
        roles=roles
    )
    

def add_raw_documents(
        session: Session, 
        documents: list[str], 
        owner_agent: str, 
        source_urls: list[str | None] | None = None, 
        roles: dict[str, list[str]] | None = None
        ) -> list[list[Document]]:
    """Insert a list of raw document strings, chunking each into smaller pieces."""
    documents_list = []
    for i, document in enumerate(documents):
        source_url = source_urls[i] if source_urls is not None else None
        document_chunks = add_raw_document(session, document, owner_agent, source_url, roles)
        documents_list.append(document_chunks)
    return documents_list

def get_by_id(session: Session, document_id: uuid.UUID) -> Document | None:
    """Direct fetch by primary key. Returns None if not found or not visible under RLS."""
    return session.get(Document, document_id)


def get_edges(session: Session, document_id: uuid.UUID) -> list[Edge]:
    """Outgoing edges from a document — what an agent could follow for more context."""
    stmt = select(Edge).where(Edge.from_document_id == document_id)
    return list(session.scalars(stmt).all())


def update_document(
    session: Session,
    document_id: uuid.UUID,
    content: str,
    embedding: list[float] | None = None,
    summary: str | None = None,
    summary_embedding: list[float] | None = None,
    source_url: str | None = None,
    description: str | None = None,
) -> Document:
    """Create a new revision of `document_id` rather than mutating it in place.

    The old row is kept as-is (so every edge that already pointed at it still
    describes exactly the content it was created against) and flagged
    `deprecated`; a `supersedes` edge links the new row back to it. This
    builds up organizational history for free — the old content, and the
    fact that it was superseded and by what, both stay on record.

    `embedding` is computed automatically from `content` if not supplied, same
    as `add_document`.
    """
    old_document = session.get(Document, document_id)
    if old_document is None:
        raise ValueError(f"no document with id {document_id}")

    new_document = add_document(
        session,
        content=content,
        embedding=embedding,
        owner_agent=old_document.owner_agent,
        summary=summary,
        summary_embedding=summary_embedding,
        source_url=source_url,
    )

    old_document.deprecated = True
    add_edge(
        session,
        from_document_id=new_document.id,
        edge_type=EdgeType.SUPERSEDES,
        description=description,
        to_document_id=old_document.id,
    )

    return new_document


def add_tag(
    session: Session, document_id: uuid.UUID, tag: str, description: str
) -> DocumentTag:
    """Attach an exact-match tag to a document. `tag` is normalized (trimmed/lowercased)."""
    document_tag = DocumentTag(
        document_id=document_id,
        tag=tag.strip().lower(),
        description=description,
    )
    session.add(document_tag)
    session.flush()
    return document_tag


def add_edge(
    session: Session,
    from_document_id: uuid.UUID,
    edge_type: str,
    description: str | None = None,
    to_document_id: uuid.UUID | None = None,
    external_url: str | None = None,
) -> Edge:
    """Attach a relationship from `from_document_id` to either another document or an external URL.

    Exactly one of `to_document_id` / `external_url` must be set. `description`
    is a short optional note on what this edge means / what's at the endpoint.
    """
    if (to_document_id is None) == (external_url is None):
        raise ValueError("exactly one of to_document_id or external_url must be set")

    edge = Edge(
        from_document_id=from_document_id,
        to_document_id=to_document_id,
        external_url=external_url,
        edge_type=edge_type,
        description=description,
    )
    session.add(edge)
    session.flush()
    return edge


def grant_access(
    session: Session, document_id: uuid.UUID, grantee_role: str, permission: str
) -> DocumentAccess:
    """Grant `read` or `write` access on a document to another role or group role."""
    grant = DocumentAccess(
        document_id=document_id,
        grantee_role=grantee_role,
        permission=permission,
    )
    session.add(grant)
    session.flush()
    return grant


def revoke_access(
    session: Session,
    document_id: uuid.UUID,
    grantee_role: str,
    permission: str | None = None,
) -> None:
    """Delete matching document_access grant(s).

    If `permission` is omitted, both `read` and `write` grants for this
    (document, grantee_role) pair are revoked.
    """
    query = session.query(DocumentAccess).filter_by(
        document_id=document_id, grantee_role=grantee_role
    )
    if permission is not None:
        query = query.filter_by(permission=permission)
    query.delete()


def log_feedback(
    session: Session,
    query_text: str,
    document_id: uuid.UUID,
    source_method: str,
    rank: int,
    judged_by: str,
    relevance: float | None = None,
) -> RetrievalFeedback:
    """Record how a document performed for a query, for later ranking-weight tuning (see §7)."""
    feedback = RetrievalFeedback(
        query_text=query_text,
        document_id=document_id,
        source_method=source_method,
        rank=rank,
        relevance=relevance,
        judged_by=judged_by,
    )
    session.add(feedback)
    session.flush()
    return feedback
