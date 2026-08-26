import uuid

from sqlalchemy import select
from sqlalchemy.orm import Session

from knowledge_grove.models import Document, DocumentAccess, DocumentTag, Edge, RetrievalFeedback


def add_document(
    session: Session,
    content: str,
    embedding: list[float],
    owner_agent: str,
    summary: str | None = None,
    summary_embedding: list[float] | None = None,
    source_url: str | None = None,
) -> Document:
    """Insert a new chunk row."""
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
    return document


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
    embedding: list[float],
    summary: str | None = None,
    summary_embedding: list[float] | None = None,
    source_url: str | None = None,
) -> Document:
    """Create a new revision of `document_id` rather than mutating it in place.

    The old row is kept as-is (so every edge that already pointed at it still
    describes exactly the content it was created against) and flagged
    `deprecated`; a `supersedes` edge links the new row back to it. This
    builds up organizational history for free — the old content, and the
    fact that it was superseded and by what, both stay on record.
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
        edge_type="supersedes",
        description="Supersedes a previous revision",
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
    description: str,
    to_document_id: uuid.UUID | None = None,
    external_url: str | None = None,
) -> Edge:
    """Attach a relationship from `from_document_id` to either another document or an external URL.

    Exactly one of `to_document_id` / `external_url` must be set.
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
