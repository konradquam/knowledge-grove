import uuid

from sqlalchemy import select
from sqlalchemy.orm import Session

from knowledge_grove.models import Document, DocumentAccess, DocumentTag, Edge, RetrievalFeedback
from knowledge_grove.utils.embedding import get_embedding_model
from knowledge_grove.constants import ContentType, EdgeType, PERMISSIONS, SHARED_READER
from knowledge_grove.utils.chunking import chunk_markdown, chunk_python, chunk_sql
from knowledge_grove.utils.hashing import hash_content

_CHUNKERS = {
    ContentType.MARKDOWN: chunk_markdown,
    ContentType.PYTHON: chunk_python,
    ContentType.SQL: chunk_sql,
}


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

    If a non-deprecated document with identical `content` already exists
    (anywhere, not just for this `owner_agent`), that document is returned
    unchanged instead of inserting a duplicate — no new embedding is computed.
    """
    content_hash = hash_content(content)
    existing = session.scalars(
        select(Document).where(
            Document.content_hash == content_hash, Document.deprecated.is_(False)
        )
    ).first()
    if existing is not None:
        return existing

    if embedding is None:
        embedding = get_embedding_model().embed_text(content)
    if summary is not None and summary_embedding is None:
        summary_embedding = get_embedding_model().embed_text(summary)
    if roles is None:
        roles = {SHARED_READER: [PERMISSIONS.READ]}

    document = Document(
        content=content,
        content_hash=content_hash,
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

def add_authored_chunks(
    session: Session,
    chunks: list[str],
    owner_agent: str,
    edges: list[dict] | None = None,
    embeddings: list[list[float]] | None = None,
    summaries: list[str] | None = None,
    summary_embeddings: list[list[float]] | None = None,
    source_urls: list[str] | None = None,
    descriptions: list[str] | None = None,
    roles: dict[str, list[str]] | None = None,
) -> list[Document]:
    """Insert a batch of already-segmented, agent-authored chunks in one call
    (§9/§11 of the design doc) -- for content an agent is originating itself
    (a decision writeup, a new note) rather than parsing an existing source,
    where the agent can compose content already correctly segmented instead
    of handing over one blob for a chunker to guess boundaries in.

    `chunks` behaves exactly like add_sequential_documents: ordered content,
    auto-linked with `prev` edges.

    `edges` covers what add_sequential_documents can't -- relationships the
    new chunks have beyond that sequence, to each other or to documents
    already in the graph. Each entry is a dict:

        {"from_index": int, "edge_type": str, "description": str | None,
         "to_document_id": uuid.UUID | None,   # an existing document
         "to_index": int | None,               # another chunk in this call
         "external_url": str | None}

    `from_index`/`to_index` are positions into `chunks` -- those chunks have
    no document id yet at call time, so edges reference them by position
    instead, resolved to real ids once every chunk is inserted. Exactly one
    of `to_document_id` / `to_index` / `external_url` must be set per edge.
    """
    documents = add_sequential_documents(
        session,
        contents=chunks,
        owner_agent=owner_agent,
        embeddings=embeddings,
        summaries=summaries,
        summary_embeddings=summary_embeddings,
        source_urls=source_urls,
        descriptions=descriptions,
        roles=roles,
    )

    for edge in edges or []:
        targets_given = sum(
            edge.get(key) is not None for key in ("to_document_id", "to_index", "external_url")
        )
        if targets_given != 1:
            raise ValueError(
                "exactly one of to_document_id, to_index, or external_url must be set per edge"
            )

        to_document_id = edge.get("to_document_id")
        if edge.get("to_index") is not None:
            to_document_id = documents[edge["to_index"]].id

        add_edge(
            session,
            from_document_id=documents[edge["from_index"]].id,
            edge_type=edge["edge_type"],
            description=edge.get("description"),
            to_document_id=to_document_id,
            external_url=edge.get("external_url"),
        )

    return documents

def add_raw_document(
        session: Session,
        document: str,
        owner_agent: str,
        source_url: str | None = None,
        roles: dict[str, list[str]] | None = None,
        content_type: str = ContentType.MARKDOWN,
        ) -> list[Document]:
    """Insert a raw document string, chunking it into smaller pieces and
    linking them with `prev` edges (add_sequential_documents) -- this holds
    regardless of `content_type`: Python/SQL chunks from the same string are
    sequentially linked exactly like markdown chunks are.

    `content_type` picks the chunker (§11 of the design doc: the right
    boundary logic differs by content type) -- "markdown" (default),
    "python", or "sql".

    If `source_url` is given and matches a previous ingestion, this reconciles
    against it rather than blindly inserting again: an unchanged document
    (identical set of chunk contents) is a no-op that returns the existing
    chunks untouched; a changed one deprecates every one of that source's
    previous chunks and inserts the new chunk sequence fresh. This is a full
    replace rather than a chunk-by-chunk patch — patching would leave the
    `prev` edge chain half old, half new, with no clean way to splice the two,
    so an actual change just supersedes the whole previous sequence at once.
    """
    if content_type not in _CHUNKERS:
        raise ValueError(f"unknown content_type {content_type!r}, expected one of {list(_CHUNKERS)}")
    chunked_documents = _CHUNKERS[content_type](document)

    if source_url is not None:
        existing_documents = list(
            session.scalars(
                select(Document)
                .where(Document.source_url == source_url, Document.deprecated.is_(False))
                .order_by(Document.created_at)
            )
        )
        if existing_documents:
            new_hashes = {hash_content(chunk) for chunk in chunked_documents}
            existing_hashes = {doc.content_hash for doc in existing_documents}
            if new_hashes == existing_hashes:
                return existing_documents

            for old_document in existing_documents:
                old_document.deprecated = True

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
        roles: dict[str, list[str]] | None = None,
        content_type: str = ContentType.MARKDOWN,
        ) -> list[list[Document]]:
    """Insert a list of raw document strings, chunking each into smaller pieces.

    `content_type` applies to every document in the batch -- see add_raw_document.
    """
    documents_list = []
    for i, document in enumerate(documents):
        source_url = source_urls[i] if source_urls is not None else None
        document_chunks = add_raw_document(session, document, owner_agent, source_url, roles, content_type)
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
    as `add_document`. If `content` is unchanged from the current revision,
    `add_document`'s own exact-match check returns that same row back, so
    there is nothing to supersede — this is a no-op.
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
    if new_document.id == old_document.id:
        return new_document

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
