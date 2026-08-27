import os
import uuid

from mcp.server.mcpserver import MCPServer

from knowledge_grove import crud
from knowledge_grove.db import get_engine, get_session
from knowledge_grove.models import Document, DocumentAccess, DocumentTag, Edge
from knowledge_grove.search import gather_context

mcp_server = MCPServer(name="knowledge-grove")
engine = get_engine(os.environ["KNOWLEDGE_GROVE_DSN"])


def _document_to_dict(doc: Document) -> dict:
    return {
        "id": str(doc.id),
        "content": doc.content,
        "summary": doc.summary,
        "source_url": doc.source_url,
        "owner_agent": doc.owner_agent,
        "deprecated": doc.deprecated,
    }


def _edge_to_dict(edge: Edge) -> dict:
    return {
        "id": str(edge.id),
        "from_document_id": str(edge.from_document_id),
        "to_document_id": str(edge.to_document_id) if edge.to_document_id else None,
        "external_url": edge.external_url,
        "edge_type": edge.edge_type,
        "description": edge.description,
    }


def _tag_to_dict(tag: DocumentTag) -> dict:
    return {
        "id": str(tag.id),
        "document_id": str(tag.document_id),
        "tag": tag.tag,
        "description": tag.description,
    }


def _grant_to_dict(grant: DocumentAccess) -> dict:
    return {
        "id": str(grant.id),
        "document_id": str(grant.document_id),
        "grantee_role": grant.grantee_role,
        "permission": grant.permission,
    }


def gather_context_tool(
    query_text: str,
    pattern: str,
    tags: list[str] | None = None,
    limit: int = 10,
    include_deprecated: bool = False,
) -> list[dict]:
    """Combined search that waits for all three methods to complete and merges results."""
    with get_session(engine) as session:
        results = gather_context(
            session, query_text=query_text, pattern=pattern,
            tags=tags, limit=limit, include_deprecated=include_deprecated,
        )
        return [{"id": str(doc.id), "content": doc.content} for doc, _score in results]


def add_document_tool(
    content: str,
    owner_agent: str,
    summary: str | None = None,
    source_url: str | None = None,
) -> dict:
    """Add a new document. The embedding is computed automatically from `content`
    (and from `summary`, if given) — there's no vector to supply."""
    with get_session(engine) as session:
        doc = crud.add_document(
            session, content=content, owner_agent=owner_agent, summary=summary, source_url=source_url,
        )
        session.commit()
        return _document_to_dict(doc)

def add_sequential_documents_tool(
    contents: list[str],
    owner_agent: str,
    summaries: list[str] | None = None,
    source_urls: list[str] | None = None,
    descriptions: list[str] | None = None,
) -> list[dict]:
    """Add a sequence of new documents, linking each to the previous one with a 'follows' edge.
    The embeddings are computed automatically from `contents` (and from `summaries`, if given)
    — there's no vector to supply."""
    with get_session(engine) as session:
        docs = crud.add_sequential_documents(
            session, contents=contents, owner_agent=owner_agent,
            summaries=summaries, source_urls=source_urls, descriptions=descriptions,
        )
        session.commit()
        return [_document_to_dict(doc) for doc in docs]


def get_by_id_tool(document_id: str) -> dict | None:
    """Fetch a document by id."""
    with get_session(engine) as session:
        doc = crud.get_by_id(session, uuid.UUID(document_id))
        return _document_to_dict(doc) if doc else None


def get_edges_tool(document_id: str) -> list[dict]:
    """Outgoing edges from a document — what you could follow for more context."""
    with get_session(engine) as session:
        edges = crud.get_edges(session, uuid.UUID(document_id))
        return [_edge_to_dict(edge) for edge in edges]


def update_document_tool(
    document_id: str,
    content: str,
    summary: str | None = None,
    source_url: str | None = None,
) -> dict:
    """Create a new revision of a document (the old one is kept, flagged deprecated,
    and linked via a `supersedes` edge). The embedding is computed automatically
    from `content` — there's no vector to supply."""
    with get_session(engine) as session:
        doc = crud.update_document(
            session, uuid.UUID(document_id), content=content, summary=summary, source_url=source_url,
        )
        session.commit()
        return _document_to_dict(doc)


def add_tag_tool(document_id: str, tag: str, description: str) -> dict:
    """Attach an exact-match tag to a document."""
    with get_session(engine) as session:
        doc_tag = crud.add_tag(session, uuid.UUID(document_id), tag, description)
        session.commit()
        return _tag_to_dict(doc_tag)


def add_edge_tool(
    from_document_id: str,
    edge_type: str,
    description: str | None = None,
    to_document_id: str | None = None,
    external_url: str | None = None,
) -> dict:
    """Attach a relationship from a document to either another document or an
    external URL. Exactly one of `to_document_id` / `external_url` must be given."""
    with get_session(engine) as session:
        edge = crud.add_edge(
            session,
            from_document_id=uuid.UUID(from_document_id),
            edge_type=edge_type,
            description=description,
            to_document_id=uuid.UUID(to_document_id) if to_document_id else None,
            external_url=external_url,
        )
        session.commit()
        return _edge_to_dict(edge)


def grant_access_tool(document_id: str, grantee_role: str, permission: str) -> dict:
    """Grant `read` or `write` access on a document to another role."""
    with get_session(engine) as session:
        grant = crud.grant_access(session, uuid.UUID(document_id), grantee_role, permission)
        session.commit()
        return _grant_to_dict(grant)


def revoke_access_tool(document_id: str, grantee_role: str, permission: str | None = None) -> dict:
    """Revoke access grant(s) on a document. Omit `permission` to revoke both read and write."""
    with get_session(engine) as session:
        crud.revoke_access(session, uuid.UUID(document_id), grantee_role, permission)
        session.commit()
        return {"revoked": True}


def log_feedback_tool(
    query_text: str,
    document_id: str,
    source_method: str,
    rank: int,
    judged_by: str,
    relevance: float | None = None,
) -> dict:
    """Record how a document performed for a query, for later ranking-weight tuning."""
    with get_session(engine) as session:
        feedback = crud.log_feedback(
            session, query_text, uuid.UUID(document_id), source_method, rank, judged_by, relevance,
        )
        session.commit()
        return {"id": str(feedback.id), "document_id": str(feedback.document_id), "rank": feedback.rank}


for _fn, _name, _description in [
    (gather_context_tool, "gather_context", "Combined search that waits for all three methods to complete and merges results."),
    (add_document_tool, "add_document", "Add a new document; the embedding is computed automatically."),
    (add_sequential_documents_tool, "add_sequential_documents", "Add a sequence of documents, each linked to the previous one with a 'prev' edge; embeddings are computed automatically."),
    (get_by_id_tool, "get_by_id", "Fetch a document by id."),
    (get_edges_tool, "get_edges", "Outgoing edges from a document."),
    (update_document_tool, "update_document", "Create a new revision of a document; the embedding is computed automatically."),
    (add_tag_tool, "add_tag", "Attach an exact-match tag to a document."),
    (add_edge_tool, "add_edge", "Attach a relationship from a document to another document or an external URL."),
    (grant_access_tool, "grant_access", "Grant read or write access on a document to another role."),
    (revoke_access_tool, "revoke_access", "Revoke access grant(s) on a document."),
    (log_feedback_tool, "log_feedback", "Record how a document performed for a query."),
]:
    mcp_server.add_tool(_fn, name=_name, description=_description)


if __name__ == "__main__":
    mcp_server.run()
