import os
import uuid

from mcp.server.mcpserver import MCPServer
from sqlalchemy import text

from knowledge_grove import crud
from knowledge_grove.db import get_engine, get_session
from knowledge_grove.models import Document, DocumentAccess, DocumentTag, Edge
from knowledge_grove.search import gather_context

mcp_server = MCPServer(name="knowledge-grove")
engine = get_engine(os.environ["KNOWLEDGE_GROVE_DSN"])


def _current_agent(session) -> str:
    """The Postgres role this server is actually connected as -- the only
    trustworthy source for who owns a document. Never take ownership from a
    caller-supplied argument: this connection is meant to represent exactly
    one agent identity (see the design doc's §4 "own role and credentials"
    model), so who's asking is a property of the connection, not something
    the caller gets to declare.
    """
    return session.execute(text("SELECT current_user")).scalar()


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
    summary: str | None = None,
    source_url: str | None = None,
    roles: dict[str, list[str]] | None = None,
) -> dict:
    """Add a new document, owned by whichever agent this server is connected
    as (there's no owner to specify -- it's a property of the connection,
    not something a caller declares). The embedding is computed automatically
    from `content` (and from `summary`, if given) — there's no vector to supply.

    `roles` controls who besides the owner can access this document: a
    mapping from grantee role name to a list of permissions ("read" and/or
    "write"), e.g. {"shared_reader": ["read"], "agent_bob": ["read", "write"]}.
    If omitted entirely, the document defaults to {"shared_reader": ["read"]}
    — readable by every agent in the shared_reader group. Pass an empty
    object ({}) to keep the document private to the owner only."""
    with get_session(engine) as session:
        doc = crud.add_document(
            session, content=content, owner_agent=_current_agent(session), summary=summary,
            source_url=source_url, roles=roles,
        )
        session.commit()
        return _document_to_dict(doc)

def add_sequential_documents_tool(
    contents: list[str],
    summaries: list[str] | None = None,
    source_urls: list[str] | None = None,
    descriptions: list[str] | None = None,
    roles: dict[str, list[str]] | None = None,
) -> list[dict]:
    """Add a sequence of new documents, linking each to the previous one with a 'follows' edge,
    owned by whichever agent this server is connected as (there's no owner to specify -- it's
    a property of the connection, not something a caller declares). The embeddings are computed
    automatically from `contents` (and from `summaries`, if given) — there's no vector to supply.

    `roles` controls who besides the owner can access every document in
    the sequence: a mapping from grantee role name to a list of permissions
    ("read" and/or "write"), e.g. {"shared_reader": ["read"], "agent_bob": ["read", "write"]}.
    The same `roles` is applied to each document in the sequence. If omitted
    entirely, every document defaults to {"shared_reader": ["read"]} —
    readable by every agent in the shared_reader group. Pass an empty object
    ({}) to keep every document in the sequence private to the owner only."""
    with get_session(engine) as session:
        docs = crud.add_sequential_documents(
            session, contents=contents, owner_agent=_current_agent(session),
            summaries=summaries, source_urls=source_urls, descriptions=descriptions,
            roles=roles,
        )
        session.commit()
        return [_document_to_dict(doc) for doc in docs]


def _edges_with_uuids(edges: list[dict] | None) -> list[dict] | None:
    """MCP callers pass to_document_id as a plain string (JSON has no UUID
    type) -- convert it the same way add_edge_tool does before it reaches crud.
    """
    if edges is None:
        return None
    converted = []
    for edge in edges:
        edge = dict(edge)
        if edge.get("to_document_id") is not None:
            edge["to_document_id"] = uuid.UUID(edge["to_document_id"])
        converted.append(edge)
    return converted


def add_authored_chunks_tool(
    chunks: list[str],
    edges: list[dict] | None = None,
    summaries: list[str] | None = None,
    source_urls: list[str] | None = None,
    descriptions: list[str] | None = None,
    roles: dict[str, list[str]] | None = None,
) -> list[dict]:
    """Add a batch of already-segmented, self-written chunks in one call --
    for content you're originating yourself (a decision writeup, a new note)
    rather than parsing an existing source. Owned by whichever agent this
    server is connected as (there's no owner to specify). `chunks` behaves
    like add_sequential_documents: ordered content, auto-linked with `prev`
    edges, embeddings computed automatically.

    `edges` covers relationships beyond that sequence -- to each other, or to
    documents already in the graph. Each entry is a dict:
        {"from_index": int, "edge_type": str, "description": str | None,
         "to_document_id": str | None,   # an existing document's id
         "to_index": int | None,         # another chunk in this same call
         "external_url": str | None}
    `from_index`/`to_index` are positions into `chunks` (those chunks have no
    id yet at call time). Exactly one of `to_document_id` / `to_index` /
    `external_url` must be set per edge.

    `roles` controls who besides the owner can access every chunk in this
    batch, same as add_sequential_documents_tool. If omitted, defaults to
    {"shared_reader": ["read"]}; pass {} to keep every chunk private."""
    with get_session(engine) as session:
        docs = crud.add_authored_chunks(
            session, chunks=chunks, owner_agent=_current_agent(session),
            edges=_edges_with_uuids(edges), summaries=summaries,
            source_urls=source_urls, descriptions=descriptions, roles=roles,
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


_GATHER_CONTEXT_DESCRIPTION = """\
Search across every retrieval method at once and return a single fused,
ranked list of documents. This is the default way to look for existing
context before writing something new, asking the user a question, or
assuming nothing already exists on a topic.

`query_text` drives semantic (embedding) and full-text search; `pattern`
drives literal substring matching (ILIKE). Pass the same string for both
unless you specifically also know an exact identifier, error code, or
config key worth matching literally alongside the semantic query.

`tags` restricts results to documents carrying at least one of the given
tags (an OR, not an AND) -- use it to narrow a search to a known category,
not as the only way to find something.

Note: this does not currently walk the document graph (`prev`/`source`/
`related` edges) to pull in neighboring chunks -- it only returns direct
search hits. If a result looks incomplete or references something else,
call get_edges on it and fetch the neighboring documents yourself.

If a result here -- or a document you only reached by jumping through its
edges via get_edges -- turns out to be genuinely relevant to something you
go on to write, link back to it: use add_edge (or add_authored_chunks's
`edges` argument, when writing something new) to attach a `related` or
`source` edge. Nothing does this automatically. The graph only grows, and
future searches only benefit from it, if you actually add that edge when
you notice the connection -- don't just rely on search to resurface it.
"""

_CHUNK_SIZE_GUIDANCE = """\
Aim for roughly a paragraph's worth of content per chunk: small enough that \
its embedding represents one coherent idea, complete enough to make sense \
on its own if it's ever retrieved without its neighbors. A few hundred \
words is a reasonable target. There's no hard limit enforced, but a chunk \
trying to cover several distinct ideas at once will embed and retrieve \
worse than several smaller ones would.\
"""

_LINK_RELATED_DOCS_GUIDANCE = """\
If this is related to, derived from, or informed by something you found \
via gather_context -- including a document you only reached by jumping \
through its edges via get_edges -- call add_edge afterward to record that \
connection. Search alone won't make it discoverable as related later; the \
edge is what does, and nothing creates it for you automatically.\
"""

_ADD_DOCUMENT_DESCRIPTION = f"""\
Add one already-written chunk as its own document. This is for a single,
self-contained piece of content -- not a large blob that still needs to be
split up. There's no automatic chunker exposed here: if you have more
material than one chunk's worth, break it into pieces yourself and use
add_sequential_documents or add_authored_chunks instead.

{_CHUNK_SIZE_GUIDANCE}

The embedding is computed automatically from `content` -- there's no vector
to supply.

`roles` grants other roles access beyond the owner: a mapping from grantee
role name to a list of permissions ("read" and/or "write"), e.g.
{{"shared_reader": ["read"], "agent_bob": ["read", "write"]}}. If omitted
entirely, defaults to {{"shared_reader": ["read"]}} -- readable by every
agent in the shared_reader group. Pass an empty object ({{}}) to keep this
document private to the owner only.

{_LINK_RELATED_DOCS_GUIDANCE}
"""

_ADD_SEQUENTIAL_DOCUMENTS_DESCRIPTION = f"""\
Add several already-written chunks in one call, in order. Use this for
content that naturally has a sequence -- the sections of one document you
wrote, a multi-step writeup -- where each chunk should link back to the one
before it. Each entry in `contents` becomes its own document, auto-linked
to the previous one with a `prev` edge; embeddings are computed
automatically.

{_CHUNK_SIZE_GUIDANCE} This applies to every entry in `contents`.

If these chunks also need edges to something other than each other in
sequence -- an existing document already in the graph, a non-adjacent chunk
in this same batch, or a tool/external URL -- use add_authored_chunks
instead, which accepts those as an `edges` argument in the same call.

`roles` applies the same grant to every document in the sequence (see
add_document for the exact semantics); defaults to {{"shared_reader":
["read"]}}, pass {{}} to keep the whole sequence private.

{_LINK_RELATED_DOCS_GUIDANCE}
"""

_ADD_AUTHORED_CHUNKS_DESCRIPTION = f"""\
Add a batch of already-segmented, self-written chunks in one call. Use this
instead of add_sequential_documents when the new chunks also need edges
beyond the automatic sequential `prev` chain: a link to an existing
document already in the graph, a non-adjacent link between two chunks in
this same batch, or a link out to a tool/external URL.

{_CHUNK_SIZE_GUIDANCE} This applies to every entry in `chunks`.

`edges` (optional) is a list of dicts, each describing one additional edge:

    {{"from_index": int, "edge_type": str, "description": str | None,
     "to_document_id": str | None,   # an existing document's id
     "to_index": int | None,         # another chunk in this same call
     "external_url": str | None}}

`from_index`/`to_index` are positions in `chunks` (0 = the first chunk
given) -- those chunks don't have ids yet while you're writing the call.
Exactly one of `to_document_id` / `to_index` / `external_url` must be set
per edge. Pick `edge_type` from "source" (this chunk was derived from that
target), "related" (a looser connection), "tool" (the target is
executable), or "supersedes" (this chunk replaces that target).

This is exactly where to record a connection to something you found via
gather_context (including a document you only reached by jumping through
its edges) -- if it's genuinely relevant, use `to_document_id` to link back
to it here rather than leaving that connection implicit.

`roles` applies the same grant to every chunk in the batch (see
add_document for the exact semantics); defaults to {{"shared_reader":
["read"]}}, pass {{}} to keep the whole batch private.
"""

_GET_BY_ID_DESCRIPTION = """\
Fetch one document by its exact id. Use this when you already know which
document you want -- e.g. its id came from a previous search result or from
an edge -- not as a way to search.
"""

_GET_EDGES_DESCRIPTION = """\
Outgoing edges from one document: what you could follow for more context (a
`prev` chunk, a `source` it was derived from, a `related` document, a
`tool` it points at). This is a single hop -- it does not recursively walk
the graph, so following a chain of several links means calling this again
on each document you land on.
"""

_UPDATE_DOCUMENT_DESCRIPTION = """\
Create a new revision of an existing document. The old row is kept as-is
and flagged deprecated rather than overwritten -- a `supersedes` edge links
the new revision back to it, so anything that already pointed at the old
content still describes exactly what it was created against.

This replaces the content of exactly one document at a time; there's no
batch form. The embedding is computed automatically from the new `content`
-- there's no vector to supply. If `content` is unchanged from the current
revision, this is a no-op that returns the existing document rather than
creating a pointless new revision.
"""

_ADD_TAG_DESCRIPTION = """\
Attach an exact-match label to a document, with a short note on why it
applies. Tags are for precise, deliberate categorization you want to filter
on exactly later (via gather_context's `tags` argument) -- use them for
things like a component name, a status, or a category worth searching on
literally, not as a substitute for writing searchable content into the
document itself. `tag` is normalized (trimmed and lowercased) automatically,
so casing/whitespace don't matter when matching later.
"""

_ADD_EDGE_DESCRIPTION = """\
Attach a relationship from one document to either another document or an
external URL -- exactly one of `to_document_id` / `external_url` must be
given. Pick `edge_type` from:
  - "prev"       this document follows another one in sequence
  - "next"       the inverse of prev; rarely needed since
                 add_sequential_documents/add_authored_chunks already
                 create prev edges for you automatically
  - "source"     this document was derived/composed from the target
  - "related"    a looser, non-specific connection
  - "tool"       the target is something executable
  - "supersedes" this document replaces the target (see update_document
                 for the usual way this gets created automatically)
"""

_GRANT_ACCESS_DESCRIPTION = """\
Grant `read` or `write` access on a document you own to another role --
either a specific agent's role name, or a group role like `shared_reader`
so every member gets access at once. Only a document's owner can grant
access to it.
"""

_REVOKE_ACCESS_DESCRIPTION = """\
Revoke access grant(s) on a document you own. Omit `permission` to revoke
both read and write for the given `grantee_role`; pass one to revoke just
that permission and leave the other intact.
"""

_LOG_FEEDBACK_DESCRIPTION = """\
Record how a document performed for a specific query: which method
surfaced it (`source_method`: "vector", "tags", "fulltext", "ilike", or
"id"), what rank it came back at, and whether it was actually judged useful
(`relevance`, `judged_by`: "explicit_llm" or "implicit_usage").

This doesn't change ranking yet, but logging consistently now means there's
real historical data to learn from once it does -- worth calling whenever
you act on (or deliberately discard) a document gather_context returned.
"""

for _fn, _name, _description in [
    (gather_context_tool, "gather_context", _GATHER_CONTEXT_DESCRIPTION),
    (add_document_tool, "add_document", _ADD_DOCUMENT_DESCRIPTION),
    (add_sequential_documents_tool, "add_sequential_documents", _ADD_SEQUENTIAL_DOCUMENTS_DESCRIPTION),
    (add_authored_chunks_tool, "add_authored_chunks", _ADD_AUTHORED_CHUNKS_DESCRIPTION),
    (get_by_id_tool, "get_by_id", _GET_BY_ID_DESCRIPTION),
    (get_edges_tool, "get_edges", _GET_EDGES_DESCRIPTION),
    (update_document_tool, "update_document", _UPDATE_DOCUMENT_DESCRIPTION),
    (add_tag_tool, "add_tag", _ADD_TAG_DESCRIPTION),
    (add_edge_tool, "add_edge", _ADD_EDGE_DESCRIPTION),
    (grant_access_tool, "grant_access", _GRANT_ACCESS_DESCRIPTION),
    (revoke_access_tool, "revoke_access", _REVOKE_ACCESS_DESCRIPTION),
    (log_feedback_tool, "log_feedback", _LOG_FEEDBACK_DESCRIPTION),
]:
    mcp_server.add_tool(_fn, name=_name, description=_description)


if __name__ == "__main__":
    mcp_server.run()
