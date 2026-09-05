"""Tests for mcp_server.py.

Unlike the other modules, mcp_server.py reads KNOWLEDGE_GROVE_DSN and builds
its engine at *import time* (module-level code), not lazily inside a
function. That means it can't be imported at file-top the way `crud`/`search`
are elsewhere in this suite — the real DSN (from the testcontainers-managed
database) isn't known until the `admin_dsn` fixture runs. The
`mcp_server_module` fixture below sets the env var and imports it fresh
inside a test, evicting any previously-cached import first.
"""
import asyncio
import importlib
import json
import os
import sys
import uuid

import pytest
from sqlalchemy import select

from knowledge_grove import crud
from knowledge_grove.db import get_session
from knowledge_grove.models import DocumentAccess

from conftest import _dsn_as


@pytest.fixture()
def mcp_server_module(admin_dsn):
    os.environ["KNOWLEDGE_GROVE_DSN"] = admin_dsn
    sys.modules.pop("knowledge_grove.mcp_server", None)
    module = importlib.import_module("knowledge_grove.mcp_server")
    yield module
    sys.modules.pop("knowledge_grove.mcp_server", None)


def test_add_document_tool_owner_agent_is_derived_from_the_connection(admin_dsn, test_roles):
    # Connect as a real, non-privileged agent role -- not the admin/superuser
    # every other test in this file uses -- to prove owner_agent is genuinely
    # derived from current_user, not something a caller could have supplied.
    os.environ["KNOWLEDGE_GROVE_DSN"] = _dsn_as(admin_dsn, "agent_alice", "alice")
    sys.modules.pop("knowledge_grove.mcp_server", None)
    module = importlib.import_module("knowledge_grove.mcp_server")
    try:
        doc = module.add_document_tool(content="whoami doc")
        assert doc["owner_agent"] == "agent_alice"

        # owner_agent isn't just defaulted -- it was removed as a parameter
        # entirely, so there's nothing for a caller to override.
        with pytest.raises(TypeError):
            module.add_document_tool(content="x", owner_agent="agent_bob")
    finally:
        sys.modules.pop("knowledge_grove.mcp_server", None)


def test_mcp_server_module_imports_without_error(mcp_server_module):
    assert mcp_server_module.mcp_server is not None


def test_gather_context_tool_is_registered(mcp_server_module):
    tools = asyncio.run(mcp_server_module.mcp_server.list_tools())
    tool_names = [t.name for t in tools]
    assert "gather_context" in tool_names


def test_gather_context_tool_returns_json_serializable_results(mcp_server_module):
    # No embedding anywhere here, on purpose -- this is how an actual agent
    # calls the tool: text in, nothing about vectors.
    session = get_session(mcp_server_module.engine)
    try:
        doc = crud.add_document(
            session, content="Retries should use exponential backoff.", owner_agent="agent_alice"
        )
        session.commit()
        doc_id = doc.id
    finally:
        session.close()

    results = mcp_server_module.gather_context_tool(query_text="backoff", pattern="backoff")

    json.dumps(results)  # must not raise -- this is what actually crosses the MCP wire
    assert any(r["id"] == str(doc_id) for r in results)


def test_all_crud_tools_are_registered(mcp_server_module):
    tools = asyncio.run(mcp_server_module.mcp_server.list_tools())
    tool_names = {t.name for t in tools}
    assert tool_names == {
        "gather_context",
        "add_document",
        "add_sequential_documents",
        "add_authored_chunks",
        "get_by_id",
        "get_edges",
        "update_document",
        "add_tag",
        "add_edge",
        "grant_access",
        "revoke_access",
        "log_feedback",
    }


def test_add_document_tool_auto_embeds_and_returns_json_serializable_dict(mcp_server_module):
    # No embedding parameter anywhere -- content in, dict out.
    result = mcp_server_module.add_document_tool(
        content="Retries should use exponential backoff.",
        summary="Retry policy",
    )
    json.dumps(result)  # must not raise

    assert result["content"] == "Retries should use exponential backoff."
    assert result["summary"] == "Retry policy"
    assert result["deprecated"] is False
    assert result["id"]  # a real uuid string was returned

    # It's genuinely embedded, not a stub -- prove it by finding it via search.
    hits = mcp_server_module.gather_context_tool(query_text="exponential backoff", pattern="backoff")
    assert any(h["id"] == result["id"] for h in hits)


def _grant_pairs_for(mcp_server_module, document_id):
    session = get_session(mcp_server_module.engine)
    try:
        grants = session.scalars(
            select(DocumentAccess).where(DocumentAccess.document_id == uuid.UUID(document_id))
        ).all()
        return {(g.grantee_role, g.permission) for g in grants}
    finally:
        session.close()


def test_add_document_tool_no_roles_arg_defaults_to_shared_reader(mcp_server_module):
    doc = mcp_server_module.add_document_tool(content="default doc")

    assert _grant_pairs_for(mcp_server_module, doc["id"]) == {("shared_reader", "read")}


def test_add_document_tool_empty_roles_dict_means_no_grants(mcp_server_module):
    doc = mcp_server_module.add_document_tool(
        content="private doc", roles={},
    )

    assert _grant_pairs_for(mcp_server_module, doc["id"]) == set()


def test_add_document_tool_custom_roles_override_the_default(mcp_server_module):
    doc = mcp_server_module.add_document_tool(
        content="custom doc", roles={"agent_bob": ["read", "write"]},
    )

    assert _grant_pairs_for(mcp_server_module, doc["id"]) == {
        ("agent_bob", "read"), ("agent_bob", "write"),
    }


def test_add_sequential_documents_tool_no_roles_arg_defaults_shared_reader_on_every_doc(mcp_server_module):
    docs = mcp_server_module.add_sequential_documents_tool(
        contents=["c0", "c1"],
    )

    for doc in docs:
        assert _grant_pairs_for(mcp_server_module, doc["id"]) == {("shared_reader", "read")}


def test_add_sequential_documents_tool_empty_roles_dict_means_no_grants_on_any_doc(mcp_server_module):
    docs = mcp_server_module.add_sequential_documents_tool(
        contents=["c0", "c1"], roles={},
    )

    for doc in docs:
        assert _grant_pairs_for(mcp_server_module, doc["id"]) == set()


def test_add_sequential_documents_tool_custom_roles_applied_to_every_doc(mcp_server_module):
    docs = mcp_server_module.add_sequential_documents_tool(
        contents=["c0", "c1"], roles={"agent_bob": ["read"]},
    )

    for doc in docs:
        assert _grant_pairs_for(mcp_server_module, doc["id"]) == {("agent_bob", "read")}


def test_add_sequential_documents_tool_is_registered(mcp_server_module):
    tools = asyncio.run(mcp_server_module.mcp_server.list_tools())
    tool_names = [t.name for t in tools]
    assert "add_sequential_documents" in tool_names


def test_add_sequential_documents_tool_auto_embeds_and_links_prev_edges(mcp_server_module):
    # No embedding parameter anywhere -- contents in, dicts out.
    docs = mcp_server_module.add_sequential_documents_tool(
        contents=["chunk one", "chunk two", "chunk three"],
    )
    json.dumps(docs)  # must not raise

    assert [d["content"] for d in docs] == ["chunk one", "chunk two", "chunk three"]

    edges = mcp_server_module.get_edges_tool(docs[1]["id"])
    assert len(edges) == 1
    assert edges[0]["edge_type"] == "prev"
    assert edges[0]["to_document_id"] == docs[0]["id"]

    # It's genuinely embedded, not a stub -- prove it by finding it via search.
    hits = mcp_server_module.gather_context_tool(query_text="chunk two", pattern="chunk two")
    assert any(h["id"] == docs[1]["id"] for h in hits)


def test_add_sequential_documents_tool_first_document_has_no_edges(mcp_server_module):
    docs = mcp_server_module.add_sequential_documents_tool(
        contents=["only one"],
    )

    assert mcp_server_module.get_edges_tool(docs[0]["id"]) == []


def test_add_authored_chunks_tool_is_registered(mcp_server_module):
    tools = asyncio.run(mcp_server_module.mcp_server.list_tools())
    tool_names = [t.name for t in tools]
    assert "add_authored_chunks" in tool_names


def test_add_authored_chunks_tool_auto_embeds_and_links_prev_edges(mcp_server_module):
    docs = mcp_server_module.add_authored_chunks_tool(
        chunks=["chunk one", "chunk two", "chunk three"],
    )
    json.dumps(docs)  # must not raise

    assert [d["content"] for d in docs] == ["chunk one", "chunk two", "chunk three"]

    edges = mcp_server_module.get_edges_tool(docs[1]["id"])
    assert len(edges) == 1
    assert edges[0]["edge_type"] == "prev"
    assert edges[0]["to_document_id"] == docs[0]["id"]

    hits = mcp_server_module.gather_context_tool(query_text="chunk two", pattern="chunk two")
    assert any(h["id"] == docs[1]["id"] for h in hits)


def test_add_authored_chunks_tool_no_edges_arg_still_creates_prev_chain_only(mcp_server_module):
    docs = mcp_server_module.add_authored_chunks_tool(chunks=["only one"])

    assert mcp_server_module.get_edges_tool(docs[0]["id"]) == []


def test_add_authored_chunks_tool_no_roles_arg_defaults_shared_reader_on_every_doc(mcp_server_module):
    docs = mcp_server_module.add_authored_chunks_tool(chunks=["c0", "c1"])

    for doc in docs:
        assert _grant_pairs_for(mcp_server_module, doc["id"]) == {("shared_reader", "read")}


def test_add_authored_chunks_tool_empty_roles_dict_means_no_grants_on_any_doc(mcp_server_module):
    docs = mcp_server_module.add_authored_chunks_tool(chunks=["c0", "c1"], roles={})

    for doc in docs:
        assert _grant_pairs_for(mcp_server_module, doc["id"]) == set()


def test_add_authored_chunks_tool_custom_roles_applied_to_every_doc(mcp_server_module):
    docs = mcp_server_module.add_authored_chunks_tool(
        chunks=["c0", "c1"], roles={"agent_bob": ["read"]},
    )

    for doc in docs:
        assert _grant_pairs_for(mcp_server_module, doc["id"]) == {("agent_bob", "read")}


def test_add_authored_chunks_tool_edge_to_existing_document_via_to_document_id(mcp_server_module):
    existing = mcp_server_module.add_document_tool(content="existing doc")

    docs = mcp_server_module.add_authored_chunks_tool(
        chunks=["new chunk"],
        edges=[{"from_index": 0, "edge_type": "source", "to_document_id": existing["id"]}],
    )

    edges = mcp_server_module.get_edges_tool(docs[0]["id"])
    assert len(edges) == 1
    assert edges[0]["edge_type"] == "source"
    assert edges[0]["to_document_id"] == existing["id"]


def test_add_authored_chunks_tool_edge_between_two_new_chunks_via_to_index(mcp_server_module):
    docs = mcp_server_module.add_authored_chunks_tool(
        chunks=["chunk 0", "chunk 1", "chunk 2"],
        edges=[{"from_index": 2, "edge_type": "related", "to_index": 0}],
    )

    edges = mcp_server_module.get_edges_tool(docs[2]["id"])
    related = [e for e in edges if e["edge_type"] == "related"]
    assert len(related) == 1
    assert related[0]["to_document_id"] == docs[0]["id"]


def test_add_authored_chunks_tool_edge_to_external_url(mcp_server_module):
    docs = mcp_server_module.add_authored_chunks_tool(
        chunks=["chunk 0"],
        edges=[{"from_index": 0, "edge_type": "tool", "external_url": "https://example.com/tool.py"}],
    )

    edges = mcp_server_module.get_edges_tool(docs[0]["id"])
    assert edges[0]["external_url"] == "https://example.com/tool.py"
    assert edges[0]["to_document_id"] is None


def test_add_authored_chunks_tool_edge_with_no_target_raises(mcp_server_module):
    with pytest.raises(ValueError):
        mcp_server_module.add_authored_chunks_tool(
            chunks=["c0"],
            edges=[{"from_index": 0, "edge_type": "related"}],
        )


def test_add_sequential_documents_tool_uses_given_descriptions(mcp_server_module):
    docs = mcp_server_module.add_sequential_documents_tool(
        contents=["c0", "c1"],
        descriptions=[None, "second chunk"],
    )

    edges = mcp_server_module.get_edges_tool(docs[1]["id"])
    assert edges[0]["description"] == "second chunk"


def test_add_sequential_documents_tool_applies_summaries_and_source_urls(mcp_server_module):
    docs = mcp_server_module.add_sequential_documents_tool(
        contents=["c0", "c1"],
        summaries=["summary 0", "summary 1"],
        source_urls=["https://example.com/0", "https://example.com/1"],
    )

    assert docs[0]["summary"] == "summary 0"
    assert docs[0]["source_url"] == "https://example.com/0"
    assert docs[1]["summary"] == "summary 1"
    assert docs[1]["source_url"] == "https://example.com/1"


def test_get_by_id_tool_returns_document(mcp_server_module):
    added = mcp_server_module.add_document_tool(content="findable content")

    fetched = mcp_server_module.get_by_id_tool(added["id"])

    assert fetched is not None
    assert fetched["id"] == added["id"]
    assert fetched["content"] == "findable content"


def test_get_by_id_tool_returns_none_for_missing(mcp_server_module):
    assert mcp_server_module.get_by_id_tool(str(uuid.uuid4())) is None


def test_get_edges_tool_returns_edges(mcp_server_module):
    doc_a = mcp_server_module.add_document_tool(content="doc a")
    doc_b = mcp_server_module.add_document_tool(content="doc b")
    mcp_server_module.add_edge_tool(
        from_document_id=doc_a["id"], edge_type="related",
        description="see also doc b", to_document_id=doc_b["id"],
    )

    edges = mcp_server_module.get_edges_tool(doc_a["id"])
    json.dumps(edges)  # must not raise

    assert len(edges) == 1
    assert edges[0]["to_document_id"] == doc_b["id"]
    assert edges[0]["description"] == "see also doc b"


def test_update_document_tool_creates_new_revision_and_auto_embeds(mcp_server_module):
    original = mcp_server_module.add_document_tool(content="v1")

    revised = mcp_server_module.update_document_tool(original["id"], content="v2 with new wording")

    assert revised["id"] != original["id"]
    assert revised["content"] == "v2 with new wording"
    assert revised["deprecated"] is False

    old = mcp_server_module.get_by_id_tool(original["id"])
    assert old["deprecated"] is True


def test_add_tag_tool(mcp_server_module):
    doc = mcp_server_module.add_document_tool(content="tagged doc")

    tag = mcp_server_module.add_tag_tool(doc["id"], "  Retries  ", "discusses retry policy")
    json.dumps(tag)  # must not raise

    assert tag["tag"] == "retries"
    assert tag["document_id"] == doc["id"]


def test_add_edge_tool_to_external_url(mcp_server_module):
    doc = mcp_server_module.add_document_tool(content="a doc")

    edge = mcp_server_module.add_edge_tool(
        from_document_id=doc["id"], edge_type="tool",
        description="points at a tool", external_url="https://example.com/tool.py",
    )
    json.dumps(edge)  # must not raise

    assert edge["external_url"] == "https://example.com/tool.py"
    assert edge["to_document_id"] is None


def test_add_edge_tool_description_is_optional(mcp_server_module):
    doc_a = mcp_server_module.add_document_tool(content="doc a")
    doc_b = mcp_server_module.add_document_tool(content="doc b")

    edge = mcp_server_module.add_edge_tool(
        from_document_id=doc_a["id"], edge_type="related", to_document_id=doc_b["id"],
    )
    json.dumps(edge)  # must not raise

    assert edge["description"] is None


def test_grant_and_revoke_access_tools(mcp_server_module):
    doc = mcp_server_module.add_document_tool(content="shared doc")

    grant = mcp_server_module.grant_access_tool(doc["id"], "agent_bob", "read")
    json.dumps(grant)  # must not raise
    assert grant["grantee_role"] == "agent_bob"
    assert grant["permission"] == "read"

    result = mcp_server_module.revoke_access_tool(doc["id"], "agent_bob")
    assert result == {"revoked": True}


def test_log_feedback_tool(mcp_server_module):
    doc = mcp_server_module.add_document_tool(content="feedback target")

    feedback = mcp_server_module.log_feedback_tool(
        query_text="how do retries work",
        document_id=doc["id"],
        source_method="vector",
        rank=1,
        judged_by="implicit_usage",
        relevance=0.9,
    )
    json.dumps(feedback)  # must not raise

    assert feedback["document_id"] == doc["id"]
    assert feedback["rank"] == 1


def test_log_feedback_tool_without_relevance(mcp_server_module):
    doc = mcp_server_module.add_document_tool(content="feedback target")

    feedback = mcp_server_module.log_feedback_tool(
        query_text="how do retries work",
        document_id=doc["id"],
        source_method="vector",
        rank=1,
        judged_by="implicit_usage",
    )
    json.dumps(feedback)  # must not raise

    assert feedback["document_id"] == doc["id"]


def test_add_edge_tool_rejects_both_targets(mcp_server_module):
    doc_a = mcp_server_module.add_document_tool(content="doc a")
    doc_b = mcp_server_module.add_document_tool(content="doc b")

    with pytest.raises(ValueError):
        mcp_server_module.add_edge_tool(
            from_document_id=doc_a["id"], edge_type="related", description="bad edge",
            to_document_id=doc_b["id"], external_url="https://example.com",
        )


def test_add_edge_tool_rejects_neither_target(mcp_server_module):
    doc = mcp_server_module.add_document_tool(content="a doc")

    with pytest.raises(ValueError):
        mcp_server_module.add_edge_tool(
            from_document_id=doc["id"], edge_type="related", description="bad edge",
        )


def test_get_edges_tool_returns_empty_list_for_no_edges(mcp_server_module):
    doc = mcp_server_module.add_document_tool(content="lonely doc")

    assert mcp_server_module.get_edges_tool(doc["id"]) == []


def test_get_edges_tool_returns_multiple_edges(mcp_server_module):
    doc_a = mcp_server_module.add_document_tool(content="doc a")
    doc_b = mcp_server_module.add_document_tool(content="doc b")
    doc_c = mcp_server_module.add_document_tool(content="doc c")
    mcp_server_module.add_edge_tool(
        from_document_id=doc_a["id"], edge_type="related", description="see b", to_document_id=doc_b["id"],
    )
    mcp_server_module.add_edge_tool(
        from_document_id=doc_a["id"], edge_type="related", description="see c", to_document_id=doc_c["id"],
    )

    edges = mcp_server_module.get_edges_tool(doc_a["id"])

    assert len(edges) == 2
    assert {e["to_document_id"] for e in edges} == {doc_b["id"], doc_c["id"]}


def test_get_edges_tool_reveals_supersedes_edge_after_update(mcp_server_module):
    # update_document_tool creates the new revision plus a `supersedes` edge
    # linking back to the old one -- confirm that's actually visible through
    # get_edges_tool, tying the two tools together the way an agent would use
    # them (update, then look at what changed).
    original = mcp_server_module.add_document_tool(content="v1")
    revised = mcp_server_module.update_document_tool(original["id"], content="v2")

    edges = mcp_server_module.get_edges_tool(revised["id"])

    assert len(edges) == 1
    assert edges[0]["edge_type"] == "supersedes"
    assert edges[0]["to_document_id"] == original["id"]


def test_revoke_access_tool_with_specific_permission_leaves_other_intact(mcp_server_module):
    doc = mcp_server_module.add_document_tool(content="shared doc")
    mcp_server_module.grant_access_tool(doc["id"], "agent_bob", "read")
    mcp_server_module.grant_access_tool(doc["id"], "agent_bob", "write")

    session = get_session(mcp_server_module.engine)
    try:
        mcp_server_module.revoke_access_tool(doc["id"], "agent_bob", permission="write")
        remaining = session.query(DocumentAccess).filter_by(
            document_id=uuid.UUID(doc["id"]), grantee_role="agent_bob"
        ).all()
        assert [g.permission for g in remaining] == ["read"]
    finally:
        session.close()


def test_gather_context_tool_respects_tags(mcp_server_module):
    tagged = mcp_server_module.add_document_tool(content="retries and backoff")
    mcp_server_module.add_tag_tool(tagged["id"], "config", "config setting")
    untagged = mcp_server_module.add_document_tool(content="retries and backoff too")

    results = mcp_server_module.gather_context_tool(
        query_text="retries", pattern="backoff", tags=["config"],
    )
    result_ids = {r["id"] for r in results}

    assert tagged["id"] in result_ids
    assert untagged["id"] not in result_ids


def test_all_registered_tools_have_descriptions(mcp_server_module):
    tools = asyncio.run(mcp_server_module.mcp_server.list_tools())
    for tool in tools:
        assert tool.description, f"{tool.name} is missing a description"
