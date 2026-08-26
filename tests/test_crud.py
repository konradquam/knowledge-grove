import uuid

import pytest
from sqlalchemy import select

from knowledge_grove import crud
from knowledge_grove.models import DocumentAccess, Edge

from conftest import vec


def test_add_document_sets_fields(alice):
    doc = crud.add_document(
        alice,
        content="Retries should use exponential backoff.",
        embedding=vec(0),
        owner_agent="agent_alice",
        summary="Retry policy",
        summary_embedding=vec(1),
        source_url="https://example.com/retry.md",
    )
    alice.commit()

    assert isinstance(doc.id, uuid.UUID)
    assert doc.content == "Retries should use exponential backoff."
    assert doc.owner_agent == "agent_alice"
    assert doc.summary == "Retry policy"
    assert doc.source_url == "https://example.com/retry.md"
    assert doc.deprecated is False
    assert doc.created_at is not None


def test_add_document_defaults_are_optional(alice):
    doc = crud.add_document(
        alice, content="minimal doc", embedding=vec(0), owner_agent="agent_alice"
    )
    alice.commit()

    assert doc.summary is None
    assert doc.summary_embedding is None
    assert doc.source_url is None


def test_get_by_id_found(alice):
    doc = crud.add_document(
        alice, content="findable", embedding=vec(0), owner_agent="agent_alice"
    )
    alice.commit()

    found = crud.get_by_id(alice, doc.id)
    assert found is not None
    assert found.id == doc.id


def test_get_by_id_missing_returns_none(alice):
    assert crud.get_by_id(alice, uuid.uuid4()) is None


def test_update_document_creates_new_revision(alice):
    original = crud.add_document(
        alice, content="v1", embedding=vec(0), owner_agent="agent_alice"
    )
    alice.commit()

    revised = crud.update_document(alice, original.id, content="v2", embedding=vec(1))
    alice.commit()

    assert revised.id != original.id
    assert revised.content == "v2"
    assert revised.deprecated is False
    assert revised.owner_agent == original.owner_agent

    old = crud.get_by_id(alice, original.id)
    assert old.deprecated is True
    assert old.content == "v1", "old revision's content must never be mutated"


def test_update_document_creates_supersedes_edge(alice):
    original = crud.add_document(
        alice, content="v1", embedding=vec(0), owner_agent="agent_alice"
    )
    alice.commit()

    revised = crud.update_document(alice, original.id, content="v2", embedding=vec(1))
    alice.commit()

    edge = alice.scalars(
        select(Edge).where(
            Edge.from_document_id == revised.id, Edge.edge_type == "supersedes"
        )
    ).one()
    assert edge.to_document_id == original.id


def test_update_document_missing_raises(alice):
    with pytest.raises(ValueError):
        crud.update_document(alice, uuid.uuid4(), content="x", embedding=vec(0))


def test_add_tag_normalizes(alice):
    doc = crud.add_document(
        alice, content="tagged doc", embedding=vec(0), owner_agent="agent_alice"
    )
    alice.commit()

    tag = crud.add_tag(alice, doc.id, "  Retries  ", "discusses retry policy")
    alice.commit()

    assert tag.tag == "retries"
    assert tag.description == "discusses retry policy"


def test_add_edge_to_document(alice):
    doc_a = crud.add_document(
        alice, content="a", embedding=vec(0), owner_agent="agent_alice"
    )
    doc_b = crud.add_document(
        alice, content="b", embedding=vec(1), owner_agent="agent_alice"
    )
    alice.commit()

    edge = crud.add_edge(
        alice,
        from_document_id=doc_a.id,
        edge_type="related",
        description="related content",
        to_document_id=doc_b.id,
    )
    alice.commit()

    assert edge.to_document_id == doc_b.id
    assert edge.external_url is None


def test_add_edge_to_external_url(alice):
    doc = crud.add_document(
        alice, content="a", embedding=vec(0), owner_agent="agent_alice"
    )
    alice.commit()

    edge = crud.add_edge(
        alice,
        from_document_id=doc.id,
        edge_type="tool",
        description="points at a tool",
        external_url="https://example.com/tool.py",
    )
    alice.commit()

    assert edge.external_url == "https://example.com/tool.py"
    assert edge.to_document_id is None


def test_add_edge_rejects_both_targets(alice):
    doc_a = crud.add_document(
        alice, content="a", embedding=vec(0), owner_agent="agent_alice"
    )
    doc_b = crud.add_document(
        alice, content="b", embedding=vec(1), owner_agent="agent_alice"
    )
    alice.commit()

    with pytest.raises(ValueError):
        crud.add_edge(
            alice,
            from_document_id=doc_a.id,
            edge_type="related",
            description="bad edge",
            to_document_id=doc_b.id,
            external_url="https://example.com",
        )


def test_add_edge_rejects_neither_target(alice):
    doc = crud.add_document(
        alice, content="a", embedding=vec(0), owner_agent="agent_alice"
    )
    alice.commit()

    with pytest.raises(ValueError):
        crud.add_edge(
            alice,
            from_document_id=doc.id,
            edge_type="related",
            description="bad edge",
        )


def test_grant_access_creates_row(alice):
    doc = crud.add_document(
        alice, content="shared doc", embedding=vec(0), owner_agent="agent_alice"
    )
    alice.commit()

    grant = crud.grant_access(alice, doc.id, "agent_bob", "read")
    alice.commit()

    assert grant.grantee_role == "agent_bob"
    assert grant.permission == "read"


def test_revoke_access_removes_matching_grant(alice):
    doc = crud.add_document(
        alice, content="shared doc", embedding=vec(0), owner_agent="agent_alice"
    )
    alice.commit()
    crud.grant_access(alice, doc.id, "agent_bob", "read")
    alice.commit()

    crud.revoke_access(alice, doc.id, "agent_bob")
    alice.commit()

    remaining = alice.scalars(
        select(DocumentAccess).where(DocumentAccess.document_id == doc.id)
    ).all()
    assert remaining == []


def test_revoke_access_with_permission_only_removes_that_permission(alice):
    doc = crud.add_document(
        alice, content="shared doc", embedding=vec(0), owner_agent="agent_alice"
    )
    alice.commit()
    crud.grant_access(alice, doc.id, "agent_bob", "read")
    crud.grant_access(alice, doc.id, "agent_bob", "write")
    alice.commit()

    crud.revoke_access(alice, doc.id, "agent_bob", permission="write")
    alice.commit()

    remaining = alice.scalars(
        select(DocumentAccess).where(DocumentAccess.document_id == doc.id)
    ).all()
    assert [g.permission for g in remaining] == ["read"]


def test_log_feedback(alice):
    doc = crud.add_document(
        alice, content="feedback target", embedding=vec(0), owner_agent="agent_alice"
    )
    alice.commit()

    feedback = crud.log_feedback(
        alice,
        query_text="how do retries work",
        document_id=doc.id,
        source_method="vector",
        rank=1,
        judged_by="implicit_usage",
        relevance=0.8,
    )
    alice.commit()

    assert feedback.query_text == "how do retries work"
    assert feedback.source_method == "vector"
    assert feedback.rank == 1
    assert feedback.relevance == 0.8
