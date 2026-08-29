import uuid

import pytest
from sqlalchemy import select

from knowledge_grove import crud
from knowledge_grove.models import DocumentAccess, Edge

from conftest import vec


def test_add_document_auto_embeds_when_not_given(alice, embedding_model):
    doc = crud.add_document(alice, content="Retries should use exponential backoff.", owner_agent="agent_alice")
    alice.commit()

    assert doc.embedding is not None
    assert len(doc.embedding) == len(embedding_model.embed_text("anything"))


def test_add_document_uses_explicit_embedding_when_given(alice):
    doc = crud.add_document(
        alice, content="near match", embedding=vec(0), owner_agent="agent_alice"
    )
    alice.commit()

    assert doc.embedding == vec(0)


def test_add_document_auto_embeds_summary_when_summary_given_without_one(alice, embedding_model):
    doc = crud.add_document(
        alice, content="content", embedding=vec(0), owner_agent="agent_alice", summary="a short summary"
    )
    alice.commit()

    assert doc.summary_embedding is not None
    assert len(doc.summary_embedding) == len(embedding_model.embed_text("anything"))


def test_add_document_no_summary_means_no_summary_embedding(alice):
    doc = crud.add_document(
        alice, content="content", embedding=vec(0), owner_agent="agent_alice"
    )
    alice.commit()

    assert doc.summary is None
    assert doc.summary_embedding is None


def test_update_document_auto_embeds_when_not_given(alice, embedding_model):
    original = crud.add_document(
        alice, content="v1", embedding=vec(0), owner_agent="agent_alice"
    )
    alice.commit()

    revised = crud.update_document(alice, original.id, content="v2 with new wording")
    alice.commit()

    assert revised.embedding is not None
    assert len(revised.embedding) == len(embedding_model.embed_text("anything"))


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


def test_update_document_supersedes_edge_description_defaults_to_none(alice):
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
    assert edge.description is None


def test_update_document_uses_given_description(alice):
    original = crud.add_document(
        alice, content="v1", embedding=vec(0), owner_agent="agent_alice"
    )
    alice.commit()

    revised = crud.update_document(
        alice, original.id, content="v2", embedding=vec(1), description="fixed a typo"
    )
    alice.commit()

    edge = alice.scalars(
        select(Edge).where(
            Edge.from_document_id == revised.id, Edge.edge_type == "supersedes"
        )
    ).one()
    assert edge.description == "fixed a typo"


def test_add_edge_description_defaults_to_none(alice):
    doc_a = crud.add_document(
        alice, content="a", embedding=vec(0), owner_agent="agent_alice"
    )
    doc_b = crud.add_document(
        alice, content="b", embedding=vec(1), owner_agent="agent_alice"
    )
    alice.commit()

    edge = crud.add_edge(
        alice, from_document_id=doc_a.id, edge_type="related", to_document_id=doc_b.id,
    )
    alice.commit()

    assert edge.description is None


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


def test_add_sequential_documents_creates_prev_edges_in_order(alice):
    docs = crud.add_sequential_documents(
        alice,
        contents=["c0", "c1", "c2"],
        owner_agent="agent_alice",
        embeddings=[vec(0), vec(1), vec(2)],
    )
    alice.commit()

    assert [d.content for d in docs] == ["c0", "c1", "c2"]

    edge_1_to_0 = alice.scalars(
        select(Edge).where(Edge.from_document_id == docs[1].id)
    ).one()
    assert edge_1_to_0.to_document_id == docs[0].id
    assert edge_1_to_0.edge_type == "prev"

    edge_2_to_1 = alice.scalars(
        select(Edge).where(Edge.from_document_id == docs[2].id)
    ).one()
    assert edge_2_to_1.to_document_id == docs[1].id

    # The first document has nothing before it -- no outgoing edge.
    assert alice.scalars(select(Edge).where(Edge.from_document_id == docs[0].id)).all() == []


def test_add_sequential_documents_auto_embeds_when_not_given(alice, embedding_model):
    docs = crud.add_sequential_documents(
        alice, contents=["c0", "c1"], owner_agent="agent_alice"
    )
    alice.commit()

    for doc in docs:
        assert doc.embedding is not None
        assert len(doc.embedding) == len(embedding_model.embed_text("anything"))


def test_add_sequential_documents_uses_explicit_embeddings_when_given(alice):
    docs = crud.add_sequential_documents(
        alice,
        contents=["c0", "c1"],
        owner_agent="agent_alice",
        embeddings=[vec(0), vec(1)],
    )
    alice.commit()

    assert docs[0].embedding == vec(0)
    assert docs[1].embedding == vec(1)


def test_add_sequential_documents_description_defaults_to_none_when_not_given(alice):
    docs = crud.add_sequential_documents(
        alice,
        contents=["c0", "c1"],
        owner_agent="agent_alice",
        embeddings=[vec(0), vec(1)],
    )
    alice.commit()

    edge = alice.scalars(select(Edge).where(Edge.from_document_id == docs[1].id)).one()
    assert edge.description is None


def test_add_sequential_documents_uses_given_descriptions(alice):
    docs = crud.add_sequential_documents(
        alice,
        contents=["c0", "c1", "c2"],
        owner_agent="agent_alice",
        embeddings=[vec(0), vec(1), vec(2)],
        descriptions=[None, "second chunk", "third chunk"],
    )
    alice.commit()

    edge_1_to_0 = alice.scalars(
        select(Edge).where(Edge.from_document_id == docs[1].id)
    ).one()
    assert edge_1_to_0.description == "second chunk"

    edge_2_to_1 = alice.scalars(
        select(Edge).where(Edge.from_document_id == docs[2].id)
    ).one()
    assert edge_2_to_1.description == "third chunk"


def test_add_sequential_documents_applies_summaries_per_index(alice, embedding_model):
    docs = crud.add_sequential_documents(
        alice,
        contents=["c0", "c1"],
        owner_agent="agent_alice",
        embeddings=[vec(0), vec(1)],
        summaries=["summary 0", "summary 1"],
    )
    alice.commit()

    assert docs[0].summary == "summary 0"
    assert docs[1].summary == "summary 1"
    assert docs[0].summary_embedding is not None
    assert len(docs[0].summary_embedding) == len(embedding_model.embed_text("anything"))


def test_add_sequential_documents_applies_source_urls_per_index(alice):
    docs = crud.add_sequential_documents(
        alice,
        contents=["c0", "c1"],
        owner_agent="agent_alice",
        embeddings=[vec(0), vec(1)],
        source_urls=["https://example.com/0", "https://example.com/1"],
    )
    alice.commit()

    assert docs[0].source_url == "https://example.com/0"
    assert docs[1].source_url == "https://example.com/1"


def test_add_sequential_documents_single_document_creates_no_edges(alice):
    docs = crud.add_sequential_documents(
        alice, contents=["only one"], owner_agent="agent_alice", embeddings=[vec(0)]
    )
    alice.commit()

    assert len(docs) == 1
    assert alice.scalars(select(Edge).where(Edge.from_document_id == docs[0].id)).all() == []


def test_add_sequential_documents_empty_list_returns_empty(alice):
    assert crud.add_sequential_documents(alice, contents=[], owner_agent="agent_alice") == []


def test_add_raw_document_chunks_by_heading_and_creates_prev_edges(alice):
    raw = "# Heading One\n\npara one\n\npara two\n\n# Heading Two\npara three\n"
    docs = crud.add_raw_document(alice, raw, owner_agent="agent_alice")
    alice.commit()

    assert [d.content for d in docs] == [
        "# Heading One\n\npara one",
        "para two",
        "# Heading Two\npara three",
    ]

    edge_1_to_0 = alice.scalars(
        select(Edge).where(Edge.from_document_id == docs[1].id)
    ).one()
    assert edge_1_to_0.to_document_id == docs[0].id
    assert edge_1_to_0.edge_type == "prev"

    edge_2_to_1 = alice.scalars(
        select(Edge).where(Edge.from_document_id == docs[2].id)
    ).one()
    assert edge_2_to_1.to_document_id == docs[1].id

    assert alice.scalars(select(Edge).where(Edge.from_document_id == docs[0].id)).all() == []


def test_add_raw_document_sets_owner_agent_on_every_chunk(alice):
    raw = "para one\n\npara two\n"
    docs = crud.add_raw_document(alice, raw, owner_agent="agent_alice")
    alice.commit()

    assert all(doc.owner_agent == "agent_alice" for doc in docs)


def test_add_raw_document_auto_embeds_every_chunk(alice, embedding_model):
    raw = "para one\n\npara two\n"
    docs = crud.add_raw_document(alice, raw, owner_agent="agent_alice")
    alice.commit()

    for doc in docs:
        assert doc.embedding is not None
        assert len(doc.embedding) == len(embedding_model.embed_text("anything"))


def test_add_raw_document_single_chunk_creates_no_edges(alice):
    docs = crud.add_raw_document(alice, "just one paragraph, no breaks\n", owner_agent="agent_alice")
    alice.commit()

    assert len(docs) == 1
    assert alice.scalars(select(Edge).where(Edge.from_document_id == docs[0].id)).all() == []


def test_add_raw_document_empty_content_returns_empty_list(alice):
    assert crud.add_raw_document(alice, "", owner_agent="agent_alice") == []


def test_add_raw_documents_returns_one_sublist_per_source_document(alice):
    raw_docs = ["# Doc A\n\npara a\n", "# Doc B\n\npara b\n"]
    result = crud.add_raw_documents(alice, raw_docs, owner_agent="agent_alice")
    alice.commit()

    assert len(result) == 2
    assert [d.content for d in result[0]] == ["# Doc A\n\npara a"]
    assert [d.content for d in result[1]] == ["# Doc B\n\npara b"]


def test_add_raw_documents_does_not_link_edges_across_source_documents(alice):
    raw_docs = ["# Doc A\n\npara a1\n\npara a2\n", "# Doc B\n\npara b1\n"]
    result = crud.add_raw_documents(alice, raw_docs, owner_agent="agent_alice")
    alice.commit()

    doc_b_first_chunk = result[1][0]
    assert alice.scalars(
        select(Edge).where(Edge.from_document_id == doc_b_first_chunk.id)
    ).all() == []


def test_add_raw_documents_sets_owner_agent_on_every_chunk(alice):
    raw_docs = ["para a\n", "para b\n"]
    result = crud.add_raw_documents(alice, raw_docs, owner_agent="agent_alice")
    alice.commit()

    assert all(doc.owner_agent == "agent_alice" for chunks in result for doc in chunks)


def test_add_raw_documents_empty_list_returns_empty(alice):
    assert crud.add_raw_documents(alice, [], owner_agent="agent_alice") == []


def test_add_raw_document_applies_source_url_to_every_chunk(alice):
    raw = "# Heading\n\npara one\n\npara two\n"
    docs = crud.add_raw_document(
        alice, raw, owner_agent="agent_alice", source_url="https://example.com/doc.md"
    )
    alice.commit()

    assert len(docs) > 1, "test needs multiple chunks to prove the source_url isn't just on the first one"
    assert all(doc.source_url == "https://example.com/doc.md" for doc in docs)


def test_add_raw_document_source_url_defaults_to_none(alice):
    docs = crud.add_raw_document(alice, "para one\n\npara two\n", owner_agent="agent_alice")
    alice.commit()

    assert all(doc.source_url is None for doc in docs)


def test_add_raw_documents_applies_source_urls_per_document(alice):
    raw_docs = ["# Doc A\n\npara a\n", "# Doc B\n\npara b\n"]
    result = crud.add_raw_documents(
        alice,
        raw_docs,
        owner_agent="agent_alice",
        source_urls=["https://example.com/a.md", "https://example.com/b.md"],
    )
    alice.commit()

    assert all(doc.source_url == "https://example.com/a.md" for doc in result[0])
    assert all(doc.source_url == "https://example.com/b.md" for doc in result[1])


def test_add_raw_documents_source_urls_can_mix_none_and_given(alice):
    raw_docs = ["para a\n", "para b\n"]
    result = crud.add_raw_documents(
        alice, raw_docs, owner_agent="agent_alice", source_urls=[None, "https://example.com/b.md"],
    )
    alice.commit()

    assert all(doc.source_url is None for doc in result[0])
    assert all(doc.source_url == "https://example.com/b.md" for doc in result[1])


def test_add_raw_documents_source_urls_defaults_to_none_when_omitted(alice):
    raw_docs = ["para a\n", "para b\n"]
    result = crud.add_raw_documents(alice, raw_docs, owner_agent="agent_alice")
    alice.commit()

    assert all(doc.source_url is None for chunks in result for doc in chunks)


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
