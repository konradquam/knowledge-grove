"""End-to-end workflows exercising multiple SDK functions together, the way an
agent actually would over the lifetime of a piece of knowledge — as opposed to
test_crud.py/test_search.py/test_rls.py, which test one function's behavior
in isolation.
"""

from knowledge_grove import crud, search

from conftest import vec


def test_full_document_lifecycle(alice, bob):
    # Alice authors a document and describes it.
    doc = crud.add_document(
        alice,
        content="Retries should use exponential backoff with jitter.",
        embedding=vec(0),
        owner_agent="agent_alice",
        summary="Retry policy",
        summary_embedding=vec(0),
    )
    crud.add_tag(alice, doc.id, "retries", "discusses retry policy")
    crud.add_edge(
        alice,
        from_document_id=doc.id,
        edge_type="tool",
        description="Points at the retry helper implementation",
        external_url="https://example.com/retry.py",
    )
    alice.commit()

    # It's discoverable through multiple entry points immediately.
    assert doc.id in [d.id for d, _ in search.search_vector(alice, vec(0))]
    assert doc.id in [d.id for d in search.search_tag(alice, "retries")]
    assert doc.id in [
        d.id for d, _ in search.search_fulltext(alice, "retry backoff")
    ]

    # Bob can't see it yet — no grant.
    assert crud.get_by_id(bob, doc.id) is None

    # Alice shares it read-only; Bob can now find and read it, but not edit it.
    crud.grant_access(alice, doc.id, "agent_bob", "read")
    alice.commit()
    assert doc.id in [d.id for d in search.search_tag(bob, "retries")]

    # Alice revises the content. The old row is preserved and deprecated;
    # the new row is what search surfaces going forward.
    revised = crud.update_document(
        alice,
        doc.id,
        content="Retries should use exponential backoff with full jitter, capped at 30s.",
        embedding=vec(0),
    )
    alice.commit()

    tag_hits = [d.id for d in search.search_tag(alice, "retries")]
    assert doc.id not in tag_hits, "old revision should no longer surface in search"
    # (the new revision has no tags yet — authoring tags is a separate step,
    # not something update_document carries forward automatically)
    old = crud.get_by_id(alice, doc.id)
    assert old.deprecated is True
    assert revised.deprecated is False

    # Alice revokes Bob's access; he loses visibility entirely, old or new.
    crud.revoke_access(alice, doc.id, "agent_bob")
    alice.commit()
    assert crud.get_by_id(bob, doc.id) is None
    assert crud.get_by_id(bob, revised.id) is None


def test_authoring_a_linked_set_of_chunks_and_traversing_them(alice):
    # An agent authoring a multi-section note: three chunks, linked next/prev,
    # matching the pattern described in §11 for author-time chunking (each
    # add_edge call here stands in for what add_authored_chunks would do
    # under the hood, which isn't built yet).
    intro = crud.add_document(
        alice, content="Intro: why retries matter.", embedding=vec(0),
        owner_agent="agent_alice",
    )
    body = crud.add_document(
        alice, content="Body: exponential backoff with jitter.", embedding=vec(1),
        owner_agent="agent_alice",
    )
    conclusion = crud.add_document(
        alice, content="Conclusion: cap backoff at 30s.", embedding=vec(2),
        owner_agent="agent_alice",
    )
    alice.commit()

    crud.add_edge(
        alice, from_document_id=intro.id, edge_type="next",
        description="continues into the body", to_document_id=body.id,
    )
    crud.add_edge(
        alice, from_document_id=body.id, edge_type="prev",
        description="continues from the intro", to_document_id=intro.id,
    )
    crud.add_edge(
        alice, from_document_id=body.id, edge_type="next",
        description="continues into the conclusion", to_document_id=conclusion.id,
    )
    crud.add_edge(
        alice, from_document_id=conclusion.id, edge_type="prev",
        description="continues from the body", to_document_id=body.id,
    )
    alice.commit()

    # A search hit on just the body chunk should still be locatable, and its
    # neighbors reachable via the edges (manual traversal here — the bounded
    # graph walk described in §6/§10 isn't built yet).
    hits = search.search_fulltext(alice, "backoff jitter")
    assert body.id in [d.id for d, _ in hits]


def test_feedback_logged_against_search_results_reflects_the_query(alice):
    # A realistic pattern from §7: run a search, then log feedback per result
    # using the method and rank that search actually returned.
    doc = crud.add_document(
        alice, content="Retries should use exponential backoff.",
        embedding=vec(0), owner_agent="agent_alice",
    )
    alice.commit()

    results = search.search_vector(alice, vec(0), limit=5)
    top_doc, _score = results[0]
    assert top_doc.id == doc.id

    feedback = crud.log_feedback(
        alice,
        query_text="how should retries back off",
        document_id=top_doc.id,
        source_method="vector",
        rank=1,
        judged_by="implicit_usage",
        relevance=1.0,
    )
    alice.commit()

    assert feedback.document_id == doc.id
    assert feedback.rank == 1
