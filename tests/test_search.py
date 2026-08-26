from knowledge_grove import crud, search

from conftest import vec


def test_search_vector_ranks_nearest_first(alice):
    near = crud.add_document(
        alice, content="near match", embedding=vec(0), owner_agent="agent_alice"
    )
    far = crud.add_document(
        alice, content="far match", embedding=vec(5), owner_agent="agent_alice"
    )
    alice.commit()

    results = search.search_vector(alice, vec(0), limit=5)
    ids_in_order = [doc.id for doc, _score in results]

    assert ids_in_order.index(near.id) < ids_in_order.index(far.id)


def test_search_vector_excludes_deprecated_by_default(alice):
    original = crud.add_document(
        alice, content="v1", embedding=vec(0), owner_agent="agent_alice"
    )
    alice.commit()
    crud.update_document(alice, original.id, content="v2", embedding=vec(0))
    alice.commit()

    results = search.search_vector(alice, vec(0), limit=5)
    assert original.id not in [doc.id for doc, _ in results]

    results_incl = search.search_vector(alice, vec(0), limit=5, include_deprecated=True)
    assert original.id in [doc.id for doc, _ in results_incl]


def test_search_tags_matches_and_normalizes(alice):
    doc = crud.add_document(
        alice, content="tagged", embedding=vec(0), owner_agent="agent_alice"
    )
    alice.commit()
    crud.add_tag(alice, doc.id, "retries", "retry policy")
    alice.commit()

    results = search.search_tags(alice, "  RETRIES  ")
    assert doc.id in [d.id for d in results]


def test_search_tags_no_match_returns_empty(alice):
    doc = crud.add_document(
        alice, content="untagged", embedding=vec(0), owner_agent="agent_alice"
    )
    alice.commit()

    assert search.search_tags(alice, "nonexistent") == []


def test_search_tags_excludes_deprecated_by_default(alice):
    original = crud.add_document(
        alice, content="v1", embedding=vec(0), owner_agent="agent_alice"
    )
    alice.commit()
    crud.add_tag(alice, original.id, "retries", "retry policy")
    alice.commit()
    crud.update_document(alice, original.id, content="v2", embedding=vec(0))
    alice.commit()

    results = search.search_tags(alice, "retries")
    assert original.id not in [d.id for d in results]


def test_search_fulltext_matches_stemmed_terms(alice):
    doc = crud.add_document(
        alice,
        content="Retries should use exponential backoff.",
        embedding=vec(0),
        owner_agent="agent_alice",
    )
    alice.commit()

    # "retry" should match "Retries" via stemming
    results = search.search_fulltext(alice, "retry backoff")
    assert doc.id in [d.id for d, _score in results]


def test_search_fulltext_no_match_returns_empty(alice):
    crud.add_document(
        alice, content="unrelated content", embedding=vec(0), owner_agent="agent_alice"
    )
    alice.commit()

    assert search.search_fulltext(alice, "nonexistent phrase xyz") == []


def test_search_ilike_matches_literal_substring(alice):
    doc = crud.add_document(
        alice,
        content="Configure RETRY_TIMEOUT_MS before deploying.",
        embedding=vec(0),
        owner_agent="agent_alice",
    )
    alice.commit()

    results = search.search_ilike(alice, "RETRY_TIMEOUT_MS")
    assert doc.id in [d.id for d, _score in results]


def test_search_ilike_treats_underscore_as_literal_not_wildcard(alice):
    exact = crud.add_document(
        alice,
        content="Call get_user(id) to fetch the record.",
        embedding=vec(0),
        owner_agent="agent_alice",
    )
    decoy = crud.add_document(
        alice,
        content="Call getXuser(id) to fetch the record.",
        embedding=vec(1),
        owner_agent="agent_alice",
    )
    alice.commit()

    results = search.search_ilike(alice, "get_user")
    ids = [d.id for d, _score in results]

    assert exact.id in ids
    assert decoy.id not in ids


def test_search_ilike_no_match_returns_empty(alice):
    crud.add_document(
        alice, content="unrelated content", embedding=vec(0), owner_agent="agent_alice"
    )
    alice.commit()

    assert search.search_ilike(alice, "NONEXISTENT_TOKEN") == []
