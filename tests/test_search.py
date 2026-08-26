from knowledge_grove import crud, search

from conftest import embedding_model, vec


def test_search_vector_ranks_nearest_first(alice, embedding_model):
    near = crud.add_document(
        alice, content="near match", embedding=embedding_model.embed_text("near match"), owner_agent="agent_alice"
    )
    far = crud.add_document(
        alice, content="far match", embedding=embedding_model.embed_text("far match"), owner_agent="agent_alice"
    )
    alice.commit()

    results = search.search_vector(alice, embedding_model.embed_text("near match"), limit=5)
    ids_in_order = [doc.id for doc, _score in results]

    assert ids_in_order.index(near.id) < ids_in_order.index(far.id)


def test_search_vector_excludes_deprecated_by_default(alice, embedding_model):
    original = crud.add_document(
        alice, content="v1", embedding=embedding_model.embed_text("v1"), owner_agent="agent_alice"
    )
    alice.commit()
    crud.update_document(alice, original.id, content="v2", embedding=embedding_model.embed_text("v2"))
    alice.commit()

    results = search.search_vector(alice, embedding_model.embed_text("v1"), limit=5)
    assert original.id not in [doc.id for doc, _ in results]

    results_incl = search.search_vector(alice, embedding_model.embed_text("v1"), limit=5, include_deprecated=True)
    assert original.id in [doc.id for doc, _ in results_incl]


def test_search_vector_filters_by_tags(alice):
    tagged = crud.add_document(
        alice, content="near match", embedding=vec(0), owner_agent="agent_alice"
    )
    crud.add_tag(alice, tagged.id, "retries", "retry policy")
    untagged = crud.add_document(
        alice, content="near match too", embedding=vec(0), owner_agent="agent_alice"
    )
    alice.commit()

    results = search.search_vector(alice, vec(0), tags=["retries"])
    result_ids = [doc.id for doc, _score in results]

    assert tagged.id in result_ids
    assert untagged.id not in result_ids


def test_search_vector_tags_no_match_returns_empty(alice):
    crud.add_document(
        alice, content="near match", embedding=vec(0), owner_agent="agent_alice"
    )
    alice.commit()

    assert search.search_vector(alice, vec(0), tags=["nonexistent"]) == []


def test_search_tag_matches_and_normalizes(alice):
    doc = crud.add_document(
        alice, content="tagged", embedding=vec(0), owner_agent="agent_alice"
    )
    alice.commit()
    crud.add_tag(alice, doc.id, "retries", "retry policy")
    alice.commit()

    results = search.search_tag(alice, "  RETRIES  ")
    assert doc.id in [d.id for d in results]


def test_search_tag_no_match_returns_empty(alice):
    doc = crud.add_document(
        alice, content="untagged", embedding=vec(0), owner_agent="agent_alice"
    )
    alice.commit()

    assert search.search_tag(alice, "nonexistent") == []


def test_search_tag_excludes_deprecated_by_default(alice):
    original = crud.add_document(
        alice, content="v1", embedding=vec(0), owner_agent="agent_alice"
    )
    alice.commit()
    crud.add_tag(alice, original.id, "retries", "retry policy")
    alice.commit()
    crud.update_document(alice, original.id, content="v2", embedding=vec(0))
    alice.commit()

    results = search.search_tag(alice, "retries")
    assert original.id not in [d.id for d in results]


def test_search_tags_matches_any_of_multiple_tags(alice):
    doc = crud.add_document(
        alice, content="tagged", embedding=vec(0), owner_agent="agent_alice"
    )
    alice.commit()
    crud.add_tag(alice, doc.id, "retries", "retry policy")
    alice.commit()

    results = search.search_tags(alice, ["config", "retries"])
    assert doc.id in [d.id for d in results]


def test_search_tags_deduplicates_document_matching_multiple_tags(alice):
    doc = crud.add_document(
        alice, content="tagged twice", embedding=vec(0), owner_agent="agent_alice"
    )
    alice.commit()
    crud.add_tag(alice, doc.id, "retries", "retry policy")
    crud.add_tag(alice, doc.id, "config", "config knob")
    alice.commit()

    results = search.search_tags(alice, ["config", "retries"])
    assert [d.id for d in results].count(doc.id) == 1


def test_search_tags_no_match_returns_empty(alice):
    doc = crud.add_document(
        alice, content="untagged", embedding=vec(0), owner_agent="agent_alice"
    )
    alice.commit()

    assert search.search_tags(alice, ["nonexistent", "also-nonexistent"]) == []


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


def test_search_fulltext_filters_by_tags(alice):
    tagged = crud.add_document(
        alice,
        content="Retries should use exponential backoff.",
        embedding=vec(0),
        owner_agent="agent_alice",
    )
    crud.add_tag(alice, tagged.id, "retries", "retry policy")
    untagged = crud.add_document(
        alice,
        content="Retries should use linear backoff.",
        embedding=vec(1),
        owner_agent="agent_alice",
    )
    alice.commit()

    results = search.search_fulltext(alice, "retry backoff", tags=["retries"])
    result_ids = [doc.id for doc, _score in results]

    assert tagged.id in result_ids
    assert untagged.id not in result_ids


def test_search_fulltext_tags_no_match_returns_empty(alice):
    crud.add_document(
        alice,
        content="Retries should use exponential backoff.",
        embedding=vec(0),
        owner_agent="agent_alice",
    )
    alice.commit()

    assert search.search_fulltext(alice, "retry backoff", tags=["nonexistent"]) == []


def test_search_fulltext_correct_ranking(alice):
    top_doc =crud.add_document(
        alice, content="This is a test document and is the best among all test documents.", 
        embedding=vec(0), owner_agent="agent_alice"
    )
    middle_doc = crud.add_document(
        alice, content="this is also a test document.", embedding=vec(1), owner_agent="agent_alice"
    )
    bottom_doc = crud.add_document(
        alice, content="this is a test for a document.", embedding=vec(2), owner_agent="agent_alice"
    )
    irrelevant_doc = crud.add_document(
        alice, content="completely unrelated content.", embedding=vec(2), owner_agent="agent_alice"
    )
    alice.commit()

    results = search.search_fulltext(alice, "test document")
    result_ids = [doc.id for doc, _score in results]

    assert top_doc.id == result_ids[0]
    assert middle_doc.id == result_ids[1]
    assert bottom_doc.id == result_ids[2]
    assert results[0][1] > results[1][1]  # top_doc should have a higher score than middle_doc
    assert results[1][1] > results[2][1]  # middle_doc should have a higher score than bottom_doc
    assert irrelevant_doc.id not in result_ids

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


def test_search_ilike_filters_by_tags(alice):
    tagged = crud.add_document(
        alice,
        content="Configure RETRY_TIMEOUT_MS before deploying.",
        embedding=vec(0),
        owner_agent="agent_alice",
    )
    crud.add_tag(alice, tagged.id, "config", "config knob")
    untagged = crud.add_document(
        alice,
        content="Also configure RETRY_TIMEOUT_MS carefully.",
        embedding=vec(1),
        owner_agent="agent_alice",
    )
    alice.commit()

    results = search.search_ilike(alice, "RETRY_TIMEOUT_MS", tags=["config"])
    result_ids = [doc.id for doc, _score in results]

    assert tagged.id in result_ids
    assert untagged.id not in result_ids


def test_search_ilike_tags_no_match_returns_empty(alice):
    crud.add_document(
        alice,
        content="Configure RETRY_TIMEOUT_MS before deploying.",
        embedding=vec(0),
        owner_agent="agent_alice",
    )
    alice.commit()

    assert search.search_ilike(alice, "RETRY_TIMEOUT_MS", tags=["nonexistent"]) == []


def test_search_ilike_ranking(alice):
    top_doc = crud.add_document(
        alice, content="This is a test document.", 
        embedding=vec(0), owner_agent="agent_alice"
    )
    middle_doc = crud.add_document(
        alice, content="This is a test document and is the second best among all test documents..", 
        embedding=vec(1), owner_agent="agent_alice"
    )
    bottom_doc = crud.add_document(
        alice, content="this is also a test document but it is not as good for Ilike because of length.", 
        embedding=vec(2), owner_agent="agent_alice"
    )
    irrelevant_doc = crud.add_document(
        alice, content="completely unrelated content.", embedding=vec(2), owner_agent="agent_alice"
    )
    alice.commit()

    results = search.search_ilike(alice, "test document")
    result_ids = [doc.id for doc, _score in results]

    assert top_doc.id == result_ids[0]
    assert middle_doc.id == result_ids[1]
    assert bottom_doc.id == result_ids[2]
    assert results[0][1] > results[1][1]  # top_doc should have a higher score than middle_doc
    assert results[1][1] > results[2][1]  # middle_doc should have a higher score than bottom_doc
    assert irrelevant_doc.id not in result_ids

def test_search_merges_results_from_multiple_methods(alice, embedding_model):
    doc_vector = crud.add_document(
        alice, content="vector match", embedding=embedding_model.embed_text("vector match"), owner_agent="agent_alice"
    )
    doc_fulltext = crud.add_document(
        alice, content="fulltext match", embedding=embedding_model.embed_text("fulltext match"), owner_agent="agent_alice"
    )
    doc_ilike = crud.add_document(
        alice, content="ilike match", embedding=embedding_model.embed_text("ilike match"), owner_agent="agent_alice"
    )
    alice.commit()

    results = search.weighted_search(
        session=alice,
        query_embedding=embedding_model.embed_text("fulltext"),
        query_text="fulltext",
        pattern="ilike",
        limit=10,
    )

    # Ensure all three documents are in the results
    result_ids = [doc.id for doc, _score in results]
    assert doc_vector.id in result_ids
    assert doc_fulltext.id in result_ids
    assert doc_ilike.id in result_ids

def test_search_ranks_merged_results_from_multiple_methods(alice, embedding_model):
    best_doc = crud.add_document(
        alice, content="here is the top ranking text", embedding=embedding_model.embed_text("here is the top ranking text"), owner_agent="agent_alice"
    )
    second_best_doc = crud.add_document(
        alice, content="second ranking text", embedding=embedding_model.embed_text("second ranking text"), owner_agent="agent_alice"
    )
    third_best_doc = crud.add_document(
        alice, content="irrelevant text", embedding=embedding_model.embed_text("irrelevant text"), owner_agent="agent_alice"
    )
    alice.commit()

    results = search.weighted_search(
        session=alice,
        query_embedding=embedding_model.embed_text("what is the top ranking text"),
        query_text="what is the top ranking text",
        pattern="ranking text",
        limit=10,
    )

    # Ensure all three documents are in the results
    result_ids = [doc.id for doc, _score in results]
    assert best_doc.id == result_ids[0]
    assert second_best_doc.id == result_ids[1]
    assert third_best_doc.id == result_ids[2]

def test_search_ranks_2_merged_results_from_multiple_methods(alice, embedding_model):
    best_doc = crud.add_document(
        alice, content="here is the top ranking text", embedding=embedding_model.embed_text("here is the top ranking text"), owner_agent="agent_alice"
    )
    second_best_doc = crud.add_document(
        alice, content="second ranking text", embedding=embedding_model.embed_text("second ranking text"), owner_agent="agent_alice"
    )
    third_best_doc = crud.add_document(
        alice, content="irrelevant text", embedding=embedding_model.embed_text("irrelevant text"), owner_agent="agent_alice"
    )
    alice.commit()

    results = search.weighted_search(
        session=alice,
        query_embedding=embedding_model.embed_text("what is the top ranking text"),
        query_text="what is the top ranking text",
        pattern="top ranking",
        limit=10,
    )

    # Ensure all three documents are in the results
    result_ids = [doc.id for doc, _score in results]
    assert best_doc.id == result_ids[0]
    assert second_best_doc.id == result_ids[1]
    assert third_best_doc.id == result_ids[2]