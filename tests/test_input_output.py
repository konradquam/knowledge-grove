from sqlalchemy import select

from knowledge_grove.models import Edge
from knowledge_grove.utils.input_output import (
    add_file_as_document,
    add_files_as_documents,
    file_to_string,
)


def test_file_to_string_reads_file_contents(tmp_path):
    path = tmp_path / "doc.md"
    path.write_text("# Heading\n\nsome body text\n")

    assert file_to_string(str(path)) == "# Heading\n\nsome body text\n"


def test_file_to_string_reads_utf8_content(tmp_path):
    path = tmp_path / "doc.md"
    path.write_text("café — naïve 字\n", encoding="utf-8")

    assert file_to_string(str(path)) == "café — naïve 字\n"


def test_add_file_as_document_chunks_and_creates_prev_edges(alice, tmp_path):
    path = tmp_path / "doc.md"
    path.write_text("# Heading One\n\npara one\n\npara two\n")

    docs = add_file_as_document(alice, str(path), owner_agent="agent_alice")
    alice.commit()

    assert [d.content for d in docs] == ["# Heading One\n\npara one", "para two"]

    edge = alice.scalars(select(Edge).where(Edge.from_document_id == docs[1].id)).one()
    assert edge.to_document_id == docs[0].id
    assert edge.edge_type == "prev"


def test_add_file_as_document_auto_embeds(alice, embedding_model, tmp_path):
    path = tmp_path / "doc.md"
    path.write_text("just one paragraph\n")

    docs = add_file_as_document(alice, str(path), owner_agent="agent_alice")
    alice.commit()

    assert docs[0].embedding is not None
    assert len(docs[0].embedding) == len(embedding_model.embed_text("anything"))


def test_add_file_as_document_sets_owner_agent(alice, tmp_path):
    path = tmp_path / "doc.md"
    path.write_text("para one\n\npara two\n")

    docs = add_file_as_document(alice, str(path), owner_agent="agent_alice")
    alice.commit()

    assert all(doc.owner_agent == "agent_alice" for doc in docs)


def test_add_file_as_document_source_url_defaults_to_none(alice, tmp_path):
    path = tmp_path / "doc.md"
    path.write_text("just one paragraph\n")

    docs = add_file_as_document(alice, str(path), owner_agent="agent_alice")
    alice.commit()

    assert all(doc.source_url is None for doc in docs)


def test_add_file_as_document_uses_given_source_url(alice, tmp_path):
    path = tmp_path / "doc.md"
    path.write_text("para one\n\npara two\n")

    docs = add_file_as_document(
        alice, str(path), owner_agent="agent_alice", source_url="https://example.com/doc.md"
    )
    alice.commit()

    assert all(doc.source_url == "https://example.com/doc.md" for doc in docs)


def test_add_file_as_document_single_chunk_creates_no_edges(alice, tmp_path):
    path = tmp_path / "doc.md"
    path.write_text("just one paragraph, no breaks\n")

    docs = add_file_as_document(alice, str(path), owner_agent="agent_alice")
    alice.commit()

    assert len(docs) == 1
    assert alice.scalars(select(Edge).where(Edge.from_document_id == docs[0].id)).all() == []


def test_add_files_as_documents_returns_one_sublist_per_file(alice, tmp_path):
    path_a = tmp_path / "a.md"
    path_a.write_text("# Doc A\n\npara a\n")
    path_b = tmp_path / "b.md"
    path_b.write_text("# Doc B\n\npara b\n")

    result = add_files_as_documents(alice, [str(path_a), str(path_b)], owner_agent="agent_alice")
    alice.commit()

    assert len(result) == 2
    assert [d.content for d in result[0]] == ["# Doc A\n\npara a"]
    assert [d.content for d in result[1]] == ["# Doc B\n\npara b"]


def test_add_files_as_documents_applies_source_urls_per_file(alice, tmp_path):
    path_a = tmp_path / "a.md"
    path_a.write_text("para a\n")
    path_b = tmp_path / "b.md"
    path_b.write_text("para b\n")

    result = add_files_as_documents(
        alice,
        [str(path_a), str(path_b)],
        owner_agent="agent_alice",
        source_urls=["https://example.com/a.md", "https://example.com/b.md"],
    )
    alice.commit()

    assert all(doc.source_url == "https://example.com/a.md" for doc in result[0])
    assert all(doc.source_url == "https://example.com/b.md" for doc in result[1])


def test_add_files_as_documents_source_urls_default_to_none_when_omitted(alice, tmp_path):
    path_a = tmp_path / "a.md"
    path_a.write_text("para a\n")
    path_b = tmp_path / "b.md"
    path_b.write_text("para b\n")

    result = add_files_as_documents(alice, [str(path_a), str(path_b)], owner_agent="agent_alice")
    alice.commit()

    assert all(doc.source_url is None for chunks in result for doc in chunks)


def test_add_files_as_documents_empty_list_returns_empty(alice):
    assert add_files_as_documents(alice, [], owner_agent="agent_alice") == []
