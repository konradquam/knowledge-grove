import sys

import pytest
from sqlalchemy import create_engine, select, text
from testcontainers.community.postgres import PostgresContainer

from knowledge_grove import cli, crud
from knowledge_grove.db import get_engine, get_session
from knowledge_grove.models import Document

PG_IMAGE = "pgvector/pgvector:pg16"


def _drop_agent_role(dsn, role_name):
    engine = create_engine(dsn)
    with engine.begin() as conn:
        conn.execute(text(
            f"REVOKE ALL ON documents, document_tags, edges, document_access, "
            f"retrieval_feedback FROM {role_name}"
        ))
        conn.execute(text(f"REVOKE shared_reader FROM {role_name}"))
        conn.execute(text(f"DROP ROLE {role_name}"))
    engine.dispose()


@pytest.fixture()
def bare_dsn():
    """A fresh Postgres container with no migrations applied -- for testing
    init_db itself, which needs to start from nothing."""
    with PostgresContainer(PG_IMAGE, driver="psycopg") as container:
        yield container.get_connection_url()


def test_init_db_creates_full_schema(bare_dsn):
    cli.init_db(bare_dsn)

    engine = create_engine(bare_dsn)
    with engine.connect() as conn:
        tables = set(
            conn.execute(
                text("SELECT tablename FROM pg_tables WHERE schemaname = 'public'")
            ).scalars()
        )
        assert {
            "documents", "document_tags", "edges", "document_access", "retrieval_feedback",
        } <= tables

        shared_reader_exists = conn.execute(
            text("SELECT 1 FROM pg_roles WHERE rolname = 'shared_reader'")
        ).scalar()
        assert shared_reader_exists == 1
    engine.dispose()


def test_init_db_is_idempotent(bare_dsn):
    cli.init_db(bare_dsn)
    cli.init_db(bare_dsn)  # must not raise


def test_create_agent_role_grants_base_privileges_and_shared_reader_membership(admin_dsn):
    cli.create_agent_role(admin_dsn, "agent_cli_test", "cli_test_pass")

    engine = create_engine(admin_dsn)
    with engine.connect() as conn:
        is_member = conn.execute(
            text("SELECT pg_has_role('agent_cli_test', 'shared_reader', 'MEMBER')")
        ).scalar()
        assert is_member is True

        privileges = set(
            conn.execute(
                text(
                    "SELECT table_name || ':' || privilege_type "
                    "FROM information_schema.role_table_grants WHERE grantee = 'agent_cli_test'"
                )
            ).scalars()
        )
        for table in ("documents", "document_tags", "edges", "document_access"):
            for perm in ("SELECT", "INSERT", "UPDATE", "DELETE"):
                assert f"{table}:{perm}" in privileges
        assert "retrieval_feedback:SELECT" in privileges
        assert "retrieval_feedback:INSERT" in privileges
    engine.dispose()

    _drop_agent_role(admin_dsn, "agent_cli_test")


def test_create_agent_role_returned_dsn_actually_works(admin_dsn):
    agent_dsn = cli.create_agent_role(admin_dsn, "agent_cli_roundtrip", "roundtrip_pass")

    assert "agent_cli_roundtrip" in agent_dsn
    assert "roundtrip_pass" in agent_dsn

    engine = get_engine(agent_dsn)
    session = get_session(engine)
    try:
        doc = crud.add_document(
            session, content="hello from a cli-provisioned role", owner_agent="agent_cli_roundtrip",
        )
        session.commit()
        found = crud.get_by_id(session, doc.id)
        assert found.content == "hello from a cli-provisioned role"
    finally:
        session.close()
        engine.dispose()

    admin_engine = create_engine(admin_dsn)
    with admin_engine.begin() as conn:
        conn.execute(text("TRUNCATE documents CASCADE"))
    admin_engine.dispose()
    _drop_agent_role(admin_dsn, "agent_cli_roundtrip")


def test_create_agent_role_escapes_password_with_special_characters(admin_dsn):
    tricky_password = "o'reilly\"s;--pass"
    agent_dsn = cli.create_agent_role(admin_dsn, "agent_cli_tricky", tricky_password)

    engine = get_engine(agent_dsn)
    session = get_session(engine)
    try:
        doc = crud.add_document(session, content="tricky password worked", owner_agent="agent_cli_tricky")
        session.commit()
        assert crud.get_by_id(session, doc.id).content == "tricky password worked"
    finally:
        session.close()
        engine.dispose()

    admin_engine = create_engine(admin_dsn)
    with admin_engine.begin() as conn:
        conn.execute(text("TRUNCATE documents CASCADE"))
    admin_engine.dispose()
    _drop_agent_role(admin_dsn, "agent_cli_tricky")


def test_main_init_db_subcommand_runs_migrations(bare_dsn, monkeypatch):
    monkeypatch.setenv("KNOWLEDGE_GROVE_DSN", bare_dsn)
    monkeypatch.setattr(sys, "argv", ["knowledge-grove", "init-db"])

    cli.main()

    engine = create_engine(bare_dsn)
    with engine.connect() as conn:
        tables = set(
            conn.execute(
                text("SELECT tablename FROM pg_tables WHERE schemaname = 'public'")
            ).scalars()
        )
        assert "documents" in tables
    engine.dispose()


def test_main_requires_dsn_env_var(monkeypatch):
    monkeypatch.delenv("KNOWLEDGE_GROVE_DSN", raising=False)
    monkeypatch.setattr(sys, "argv", ["knowledge-grove", "init-db"])

    with pytest.raises(SystemExit):
        cli.main()


def test_main_create_agent_role_subcommand_with_password_flag(admin_dsn, monkeypatch, capsys):
    monkeypatch.setenv("KNOWLEDGE_GROVE_DSN", admin_dsn)
    monkeypatch.setattr(
        sys, "argv",
        ["knowledge-grove", "create-agent-role", "agent_cli_main_flag", "--password", "mainpass"],
    )

    cli.main()

    captured = capsys.readouterr()
    assert "agent_cli_main_flag" in captured.out
    assert "mainpass" in captured.out

    engine = create_engine(admin_dsn)
    with engine.connect() as conn:
        is_member = conn.execute(
            text("SELECT pg_has_role('agent_cli_main_flag', 'shared_reader', 'MEMBER')")
        ).scalar()
        assert is_member is True
    engine.dispose()

    _drop_agent_role(admin_dsn, "agent_cli_main_flag")


def test_main_create_agent_role_prompts_for_password_when_flag_omitted(admin_dsn, monkeypatch):
    monkeypatch.setenv("KNOWLEDGE_GROVE_DSN", admin_dsn)
    monkeypatch.setattr(sys, "argv", ["knowledge-grove", "create-agent-role", "agent_cli_main_prompt"])
    monkeypatch.setattr(cli.getpass, "getpass", lambda prompt="": "prompted_pass")

    cli.main()

    engine = create_engine(admin_dsn)
    with engine.connect() as conn:
        exists = conn.execute(
            text("SELECT 1 FROM pg_roles WHERE rolname = 'agent_cli_main_prompt'")
        ).scalar()
        assert exists == 1
    engine.dispose()

    _drop_agent_role(admin_dsn, "agent_cli_main_prompt")


@pytest.fixture()
def ingest_agent(admin_dsn):
    """A provisioned agent role + its DSN, for ingest_files tests."""
    role_name = "agent_cli_ingest"
    agent_dsn = cli.create_agent_role(admin_dsn, role_name, "ingest_pass")

    yield agent_dsn, role_name

    admin_engine = create_engine(admin_dsn)
    with admin_engine.begin() as conn:
        conn.execute(text("TRUNCATE documents CASCADE"))
    admin_engine.dispose()
    _drop_agent_role(admin_dsn, role_name)


def _all_documents(admin_dsn):
    engine = create_engine(admin_dsn)
    with engine.connect() as conn:
        docs = conn.execute(
            select(Document.content, Document.source_url, Document.owner_agent, Document.deprecated)
        ).all()
    engine.dispose()
    return docs


def test_ingest_files_defaults_source_url_to_filename(admin_dsn, ingest_agent, tmp_path):
    agent_dsn, role_name = ingest_agent
    file_path = tmp_path / "notes.md"
    file_path.write_text("just one paragraph, no breaks\n")

    cli.ingest_files(agent_dsn, [str(file_path)])

    docs = _all_documents(admin_dsn)
    assert len(docs) == 1
    assert docs[0].source_url == "notes.md"
    assert docs[0].owner_agent == role_name
    assert docs[0].content == "just one paragraph, no breaks"


def test_ingest_files_uses_explicit_source_url(admin_dsn, ingest_agent, tmp_path):
    agent_dsn, _ = ingest_agent
    file_path = tmp_path / "notes.md"
    file_path.write_text("content\n")

    cli.ingest_files(agent_dsn, [str(file_path)], source_urls=["https://example.com/notes"])

    docs = _all_documents(admin_dsn)
    assert docs[0].source_url == "https://example.com/notes"


def test_ingest_files_chunks_the_file(admin_dsn, ingest_agent, tmp_path):
    agent_dsn, _ = ingest_agent
    file_path = tmp_path / "notes.md"
    file_path.write_text("# Heading\n\npara one\n\npara two\n")

    cli.ingest_files(agent_dsn, [str(file_path)])

    docs = _all_documents(admin_dsn)
    assert [d.content for d in docs] == ["# Heading\n\npara one", "para two"]


def test_ingest_files_processes_multiple_files_with_their_own_filenames(admin_dsn, ingest_agent, tmp_path):
    agent_dsn, _ = ingest_agent
    file_a = tmp_path / "a.md"
    file_a.write_text("content a\n")
    file_b = tmp_path / "b.md"
    file_b.write_text("content b\n")

    cli.ingest_files(agent_dsn, [str(file_a), str(file_b)])

    docs = _all_documents(admin_dsn)
    assert {d.source_url for d in docs} == {"a.md", "b.md"}


def test_ingest_files_warns_when_source_url_already_has_documents(admin_dsn, ingest_agent, tmp_path, monkeypatch, capsys):
    agent_dsn, _ = ingest_agent
    file_path = tmp_path / "notes.md"
    file_path.write_text("content\n")
    cli.ingest_files(agent_dsn, [str(file_path)])

    monkeypatch.setattr("builtins.input", lambda prompt="": "y")
    cli.ingest_files(agent_dsn, [str(file_path)])

    captured = capsys.readouterr()
    assert "Warning" in captured.out
    assert "notes.md" in captured.out


def test_ingest_files_confirming_prompt_over_identical_content_stays_a_noop(admin_dsn, ingest_agent, tmp_path, monkeypatch):
    agent_dsn, _ = ingest_agent
    file_path = tmp_path / "notes.md"
    file_path.write_text("content\n")
    cli.ingest_files(agent_dsn, [str(file_path)])
    first_count = len(_all_documents(admin_dsn))

    monkeypatch.setattr("builtins.input", lambda prompt="": "y")
    cli.ingest_files(agent_dsn, [str(file_path)])

    assert len(_all_documents(admin_dsn)) == first_count


def test_ingest_files_declining_prompt_skips_the_file(admin_dsn, ingest_agent, tmp_path, monkeypatch, capsys):
    agent_dsn, _ = ingest_agent
    file_path = tmp_path / "notes.md"
    file_path.write_text("content\n")
    cli.ingest_files(agent_dsn, [str(file_path)])

    monkeypatch.setattr("builtins.input", lambda prompt="": "n")
    cli.ingest_files(agent_dsn, [str(file_path)])

    captured = capsys.readouterr()
    assert "Skipped" in captured.out


def test_ingest_files_reusing_source_url_for_a_different_file_deprecates_the_old_one_when_confirmed(
    admin_dsn, ingest_agent, tmp_path, monkeypatch
):
    agent_dsn, _ = ingest_agent
    file_a = tmp_path / "a.md"
    file_a.write_text("original content\n")
    cli.ingest_files(agent_dsn, [str(file_a)])

    file_b = tmp_path / "b.md"
    file_b.write_text("replacement content\n")
    monkeypatch.setattr("builtins.input", lambda prompt="": "y")
    cli.ingest_files(agent_dsn, [str(file_b)], source_urls=["a.md"])

    docs = _all_documents(admin_dsn)
    original = next(d for d in docs if d.content == "original content")
    replacement = next(d for d in docs if d.content == "replacement content")
    assert original.deprecated is True
    assert replacement.deprecated is False
    assert replacement.source_url == "a.md"


def test_ingest_files_assume_yes_skips_the_prompt(admin_dsn, ingest_agent, tmp_path, monkeypatch):
    agent_dsn, _ = ingest_agent
    file_path = tmp_path / "notes.md"
    file_path.write_text("content\n")
    cli.ingest_files(agent_dsn, [str(file_path)])

    def _fail_if_called(prompt=""):
        raise AssertionError("input() should not be called when assume_yes=True")

    monkeypatch.setattr("builtins.input", _fail_if_called)
    cli.ingest_files(agent_dsn, [str(file_path)], assume_yes=True)

    # Still a no-op content-wise (identical content), just without prompting.
    assert len(_all_documents(admin_dsn)) == 1


def test_main_ingest_subcommand_end_to_end(admin_dsn, ingest_agent, tmp_path, monkeypatch):
    agent_dsn, role_name = ingest_agent
    file_path = tmp_path / "notes.md"
    file_path.write_text("# Heading\n\npara one\n\npara two\n")

    monkeypatch.setenv("KNOWLEDGE_GROVE_DSN", agent_dsn)
    monkeypatch.setattr(sys, "argv", ["knowledge-grove", "ingest", str(file_path)])

    cli.main()

    docs = _all_documents(admin_dsn)
    assert [d.content for d in docs] == ["# Heading\n\npara one", "para two"]
    assert all(d.owner_agent == role_name for d in docs)


def test_main_ingest_rejects_mismatched_source_url_count(admin_dsn, ingest_agent, tmp_path, monkeypatch):
    agent_dsn, _ = ingest_agent
    file_a = tmp_path / "a.md"
    file_a.write_text("a\n")
    file_b = tmp_path / "b.md"
    file_b.write_text("b\n")

    monkeypatch.setenv("KNOWLEDGE_GROVE_DSN", agent_dsn)
    monkeypatch.setattr(
        sys, "argv",
        ["knowledge-grove", "ingest", str(file_a), str(file_b), "--source-url", "only-one"],
    )

    with pytest.raises(SystemExit):
        cli.main()
