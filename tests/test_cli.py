import sys

import pytest
from sqlalchemy import create_engine, text
from testcontainers.community.postgres import PostgresContainer

from knowledge_grove import cli, crud
from knowledge_grove.db import get_engine, get_session

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
