"""CLI commands: bootstrapping (`init-db`, `create-agent-role`, §14 of the
design doc, ops-facing one-time-per-database/per-agent setup) and content
ingestion (`ingest`, a thin wrapper over add_file_as_document/§13 dedup for
use from a shell rather than another agent's own code).
"""
import argparse
import getpass
import os
from pathlib import Path

from alembic import command
from alembic.config import Config
from psycopg import sql
from sqlalchemy import create_engine, func, select, text
from sqlalchemy.engine import make_url

import knowledge_grove
from knowledge_grove.constants import SHARED_READER
from knowledge_grove.db import get_engine, get_session
from knowledge_grove.models import Document
from knowledge_grove.utils.input_output import add_file_as_document


def _alembic_config(dsn: str) -> Config:
    """Build an Alembic Config pointing at this package's own bundled
    migrations directly, rather than relying on alembic.ini being present on
    disk -- that file lives at the repo root, outside the installed package,
    so it won't exist for a real (non-editable) install of knowledge-grove
    used from another project.
    """
    migrations_dir = Path(knowledge_grove.__file__).parent / "migrations"
    cfg = Config()
    cfg.set_main_option("script_location", str(migrations_dir))
    cfg.set_main_option("sqlalchemy.url", dsn)
    return cfg


def init_db(dsn: str) -> None:
    """Run every bundled migration up to head against `dsn`.

    Must be run by a role with CREATE EXTENSION / CREATE POLICY / table-owner
    privileges -- never the role an ordinary agent connects as, since table
    owners bypass RLS by default (see the initial migration's own docstring).
    """
    command.upgrade(_alembic_config(dsn), "head")


def create_agent_role(dsn: str, role_name: str, password: str) -> str:
    """Provision a new agent's Postgres role: LOGIN credentials, the base
    grants every agent needs, and membership in `shared_reader` so it can
    read whatever's been shared into that group by default.

    `dsn` must belong to a role with privileges to create roles and grant
    table access (the same setup-only role `init_db` requires), not an
    ordinary agent role. Returns the DSN the new agent should connect with.

    Role names and the password can't be passed as ordinary bind parameters
    -- CREATE ROLE / GRANT are utility statements, not DML, so Postgres
    doesn't accept protocol-level placeholders for them. `psycopg.sql`
    composes them safely instead (proper identifier quoting for the role
    name, proper literal escaping for the password).
    """
    engine = create_engine(dsn)
    try:
        with engine.begin() as conn:
            cur = conn.connection.dbapi_connection.cursor()
            cur.execute(
                sql.SQL("CREATE ROLE {} LOGIN PASSWORD {}").format(
                    sql.Identifier(role_name), sql.Literal(password)
                )
            )
            cur.execute(
                sql.SQL(
                    "GRANT SELECT, INSERT, UPDATE, DELETE ON "
                    "documents, document_tags, edges, document_access TO {}"
                ).format(sql.Identifier(role_name))
            )
            cur.execute(
                sql.SQL("GRANT SELECT, INSERT ON retrieval_feedback TO {}").format(
                    sql.Identifier(role_name)
                )
            )
            cur.execute(
                sql.SQL("GRANT {} TO {}").format(
                    sql.Identifier(SHARED_READER), sql.Identifier(role_name)
                )
            )
    finally:
        engine.dispose()

    agent_url = make_url(dsn).set(username=role_name, password=password)
    return agent_url.render_as_string(hide_password=False)


def _current_agent(session) -> str:
    """The Postgres role this connection is actually authenticated as -- see
    mcp_server.py's identical helper for why document ownership always comes
    from the connection itself, never a caller-supplied argument.
    """
    return session.execute(text("SELECT current_user")).scalar()


def _active_document_count(session, source_url: str) -> int:
    return session.scalar(
        select(func.count()).select_from(Document).where(
            Document.source_url == source_url, Document.deprecated.is_(False)
        )
    )


def ingest_files(
    dsn: str,
    file_paths: list[str],
    source_urls: list[str | None] | None = None,
    assume_yes: bool = False,
) -> None:
    """Ingest one or more files as documents, chunking each and computing
    embeddings (add_file_as_document). `dsn` should be an ordinary agent's
    connection string, not the admin/setup one `init_db`/`create_agent_role`
    need -- ownership is derived from whichever role `dsn` authenticates as.

    Each file's `source_url` defaults to its own filename (not a real URL,
    just a stable identifier re-ingesting the same file later will match
    again -- see add_raw_document's §13 reconciliation). If a source_url
    already has existing documents, this asks for confirmation before
    proceeding (unless `assume_yes`): re-using an existing source_url for a
    genuinely different file would make add_raw_document treat it as a
    changed revision and deprecate that unrelated content.
    """
    resolved_source_urls = [
        source_urls[i] if source_urls and source_urls[i] else Path(file_paths[i]).name
        for i in range(len(file_paths))
    ]

    engine = get_engine(dsn)
    session = get_session(engine)
    try:
        owner_agent = _current_agent(session)

        for path, source_url in zip(file_paths, resolved_source_urls):
            existing_count = _active_document_count(session, source_url)
            if existing_count > 0 and not assume_yes:
                print(
                    f"Warning: {existing_count} existing document(s) already use "
                    f"source_url '{source_url}'. Continuing will treat '{path}' as a "
                    f"new revision of that same source: identical content is a "
                    f"no-op, but different content will deprecate the existing "
                    f"chunks. If '{path}' is not actually a revision of that source, "
                    f"answer no and re-run with a different --source-url."
                )
                answer = input("Proceed? [y/N] ").strip().lower()
                if answer != "y":
                    print(f"Skipped '{path}'.")
                    continue

            docs = add_file_as_document(session, path, owner_agent=owner_agent, source_url=source_url)
            session.commit()
            print(f"Ingested '{path}' as {len(docs)} chunk(s) under source_url '{source_url}'.")
    finally:
        session.close()
        engine.dispose()


def main() -> None:
    parser = argparse.ArgumentParser(prog="knowledge-grove")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser(
        "init-db", help="Run the bundled migrations against KNOWLEDGE_GROVE_DSN."
    )

    role_parser = subparsers.add_parser(
        "create-agent-role", help="Provision a new agent's Postgres role."
    )
    role_parser.add_argument("role_name")
    role_parser.add_argument(
        "--password", help="If omitted, you'll be prompted (not echoed)."
    )

    ingest_parser = subparsers.add_parser(
        "ingest", help="Add one or more files as documents, chunking each and computing embeddings."
    )
    ingest_parser.add_argument("files", nargs="+", help="Path(s) to the file(s) to ingest.")
    ingest_parser.add_argument(
        "--source-url", action="append", default=None,
        help=(
            "Source identifier, given once per file in the same order as `files`. "
            "Defaults to each file's own filename (not a real URL -- just a stable "
            "identifier so re-ingesting the same file later is recognized as an "
            "update rather than a new, unrelated document)."
        ),
    )
    ingest_parser.add_argument(
        "-y", "--yes", action="store_true",
        help="Don't prompt for confirmation when a source_url already has existing documents.",
    )

    args = parser.parse_args()

    dsn = os.environ.get("KNOWLEDGE_GROVE_DSN")
    if not dsn:
        parser.error("KNOWLEDGE_GROVE_DSN must be set to a connection string")

    if args.command == "init-db":
        init_db(dsn)
        print("Migrations applied.")
    elif args.command == "create-agent-role":
        password = args.password or getpass.getpass(f"Password for {args.role_name}: ")
        agent_dsn = create_agent_role(dsn, args.role_name, password)
        print(f"Role '{args.role_name}' created.")
        print(f"Agent DSN: {agent_dsn}")
    elif args.command == "ingest":
        if args.source_url and len(args.source_url) != len(args.files):
            parser.error("--source-url must be given once per file, or omitted entirely")
        ingest_files(dsn, args.files, source_urls=args.source_url, assume_yes=args.yes)


if __name__ == "__main__":
    main()
