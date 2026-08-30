"""Bootstrapping commands (§14 of the design doc): `init-db` and
`create-agent-role`. These are ops-facing, one-time-per-database/per-agent
setup steps -- not part of the runtime SDK agents call.
"""
import argparse
import getpass
import os
from pathlib import Path

from alembic import command
from alembic.config import Config
from psycopg import sql
from sqlalchemy import create_engine
from sqlalchemy.engine import make_url

import knowledge_grove
from knowledge_grove.constants import SHARED_READER


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

    args = parser.parse_args()

    dsn = os.environ.get("KNOWLEDGE_GROVE_DSN")
    if not dsn:
        parser.error("KNOWLEDGE_GROVE_DSN must be set to an admin/setup connection string")

    if args.command == "init-db":
        init_db(dsn)
        print("Migrations applied.")
    elif args.command == "create-agent-role":
        password = args.password or getpass.getpass(f"Password for {args.role_name}: ")
        agent_dsn = create_agent_role(dsn, args.role_name, password)
        print(f"Role '{args.role_name}' created.")
        print(f"Agent DSN: {agent_dsn}")


if __name__ == "__main__":
    main()
