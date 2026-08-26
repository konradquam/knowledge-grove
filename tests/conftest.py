from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url
from testcontainers.community.postgres import PostgresContainer

from knowledge_grove.db import get_engine, get_session

REPO_ROOT = Path(__file__).resolve().parent.parent
PG_IMAGE = "pgvector/pgvector:pg16"


def _dsn_as(base_dsn: str, username: str, password: str) -> str:
    """Same server/database as `base_dsn`, but connecting as a different role.

    Uses render_as_string(hide_password=False) rather than str()/repr() —
    SQLAlchemy's URL deliberately masks the password as `***` for those
    (a sensible default for logging), which would otherwise silently bake
    the literal text "***" in as the password here.
    """
    url = make_url(base_dsn).set(username=username, password=password)
    return url.render_as_string(hide_password=False)


@pytest.fixture(scope="session")
def admin_dsn():
    """A fresh, migrated pgvector/Postgres container, connected as its superuser."""
    with PostgresContainer(PG_IMAGE, driver="psycopg") as container:
        dsn = container.get_connection_url()

        cfg = Config(str(REPO_ROOT / "alembic.ini"))
        cfg.set_main_option("sqlalchemy.url", dsn)
        command.upgrade(cfg, "head")

        yield dsn


@pytest.fixture(scope="session")
def admin_engine(admin_dsn):
    engine = create_engine(admin_dsn)
    yield engine
    engine.dispose()


@pytest.fixture(scope="session")
def test_roles(admin_engine):
    """Two agent roles used throughout the suite, each with ordinary base grants
    (no superuser/BYPASSRLS) so tests exercise RLS the same way a real agent would.
    """
    with admin_engine.begin() as conn:
        conn.execute(text("CREATE ROLE agent_alice LOGIN PASSWORD 'alice'"))
        conn.execute(text("CREATE ROLE agent_bob LOGIN PASSWORD 'bob'"))
        for role in ("agent_alice", "agent_bob"):
            conn.execute(
                text(
                    f"GRANT SELECT, INSERT, UPDATE, DELETE ON "
                    f"documents, document_tags, edges, document_access TO {role}"
                )
            )
            conn.execute(text(f"GRANT SELECT, INSERT ON retrieval_feedback TO {role}"))
    yield


@pytest.fixture(scope="session")
def alice_engine(admin_dsn, test_roles):
    engine = get_engine(_dsn_as(admin_dsn, "agent_alice", "alice"))
    yield engine
    engine.dispose()


@pytest.fixture(scope="session")
def bob_engine(admin_dsn, test_roles):
    engine = get_engine(_dsn_as(admin_dsn, "agent_bob", "bob"))
    yield engine
    engine.dispose()


@pytest.fixture(autouse=True)
def _clean_tables(admin_engine):
    """Truncate all data between tests so each test starts from an empty database.

    Runs as the admin/superuser role (bypasses RLS) since the agent roles only
    have DML grants, not TRUNCATE.
    """
    yield
    with admin_engine.begin() as conn:
        conn.execute(text("TRUNCATE documents CASCADE"))


@pytest.fixture()
def alice(alice_engine):
    session = get_session(alice_engine)
    yield session
    session.rollback()
    session.close()


@pytest.fixture()
def bob(bob_engine):
    session = get_session(bob_engine)
    yield session
    session.rollback()
    session.close()


def vec(dominant_dim: int, dim: int = 768) -> list[float]:
    """A synthetic 768-dim embedding, mostly distinguished by one dominant dimension.

    Real ingestion would use an actual embedding model (§15 leaves that to the
    caller) — these are just cheap, deterministic stand-ins for testing ranking
    behavior without depending on model inference.
    """
    v = [0.01] * dim
    v[dominant_dim] = 0.9
    return v
