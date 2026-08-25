from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker


def get_engine(dsn: str, **kwargs) -> Engine:
    """Create an engine for the given DSN.

    `dsn` should already be a fully resolved connection string, e.g.
    "postgresql+psycopg://agent_role:password@host:5432/dbname" — resolving
    it from Vault/a secrets manager/env is the caller's responsibility (§15).
    """
    return create_engine(dsn, **kwargs)


def get_session(engine: Engine) -> Session:
    """Create a new session bound to `engine`.

    Each agent connects as its own Postgres role, so the session returned
    here carries whatever row-level-security scope that role's credentials
    grant it — callers should not share one session/engine across agents.
    """
    session_factory = sessionmaker(bind=engine)
    return session_factory()
