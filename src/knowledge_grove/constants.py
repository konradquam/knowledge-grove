import os
from enum import StrEnum

# Dimension of the `embedding` / `summary_embedding` vector columns.
# Baked into the schema at migration time (pgvector requires a fixed size per
# column), so changing this after tables exist requires a migration that
# rebuilds the vector columns and re-embeds existing content.
#
# Default matches intfloat/e5-base-v2 (768-dim, open source, MIT licensed).
EMBEDDING_DIM = int(os.environ.get("KNOWLEDGE_GROVE_EMBEDDING_DIM", 768))

# Postgres enum values. Defined once here so models.py (SQLAlchemy Enum
# types) and the initial migration (raw CREATE TYPE ... AS ENUM statements)
# can't drift apart from each other.
SOURCE_METHODS = ("vector", "tags", "fulltext", "ilike", "id")
JUDGED_BY_VALUES = ("explicit_llm", "implicit_usage")
SHARED_READER = "shared_reader"

class EdgeType(StrEnum):
    NEXT = "next"
    PREV = "prev"
    SOURCE = "source"
    TOOL = "tool"
    RELATED = "related"
    SUPERSEDES = "supersedes"

class PERMISSIONS(StrEnum):
    READ = "read"
    WRITE = "write"