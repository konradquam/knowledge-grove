# Knowledge Grove

A Postgres-backed knowledge schema and SDK that lets agents — in one repo and across repos — store, discover, and act on shared context: document chunks, embeddings, exact-match tags, and a graph of links between them, including links out to executable tools.

**Status:** Core schema, SDK, CLI, and MCP server are implemented and tested (real Postgres, not mocks). A few sections of the original design are still open — see [Implementation status](knowledge-grove-design.md#implementation-status) in the design doc for the precise list.

## Design commitments

- **Postgres is the only system of record.** Vector search (`pgvector`), full-text search, exact-match lookups, and the relationship graph all live in one transactional database.
- **The schema is normalized.** Every relationship — a tag, a link, an access grant — is its own row in its own table.
- **Access control is enforced by Postgres itself**, via row-level security tied to each agent's own database role. Each agent authenticates with its own role and credentials — never one shared service account.

## Core data model

Four tables carry the whole system: `documents` (one row per chunk, with content, a content hash for exact-match dedup, an embedding, and full-text search columns), `document_tags` (exact-match labels), `edges` (a generic relationship graph — next/prev, source, tool links, related, supersedes), and `document_access` (per-document read/write grants). A fifth, `retrieval_feedback`, logs how documents perform for a query, for later ranking-weight tuning.

## Entry points

Five ways to land in the graph, each catching a case the others miss: direct ID lookup, tag matching, ILIKE/trigram search, full-text search, and embedding similarity — fused via weighted Reciprocal Rank Fusion. (A bounded, personalized-PageRank walk over the edge graph to expand past those entrance nodes is designed but not yet implemented — see the status link above.)

## Installation

Not yet published to PyPI. Install from source:

```bash
git clone https://github.com/konradquam/knowledge-grove.git
cd knowledge-grove
pip install .          # or: pip install -e . for an editable install
```

Requires Python 3.11+ and a Postgres server with the `pgvector` and `pg_trgm` extensions available (the bootstrapping step below creates them for you if your connecting role has privilege to).

## Quickstart

**1. Stand up the schema.** Run once per database, using a role with `CREATE EXTENSION`/table-owner privileges — never the role an ordinary agent connects as, since table owners bypass row-level security by default.

```bash
export KNOWLEDGE_GROVE_DSN="postgresql+psycopg://admin_role:password@host:5432/dbname"
knowledge-grove init-db
```

**2. Provision an agent.** Each agent gets its own Postgres role, added to the `shared_reader` group so it can read whatever's been shared into that group by default. Still using the admin DSN:

```bash
knowledge-grove create-agent-role my_agent
# Role 'my_agent' created.
# Agent DSN: postgresql+psycopg://my_agent:<generated-password>@host:5432/dbname
```

Save that returned DSN somewhere your agent can read it (an env var, a secrets manager — resolving the secret is your project's job, not this package's; see §15 of the design doc).

**3. Use it.** Either point an MCP-capable agent at the server (below), ingest existing files from a shell, or call the SDK directly from Python.

```bash
export KNOWLEDGE_GROVE_DSN="postgresql+psycopg://my_agent:<password>@host:5432/dbname"
knowledge-grove ingest README.md docs/notes.py
```

## CLI reference

All commands read their connection string from `KNOWLEDGE_GROVE_DSN`.

| Command | Purpose |
|---|---|
| `init-db` | Run the bundled Alembic migrations: creates the schema, indexes, and RLS policies. Needs an admin/setup role. |
| `create-agent-role <name> [--password PW]` | Provision a new agent's Postgres role and `shared_reader` membership. Prompts for a password (hidden) if `--password` is omitted. Needs an admin/setup role. |
| `ingest <files...> [--source-url URL...] [--content-type {markdown,python,sql}...] [--roles JSON] [-y]` | Chunk and add one or more files as documents. `--source-url`/`--content-type` are given once per file (positionally matched); each defaults respectively to the file's own name and a guess from its extension. Re-ingesting a file under the same `source_url` reconciles against what's already there (a no-op if unchanged, a full replace if not) rather than duplicating it — warns and asks for confirmation first unless `-y` is given. `--roles` is a JSON object (e.g. `'{"shared_reader": ["read"]}'`, the default) applied to every document ingested in the call. |

## MCP server

`knowledge_grove.mcp_server` exposes the SDK as MCP tools over stdio (the default transport). Run it directly, or point an MCP-capable client at it:

```json
{
  "mcpServers": {
    "knowledge-grove": {
      "command": "/path/to/venv/bin/python",
      "args": ["-m", "knowledge_grove.mcp_server"],
      "env": { "KNOWLEDGE_GROVE_DSN": "postgresql+psycopg://my_agent:<password>@host:5432/dbname" }
    }
  }
}
```

Each entry in a client's MCP config gets its own subprocess, so each agent identity that needs its own Postgres role should get its own entry with its own DSN — one running server process, one fixed identity for its lifetime; nothing multiplexes several agents through a single connection.

Tools exposed:

| Tool | Purpose |
|---|---|
| `gather_context` | Fused search across all four entry points; the default way to look for existing context. |
| `add_document` | Add one already-written chunk. |
| `add_sequential_documents` | Add several chunks in order, auto-linked with `prev` edges. |
| `add_authored_chunks` | Add several chunks in order, plus arbitrary extra edges (to existing documents, to each other, or to a URL) in the same call. |
| `get_by_id` | Fetch a document by id. |
| `get_edges` | Outgoing edges from a document (one hop). |
| `update_document` | Create a new revision of a document (old one kept, flagged deprecated, linked via `supersedes`). |
| `add_tag` | Attach an exact-match tag. |
| `add_edge` | Attach a relationship to another document or an external URL. |
| `grant_access` / `revoke_access` | Manage a document's access grants. |
| `log_feedback` | Record how a document performed for a query. |

Every tool's full description (usage guidance, argument shapes, when to prefer one over another) is visible to any MCP client that lists tools, and is worth reading directly in `src/knowledge_grove/mcp_server.py` if you're integrating one.

## Using the SDK directly

Everything the MCP server exposes is a thin wrapper over `knowledge_grove.crud` and `knowledge_grove.search` — call those directly from Python for non-MCP integrations (e.g. a Temporal activity):

```python
from knowledge_grove.db import get_engine, get_session
from knowledge_grove import crud, search

engine = get_engine("postgresql+psycopg://my_agent:password@host:5432/dbname")
session = get_session(engine)

doc = crud.add_document(session, content="Retries should use exponential backoff.", owner_agent="my_agent")
session.commit()

hits = search.gather_context(session, query_text="how do retries work", pattern="retry")
```

## Design document

See [knowledge-grove-design.md](knowledge-grove-design.md) for the full design rationale — auth model, ranking math, chunking strategy, bootstrapping, and the [Implementation status](knowledge-grove-design.md#implementation-status) section tracking what's built against what's still designed-but-open.

## License

MIT — see [LICENSE](LICENSE).
