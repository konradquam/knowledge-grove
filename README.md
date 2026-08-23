# Knowledge Grove

A Postgres-backed knowledge schema and SDK that lets agents — in one repo and across repos — store, discover, and act on shared context: document chunks, embeddings, exact-match tags, and a graph of links between them, including links out to executable tools.

**Status:** Plan phase. See [knowledge-grove-design.md](knowledge-grove-design.md) for the full design — nothing described there is implemented yet.

## Design commitments

- **Postgres is the only system of record.** Vector search (`pgvector`), full-text search, exact-match lookups, and the relationship graph all live in one transactional database.
- **The schema is normalized.** Every relationship — a tag, a link, an access grant — is its own row in its own table.
- **Access control is enforced by Postgres itself**, via row-level security tied to each agent's own database role.

## Core data model

Four tables carry the whole system: `documents` (one row per chunk, with content, embedding, and full-text search columns), `document_tags` (exact-match labels), `edges` (a generic relationship graph — next/prev, source, tool links, related), and `document_access` (per-document read/write grants).

## Entry points

Five ways to land in the graph, each catching a case the others miss: direct ID lookup, tag matching, ILIKE/trigram search, full-text search, and embedding similarity — fused via weighted Reciprocal Rank Fusion and expanded through a bounded, personalized-PageRank walk over the edge graph.

See the [design document](knowledge-grove-design.md) for the full picture, including auth, ranking, tool discovery via MCP, the SDK surface, chunking strategy, and bootstrapping.

## License

MIT — see [LICENSE](LICENSE).
