# Knowledge Grove — Design Document

**Status:** Plan phase. Nothing described here is implemented yet. Open points are marked explicitly as **Open decision** or **Optional** rather than presented as settled — resolve those before or during implementation of the relevant piece, don't assume a default silently.

A Postgres-backed knowledge schema and SDK that lets agents — in one repo and across repos — store, discover, and act on shared context: document chunks, embeddings, exact-match tags, and a graph of links between them, including links out to executable tools.

## Table of contents

1. Overview & goals
2. Core data model
3. Entry points
4. Auth & access control
5. Ranking & fusion
6. Graph-topology ranking
7. Learning from usage
8. Tool discovery (MCP)
9. SDK surface
10. Gathering context
11. Chunking strategy
12. Adding & updating
13. Importing
14. Bootstrapping
15. Package & distribution
16. Explicit non-goals
17. Ideas for later

---

## 1. Overview & goals

The problem this solves: agents working across several repositories keep re-deriving context that was already worked out somewhere else — prior decisions, documentation, tool capabilities, source material a document was composed from. Right now that knowledge either lives nowhere durable, or lives in a form only a human can search. The goal is a single, reusable substrate that agents can query for context the same way they'd query any other structured store, with enough structure that retrieval is precise rather than a guess.

Three design commitments run through everything below. First, **Postgres is the only system of record** — vector search (via `pgvector`), full-text search, exact-match lookups, and the graph of relationships between documents all live in one transactional database, rather than being split across a vector store, a graph store, and an application database that have to be kept in sync. Second, **the schema is normalized** — every relationship (a tag, a link, an access grant) is its own row in its own table, never a list stuffed into a cell, because that's what lets Postgres index, join, and enforce referential integrity on it. Third, **access control is enforced by Postgres itself**, via row-level security tied to each agent's own database role, not by application code that has to remember to filter correctly every time.

The package's job is to own this schema — its creation, its evolution, and a client SDK for reading and writing it — so that adopting it in a new project is "point at a Postgres server and authenticate," not "re-derive this whole design again."

## 2. Core data model

Four tables carry the whole system. Everything in later sections — entry points, ranking, access control — is built on top of these.

### documents

One row per chunk. This is the atomic unit everything else attaches to.

| Column | Type | Notes |
|---|---|---|
| `id` | uuid, pk | — |
| `content` | text | The chunk itself |
| `content_tsv` | tsvector, generated | Generated from `content`, GIN-indexed — backs full-text search (§3) |
| `embedding` | vector | pgvector column, HNSW-indexed |
| `summary` | text, null | Optional short summary of the chunk |
| `summary_embedding` | vector, null | Embedding of the summary, when present |
| `source_url` | text, null | GitHub link, if this chunk lives in a repo |
| `owner_agent` | text | Namespace this document belongs to — the Postgres role name of its owning agent |
| `created_at` | timestamptz | — |

`content_tsv` is `GENERATED ALWAYS AS (to_tsvector('english', content)) STORED` — Postgres keeps it current automatically whenever `content` changes, so nothing in the SDK has to maintain it. The ILIKE/trigram entry point (§3) needs no equivalent column — its GIN index sits directly on `content`.

### document_tags

One row per (document, tag) pair — the exact-match label a document carries, plus why it applies. No separate tag dictionary table; tags are plain strings normalized (lowercased/trimmed) on write.

| Column | Type | Notes |
|---|---|---|
| `id` | uuid, pk | — |
| `document_id` | uuid, fk → documents.id | — |
| `tag` | text | The exact string to match on |
| `description` | text | Why this tag applies to this document |
| `created_at` | timestamptz | — |

### edges

One row per relationship — next/prev, source composition, tool links, and general related-document links, all in one generic table rather than one column per relationship type. New relationships are new rows, not new columns.

| Column | Type | Notes |
|---|---|---|
| `id` | uuid, pk | — |
| `from_document_id` | uuid, fk → documents.id | — |
| `to_document_id` | uuid, fk → documents.id, null | Set when the target is another row in this database |
| `external_url` | text, null | Set when the target is outside the database (a GitHub file, a tool script) |
| `edge_type` | enum | `next` · `prev` · `source` · `tool` · `related` |
| `description` | text | Short note on what this specific edge means |
| `created_at` | timestamptz | — |

A check constraint enforces that exactly one of `to_document_id` / `external_url` is set on every row.

### document_access

One row per grant — who besides the owner can read or write a document. Detailed further in §4.

| Column | Type | Notes |
|---|---|---|
| `id` | uuid, pk | Identifies the grant row itself, not a person |
| `document_id` | uuid, fk → documents.id | — |
| `grantee_role` | text | A Postgres role name — an individual agent or a group role |
| `permission` | enum | `read` · `write` |
| `granted_at` | timestamptz | — |

## 3. Entry points

Five distinct ways to land in the graph, each backed by its own index type, each catching a case the others miss.

| Entry point | Index | Catches |
|---|---|---|
| Document ID | primary key | Direct fetch — the caller already knows exactly what it wants |
| Tags / metadata | B-tree on `tag` | Deliberately curated, exact structured labels |
| ILIKE / substring | `pg_trgm` GIN | Exact identifiers, error codes, config keys — things word-tokenizing search would break apart |
| Full-text search | `tsvector` GIN | Natural-language prose — word/stem matching with relevance ranking |
| Embeddings | pgvector HNSW | Conceptual/semantic similarity, no literal term overlap required |

Full-text search and ILIKE are easy to conflate but solve different problems. Postgres's `tsvector` parser tokenizes on word boundaries and stems — a search index built that way would split `retry_after_ms` into `retry`, `after`, `ms` as separate lexemes, rather than keeping it as one atomic token. That's the right behavior for prose, wrong for a literal identifier. `ILIKE` backed by a trigram index is the closer analog to `grep` with an index — it matches the literal substring, which is what you want for a code symbol or an error code someone already knows.

Embeddings, meanwhile, are weakest exactly where ILIKE is strongest. A chunk's embedding represents a blended sense of the whole passage, not any single token in it, so a chunk that discusses a topic at length in different words can rank more "central" to that topic than a chunk where the exact search term appears once in passing. Rare, non-dictionary tokens (an error code, an all-caps constant) also tend to get split into odd subword fragments by the tokenizer underneath most embedding models, which are trained mostly on natural language. Lexical and trigram search guarantee an exact hit; semantic search only guarantees "conceptually similar" — for a precise identifier, that's a materially weaker guarantee.

## 4. Auth & access control

Each agent authenticates with its own Postgres role and its own credentials — never one shared service account. Access is then enforced by Postgres itself, not by application code.

### Row-level security, not application-level filtering

With every agent connecting as its own role, Postgres row-level security (RLS) can reference `current_user` directly. A policy on `documents` combines ownership with the grants table:

```sql
ALTER TABLE documents ENABLE ROW LEVEL SECURITY;

CREATE POLICY document_read_policy ON documents
FOR SELECT
USING (
  owner_agent = current_user
  OR EXISTS (
    SELECT 1 FROM document_access
    WHERE document_access.document_id = documents.id
      AND pg_has_role(current_user, document_access.grantee_role, 'MEMBER')
  )
);
```

Using `pg_has_role(...)` rather than a plain string comparison means a grant can name either an individual agent's role or a group role — any agent that's a member of that group satisfies the check automatically, without a row per member. This leans on Postgres's own role hierarchy for group membership rather than re-implementing it.

### Group roles for shared content

Rather than granting shared-namespace access to every agent individually, one group role (e.g. `shared_reader`) holds the shared-namespace grants, and each agent role is simply made a member of it. Provisioning a new agent then reduces to: create its role, add it to `shared_reader`, and its own `owner_agent`-based policy handles the rest automatically.

### document_access needs its own protection

> If any agent role can freely write to `document_access`, the whole scheme has a hole — nothing stops an agent from granting itself access to someone else's document. `document_access` needs either its own RLS policy (only a document's owner or an admin role may insert/modify grants for it) or tighter base privileges: agents get `SELECT` only, an admin/setup role holds `INSERT`/`UPDATE`/`DELETE`.

### Three layers, all required

1. **Enable RLS** on the table — off by default even with policies present.
2. **Define the policy** — the actual row-filtering logic.
3. **Base `GRANT`s** — RLS only narrows privileges a role already has at the table level; it doesn't substitute for `GRANT SELECT` / `INSERT` in the first place.

One operational trap worth naming: table owners and superusers bypass RLS by default (the `BYPASSRLS` attribute). The role that runs migrations and setup should never be the same role an agent connects as, or RLS silently stops applying to it.

## 5. Ranking & fusion

Combining results from five entry points that don't produce comparable scores.

Cosine similarity from vector search, `ts_rank` from full-text search, and trigram `similarity()` from ILIKE all live on different, non-comparable scales — a 0.82 cosine similarity doesn't mean the same thing as a `ts_rank` of 0.4. Averaging raw scores across methods isn't meaningful.

The standard fix — used by most production hybrid-search systems — is **Reciprocal Rank Fusion (RRF)**: run each method independently to get its own ranked list, then combine *ranks* instead of raw scores. A document's fused score is the sum, over every method that surfaced it, of one over a damping constant plus its rank in that method's list:

```
fused_score(d) = Σ_m  w_m / (k + rank_m(d))

k ≈ 60 · w_m is a per-method weight, 1 by default
```

This sidesteps the scale problem entirely — it never looks at a raw score, only where a document sits in each method's own ordering. A document ranked near the top by two or three methods outranks one that's a strong hit by only one, which is generally the outcome you want.

**On `w_m`:** it is not derived from any raw score. It's a static, per-method configuration value — "how much do we trust this method in general" — set to `1` for every method by default, later tuned offline from aggregated feedback (§7), never computed per-query from a cosine similarity or `ts_rank` value. `rank_m(d)` (an integer position) is the only per-document input that comes from a raw score; the raw score itself is discarded once it's produced that ordering.

**On `k`:** also a fixed constant, not derived from data — but it plays a different role than `w_m`. It controls how much the fused score is dominated by top ranks vs. spread evenly across lower ones (small `k` → top rank dominates heavily; larger `k` → the curve flattens). `k = 60` is the empirically-common default from the original RRF literature, usually left as-is rather than tuned.

### Not every entry point belongs inside the fusion

- **Vector search, full-text search, ILIKE** — genuine ranked lists, feed directly into RRF.
- **Document ID** — not a search at all. A direct fetch, handled outside ranking entirely.
- **Tags** — naturally binary (matched or not), and deliberately curated, so generally higher-precision than a fuzzy hit. Treated as a hard pre-filter or an automatic top-rank boost, rather than one equally-weighted voice among the fuzzy methods — folding it into RRF at equal weight would dilute a strong curated signal down to the level of a so-so vector match.

Per-method weights (`w_m` above) make the fusion tunable later without changing its shape — see §7 for how those weights are meant to move over time.

## 6. Graph-topology ranking

Using the shape of the edge graph itself as a ranking signal, not just as a place to expand into after ranking is done.

Fusion (§5) ranks entrance nodes on how well they matched the query directly. It says nothing about which of those nodes sit in a well-connected, mutually-reinforcing neighborhood of the graph versus which are only reachable by one thin path. A **personalized PageRank** (PPR) pass over the edge graph captures exactly that.

Picture a random walker moving through `edges`: at each step it either follows an outgoing edge, or — instead of teleporting to a uniformly random node the way plain PageRank would — teleports back to the query's entrance nodes, weighted by their fusion score. A node that keeps getting revisited from several different entrance points, because it sits between them or is linked to several at once, accumulates a higher score than a node reachable through only one path. `edge_type` can weight the walk further — a `source`/derivation edge is a stronger relationship than a loose `related` one, and can be given more weight in the walk accordingly.

This doesn't require computing PageRank over the whole database on every query. A local, approximate version — short random walks, or a "push"-style computation confined to the bounded neighborhood already being pulled in for traversal — keeps it cheap, and composes with the hop-depth limit from §10 rather than replacing it. Because edge traversal reads back through `documents`, RLS already scopes which nodes the walk can even see (§4) — a walk never surfaces a document the caller's role couldn't read directly.

**Rejected idea, worth recording so it doesn't get re-litigated:** deliberately promoting "important" (e.g. graph-central) vectors to HNSW's top layer, to make them easier for vector search to find. HNSW's per-vector layer assignment is a random, content-blind draw (`level = floor(-ln(U) × mL)`, independent of the vector's position or importance) that exists specifically to keep the top layers a statistically unbiased map of the embedding space's geometry. Overriding it with a content-graph-centrality heuristic would likely skew the top layer toward wherever "important" documents happen to sit in vector space, degrading recall for unrelated queries, while conflating two unrelated graphs — vector-space proximity and content-link centrality don't correlate. The correct place for "importance" to influence results is exactly this section's PPR pass, applied after retrieval, not baked into the ANN index's internals. `pgvector` also doesn't expose per-vector layer assignment as a tunable, so this isn't achievable without patching the extension regardless.

## 7. Learning from usage

Letting the per-method weights in §5 move over time based on which results actually turned out to be useful — sequenced conservatively, given the real risks involved.

### Capturing feedback

Two signals, not mutually exclusive:

- **Explicit** — an LLM judgment pass over a query's returned documents, rating relevance or identifying which were actually used. Clean signal, but costs extra calls; better suited to an offline batch pass over logged query-result pairs than blocking every live retrieval.
- **Implicit** — logging which documents an agent went on to actually use, cite, or traverse further from, versus ones it ignored. Free, but noisier — not using a document doesn't cleanly mean it was irrelevant.

### retrieval_feedback

| Column | Type | Notes |
|---|---|---|
| `id` | uuid, pk | — |
| `query_text` | text | — |
| `document_id` | uuid, fk → documents.id | — |
| `source_method` | enum | vector · tags · fulltext · ilike · id |
| `rank` | integer | Position within that method's own result list |
| `relevance` | real, null | Explicit or implicit judgment score |
| `judged_by` | enum | explicit_llm · implicit_usage |
| `created_at` | timestamptz | — |

Worth logging from day one regardless of when weight-adjustment actually begins — historical feedback can't be generated retroactively.

Worth logging more than just the fused outcome, too: the per-method raw rankings that fed into fusion (§5), and — once an experimental ranker exists but isn't live yet — its shadow output computed alongside the real one, without being used to change what's actually returned. That's cheap to capture, and it means evaluating whether something like §6's graph-topology ranking is worth turning on doesn't require waiting for fresh data collected under it after the fact; it can be checked retroactively against outcomes already on file, using historical queries, before ever deploying it live.

### Three stages, in order

1. **Manual, offline.** Periodically review the feedback table, adjust weights by hand, redeploy. Slow, but fully inspectable — the right starting point.
2. **Automated, offline, human-reviewed.** A scheduled batch job fits new weights against historical feedback; a person reviews before they go live. Now genuinely learning, but nothing changes mid-session.
3. **Online, adaptive.** Weights shift continuously as feedback streams in. The most powerful version, and the riskiest — not the starting point.

> **Risks worth naming before stage 3**
>
> **LLM-as-judge bias.** The judge has its own systematic preferences (favoring longer or more confident-sounding chunks, say) unrelated to real usefulness — an online loop will happily optimize toward those biases unless checked periodically against human judgment.
>
> **Self-reinforcing feedback loops.** Whichever method starts slightly ahead gets shown more, gets judged more, and entrenches — starving other methods of enough data to prove themselves on query types they'd genuinely handle better. Mitigated by holding out a slice of traffic on fixed baseline weights, or deliberately injecting exploration.
>
> **One global weight is probably wrong.** "Find the function named X" wants ILIKE/tags dominant; "explain how retries work here" wants vector/full-text dominant. Per-query-type weighting is a natural extension, but not a day-one requirement.

One advantage specific to the shared-database decision in §4: because every agent's namespace shares the same feedback table, a pattern learned from one agent's usage — "vector search underperforms on snake_case-heavy queries," say — can inform ranking for every other agent, not just the one that generated the feedback. Isolated per-project retrieval setups don't get that for free.

**Note on the other entry points:** it does not make sense to try to impose this kind of usage-driven "structural" updating on the B-tree (tags) or GIN (full-text, trigram) indexes themselves, the way §6 considered (and rejected) for HNSW. B-tree and GIN are exact structures — given a query they return the complete, correct match set every time, with no approximation and therefore no recall to improve by restructuring based on usage. The per-method RRF weight (`w_m`) already covers all four search-based methods symmetrically; that's the correct and sufficient lever, not something specific to vector search.

## 8. Tool discovery (MCP)

A `tool`-typed edge points at something executable. Discovery and execution are deliberately kept separate.

A row in `edges` pointing at a tool is metadata — a pointer and a description. It doesn't, by itself, grant an agent the ability to execute anything; reading a decorator's source text out of a database row doesn't make the underlying function callable. Execution requires a real invocation path: the tool's repo installed as a dependency, a remote procedure call, or — the approach adopted here — an MCP server that resolves a `tool_id` to the actual callable.

### Gated discovery, not a static catalog

Exposing every available tool through MCP up front risks overwhelming an agent with possibilities it doesn't need for the task at hand — and, separately from context cost, model function-calling accuracy measurably degrades as the number of available options grows. Instead, the MCP server exposes exactly two tools, always:

| Tool | Purpose |
|---|---|
| `describe_tool(tool_id)` | Returns the real argument schema for a specific tool |
| `invoke_tool(tool_id, args)` | Dispatches to the actual underlying callable |

An agent finds a `tool_id` only by traversing the graph — a `tool`-type edge or document carries the id and a description. The full catalog is never listed anywhere the agent can see it directly; capability surfaces only through context retrieval, using the same search-and-traverse machinery that surfaces documents, rather than a separate bespoke tool registry. The `tool_id` stored in an edge is the same key the dispatcher resolves — the graph and the MCP surface stay a single source of truth with nothing to keep in sync.

This was chosen over MCP's native `tools/list_changed` notification (dynamically unlocking specific tools mid-session) because not every MCP client handles that notification reliably, and a generic dispatcher needs no session-scoped "what's currently unlocked" state to track against the graph traversal — it's simpler with fewer moving parts, at the cost of the host UI's native per-tool argument autocomplete and at the cost of an extra round trip (`describe_tool` then `invoke_tool`) versus a single native tool call. A tool is also only as discoverable as the graph is well-curated: an edge that's never created leaves a real capability invisible, regardless of how good the dispatcher is.

## 9. SDK surface

The client library any project depends on. Naming below is illustrative — the shape, not the final signatures.

| Function | Does |
|---|---|
| `add_document(...)` | Insert a chunk, its embedding, and metadata |
| `add_tag(document_id, tag, description)` | Attach an exact-match tag |
| `add_edge(from_id, to_id \| external_url, edge_type, description)` | Attach a relationship |
| `add_authored_chunks(chunks, edges)` | Agent-facing authoring tool — submits pre-chunked, self-written content in one call. See §11. |
| `grant_access` / `revoke_access` | Manage a document's `document_access` rows |
| `get_by_id(document_id)` | Direct fetch entry point |
| `search_vector` / `search_tags` / `search_fulltext` / `search_ilike` | The other four entry points, each returning its own ranked or matched list |
| `gather_context(query, ...)` | The full pipeline — fuse entry points, rank, traverse the graph. See §10. |
| `log_feedback(query, document_id, source_method, rank, relevance)` | Write a `retrieval_feedback` row |
| `update_document(...)` | See §12 for what "update" means here |
| `import_repo(...)` | Bulk ingestion helper — see §13 |

Plus CLI-facing commands that aren't part of the runtime SDK at all — `init-db` and `create-agent-role` — covered in §14.

## 10. Gathering context

What `gather_context` actually does, end to end.

1. Run the four search-based entry points (vector, tags, full-text, ILIKE) as independent queries. Each is automatically scoped by RLS to whatever the caller's role can see — no separate filtering step needed.
2. Fuse the ranked lists via weighted RRF (§5); fold tag matches in as a pre-filter or boost rather than an equal-weight voice.
3. Take the top-N fused results as entrance nodes into the graph.
4. Run a bounded, entrance-weighted walk over `edges` (§6) — capped by hop depth and/or node count, so one densely-linked document can't pull in a disproportionate share of the graph.
5. Return the assembled, ranked set of documents — search hits plus their graph neighborhood, already access-filtered.
6. Optionally log a `retrieval_feedback` row per returned document for §7 to use later.

## 11. Chunking strategy

How content becomes rows in `documents` in the first place — this affects retrieval quality more than the ranking machinery built on top of it, so it's worth treating as a first-class design decision rather than an implementation detail of the ingestion layer.

### Chunk by structure, not by size

Splitting text every N characters or tokens regardless of content is the naive baseline, and it fails badly on a meaningful share of what this system stores: code, SQL, and tool definitions. Cutting a function or a SQL statement in half mid-body destroys its meaning and produces an embedding for a fragment that represents nothing coherent. The boundary logic should differ by content type rather than applying one splitter everywhere:

| Content type | Boundary unit | How |
|---|---|---|
| Code / scripts / SQL | One function, class, or statement | A real parser (`ast`, or `tree-sitter` for multiple languages) — never a text splitter |
| Tool definitions | One tool per chunk | Matches §8 directly — one `tool_id`, one chunk |
| Prose / documentation | One section or paragraph | Recursive split — headers first, then paragraphs, with a size cutoff only as a last-resort fallback |

The ingestion layer (§13) should accept a pluggable chunker per content type rather than one hardcoded algorithm — the right boundary logic for a Python function and for a paragraph of a design doc are different problems, and forcing one splitter to handle both is where naive chunking usually goes wrong in practice.

### Semantic-shift fallback for prose — *Optional*

For the prose fallback specifically — where structure runs out and a plain size cutoff would otherwise apply — a sharper option is to embed small units within the section (sentence by sentence, or over short windows) and break where meaning actually shifts, rather than where a token count happens to land. Plain semantic chunking has a known weakness though: a section that stays thematically consistent for a long stretch never produces a dissimilarity spike large enough to trigger a break, so it can grow unboundedly. The fix is a combined score — semantic dissimilarity plus a penalty that grows with how long the current chunk has been running — that breaks once the sum crosses a threshold, so a chunk that's already long only needs a mild shift to justify ending, not a dramatic one. An absolute max size still sits on top as a hard ceiling, for the case where even that combined score never crosses threshold.

This is opt-in, not part of the default chunking path — it's real additional cost (embedding many small units per document at ingestion time, on top of whatever the final chunk embeddings already cost) and real additional tuning surface (the dissimilarity metric, the penalty's growth rate, the crossing threshold, on top of the size/overlap tuning already called out below), worth reaching for only where plain structural splitting is demonstrably leaving prose chunks poorly bounded. It has no role in code/SQL/tool chunking, where the parser-defined unit is already atomic, and it's redundant for content added through author-time chunking (below) — there, the agent is already placing semantic boundaries as it writes.

### Small chunks, recovered by the graph

Most retrieval systems have to trade off between small chunks (precise embeddings, but each hit loses surrounding context) and large chunks (more context per hit, but a diluted embedding — and per §3, dilution is exactly what already makes vector search weak on specific terms). This schema doesn't have to make that trade the same way, because the `next`/`prev` edges (§2) already exist: chunk small for precision, and let `gather_context`'s bounded traversal (§10) pull in the neighboring chunk when a hit needs the surrounding thought, rather than inflating chunk size against every possible case in advance.

Worth being precise about what this does and doesn't fix. `next`/`prev` traversal recovers a thought that got cut off at a chunk boundary — that's its real value. It does not fix dilution caused by oversized chunks in the first place; that's addressed by chunking small, not by linking. And it doesn't rescue a chunk that never matched any entry point at all — traversal only starts from an entrance node, so if nothing in a document matched well enough to become one, there's nothing to walk from to reach its neighbors either.

### Summary fields for a coarser signal

For a long source document split into many small, precise chunks, the `summary`/`summary_embedding` columns give a cheap document-level match before drilling into individual chunks — an LLM-generated summary of the whole document, embedded separately. This is semantically compressed rather than mechanically averaged, which is generally the stronger of the two — see the vector-space centroid idea in §17 for the mechanical alternative.

### Author-time chunking, for agent-originated knowledge

Everything above assumes the content already exists and has to be parsed after the fact — the right approach for importing pre-existing material (§13), where a file already exists and has to be reverse-engineered into chunks. It isn't the only path content takes into the graph, though. When an agent is originating brand-new knowledge itself — writing up a decision, documenting something it just learned — there's nothing to reverse-engineer: the agent can compose the content already correctly segmented, since it's authoring it from scratch anyway.

This is exposed as its own tool rather than a raw `add_document` call, with a description that instructs the calling LLM to write in a general hierarchical format: compose content as nested sections, where each leaf section becomes one chunk — small enough to embed precisely, complete enough to stand alone if it's ever retrieved without its neighbors. The LLM submits a list of chunks in one call, in the order it wrote them, rather than one long blob for an ingestion layer to re-split afterward.

| Argument | Carries |
|---|---|
| `chunks` | An ordered list of pre-segmented pieces of content, each becoming one `documents` row — `next`/`prev` edges between them are inferred automatically from the order given |
| `edges` | Relationships the new chunks have beyond sequence: to each other, and to existing `document_id`s already in the graph — `source`, `related`, or a supersession link if this content replaces something earlier (§12) |

The advantage over parsing after the fact: no heuristics are needed to guess where a boundary belongs, because the author already knows — and the same call that's composing the content is well-placed to reason about what it relates to, producing tags and edges as a byproduct of writing rather than something a separate pass has to infer from raw text later. This complements the parser-based path above rather than replacing it: pre-existing external content still needs to be parsed, since it wasn't authored through this tool and can't retroactively be asked to have pre-chunked itself.

> **Open decision:** exact size targets and overlap between adjacent chunks need empirical tuning per content type rather than a fixed number chosen up front — a reasonable starting point for prose is on the order of a few hundred tokens with modest overlap, but this should be measured against real retrieval quality once there's usage data to measure against, not locked in speculatively.

## 12. Adding & updating

**Adding** is straightforward: a new chunk is a new row in `documents`, plus whatever `document_tags` and `edges` rows describe how it relates to what's already there. Nothing about this is different from ordinary inserts into a normalized schema.

**Updating** is less settled, and worth deciding deliberately before implementation rather than defaulting to in-place mutation.

> **Open decision:** in-place edits to `content` or `embedding` are simple, but they silently change what every existing edge into that document points at — a `source` edge recorded against a specific version of a chunk may no longer describe what's actually there. The alternative is treating chunks as effectively immutable: a revision becomes a new row, linked to its predecessor by an edge (or a dedicated version pointer), leaving old edges pointing at exactly the content they were created against. That preserves history and referential meaning at the cost of the table growing with every edit rather than shrinking to just current state. Worth resolving explicitly before the schema is implemented, not left as an accident of how `add_document` happens to be written.

## 13. Importing

Bulk ingestion — scanning a repository for documents and `@tool`-decorated functions, chunking (§11), embedding, and inserting via the SDK — is ordinary Python calling `add_document`/`add_tag`/`add_edge` in a loop. It isn't a data-warehouse transformation problem, so tools built for that shape of work don't fit: **dbt** compiles SQL `SELECT`s into materialized tables from data already sitting in a warehouse — it has nothing to do with chunking text or calling an embedding model. **Airflow** is a scheduler for exactly this kind of batch job, but standing up a new orchestration platform is a heavier commitment than the job warrants when this repo already runs Temporal — a scheduled Temporal workflow calling the ingestion function is the more natural fit here, with plain cron or a scheduled GitHub Actions workflow as the lighter-weight option for a simpler periodic sync elsewhere.

Both dbt and Airflow are free, open-source software — but that's beside the point, since neither is the right tool for this job regardless of cost. Running either one also isn't actually free in the total sense: it means hosting a scheduler and its metadata store yourself, or paying for a managed version (dbt Cloud, Astronomer, MWAA, Cloud Composer).

## 14. Bootstrapping

Standing up a new project's database is meant to be a small, versioned, one-time-per-database sequence, owned entirely by the package:

1. **Enable `pgvector`** — `CREATE EXTENSION vector`, run once per database by a role with sufficient privilege (superuser, or a role explicitly granted extension-creation rights). Most managed Postgres offerings support this, but it sits outside what an ordinary application role can do on its own.
2. **`init-db`** — runs the bundled Alembic migrations: creates `documents`, `document_tags`, `edges`, `document_access`, `retrieval_feedback`, their indexes (HNSW, GIN × 2, B-tree), enables RLS, and creates the policies from §4.
3. **`create-agent-role`** — provisions a new agent's Postgres role, adds it to the `shared_reader` group role, and hands back credentials for that agent to authenticate with directly.

Because schema changes ship as Alembic migrations rather than being hand-run per project, evolving the schema later — a new column, a new `edge_type` — is a repeatable upgrade any project can run, not something re-derived by hand each time.

## 15. Package & distribution

Ships as a single pip-installable package with three parts: the Alembic migrations and RLS/policy definitions (§14), the client SDK (§9), and an optional MCP server entrypoint (§8) built on the same client. Authentication is just a DSN — the package doesn't invent its own auth system, it connects with whatever Postgres role and credentials it's given, which is what makes "bring your own agent credentials" (§4) possible without any special-casing in the package itself.

Secret *resolution* is deliberately not the package's job. It's a library, not a deployed service — it has no runtime identity of its own and nowhere more secure to keep a credential than wherever the importing project already keeps its secrets. The client accepts a resolved DSN as an argument (with an env-var read as a local-dev convenience, not a secrets mechanism); pulling the real value out of Vault, a cloud secrets manager, or a Kubernetes secret is entirely the responsibility of whichever project is being deployed, using whatever mechanism it already has for every other credential it manages. This mirrors how ordinary DB client libraries work — none of them integrate with a specific secrets backend, and this package doesn't either. The one place this touches credential lifecycle at all is `create-agent-role` (§14): it generates and hands back a new role's password at creation time, and its responsibility ends there — getting that value into the deploying project's vault is a step that project takes, not something the CLI automates.

One thing explicitly outside the package's reach: connection pooling infrastructure (PgBouncer or similar) is a separate process someone deploys in front of Postgres, not something a Python package installs. The package should be written to behave correctly under a pooler — not relying on session state that breaks under transaction-mode pooling — but standing the pooler up is an operational decision for whoever's running the database, made if and when enough concurrent agents make direct connections impractical.

## 16. Explicit non-goals

- **Not a general orchestration platform.** Ingestion scheduling belongs to whatever workflow system a project already runs (§13) — the package provides the ingestion function, not the scheduler.
- **Not a connection pooler.** Infrastructure that sits in front of Postgres is out of scope by nature, covered in §15.
- **Not its own auth system.** The package trusts Postgres's own authentication and role system entirely (§4) rather than layering anything on top.

## 17. Ideas for later

Speculative, more research-shaped directions — not committed, not sequenced, worth revisiting once the core system above is running on real usage data.

### Relations as group elements

Knowledge-graph embedding models (TransE, RotatE) represent a relation itself as an element of a group acting on entity vectors — TransE as a translation, using the vector space's addition (`head + relation ≈ tail`); RotatE as a rotation, using the structure of the rotation group. The schema's `next`/`prev` edge type is a clean example of a pair of *inverse* relations — exactly the structure these models are built around: apply `next`, then its group inverse, and you're back where you started. Worth exploring if relation-aware embeddings (not just content embeddings) ever become useful for understanding multi-hop composition — "two `next` hops," "the source of a source" — but this means training a separate relation-embedding model, a real step up in complexity from anything else in this document.

### Vector-space centroids

Embeddings already live in a genuine vector space, so linear combination is well-defined — a document-level centroid (the mean of its chunk embeddings) is a legitimate coarse pre-filter before drilling into individual chunks. Modest, but unused by the design above and easy to add later.

### Permissions as a lattice, not a group

Worth naming precisely rather than reaching for group theory where it doesn't fit: `document_access` composes by inclusion (a partial order / lattice), not by invertible operations — `REVOKE` isn't the group-inverse of `GRANT` in any clean symmetric sense. A different piece of algebra than the relation-embedding idea above, and the honest one for this part of the schema.

### Rank aggregation and permutation groups

RRF (§5) is a form of rank aggregation, and rank aggregation is formally studied via permutation groups — rankings as elements of the symmetric group *S_n* — in social choice theory. Interesting as a theoretical lens on why methods like RRF behave consistently; not something that changes what actually gets implemented.

---

*Knowledge Grove · design document, plan phase · last compiled 2026-08-23*
