from sqlalchemy import func, select
from sqlalchemy.orm import Session

from knowledge_grove.models import Document, DocumentTag

K = 60
METHOD_WEIGHTS = {"vector": 1, "fulltext": 1, "ilike": 1}  # Weights for vector, fulltext, and ilike scores respectively
LIMIT = 10  # Default limit for search results

def search_vector(
    session: Session,
    query_embedding: list[float],
    limit: int = LIMIT,
    tags: list[str] | None = None,
    include_deprecated: bool = False,
) -> list[tuple[Document, float]]:
    """Nearest neighbors by cosine similarity. Returns (document, similarity) pairs, best first."""
    distance = Document.embedding.cosine_distance(query_embedding)
    stmt = select(Document, distance).order_by(distance).limit(limit)
    if tags:
        filtered_doc_ids = [doc.id for doc in search_tags(session, tags, limit, include_deprecated)]
        stmt = stmt.where(Document.id.in_(filtered_doc_ids))
    if not include_deprecated:
        stmt = stmt.where(Document.deprecated.is_(False))

    return [(doc, 1 - dist) for doc, dist in session.execute(stmt).all()]


def search_tag(
    session: Session,
    tag: str,
    limit: int = LIMIT,
    include_deprecated: bool = False,
) -> list[Document]:
    """Exact-match documents carrying `tag` (normalized the same way add_tag stores it)."""
    normalized = tag.strip().lower()
    stmt = (
        select(Document)
        .join(DocumentTag, DocumentTag.document_id == Document.id)
        .where(DocumentTag.tag == normalized)
        .order_by(Document.created_at.desc())
        .limit(limit)
    )
    if not include_deprecated:
        stmt = stmt.where(Document.deprecated.is_(False))

    return list(session.scalars(stmt).all())

def search_tags(
    session: Session,
    tags: list[str],
    limit: int = LIMIT,
    include_deprecated: bool = False,
) -> list[Document]:
    """Exact-match documents carrying any of the `tags` (normalized the same way add_tag stores them)."""
    normalized_tags = [tag.strip().lower() for tag in tags]
    docs = set()
    for tag in normalized_tags:
        docs.update(search_tag(session, tag, limit, include_deprecated))
    return list(docs)


def search_fulltext(
    session: Session,
    query_text: str,
    limit: int = LIMIT,
    tags: list[str] | None = None,
    include_deprecated: bool = False,
) -> list[tuple[Document, float]]:
    """Word/stem matching over `content` via tsvector. Returns (document, ts_rank) pairs, best first."""
    tsquery = func.plainto_tsquery("english", query_text)
    rank = func.ts_rank(Document.content_tsv, tsquery)
    stmt = (
        select(Document, rank)
        .where(Document.content_tsv.op("@@")(tsquery))
        .order_by(rank.desc())
        .limit(limit)
    )
    if tags:
        filtered_doc_ids = [doc.id for doc in search_tags(session, tags, limit, include_deprecated)]
        stmt = stmt.where(Document.id.in_(filtered_doc_ids))
    if not include_deprecated:
        stmt = stmt.where(Document.deprecated.is_(False))

    return [(doc, score) for doc, score in session.execute(stmt).all()]


def search_ilike(
    session: Session,
    pattern: str,
    limit: int = LIMIT,
    tags: list[str] | None = None,
    include_deprecated: bool = False,
) -> list[tuple[Document, float]]:
    """Literal substring match over `content` (`%`/`_` in `pattern` are treated as literal characters,
    not wildcards), ranked by trigram similarity. Returns (document, similarity) pairs, best first.
    """
    escaped = pattern.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    similarity = func.similarity(Document.content, pattern)
    stmt = (
        select(Document, similarity)
        .where(Document.content.ilike(f"%{escaped}%", escape="\\"))
        .order_by(similarity.desc())
        .limit(limit)
    )
    if tags:
        filtered_doc_ids = [doc.id for doc in search_tags(session, tags, limit, include_deprecated)]
        stmt = stmt.where(Document.id.in_(filtered_doc_ids))
    if not include_deprecated:
        stmt = stmt.where(Document.deprecated.is_(False))

    return [(doc, score) for doc, score in session.execute(stmt).all()]

def weighted_search(
    session: Session,
    query_embedding: list[float],
    query_text: str,
    pattern: str,
    limit: int = LIMIT,
    include_deprecated: bool = False,
) -> list[tuple[Document, float]]:
    """Combined search that waits for all three methods to complete and merges results."""
    vector_results = search_vector(session, query_embedding, limit, include_deprecated)
    fulltext_results = search_fulltext(session, query_text, limit, include_deprecated)
    ilike_results = search_ilike(session, pattern, limit, include_deprecated)

    # Merge results by document ID and take the best score from each method
    merged_scores = {}
    #add in vector results
    for rank, (doc, score) in enumerate(vector_results):
        if doc.id not in merged_scores:
            merged_scores[doc.id] = (doc, adjusted_score("vector", rank+1))
        else:
            merged_scores[doc.id] = (doc, merged_scores[doc.id][1] + adjusted_score("vector", rank+1))
    # add in fulltext results
    for rank, (doc, score) in enumerate(fulltext_results):
        if doc.id not in merged_scores:
            merged_scores[doc.id] = (doc, adjusted_score("fulltext", rank+1))
        else:
            merged_scores[doc.id] = (doc, merged_scores[doc.id][1] + adjusted_score("fulltext", rank+1))
    # add in ilike results
    for rank, (doc, score) in enumerate(ilike_results):
        if doc.id not in merged_scores:
            merged_scores[doc.id] = (doc, adjusted_score("ilike", rank+1))
        else:
            merged_scores[doc.id] = (doc, merged_scores[doc.id][1] + adjusted_score("ilike", rank+1))

    # Sort by score descending and return top `limit` results
    sorted_results = sorted(merged_scores.values(), key=lambda x: x[1], reverse=True)
    return sorted_results[:limit]

def adjusted_score(method: str, rank: float) -> float:
    """Adjust the score to a common scale for merging results from different search methods."""
    return METHOD_WEIGHTS[method] / (K + rank)