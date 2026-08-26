from sqlalchemy import func, select
from sqlalchemy.orm import Session

from knowledge_grove.models import Document, DocumentTag


def search_vector(
    session: Session,
    query_embedding: list[float],
    limit: int = 10,
    include_deprecated: bool = False,
) -> list[tuple[Document, float]]:
    """Nearest neighbors by cosine similarity. Returns (document, similarity) pairs, best first."""
    distance = Document.embedding.cosine_distance(query_embedding)
    stmt = select(Document, distance).order_by(distance).limit(limit)
    if not include_deprecated:
        stmt = stmt.where(Document.deprecated.is_(False))

    return [(doc, 1 - dist) for doc, dist in session.execute(stmt).all()]


def search_tags(
    session: Session,
    tag: str,
    limit: int = 10,
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


def search_fulltext(
    session: Session,
    query_text: str,
    limit: int = 10,
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
    if not include_deprecated:
        stmt = stmt.where(Document.deprecated.is_(False))

    return [(doc, score) for doc, score in session.execute(stmt).all()]


def search_ilike(
    session: Session,
    pattern: str,
    limit: int = 10,
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
    if not include_deprecated:
        stmt = stmt.where(Document.deprecated.is_(False))

    return [(doc, score) for doc, score in session.execute(stmt).all()]
