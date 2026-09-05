import uuid

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    Boolean,
    CheckConstraint,
    Computed,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Index,
    Integer,
    Text,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import TSVECTOR, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from knowledge_grove.constants import (
    EdgeType,
    EMBEDDING_DIM,
    JUDGED_BY_VALUES,
    PERMISSIONS,
    SOURCE_METHODS,
)


class Base(DeclarativeBase):
    pass


def _uuid_pk() -> Mapped[uuid.UUID]:
    return mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        server_default=text("gen_random_uuid()"),
    )


class Document(Base):
    """One row per chunk. The atomic unit everything else attaches to."""

    __tablename__ = "documents"

    id: Mapped[uuid.UUID] = _uuid_pk()
    content: Mapped[str] = mapped_column(Text, nullable=False)
    # sha256 hex digest of `content` -- see utils/hashing.py. Used for exact-match
    # duplicate detection (add_document, add_raw_document reconciliation) via a
    # plain btree index, which a fixed-width hash supports but the unbounded
    # `content` column itself couldn't reliably.
    content_hash: Mapped[str] = mapped_column(Text, nullable=False)
    content_tsv: Mapped[str] = mapped_column(
        TSVECTOR,
        Computed("to_tsvector('english', content)", persisted=True),
        nullable=False,
    )
    embedding: Mapped[list[float]] = mapped_column(
        Vector(EMBEDDING_DIM), nullable=False
    )
    summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    summary_embedding: Mapped[list[float] | None] = mapped_column(
        Vector(EMBEDDING_DIM), nullable=True
    )
    source_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    owner_agent: Mapped[str] = mapped_column(Text, nullable=False)
    # Set when a newer revision supersedes this row (see the `supersedes` edge
    # created by update_document in crud.py). Old revisions are kept, never
    # deleted or mutated — this is what lets the graph double as an
    # organizational history, not just a filter for "is this current".
    deprecated: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default=text("false")
    )
    created_at: Mapped["DateTime"] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    tags: Mapped[list["DocumentTag"]] = relationship(
        back_populates="document", cascade="all, delete-orphan"
    )
    access_grants: Mapped[list["DocumentAccess"]] = relationship(
        back_populates="document", cascade="all, delete-orphan"
    )

    __table_args__ = (
        Index(
            "ix_documents_content_trgm",
            "content",
            postgresql_using="gin",
            postgresql_ops={"content": "gin_trgm_ops"},
        ),
        Index("ix_documents_content_tsv", "content_tsv", postgresql_using="gin"),
        Index("ix_documents_content_hash", "content_hash"),
        Index(
            "ix_documents_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )


class DocumentTag(Base):
    """One row per (document, tag) pair — an exact-match label plus why it applies."""

    __tablename__ = "document_tags"

    id: Mapped[uuid.UUID] = _uuid_pk()
    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False
    )
    tag: Mapped[str] = mapped_column(Text, nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped["DateTime"] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    document: Mapped["Document"] = relationship(back_populates="tags")

    __table_args__ = (Index("ix_document_tags_tag", "tag"),)


edge_type_enum = Enum(*(e.value for e in EdgeType), name="edge_type", create_type=False)


class Edge(Base):
    """One row per relationship between documents, or from a document out to a tool/URL."""

    __tablename__ = "edges"

    id: Mapped[uuid.UUID] = _uuid_pk()
    from_document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False
    )
    to_document_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), nullable=True
    )
    external_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    edge_type: Mapped[str] = mapped_column(edge_type_enum, nullable=False)
    # Short note on what this edge means / what's at the endpoint. Optional --
    # e.g. a `prev` edge in a chunk sequence is self-explanatory and doesn't
    # need one.
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped["DateTime"] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    __table_args__ = (
        CheckConstraint(
            "(to_document_id IS NOT NULL) != (external_url IS NOT NULL)",
            name="ck_edges_exactly_one_target",
        ),
        Index("ix_edges_from_document_id", "from_document_id"),
        Index("ix_edges_to_document_id", "to_document_id"),
    )


permission_enum = Enum(*PERMISSIONS, name="permission", create_type=False)


class DocumentAccess(Base):
    """One row per grant — who besides the owner can read or write a document."""

    __tablename__ = "document_access"

    id: Mapped[uuid.UUID] = _uuid_pk()
    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False
    )
    grantee_role: Mapped[str] = mapped_column(Text, nullable=False)
    permission: Mapped[str] = mapped_column(permission_enum, nullable=False)
    granted_at: Mapped["DateTime"] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    document: Mapped["Document"] = relationship(back_populates="access_grants")

    __table_args__ = (
        Index("ix_document_access_document_id", "document_id"),
        Index("ix_document_access_grantee_role", "grantee_role"),
    )


source_method_enum = Enum(*SOURCE_METHODS, name="source_method", create_type=False)
judged_by_enum = Enum(*JUDGED_BY_VALUES, name="judged_by", create_type=False)


class RetrievalFeedback(Base):
    """One row per (query, returned document) pair — feeds the ranking weights in §7."""

    __tablename__ = "retrieval_feedback"

    id: Mapped[uuid.UUID] = _uuid_pk()
    query_text: Mapped[str] = mapped_column(Text, nullable=False)
    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False
    )
    source_method: Mapped[str] = mapped_column(source_method_enum, nullable=False)
    rank: Mapped[int] = mapped_column(Integer, nullable=False)
    relevance: Mapped[float | None] = mapped_column(Float, nullable=True)
    judged_by: Mapped[str] = mapped_column(judged_by_enum, nullable=False)
    created_at: Mapped["DateTime"] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    __table_args__ = (
        Index("ix_retrieval_feedback_document_id", "document_id"),
    )
