"""initial schema

Revision ID: ec7d68ee91b4
Revises:
Create Date: 2026-08-23 15:01:59.437062

Creates the four core tables plus retrieval_feedback (documents,
document_tags, edges, document_access, retrieval_feedback), their indexes,
and row-level security.

Must be run by a role with CREATE EXTENSION / CREATE POLICY / table-owner
privileges (§14, §4 of the design doc) — never the role an ordinary agent
connects as, since table owners bypass RLS by default and this migration
relies on that: the `kg_has_document_grant` helper function below is
SECURITY DEFINER precisely so its internal lookup against `document_access`
bypasses that table's own RLS policy. See the comment above its definition
for why it's deliberately scoped to document_access only, never documents —
that split is what avoids both a documents/document_access RLS recursion
cycle and a separate INSERT...RETURNING visibility gotcha.

`retrieval_feedback` is deliberately left without RLS: §7 of the design doc
is explicit that this table is shared across every agent's namespace so
usage patterns learned from one agent's queries can inform ranking for
everyone else — scoping it per-owner would defeat that.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

from knowledge_grove.models import (
    Document,
    DocumentTag,
    Edge,
    DocumentAccess,
    RetrievalFeedback,
    Base,
)
from knowledge_grove.constants import (
    EdgeType,
    PERMISSIONS,
    SOURCE_METHODS,
    JUDGED_BY_VALUES,
    SHARED_READER,
)


def _create_enum_sql(type_name: str, values: tuple[str, ...]) -> str:
    joined = ", ".join(f"'{v}'" for v in values)
    return f"CREATE TYPE {type_name} AS ENUM ({joined})"

# revision identifiers, used by Alembic.
revision: str = 'ec7d68ee91b4'
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()

    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")

    op.execute(_create_enum_sql("edge_type", tuple(e.value for e in EdgeType)))
    op.execute(_create_enum_sql("permission", PERMISSIONS))
    op.execute(_create_enum_sql("source_method", SOURCE_METHODS))
    op.execute(_create_enum_sql("judged_by", JUDGED_BY_VALUES))

    Base.metadata.create_all(
        bind=bind,
        tables=[
            Document.__table__,
            DocumentTag.__table__,
            Edge.__table__,
            DocumentAccess.__table__,
            RetrievalFeedback.__table__,
        ],
    )

    # --- RLS helper function -------------------------------------------
    # SECURITY DEFINER, and deliberately narrow: it only ever queries
    # document_access, never documents. Two reasons that narrowness matters:
    #
    # 1. Recursion: documents' own policies (below) call this to check
    #    grants; document_access's own policies (below) separately query
    #    documents directly. If this function *also* queried documents, the
    #    two RLS-guarded tables would reference each other in a cycle.
    #    Because SECURITY DEFINER makes its internal query run as the
    #    (RLS-bypassing) owner role that applied this migration, restricting
    #    that bypass to document_access alone breaks the cycle without
    #    needing to bypass documents' own RLS anywhere.
    # 2. INSERT ... RETURNING: Postgres evaluates a table's SELECT policy
    #    against a row that table's own INSERT just produced, in the same
    #    command — and a command cannot see rows it inserted itself via an
    #    ordinary re-SELECT of that same table. A function that re-queried
    #    documents from within documents' own policy would therefore see no
    #    row at all right after add_document() and always deny it. Since
    #    this function never touches documents, documents' policies below
    #    check ownership via the row's own `owner_agent` column directly
    #    (no query needed) and only delegate to this function for the
    #    grant-based path.
    #
    # Note the explicit `caller` parameter: inside a SECURITY DEFINER
    # function, `current_user` resolves to the function's *owner*, not the
    # original caller, so it can't be referenced directly in the body.
    # Callers pass `current_user` in from the policy clause itself, where it
    # still correctly resolves to the real invoking role.
    op.execute(
        """
        CREATE FUNCTION kg_has_document_grant(doc_id uuid, caller name, require_write boolean)
        RETURNS boolean AS $$
            SELECT EXISTS (
                SELECT 1 FROM document_access
                WHERE document_access.document_id = doc_id
                  AND (NOT require_write OR document_access.permission = 'write')
                  AND pg_has_role(caller, document_access.grantee_role, 'MEMBER')
            )
        $$ LANGUAGE sql STABLE SECURITY DEFINER SET search_path = public
        """
    )

    # --- documents ---------------------------------------------------------
    op.execute("ALTER TABLE documents ENABLE ROW LEVEL SECURITY")
    op.execute(
        """
        CREATE POLICY document_select_policy ON documents
        FOR SELECT USING (
            owner_agent = current_user
            OR kg_has_document_grant(id, current_user, false)
        )
        """
    )
    op.execute(
        """
        CREATE POLICY document_insert_policy ON documents
        FOR INSERT WITH CHECK (owner_agent = current_user)
        """
    )
    op.execute(
        """
        CREATE POLICY document_update_policy ON documents
        FOR UPDATE USING (
            owner_agent = current_user
            OR kg_has_document_grant(id, current_user, true)
        )
        WITH CHECK (
            owner_agent = current_user
            OR kg_has_document_grant(id, current_user, true)
        )
        """
    )
    op.execute(
        """
        CREATE POLICY document_delete_policy ON documents
        FOR DELETE USING (owner_agent = current_user)
        """
    )

    # --- document_tags ------------------------------------------------------
    # document_id refers to a document from a *different*, already-committed
    # INSERT (add_tag always runs after add_document's own flush), so a
    # plain subquery into documents — filtered by documents' own RLS above —
    # is both safe from recursion and unaffected by the same-command
    # visibility issue documents' policies had to route around.
    op.execute("ALTER TABLE document_tags ENABLE ROW LEVEL SECURITY")
    op.execute(
        """
        CREATE POLICY document_tags_select_policy ON document_tags
        FOR SELECT USING (
            EXISTS (SELECT 1 FROM documents WHERE documents.id = document_tags.document_id)
        )
        """
    )
    op.execute(
        """
        CREATE POLICY document_tags_write_policy ON document_tags
        FOR ALL USING (
            EXISTS (
                SELECT 1 FROM documents
                WHERE documents.id = document_tags.document_id
                  AND documents.owner_agent = current_user
            )
            OR kg_has_document_grant(document_tags.document_id, current_user, true)
        )
        WITH CHECK (
            EXISTS (
                SELECT 1 FROM documents
                WHERE documents.id = document_tags.document_id
                  AND documents.owner_agent = current_user
            )
            OR kg_has_document_grant(document_tags.document_id, current_user, true)
        )
        """
    )

    # --- edges ---------------------------------------------------------------
    # Visibility follows the source (`from_document_id`) side of the edge —
    # an edge is metadata attached to the document it originates from.
    op.execute("ALTER TABLE edges ENABLE ROW LEVEL SECURITY")
    op.execute(
        """
        CREATE POLICY edges_select_policy ON edges
        FOR SELECT USING (
            EXISTS (SELECT 1 FROM documents WHERE documents.id = edges.from_document_id)
        )
        """
    )
    op.execute(
        """
        CREATE POLICY edges_write_policy ON edges
        FOR ALL USING (
            EXISTS (
                SELECT 1 FROM documents
                WHERE documents.id = edges.from_document_id
                  AND documents.owner_agent = current_user
            )
            OR kg_has_document_grant(edges.from_document_id, current_user, true)
        )
        WITH CHECK (
            EXISTS (
                SELECT 1 FROM documents
                WHERE documents.id = edges.from_document_id
                  AND documents.owner_agent = current_user
            )
            OR kg_has_document_grant(edges.from_document_id, current_user, true)
        )
        """
    )

    # --- document_access -------------------------------------------------
    # Flagged explicitly in §4 of the design doc: without this, any agent
    # could grant itself access to someone else's document. Only a
    # document's owner may create/modify/delete its grants; a grantee may
    # see grants naming it, so it can tell what it's been given access to.
    op.execute("ALTER TABLE document_access ENABLE ROW LEVEL SECURITY")
    op.execute(
        """
        CREATE POLICY document_access_select_policy ON document_access
        FOR SELECT USING (
            EXISTS (
                SELECT 1 FROM documents
                WHERE documents.id = document_access.document_id
                  AND documents.owner_agent = current_user
            )
            OR pg_has_role(current_user, grantee_role, 'MEMBER')
        )
        """
    )
    op.execute(
        """
        CREATE POLICY document_access_owner_write_policy ON document_access
        FOR ALL USING (
            EXISTS (
                SELECT 1 FROM documents
                WHERE documents.id = document_access.document_id
                  AND documents.owner_agent = current_user
            )
        )
        WITH CHECK (
            EXISTS (
                SELECT 1 FROM documents
                WHERE documents.id = document_access.document_id
                  AND documents.owner_agent = current_user
            )
        )
        """
    )
    op.execute(
        f"""
        CREATE ROLE {SHARED_READER}
        """
    )


def downgrade() -> None:
    bind = op.get_bind()

    # Tables first — dropping them drops their policies too, which must
    # happen before the functions those policies call are dropped.
    Base.metadata.drop_all(
        bind=bind,
        tables=[
            RetrievalFeedback.__table__,
            DocumentAccess.__table__,
            Edge.__table__,
            DocumentTag.__table__,
            Document.__table__,
        ],
    )

    op.execute("DROP FUNCTION IF EXISTS kg_has_document_grant(uuid, name, boolean)")

    for enum_name in ("edge_type", "permission", "source_method", "judged_by"):
        op.execute(f"DROP TYPE IF EXISTS {enum_name}")
