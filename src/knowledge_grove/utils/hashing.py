import hashlib


def hash_content(content: str) -> str:
    """A stable, fixed-width fingerprint of `content`.

    Used for exact-match duplicate detection instead of indexing/comparing
    the (unbounded-length) `content` column directly -- Postgres btree
    indexes cap out at roughly 2.7KB per row, which a large chunk could
    exceed, but a hash never will.
    """
    return hashlib.sha256(content.encode()).hexdigest()
