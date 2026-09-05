from knowledge_grove.utils.hashing import hash_content


def test_hash_content_is_deterministic():
    assert hash_content("hello world") == hash_content("hello world")


def test_hash_content_differs_for_different_content():
    assert hash_content("hello world") != hash_content("hello world!")


def test_hash_content_is_a_64_char_hex_digest():
    digest = hash_content("anything")
    assert len(digest) == 64
    assert all(c in "0123456789abcdef" for c in digest)
