import pytest

from knowledge_grove.utils.chunking import chunk_markdown, chunk_python, chunk_sql


def test_single_paragraph_no_heading_is_one_chunk():
    assert chunk_markdown("just one paragraph\nsecond line\n") == [
        "just one paragraph\nsecond line"
    ]


def test_heading_immediately_followed_by_body_merges_into_one_chunk():
    content = "# Heading One\npara one line one\npara one line two\n"
    assert chunk_markdown(content) == ["# Heading One\npara one line one\npara one line two"]


def test_heading_with_single_blank_line_before_body_still_merges():
    content = "# H1\n\npara one\n"
    assert chunk_markdown(content) == ["# H1\n\npara one"]


def test_heading_with_multiple_blank_lines_before_body_still_merges():
    content = "# H1\n\n\npara one\n"
    assert chunk_markdown(content) == ["# H1\n\n\npara one"]


def test_two_paragraphs_separated_by_single_blank_line_split():
    content = "para a\n\npara b\n"
    assert chunk_markdown(content) == ["para a", "para b"]


def test_two_paragraphs_separated_by_multiple_blank_lines_still_split_cleanly():
    content = "para a\n\n\npara b\n"
    assert chunk_markdown(content) == ["para a", "para b"]


def test_multiple_headings_each_start_their_own_chunk():
    content = "# Heading One\n\npara one line one\npara one line two\n\npara two\n\n# Heading Two\npara three\n"
    assert chunk_markdown(content) == [
        "# Heading One\n\npara one line one\npara one line two",
        "para two",
        "# Heading Two\npara three",
    ]


def test_heading_followed_immediately_by_another_heading_is_its_own_chunk():
    content = "# H1\n\n## H2\ncontent\n"
    assert chunk_markdown(content) == ["# H1", "## H2\ncontent"]


def test_leading_blank_line_does_not_crash():
    assert chunk_markdown("\nHello world\n") == ["Hello world"]


def test_leading_blank_lines_before_a_heading_do_not_crash():
    assert chunk_markdown("\n\n# H1\ncontent\n") == ["# H1\ncontent"]


def test_empty_string_returns_no_chunks():
    assert chunk_markdown("") == []


def test_only_whitespace_returns_no_chunks():
    assert chunk_markdown("\n\n   \n\n") == []


def test_trailing_blank_lines_do_not_produce_empty_chunk():
    content = "para a\n\n\n\n"
    assert chunk_markdown(content) == ["para a"]


def test_lines_are_stripped_of_surrounding_whitespace():
    content = "   para with leading spaces   \nsecond line\t\n"
    assert chunk_markdown(content) == ["para with leading spaces\nsecond line"]


def test_bare_hash_line_counts_as_a_heading():
    content = "para a\n\n#\ncontent under bare heading\n"
    assert chunk_markdown(content) == ["para a", "#\ncontent under bare heading"]


def test_no_trailing_newline_still_captures_last_line():
    assert chunk_markdown("para a\nsecond line") == ["para a\nsecond line"]


def test_chunks_preserve_document_order():
    content = "# First\nbody 1\n\n# Second\nbody 2\n\n# Third\nbody 3\n"
    result = chunk_markdown(content)
    assert result == [
        "# First\nbody 1",
        "# Second\nbody 2",
        "# Third\nbody 3",
    ]


def test_oversized_paragraph_gets_split_by_max_chars():
    content = ("word " * 1000).strip() + "\n"
    result = chunk_markdown(content, max_chars=100)
    assert len(result) > 1
    assert all(len(chunk) <= 100 for chunk in result)


def test_max_chars_split_does_not_break_a_word_in_half():
    content = ("word " * 1000).strip() + "\n"
    result = chunk_markdown(content, max_chars=100)
    for chunk in result:
        for token in chunk.split():
            assert token == "word"


def test_max_chars_none_disables_the_fallback():
    content = ("word " * 1000).strip() + "\n"
    result = chunk_markdown(content, max_chars=None)
    assert len(result) == 1
    assert len(result[0]) > 2000


def test_oversized_chunk_with_no_whitespace_hard_cuts_at_max_chars():
    content = "a" * 250 + "\n"
    result = chunk_markdown(content, max_chars=100)
    assert result == ["a" * 100, "a" * 100, "a" * 50]


def test_structural_split_still_applies_before_size_fallback():
    content = "# Heading\n\npara one\n\npara two\n"
    result = chunk_markdown(content, max_chars=100)
    assert result == ["# Heading\n\npara one", "para two"]


def test_default_max_chars_leaves_ordinary_content_untouched():
    content = "just one paragraph\nsecond line\n"
    assert chunk_markdown(content) == ["just one paragraph\nsecond line"]


def test_chunk_python_splits_top_level_functions_into_separate_chunks():
    content = "def foo():\n    return 1\n\n\ndef bar():\n    return 2\n"
    assert chunk_python(content) == ["def foo():\n    return 1", "def bar():\n    return 2"]


def test_chunk_python_splits_top_level_classes_into_separate_chunks():
    content = "class Foo:\n    pass\n\n\nclass Bar:\n    pass\n"
    assert chunk_python(content) == ["class Foo:\n    pass", "class Bar:\n    pass"]


def test_chunk_python_keeps_decorator_with_its_function():
    content = "@tool\ndef foo(x):\n    return x + 1\n"
    assert chunk_python(content) == ["@tool\ndef foo(x):\n    return x + 1"]


def test_chunk_python_keeps_multiple_stacked_decorators_with_its_function():
    content = "@staticmethod\n@another_decorator\ndef bar():\n    pass\n"
    assert chunk_python(content) == ["@staticmethod\n@another_decorator\ndef bar():\n    pass"]


def test_chunk_python_collapses_consecutive_top_level_statements_into_one_chunk():
    content = "import os\nimport sys\n\nCONST = 5\n\ndef foo():\n    pass\n"
    result = chunk_python(content)
    assert result == ["import os\nimport sys\nCONST = 5", "def foo():\n    pass"]


def test_chunk_python_preserves_document_order():
    content = "import os\n\ndef foo():\n    pass\n\nCONST = 5\n\nclass Bar:\n    pass\n"
    result = chunk_python(content)
    assert result == [
        "import os",
        "def foo():\n    pass",
        "CONST = 5",
        "class Bar:\n    pass",
    ]


def test_chunk_python_does_not_split_out_nested_functions():
    content = "def outer():\n    def inner():\n        return 1\n    return inner()\n"
    result = chunk_python(content)
    assert result == ["def outer():\n    def inner():\n        return 1\n    return inner()"]


def test_chunk_python_empty_string_returns_no_chunks():
    assert chunk_python("") == []


def test_chunk_python_invalid_syntax_raises_syntax_error():
    with pytest.raises(SyntaxError):
        chunk_python("def foo(:\n    pass\n")


def test_chunk_sql_splits_multiple_statements():
    content = "SELECT 1;\n\nINSERT INTO logs (msg) VALUES ('done');\n"
    assert chunk_sql(content) == ["SELECT 1;", "INSERT INTO logs (msg) VALUES ('done');"]


def test_chunk_sql_does_not_split_on_semicolon_inside_string_literal():
    content = "SELECT * FROM users WHERE name = 'a;b';\n"
    assert chunk_sql(content) == ["SELECT * FROM users WHERE name = 'a;b';"]


def test_chunk_sql_single_statement_without_trailing_semicolon():
    assert chunk_sql("SELECT 1") == ["SELECT 1"]


def test_chunk_sql_empty_string_returns_no_chunks():
    assert chunk_sql("") == []


def test_chunk_sql_whitespace_only_returns_no_chunks():
    assert chunk_sql("   \n\n  ") == []
