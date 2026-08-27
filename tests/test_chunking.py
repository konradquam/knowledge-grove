from knowledge_grove.utils.chunking import chunk_markdown


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
