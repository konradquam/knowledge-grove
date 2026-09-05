import ast

import sqlparse

DEFAULT_MAX_CHARS = 2000


def _split_oversized(chunk: str, max_chars: int) -> list[str]:
    """Break `chunk` into pieces no longer than `max_chars`, cutting at the
    nearest preceding whitespace rather than mid-word. Last-resort fallback
    only -- see chunk_markdown's docstring for why this never runs first.
    """
    pieces = []
    remaining = chunk
    while len(remaining) > max_chars:
        split_at = remaining.rfind(" ", 0, max_chars)
        if split_at <= 0:
            split_at = max_chars
        pieces.append(remaining[:split_at].strip())
        remaining = remaining[split_at:].strip()
    if remaining:
        pieces.append(remaining)
    return pieces


def chunk_markdown(
    content: str,
    max_chars: int | None = DEFAULT_MAX_CHARS,
) -> list[str]:
    """Split a markdown string into chunks based on headings and paragraphs.

    `max_chars` is a last-resort fallback, applied only after structural
    splitting: a chunk that's still too large (a single huge paragraph with
    no blank lines, say) gets broken further at word boundaries. Structure
    always wins when it's available -- this only ever kicks in when there's
    no heading/paragraph break to use instead. Pass `None` to disable it.
    """
    # Split the content into lines
    lines = content.splitlines()
    chunks = []
    current_chunk = []
    previous_line = None

    for line in lines:
        stripped_line = line.strip()
        if stripped_line.startswith("#"):
            # If we encounter a heading, we start a new chunk
            if current_chunk:
                chunks.append("".join(current_chunk).strip())
            current_chunk = []
            current_chunk.append(f'{stripped_line}\n')
        elif stripped_line == "" and previous_line and not previous_line.startswith("#"):
            # If we encounter an empty line, we consider it as a paragraph break
            if current_chunk:
                chunks.append("".join(current_chunk).strip())
            current_chunk = []
        else:
            # Otherwise, we add the line to the current chunk
            current_chunk.append(f'{stripped_line}\n')
        previous_line = stripped_line

    # Add any remaining content as the last chunk
    if current_chunk:
        chunks.append("".join(current_chunk).strip())

    chunks = [chunk for chunk in chunks if chunk]  # Filter out any empty chunks

    if max_chars is not None:
        chunks = [piece for chunk in chunks for piece in _split_oversized(chunk, max_chars)]

    return chunks


def _source_span(node: ast.AST, source_lines: list[str]) -> str:
    """The exact source text for `node`, including any decorators.

    ast.get_source_segment doesn't include decorators in a decorated
    function/class's span (node.lineno points at the `def`/`class` keyword,
    not the first `@decorator` line) -- which would silently drop a
    `@tool` decorator (relevant to §13's tool discovery) from its chunk.
    """
    start_line = node.lineno
    if getattr(node, "decorator_list", None):
        start_line = min(start_line, min(d.lineno for d in node.decorator_list))
    return "\n".join(source_lines[start_line - 1 : node.end_lineno])


def chunk_python(content: str) -> list[str]:
    """Split Python source into one chunk per top-level function/class
    definition, using the real parser (`ast`) rather than a text splitter --
    cutting a function or class body in half destroys its meaning (§11 of
    the design doc). Consecutive plain top-level statements (imports,
    module-level constants, etc.) are collapsed into one chunk each run,
    rather than one chunk per line.

    Raises SyntaxError on invalid Python -- there's no safe text-splitter
    fallback to fall back to without violating the "never a text splitter"
    rule this function exists to satisfy.
    """
    tree = ast.parse(content)
    source_lines = content.splitlines()
    chunks = []
    pending: list[str] = []

    def flush_pending():
        if pending:
            chunks.append("\n".join(pending).strip())
            pending.clear()

    for node in tree.body:
        segment = _source_span(node, source_lines)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            flush_pending()
            chunks.append(segment.strip())
        else:
            pending.append(segment)

    flush_pending()
    return [chunk for chunk in chunks if chunk]


def chunk_sql(content: str) -> list[str]:
    """Split a SQL script into one chunk per statement, using sqlparse's
    statement-aware splitting -- correctly handles a semicolon inside a
    string literal or comment, unlike a naive text split on ';' (§11 of the
    design doc: a real parser, never a text splitter).
    """
    return [stmt.strip() for stmt in sqlparse.split(content) if stmt.strip()]
