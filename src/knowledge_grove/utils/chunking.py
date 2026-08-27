def chunk_markdown(
    content: str,
) -> list[str]:
    """Split a markdown string into chunks based on headings and paragraphs."""
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

    return [chunk for chunk in chunks if chunk]  # Filter out any empty chunks