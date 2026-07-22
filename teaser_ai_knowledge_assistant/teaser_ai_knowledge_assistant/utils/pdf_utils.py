from typing import TypedDict

import pymupdf


class PageText(TypedDict):
    page: int
    text: str


class TextChunk(TypedDict):
    text: str
    page: int
    chunk: int


def extract_pages_from_pdf(
    pdf_bytes: bytes,
) -> list[PageText]:
    """Extract readable text from each PDF page."""

    try:
        document = pymupdf.open(
            stream=pdf_bytes,
            filetype="pdf",
        )

    except Exception as exc:
        raise ValueError(
            f"The file could not be opened as a PDF: {exc}"
        ) from exc

    pages: list[PageText] = []

    try:
        for page_number, page in enumerate(
            document,
            start=1,
        ):
            text = page.get_text(
                "text",
                sort=True,
            ).strip()

            if text:
                pages.append(
                    {
                        "page": page_number,
                        "text": text,
                    }
                )

    finally:
        document.close()

    if not pages:
        raise ValueError(
            "No selectable text was found. The PDF may contain "
            "scanned images and would require OCR."
        )

    return pages


def chunk_pages(
    pages: list[PageText],
    chunk_size: int = 900,
    overlap: int = 150,
) -> list[TextChunk]:
    """Split page text into overlapping chunks."""

    if chunk_size <= 0:
        raise ValueError(
            "chunk_size must be greater than zero."
        )

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError(
            "overlap must be smaller than chunk_size."
        )

    chunks: list[TextChunk] = []
    global_chunk_number = 0

    for page in pages:
        text = " ".join(
            page["text"].split()
        )

        start = 0

        while start < len(text):
            end = min(
                start + chunk_size,
                len(text),
            )

            chunk_text = text[start:end].strip()

            if chunk_text:
                chunks.append(
                    {
                        "text": chunk_text,
                        "page": page["page"],
                        "chunk": global_chunk_number,
                    }
                )

                global_chunk_number += 1

            if end == len(text):
                break

            start = end - overlap

    return chunks