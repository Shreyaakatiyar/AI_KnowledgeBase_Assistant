from dataclasses import dataclass, field
from pathlib import Path
from pypdf import PdfReader

from app.core.config import get_settings
from app.core.exceptions import DocumentIngestionError, EmptyDocumentError
from app.core.logging_config import get_logger

logger = get_logger(__name__)
settings = get_settings()


@dataclass
class Chunk:
    text: str
    source: str         
    chunk_id: str        
    page_number: int
    metadata: dict = field(default_factory=dict)


def extract_text_by_page(pdf_path: Path) -> list[tuple[int, str]]:
    try:
        reader = PdfReader(str(pdf_path))
    except Exception as e:
        raise DocumentIngestionError(f"Failed to open PDF '{pdf_path.name}': {e}") from e

    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            pages.append((i, text))

    if not pages:
        raise EmptyDocumentError(
            f"'{pdf_path.name}' has no extractable text. "
            "It may be a scanned image PDF requiring OCR."
        )

    return pages


def recursive_split(text: str, chunk_size: int, overlap: int) -> list[str]:
    separators = ["\n\n", "\n", ". ", " ", ""]

    def _split(text: str, seps: list[str]) -> list[str]:
        if len(text) <= chunk_size:
            return [text]
        if not seps:
            return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

        sep = seps[0]
        parts = text.split(sep) if sep else list(text)

        chunks, current = [], ""
        for part in parts:
            candidate = current + (sep if current else "") + part
            if len(candidate) <= chunk_size:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                if len(part) > chunk_size:
                    chunks.extend(_split(part, seps[1:]))
                    current = ""
                else:
                    current = part
        if current:
            chunks.append(current)
        return chunks

    raw_chunks = _split(text, separators)

    overlapped = []
    for i, chunk in enumerate(raw_chunks):
        if i == 0:
            overlapped.append(chunk)
        else:
            prev_tail = raw_chunks[i - 1][-overlap:] if overlap > 0 else ""
            overlapped.append(prev_tail + chunk)
    return [c.strip() for c in overlapped if c.strip()]


def ingest_pdf(pdf_path: Path) -> list[Chunk]:
    logger.info(f"Ingesting document: {pdf_path.name}")
    pages = extract_text_by_page(pdf_path)

    chunks: list[Chunk] = []
    chunk_counter = 0
    for page_number, page_text in pages:
        page_chunks = recursive_split(page_text, settings.chunk_size, settings.chunk_overlap)
        for chunk_text in page_chunks:
            chunk_counter += 1
            chunks.append(Chunk(
                text=chunk_text,
                source=pdf_path.name,
                chunk_id=f"{pdf_path.name}::chunk_{chunk_counter}",
                page_number=page_number,
                metadata={"source": pdf_path.name, "page": page_number},
            ))

    logger.info(f"Created {len(chunks)} chunks from {pdf_path.name}")
    return chunks


def ingest_directory(directory: Path) -> list[Chunk]:
    pdf_files = list(directory.glob("*.pdf"))
    if not pdf_files:
        raise DocumentIngestionError(f"No PDF files found in {directory}")

    all_chunks: list[Chunk] = []
    for pdf_path in pdf_files:
        try:
            all_chunks.extend(ingest_pdf(pdf_path))
        except EmptyDocumentError as e:
            logger.warning(str(e))
            continue

    logger.info(f"Total chunks across all documents: {len(all_chunks)}")
    return all_chunks


if __name__ == "__main__":
    setup_logging_needed = True
    from app.core.logging_config import setup_logging
    setup_logging()

    docs_dir = Path(settings.vector_store_path).parent / "documents"
    chunks = ingest_directory(docs_dir)
    print(f"\nSample chunk:\n{'-'*50}")
    print(f"Source: {chunks[0].source} (page {chunks[0].page_number})")
    print(f"Text: {chunks[0].text[:200]}...")