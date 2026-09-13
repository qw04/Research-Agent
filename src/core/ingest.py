import json
import re
from pathlib import Path
from unstructured.partition.pdf import partition_pdf
from langchain_text_splitters import RecursiveCharacterTextSplitter

PAPERS_DIR = Path("papers")
OUTPUT_FILE = Path("data/chunks.json")

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

# Element categories that add noise, not signal — drop these before chunking.
# "Table" is excluded here because bare table captions/fragments rank close
# to real queries without adding useful content; proper table extraction
# would need separate handling if you want it searchable later.
SKIP_CATEGORIES = {"Header", "Footer", "PageNumber", "Table"}

# Matches section headings like "References", "7. References", "Bibliography",
# "Works Cited" — case-insensitive, tolerant of leading numbering/punctuation.
# Checked against ANY element category, not just Title/Header, because
# unstructured's strategy="fast" layout detection sometimes misclassifies
# the heading as NarrativeText.
REFERENCES_HEADING_RE = re.compile(
    r"^\s*(\d+[\.\)]?\s*)?(references|bibliography|works cited)\s*[:.]?\s*$",
    re.IGNORECASE,
)

# Fallback signal for papers where no heading is detected at all (confirmed
# to happen — some papers' reference lists are a stream of plain NarrativeText
# elements with no heading element preceding them). Matches individual
# reference-list entries in either IEEE style ("[6] Author...") or
# APA style ("Surname, A. B. ... (YYYY)").
REFERENCE_ENTRY_RE = re.compile(
    r"^\s*\[\d+\]\s+[A-Z]"
    r"|^[A-Z][a-zA-Z\-]+,\s[A-Z]\.\s*[A-Z]?\.?,?\s.*\(\d{4}\)"
)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", " ", ""],
    length_function=len,
)


def is_references_heading(el) -> bool:
    text = (el.text or "").strip()
    return bool(REFERENCES_HEADING_RE.match(text))


def extract_text(pdf_path: Path) -> str:
    """Partition the PDF into elements, drop noise categories, and stop
    once the References/Bibliography section starts — detected either via
    an explicit heading element, or via a run of reference-shaped entries
    when no clean heading exists."""
    elements = partition_pdf(filename=str(pdf_path), strategy="fast")

    parts = []
    consecutive_ref_entries = 0

    for el in elements:
        text = (el.text or "").strip()
        if not text:
            continue

        # Primary signal: an explicit References/Bibliography heading
        if is_references_heading(el):
            break

        # Fallback signal: a run of reference-shaped entries with no
        # detected heading — two in a row is enough, since body text
        # essentially never has two consecutive elements each starting
        # with a bracketed citation number or "Surname, A. (YYYY)"
        if REFERENCE_ENTRY_RE.match(text):
            consecutive_ref_entries += 1
            if consecutive_ref_entries >= 2:
                break
        else:
            consecutive_ref_entries = 0

        if el.category in SKIP_CATEGORIES:
            continue

        parts.append(text)

    return "\n\n".join(parts)


def chunk_text(text: str, source: str) -> list[dict]:
    raw_chunks = splitter.split_text(text)
    chunks = []
    for idx, chunk in enumerate(raw_chunks):
        chunk = chunk.strip()
        if chunk:
            chunks.append({
                "id": f"{source}_{idx}",
                "source": source,
                "text": chunk
            })
    return chunks


def main():
    all_chunks = []
    pdf_files = list(PAPERS_DIR.glob("*.pdf"))
    print(f"Found {len(pdf_files)} PDFs")

    for pdf_path in pdf_files:
        print(f"Processing {pdf_path.name}...")
        try:
            text = extract_text(pdf_path)
            chunks = chunk_text(text, pdf_path.stem)
            all_chunks.extend(chunks)
            print(f"  -> {len(chunks)} chunks")
        except Exception as e:
            print(f"  FAILED: {e}")

    OUTPUT_FILE.parent.mkdir(exist_ok=True, parents=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(all_chunks, f, indent=2)

    print(f"\nTotal: {len(all_chunks)} chunks written to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()