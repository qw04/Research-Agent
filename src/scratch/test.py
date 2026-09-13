import json, re

REFERENCE_ENTRY_RE = re.compile(
    r"^\s*\[\d+\]\s+[A-Z]"
    r"|^[A-Z][a-zA-Z\-]+,\s[A-Z]\.\s*[A-Z]?\.?,?\s.*\(\d{4}\)"
)

data = json.load(open("data/chunks.json"))
suspects = [c for c in data if REFERENCE_ENTRY_RE.match(c["text"])]
print(f"{len(suspects)} chunks still look reference-shaped")
for c in suspects[:20]:
    print(c["id"], "->", c["text"][:80])