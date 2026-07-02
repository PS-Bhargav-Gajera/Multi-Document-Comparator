"""Baseline audit of CURRENT chunker (before improvement). Loader+Chunker only, no embeddings."""
import sys, io, re, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from rag.loader import PDFLoader
from rag.chunker import SemanticChunker

loader = PDFLoader()
chunker = SemanticChunker()

pages = loader.load_all(["Pdfs/ateension all you need.pdf", "Pdfs/bert.pdf"])
chunks = chunker.chunk(pages)

print("Pages:", len(pages), "| Raw chunks (current):", len(chunks))

REVERSED_WORDS = {
    "tahw", "era", "gnissim", "si", "ot", "ni", "rof", "eht", "taht",
    "htiw", "ecno", "neht", "naht", "htob", "esu", "rae", "ton",
    "dna", "gnidulcni", "gnisufnoc", "derevocsid", "dereffid",
    "tset", "gnitset", "laer", "eludom", "retcarahc", "tnerrucer",
}
BIB = {"references", "bibliography", "acknowledgments", "appendix"}


def wlist(s):
    return re.findall(r"[A-Za-z]+", s)


def issues_for(text, meta):
    iss = []
    w = text.split()
    if not w:
        return ["EMPTY"]
    rev = [x for x in wlist(text) if len(x) > 2 and x.lower() in REVERSED_WORDS]
    if rev:
        iss.append("REVERSED:" + str(rev[:6]))
    reps = sum(1 for j in range(1, len(w)) if w[j].lower() == w[j - 1].lower())
    if reps > 3:
        iss.append("REPEATS:" + str(reps))
    alpha = sum(1 for c in text if c.isalpha())
    ar = alpha / max(len(text), 1)
    if ar < 0.4:
        iss.append("LOW_ALPHA:%.2f" % ar)
    sym = sum(1 for c in text if not c.isalnum() and not c.isspace()) / max(len(text), 1)
    if sym > 0.2:
        iss.append("HIGH_SYMBOL:%.2f" % sym)
    h = meta.get("heading", "").lower()
    if any(b in h for b in BIB):
        iss.append("BIBLIOGRAPHY")
    fw = w[0].lower() if w else ""
    if fw in ("figure", "table", "fig.", "fig", "algorithm", "algorithms", "table:"):
        iss.append("FIGURE/TABLE")
    mean = sum(1 for x in wlist(text) if len(x) > 2)
    if mean < 40:
        iss.append("LOW_MEANINGFUL:" + str(mean))
    sr = text.count(" ") / max(len(text), 1)
    if sr < 0.03:
        iss.append("NO_SPACES:%.3f" % sr)
    if text and text[0].islower() and len(text) > 10:
        iss.append("MID_SENTENCE")
    dig = sum(1 for c in text if c.isdigit()) / max(len(text), 1)
    if dig > 0.4:
        iss.append("DIGIT_HEAVY")
    return iss


allt = []
prob = 0
for i, c in enumerate(chunks):
    m = c["metadata"]
    iss = issues_for(c["text"], m)
    allt.append(c["text"])
    line = "%3d %-22s p%-3d %4dtok %3dw %s" % (
        i, m["document_name"][:22], m["page_number"],
        m["token_count"], len(c["text"].split()),
        "!! " + "; ".join(iss) if iss else "OK",
    )
    print(line)
    if iss:
        prob += 1
        print("      >>", c["text"][:200].replace(chr(10), " "))


def norm(s):
    return " ".join(s.lower().split())


seen = {}
dups = 0
for i, t in enumerate(allt):
    n = norm(t)
    if n in seen:
        dups += 1
        print("  DUP chunk", i, "==", seen[n])
    else:
        seen[n] = i

print("SUMMARY: total=%d problem_chunks=%d duplicates=%d" % (len(chunks), prob, dups))

with open("baseline_chunks.json", "w", encoding="utf-8") as f:
    json.dump(
        [{"text": c["text"], "metadata": c["metadata"]} for c in chunks],
        f, ensure_ascii=False, indent=2,
    )
print("Saved baseline_chunks.json")

