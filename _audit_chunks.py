"""Audit current chunking output in detail."""
import sys, io, re
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from rag.pipeline import RAGPipeline
pipeline = RAGPipeline()
pipeline.clear_all()

result = pipeline.ingest([
    "Pdfs/ateension all you need.pdf",
    "Pdfs/bert.pdf"
])

print(f"Total chunks ingested: {result['total_chunks']}\n")

REVERSED_WORDS = {
    "tahw", "era", "gnissim", "si", "ot", "ni", "rof", "eht", "taht",
    "htiw", "ecno", "neht", "naht", "htob", "esu", "rae", "ton",
    "dna", "gnidulcni", "gnisufnoc", "derevocsid", "dereffid",
}
BIB_HEADINGS = {"references", "bibliography", "acknowledgments", "appendix"}

def passes_quick(text, meta):
    words = text.split()
    if not words or len(words) < 30:
        return False
    alpha = sum(1 for c in text if c.isalpha())
    if alpha / max(len(text), 1) < 0.35:
        return False
    meaningful = sum(1 for w in words if len(w) > 2 and w.isalpha())
    if meaningful < 30:
        return False
    return True

print(f"{'CHUNK':<8} {'DOC':<25} {'PG':<3} {'TOK':<5} {'PASS':<5} {'QUALITY ISSUES'}")
print("-"*90)

for i, (text, meta) in enumerate(zip(pipeline.vector_db.texts, pipeline.vector_db.metadata)):
    issues = []
    words = text.split()
    tok = meta["token_count"]
    
    reversed_found = [w for w in words[:50] if w.lower().strip(".,;:!?") in REVERSED_WORDS and len(w) > 2]
    if reversed_found:
        issues.append(f"REVERSED:{reversed_found[:5]}")
    
    repeats = sum(1 for j in range(1, len(words)) if words[j].lower() == words[j-1].lower())
    if repeats > 3:
        issues.append(f"REPEATS:{repeats}")
    
    alpha = sum(1 for c in text if c.isalpha())
    alpha_ratio = alpha / max(len(text), 1)
    if alpha_ratio < 0.4:
        issues.append(f"LOW_ALPHA:{alpha_ratio:.2f}")
    
    symbol_ratio = sum(1 for c in text if not c.isalnum() and not c.isspace()) / max(len(text), 1)
    if symbol_ratio > 0.2:
        issues.append(f"HIGH_SYMBOL:{symbol_ratio:.2f}")
    
    heading_lower = meta.get("heading", "").lower()
    if any(b in heading_lower for b in BIB_HEADINGS):
        issues.append("BIBLIOGRAPHY")
    
    first_word = words[0].lower() if words else ""
    if first_word in ("figure", "table", "fig.", "table:"):
        issues.append("FIGURE/TABLE")
    
    meaningful = sum(1 for w in words if len(w) > 2 and w.isalpha())
    if meaningful < 40:
        issues.append(f"LOW_MEANINGFUL:{meaningful}")
    
    spaces = text.count(" ")
    space_ratio = spaces / max(len(text), 1)
    if space_ratio < 0.03:
        issues.append(f"NO_SPACES:{space_ratio:.3f}")
    
    if text and text[0].islower() and len(text) > 10:
        issues.append("MID_SENTENCE")
    
    p = passes_quick(text, meta)
    issue_str = "; ".join(issues) if issues else "OK"
    doc_short = meta["document_name"][:24]
    print(f"{i:<8} {doc_short:<25} {meta['page_number']:<3} {tok:<5} {'PASS' if p else 'FAIL':<5} {issue_str}")
    if not p or issues:
        print(f"         TEXT: {text[:200]}...")

passed = sum(1 for t, m in zip(pipeline.vector_db.texts, pipeline.vector_db.metadata) if passes_quick(t, m))
failed = sum(1 for t, m in zip(pipeline.vector_db.texts, pipeline.vector_db.metadata) if not passes_quick(t, m))
print(f"\nTotal: {len(pipeline.vector_db.texts)} chunks, Passed: {passed}, Failed: {failed}")
