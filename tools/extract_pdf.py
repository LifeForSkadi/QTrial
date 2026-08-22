"""Extract text from PDFs in docs/ using PyMuPDF and save as .txt files."""
import fitz  # PyMuPDF
from pathlib import Path

DOCS = Path(r"f:\Study\Quantum Computing\信安竞赛\QTrial\docs")
OUT = DOCS / "extracted"
OUT.mkdir(exist_ok=True)

pdfs = sorted(DOCS.glob("*.pdf"))
for pdf in pdfs:
    doc = fitz.open(pdf)
    n_pages = len(doc)
    parts = []
    total_chars = 0
    for i, page in enumerate(doc):
        text = page.get_text("text")
        parts.append(f"\n===== PAGE {i + 1}/{n_pages} =====\n{text}")
        total_chars += len(text.strip())
    doc.close()

    out_txt = OUT / (pdf.stem + ".txt")
    out_txt.write_text("".join(parts), encoding="utf-8")
    print(f"{pdf.name}: {n_pages} pages, {total_chars} chars extracted -> {out_txt.name}")
