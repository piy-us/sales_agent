"""
════════════════════════════════════════════════════════════════════════════════
PARSER 3 — Docling  (IBM Research)
════════════════════════════════════════════════════════════════════════════════
What this demo shows
────────────────────
 • Document conversion with full structural output (DoclingDocument)
 • Hierarchical document tree (body → sections → groups → items)
 • Rich Markdown export (preserves tables, headings, lists)
 • JSON export of the full document model
 • Per-item label classification (title / section_header / text / table / picture / …)
 • Table cell-level extraction with row/col spans
 • Image detection and saving
 • Hierarchical chunking with HybridChunker (title-aware, token-aware)
 • Metadata: file info, page count, origin
 • Capability verdict

Install:
    pip install docling

For GPU-accelerated table/image recognition (optional):
    pip install docling[gpu]

Run:
    python parser3_docling.py [your_file.pptx]
"""

import sys, json, textwrap
from pathlib import Path
import re
# ── colour helpers ────────────────────────────────────────────────────────────
RESET = "\033[0m"; BOLD = "\033[1m"; DIM = "\033[2m"
CYAN = "\033[36m"; GREEN = "\033[32m"; YELLOW = "\033[33m"
MAGENTA = "\033[35m"; RED = "\033[31m"; BLUE = "\033[34m"
class StripAnsi:
    def __init__(self, stream):
        self.stream = stream
    def write(self, text):
        self.stream.write(re.sub(r'\033\[[0-9;]*m', '', text))
    def flush(self):
        self.stream.flush()

sys.stdout = StripAnsi(open("output_parser3_python_pptx.txt", "w", encoding="utf-8"))

def hdr(title, color=CYAN):
    bar = "─" * 70
    print(f"\n{color}{BOLD}{bar}{RESET}")
    print(f"{color}{BOLD}  {title}{RESET}")
    print(f"{color}{BOLD}{bar}{RESET}")

def sub(title):
    print(f"\n  {YELLOW}{BOLD}▸ {title}{RESET}")

def info(label, value, indent=4):
    sp = " " * indent
    print(f"{sp}{GREEN}{label:<40}{RESET}{value}")


# ════════════════════════════════════════════════════════════════════════════
# import check
# ════════════════════════════════════════════════════════════════════════════
try:
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.chunking import HybridChunker
    DOCLING_OK = True
except ImportError:
    DOCLING_OK = False

try:
    from docling_core.types.doc import DocItemLabel
    LABELS_OK = True
except ImportError:
    LABELS_OK = False


# ════════════════════════════════════════════════════════════════════════════
# helpers
# ════════════════════════════════════════════════════════════════════════════

LABEL_COLORS = {
    "title":            CYAN,
    "section_header":   BLUE,
    "text":             GREEN,
    "list_item":        MAGENTA,
    "table":            YELLOW,
    "picture":          BLUE,
    "caption":          DIM,
    "footnote":         DIM,
    "page_header":      DIM,
    "page_footer":      DIM,
    "formula":          RED,
    "code":             RED,
}

def label_color(label_str):
    return LABEL_COLORS.get(str(label_str).lower(), RESET)


def iter_items(doc):
    """Yield (item, level) for every item in the document body."""
    try:
        # Docling >= 2.x
        for item, level in doc.iterate_items():
            yield item, level
    except AttributeError:
        # Older API fallback
        try:
            for item in doc.body.children:
                yield item, 0
        except Exception:
            pass


def safe_label(item):
    try:
        return str(item.label)
    except Exception:
        return type(item).__name__


def safe_text(item):
    try:
        return item.text or ""
    except Exception:
        return ""


def safe_page(item):
    try:
        refs = item.prov
        if refs:
            return refs[0].page_no
    except Exception:
        pass
    return "?"


# ════════════════════════════════════════════════════════════════════════════
# section printers
# ════════════════════════════════════════════════════════════════════════════

def print_document_tree(doc, max_items=80):
    """Print the hierarchical document tree."""
    count = 0
    for item, level in iter_items(doc):
        if count >= max_items:
            print(f"\n  {DIM}  … (truncated at {max_items} items){RESET}")
            break
        label = safe_label(item)
        text  = safe_text(item).strip()
        page  = safe_page(item)
        color = label_color(label)
        indent = "  " * level

        if "table" in label.lower():
            try:
                rows = len(item.data.grid)
                cols = len(item.data.grid[0]) if rows else 0
                print(f"\n  {indent}{color}{BOLD}[{label}]{RESET}  slide={page}"
                      f"  {rows}×{cols} table")
            except Exception:
                print(f"\n  {indent}{color}{BOLD}[{label}]{RESET}  slide={page}")
            count += 1
            continue

        if "picture" in label.lower() or "image" in label.lower():
            print(f"\n  {indent}{color}{BOLD}[{label}]{RESET}  slide={page}"
                  f"  (embedded image)")
            count += 1
            continue

        wrapped = textwrap.fill(text[:160], width=80,
                                initial_indent=f"  {indent}  ",
                                subsequent_indent=f"  {indent}  ")
        print(f"\n  {indent}{color}{BOLD}[{label}]{RESET}  slide={page}")
        if wrapped.strip():
            print(wrapped)
        count += 1


def print_table_detail(doc, max_tables=3):
    """Print cell-level table content."""
    table_count = 0
    for item, _ in iter_items(doc):
        label = safe_label(item)
        if "table" not in label.lower():
            continue
        table_count += 1
        if table_count > max_tables:
            break
        page = safe_page(item)
        sub(f"Table {table_count}  (slide {page})")
        try:
            grid = item.data.grid
            for r, row in enumerate(grid):
                row_txt = []
                for cell in row:
                    txt = cell.text.strip() if hasattr(cell, "text") else str(cell)
                    row_txt.append(f"{txt[:18]:<20}")
                marker = f"  {BOLD}HDR{RESET}" if r == 0 else "     "
                print(f"  {marker}  " + " | ".join(row_txt))
        except Exception as e:
            print(f"    {DIM}(Could not render grid: {e}){RESET}")
        # also show exported markdown table
        try:
            md_table = item.export_to_markdown()
            print(f"\n  {DIM}Markdown export (first 500 chars):{RESET}")
            print(f"  {DIM}{md_table[:500]}{RESET}")
        except Exception:
            pass


def print_chunks(chunks, label):
    sub(f"Chunking: {BOLD}{label}{RESET}")
    for i, chunk in enumerate(chunks, 1):
        text = (chunk.text or "").strip()
        meta = chunk.meta if hasattr(chunk, "meta") else {}
        headings = []
        try:
            headings = [h.text for h in chunk.meta.headings] if chunk.meta.headings else []
        except Exception:
            pass
        page = "?"
        try:
            page = chunk.meta.doc_items[0].prov[0].page_no
        except Exception:
            pass
        print(f"\n    {CYAN}── Chunk {i:02d}  slide={page}{RESET}")
        if headings:
            print(f"    {BLUE}  headings: {' > '.join(headings)}{RESET}")
        print(textwrap.fill(text[:400], width=76,
                            initial_indent="    ",
                            subsequent_indent="    "))
        if i >= 10:
            print(f"  {DIM}  … (showing first 10 chunks){RESET}")
            break


# ════════════════════════════════════════════════════════════════════════════
# main
# ════════════════════════════════════════════════════════════════════════════

def parse_with_docling(path: str):
    hdr(f"Docling  |  {Path(path).name}", GREEN)

    if not DOCLING_OK:
        print(f"\n  {RED}❌  docling not installed.{RESET}")
        print("  Run:  pip install docling")
        return

    # ── 1. Convert ───────────────────────────────────────────────────────────
    hdr("1. DOCUMENT CONVERSION", BLUE)
    print("  Running DocumentConverter().convert() …")

    converter = DocumentConverter()
    result    = converter.convert(path)
    doc       = result.document

    print(f"  → Conversion complete")

    # ── 2. Document metadata ─────────────────────────────────────────────────
    hdr("2. DOCUMENT METADATA", BLUE)
    try:
        info("Document name",   doc.name or "—")
    except Exception: pass
    try:
        info("Origin filename", str(doc.origin.filename) if doc.origin else "—")
    except Exception: pass
    try:
        info("Mimetype",        doc.origin.mimetype if doc.origin else "—")
    except Exception: pass
    # count items by label
    label_counts = {}
    for item, _ in iter_items(doc):
        k = safe_label(item)
        label_counts[k] = label_counts.get(k, 0) + 1
    info("Total items", str(sum(label_counts.values())))
    for k, v in sorted(label_counts.items(), key=lambda x: -x[1]):
        color = label_color(k)
        print(f"    {color}{k:<30}{RESET}  {v}")

    # ── 3. Document tree ─────────────────────────────────────────────────────
    hdr("3. HIERARCHICAL DOCUMENT TREE", CYAN)
    print_document_tree(doc, max_items=80)

    # ── 4. Tables ────────────────────────────────────────────────────────────
    hdr("4. TABLE CELL-LEVEL EXTRACTION", YELLOW)
    print_table_detail(doc, max_tables=3)

    # ── 5. Markdown export ───────────────────────────────────────────────────
    hdr("5. MARKDOWN EXPORT  (first 2000 chars)", GREEN)
    try:
        markdown = doc.export_to_markdown()
        print(markdown[:2000])
        print(f"\n  {DIM}… total markdown length: {len(markdown)} chars{RESET}")
    except Exception as e:
        print(f"  {RED}Markdown export failed: {e}{RESET}")

    # ── 6. JSON export ───────────────────────────────────────────────────────
    hdr("6. JSON EXPORT  (first 1500 chars)", GREEN)
    try:
        doc_json = doc.export_to_dict()
        pretty   = json.dumps(doc_json, indent=2, ensure_ascii=False)
        print(pretty[:1500])
        print(f"\n  {DIM}… total JSON length: {len(pretty)} chars{RESET}")
    except Exception as e:
        print(f"  {RED}JSON export failed: {e}{RESET}")

    # ── 7. HybridChunker ────────────────────────────────────────────────────
    hdr("7. CHUNKING  —  HybridChunker (title-aware + token-aware)", MAGENTA)
    try:
        chunker = HybridChunker(
            tokenizer="BAAI/bge-small-en-v1.5",  # any HF tokenizer
            max_tokens=256,
            merge_peers=True,
        )
        chunks = list(chunker.chunk(dl_doc=doc))
        print(f"  → {len(chunks)} chunks produced")
        print_chunks(chunks, "HybridChunker(max_tokens=256)")
    except Exception as e:
        print(f"  {YELLOW}HybridChunker unavailable ({e}).{RESET}")
        print(f"  Falling back to iterate_items() manual chunking …")
        # manual fallback
        chunks = []
        buf, cur_head = [], "—"
        for item, _ in iter_items(doc):
            label = safe_label(item)
            text  = safe_text(item).strip()
            if not text:
                continue
            if "title" in label.lower() or "header" in label.lower():
                if buf:
                    chunks.append({"heading": cur_head, "text": " ".join(buf)})
                    buf = []
                cur_head = text
            else:
                buf.append(text)
        if buf:
            chunks.append({"heading": cur_head, "text": " ".join(buf)})
        print(f"  → {len(chunks)} manual chunks")
        for i, c in enumerate(chunks[:8], 1):
            print(f"\n    {CYAN}── Chunk {i:02d}  heading: {c['heading']}{RESET}")
            print(textwrap.fill(c["text"][:300], width=76,
                                initial_indent="    ",
                                subsequent_indent="    "))

    # ── 8. Picture items ─────────────────────────────────────────────────────
    hdr("8. PICTURE / IMAGE ITEMS", BLUE)
    pic_count = 0
    for item, _ in iter_items(doc):
        label = safe_label(item)
        if "picture" in label.lower() or "image" in label.lower():
            pic_count += 1
            page = safe_page(item)
            print(f"  {BLUE}[Picture {pic_count}]{RESET}  slide={page}")
            try:
                ref = item.image.uri if item.image else "—"
                info("  URI / ref", str(ref)[:80], indent=6)
            except Exception:
                pass
    if pic_count == 0:
        print(f"  {DIM}(No picture items detected in this file){RESET}")

    # ── 9. Capabilities verdict ──────────────────────────────────────────────
    hdr("CAPABILITIES VERDICT", GREEN)
    caps = {
        "Hierarchical document model":    "✅  Full DoclingDocument tree",
        "Label classification":           "✅  title/section_header/text/list_item/table/picture/…",
        "Text extraction":                "✅  Per item with nesting level",
        "Bullet hierarchy":               "✅  list_item + nesting level",
        "Table cell extraction":          "✅  Grid model + row/col spans",
        "Table Markdown export":          "✅  item.export_to_markdown()",
        "Full Markdown export":           "✅  doc.export_to_markdown() — headings, tables, lists",
        "JSON export":                    "✅  doc.export_to_dict() — full model",
        "Image / picture detection":      "✅  Picture items with URI ref",
        "Image OCR":                      "✅  Built-in via EasyOCR/Tesseract pipeline",
        "Speaker notes":                  "⚠️   Extracted as text items (no special label)",
        "HybridChunker":                  "✅  Token-aware + title-boundary chunking",
        "Heading context in chunks":      "✅  chunk.meta.headings breadcrumb",
        "Slide-level boundaries":         "✅  prov[0].page_no on every item",
        "Rich formatting (bold/color)":   "⚠️   Partial — structure yes, font attrs limited",
        "Chart data extraction":          "❌  Charts treated as images",
        "Hyperlinks":                     "⚠️   Available in hyperlink items",
        "GPU acceleration":               "✅  pip install docling[gpu]",
        "PDF, DOCX, HTML inputs too":     "✅  Universal converter",
        "SmartArt":                       "⚠️   Text extracted, visual layout lost",
    }
    for k, v in caps.items():
        info(k, v)
    print()


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "Ch 1 Intro to Internet Technology.pptx"
    if not Path(target).exists():
        print(f"  File not found: {target}")
        print("  Run  python create_sample_pptx.py  first.")
        sys.exit(1)
    parse_with_docling(target)