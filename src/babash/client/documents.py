"""Turning a document's bytes into text an agent can read.

Text, and only text. MCP has no `document` content block — a tool result can
carry TextContent, ImageContent, AudioContent, ResourceLink or EmbeddedResource,
and nothing else. Handing over a PDF as an EmbeddedResource blob does not give
the model the document: Claude Code spools the bytes to a file and shows the
model a path, and Claude Desktop validates every blob as an image and *rejects
the whole tool response* — including any text sent alongside it. So extraction
happens here, and what goes back is text.

Formats are recognised by their leading bytes, never by their extension. That is
not pedantry: `textutil` and `catdoc` both exit 0 on a PDF and emit binary
garbage, so a misidentified file fails silently and convincingly, while a file
we decline to read fails loudly.
"""

import io
import zipfile
from dataclasses import dataclass
from xml.etree import ElementTree

# OOXML namespaces. Word, Excel and PowerPoint each name their text element
# differently, which is the only reason this module knows about all three.
_W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
_A = "{http://schemas.openxmlformats.org/drawingml/2006/main}"
_S = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"


class DocumentError(Exception):
    """The document could not be turned into text, and the agent should be told
    why — and, where possible, what to do instead."""


@dataclass(frozen=True)
class Document:
    text: str
    kind: str
    note: str = ""
    """Anything the agent needs to know about *how* this was read — most
    importantly, that a PDF turned out to be scanned and has no text in it."""


def _first(names: list[str], candidates: list[str]) -> list[str]:
    return sorted(n for n in names if n in candidates)


def _docx_text(zf: zipfile.ZipFile) -> str:
    root = ElementTree.fromstring(zf.read("word/document.xml"))
    # One line per paragraph: a <w:p> may split a sentence across many <w:t>
    # runs wherever formatting changes, and those runs are not line breaks.
    lines = []
    for para in root.iter(f"{_W}p"):
        text = "".join(node.text or "" for node in para.iter(f"{_W}t"))
        lines.append(text)
    return "\n".join(lines)


def _xlsx_text(zf: zipfile.ZipFile) -> str:
    shared: list[str] = []
    if "xl/sharedStrings.xml" in zf.namelist():
        root = ElementTree.fromstring(zf.read("xl/sharedStrings.xml"))
        for item in root.iter(f"{_S}si"):
            shared.append("".join(node.text or "" for node in item.iter(f"{_S}t")))

    sheets = _first(
        zf.namelist(),
        [n for n in zf.namelist() if n.startswith("xl/worksheets/sheet")],
    )
    blocks = []
    for sheet in sheets:
        root = ElementTree.fromstring(zf.read(sheet))
        rows = []
        for row in root.iter(f"{_S}row"):
            cells = []
            for cell in row.iter(f"{_S}c"):
                cells.append(_cell_text(cell, shared))
            rows.append("\t".join(cells))
        if rows:
            blocks.append(f"--- {sheet.rsplit('/', 1)[-1]} ---\n" + "\n".join(rows))
    return "\n\n".join(blocks)


def _cell_text(cell: ElementTree.Element, shared: list[str]) -> str:
    kind = cell.get("t")
    if kind == "s":
        # A shared string: <v> holds an index into the shared table, not a value.
        value = cell.find(f"{_S}v")
        if value is not None and value.text and value.text.isdigit():
            index = int(value.text)
            return shared[index] if index < len(shared) else ""
        return ""
    if kind == "inlineStr":
        return "".join(node.text or "" for node in cell.iter(f"{_S}t"))
    # Anything else — a number, a date, a formula — is in <v> as its cached
    # value. Formulas are deliberately not evaluated: the cached value is what
    # the spreadsheet last showed, which is what a reader would have seen.
    value = cell.find(f"{_S}v")
    return value.text or "" if value is not None else ""


def _pptx_text(zf: zipfile.ZipFile) -> str:
    slides = sorted(
        n for n in zf.namelist() if n.startswith("ppt/slides/slide") and n.endswith(".xml")
    )
    blocks = []
    for slide in slides:
        root = ElementTree.fromstring(zf.read(slide))
        lines = [node.text for node in root.iter(f"{_A}t") if node.text]
        if lines:
            blocks.append(f"--- {slide.rsplit('/', 1)[-1]} ---\n" + "\n".join(lines))
    return "\n\n".join(blocks)


def _ooxml(data: bytes) -> Document:
    """docx, xlsx and pptx are all a zip of XML, so the stdlib is the whole
    dependency. Which of the three it is shows in what's inside."""
    try:
        zf = zipfile.ZipFile(io.BytesIO(data))
        names = zf.namelist()
        if "word/document.xml" in names:
            return Document(text=_docx_text(zf), kind="docx")
        if "xl/workbook.xml" in names:
            return Document(text=_xlsx_text(zf), kind="xlsx")
        if any(n.startswith("ppt/slides/slide") for n in names):
            return Document(text=_pptx_text(zf), kind="pptx")
    except (zipfile.BadZipFile, KeyError, ElementTree.ParseError) as e:
        raise DocumentError(f"Malformed OOXML document: {e}")

    raise DocumentError(
        "This is a zip archive, but not a Word/Excel/PowerPoint document. "
        "Unzip it with run_command and read the parts you want."
    )


def _pdf(data: bytes) -> Document:
    from pypdf import PdfReader
    from pypdf.errors import PdfReadError

    try:
        reader = PdfReader(io.BytesIO(data))
        pages = [page.extract_text() or "" for page in reader.pages]
    except PdfReadError as e:
        raise DocumentError(f"Could not read PDF: {e}")

    text = "\n\n".join(
        f"--- page {i} ---\n{page}" for i, page in enumerate(pages, 1) if page.strip()
    )
    if text:
        return Document(text=text, kind="pdf")

    # A PDF with no extractable text is a scan: pages of pixels, no characters.
    # There is nothing to fix here — but the agent already has the two tools it
    # needs to see it, so say so rather than returning an empty string.
    return Document(
        text="",
        kind="pdf",
        note=(
            f"This PDF has {len(pages)} page(s) but no extractable text — it is a scan "
            f"or is made of images. To read it, rasterise a page and look at it:\n"
            f"  run_command: pdftoppm -png -r 150 -f 1 -l 1 <file> /tmp/page   "
            f"(or, on macOS: sips -s format png <file> --out /tmp/page.png)\n"
            f"  then: read_image(file_path='/tmp/page-1.png')"
        ),
    )


_LEGACY_DOC_ADVICE = (
    "This is a legacy OLE2 document (.doc/.xls/.ppt from Office 97-2003), which no "
    "pure-Python reader handles. Convert it in the shell first, then read the result:\n"
    "  macOS:  textutil -convert txt <file> -output /tmp/out.txt\n"
    "  Linux:  antiword <file> > /tmp/out.txt   (apt install antiword)\n"
    "          or: libreoffice --headless --convert-to txt --outdir /tmp <file>\n"
    "then: read_file(file_path='/tmp/out.txt')"
)


def extract(data: bytes) -> Document:
    """Read a document's bytes as text, or explain why that can't be done.

    Dispatch is on the leading bytes, so a `.docx` that is really a PDF is read
    as a PDF, and a `.pdf` that is really an OLE2 file is refused rather than
    silently mangled.
    """
    if data.startswith(b"%PDF-"):
        return _pdf(data)
    if data.startswith(b"PK\x03\x04"):
        return _ooxml(data)
    if data.startswith(b"\xd0\xcf\x11\xe0"):
        raise DocumentError(_LEGACY_DOC_ADVICE)
    raise DocumentError(
        "Not a document babash can read (PDF, docx, xlsx, pptx). "
        "If it is text, use read_file; if it is an image, use read_image."
    )
