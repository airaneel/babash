"""Turning documents into text.

The OOXML fixtures are hand-built with zipfile: python-docx and friends are
exactly the dependencies this module exists in order not to have, and pulling
them in as test deps to prove we don't need them would be a joke at our own
expense. (The extractor has been checked against real files produced by those
libraries; what is pinned here is the parsing, which is where the bugs live.)
"""

import io
import zipfile

import pytest
from pypdf import PdfWriter

from babash.client.documents import DocumentError, extract


def _ooxml(parts: dict[str, str]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, body in parts.items():
            zf.writestr(name, body)
    return buf.getvalue()


DOCX = _ooxml(
    {
        "word/document.xml": """<?xml version="1.0"?>
        <w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
          <w:body>
            <w:p><w:r><w:t>Quarterly </w:t></w:r><w:r><w:t>Report</w:t></w:r></w:p>
            <w:p><w:r><w:t>Revenue grew 42%.</w:t></w:r></w:p>
          </w:body>
        </w:document>"""
    }
)

XLSX = _ooxml(
    {
        "xl/workbook.xml": "<workbook/>",
        "xl/sharedStrings.xml": """<?xml version="1.0"?>
        <sst xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
          <si><t>Region</t></si><si><t>EMEA</t></si>
        </sst>""",
        "xl/worksheets/sheet1.xml": """<?xml version="1.0"?>
        <worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
          <sheetData>
            <row><c t="s"><v>0</v></c><c><v>2024</v></c></row>
            <row><c t="s"><v>1</v></c><c><v>120000</v></c></row>
          </sheetData>
        </worksheet>""",
    }
)

PPTX = _ooxml(
    {
        "ppt/slides/slide1.xml": """<?xml version="1.0"?>
        <sld xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">
          <a:t>Roadmap 2026</a:t><a:t>Ship babash</a:t>
        </sld>"""
    }
)


def test_docx_joins_runs_within_a_paragraph() -> None:
    """Word splits a sentence across <w:t> runs wherever formatting changes.
    Those are not line breaks, and treating them as such shreds the text."""
    doc = extract(DOCX)
    assert doc.kind == "docx"
    assert "Quarterly Report" in doc.text
    assert "Revenue grew 42%." in doc.text


def test_xlsx_resolves_shared_strings() -> None:
    """A cell with t="s" holds an *index* into the shared string table, not text.
    Read naively, a spreadsheet comes out as a column of integers."""
    doc = extract(XLSX)
    assert doc.kind == "xlsx"
    assert "Region\t2024" in doc.text
    assert "EMEA\t120000" in doc.text


def test_pptx_reads_slide_text() -> None:
    doc = extract(PPTX)
    assert doc.kind == "pptx"
    assert "Roadmap 2026" in doc.text
    assert "Ship babash" in doc.text


def _blank_pdf() -> bytes:
    """A valid PDF with a page and no text — i.e. what a scan looks like."""
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


def test_format_comes_from_the_bytes_not_the_name() -> None:
    """`textutil` and `catdoc` exit 0 on a PDF and emit binary garbage. Trusting
    an extension is how a file gets silently mangled instead of loudly refused."""
    assert extract(_blank_pdf()).kind == "pdf", "a PDF is a PDF whatever it is called"


def test_legacy_doc_says_how_to_convert_it() -> None:
    """OLE2 (.doc from Office 97-2003) has no pure-Python reader. The agent has a
    shell — point it at one rather than failing blankly."""
    with pytest.raises(DocumentError, match="legacy OLE2"):
        extract(b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1" + b"\x00" * 64)


def test_unknown_bytes_are_refused_with_advice() -> None:
    with pytest.raises(DocumentError, match="read_file"):
        extract(b"just some plain text, not a document at all")


def test_a_zip_that_is_not_an_office_document() -> None:
    with pytest.raises(DocumentError, match="not a Word/Excel/PowerPoint"):
        extract(_ooxml({"random/thing.txt": "hello"}))


def test_a_scanned_pdf_points_at_read_image() -> None:
    """A PDF with no extractable text is a picture of a document. There is
    nothing to fix — but the agent already owns the two tools that can read it."""
    doc = extract(_blank_pdf())
    assert doc.kind == "pdf"
    assert doc.text == ""
    assert "no extractable text" in doc.note
    assert "read_image" in doc.note
