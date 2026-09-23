"""DOCX parsing — paragraphs + tables to Markdown (SPEC ingest.parse)."""

from __future__ import annotations

from pathlib import Path

from docx import Document as DocxDocument

_W_NS = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}


def parse_docx(file_path: str | Path) -> str:
    """Parse a DOCX file into plain text; tables become Markdown tables."""
    doc = DocxDocument(str(file_path))
    all_text: list[str] = []

    for element in doc.element.body:
        if element.tag.endswith("p"):
            paragraph_text = ""
            for run in element.findall(".//w:t", _W_NS):
                paragraph_text += run.text if run.text else ""
            if paragraph_text.strip():
                all_text.append(paragraph_text.strip())

        elif element.tag.endswith("tbl"):
            table = [t for t in doc.tables if t._element is element][0]
            if table.rows:
                md_table: list[str] = []
                header = [cell.text.strip() for cell in table.rows[0].cells]
                md_table.append("| " + " | ".join(header) + " |")
                md_table.append("|" + "---|" * len(header))
                for row in table.rows[1:]:
                    row_data = [cell.text.strip() for cell in row.cells]
                    md_table.append("| " + " | ".join(row_data) + " |")
                all_text.append("\n".join(md_table))

    return "\n".join(all_text)
