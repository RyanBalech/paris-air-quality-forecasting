"""Render the generated Markdown assignment report as a shareable PDF."""
from __future__ import annotations

import html
import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image,
    KeepTogether,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

SOURCE = Path("docs/assignment/report.md")
DESTINATION = Path("docs/assignment/report.pdf")


def rich(text: str) -> str:
    text = html.escape(text)
    text = re.sub(r"\[([^]]+)\]\(([^)]+)\)", r'<link href="\2">\1</link>', text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"`(.+?)`", r'<font face="Courier">\1</font>', text)
    return text


def render() -> Path:
    fonts = [
        (Path("C:/Windows/Fonts/arial.ttf"), Path("C:/Windows/Fonts/arialbd.ttf")),
        (Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
         Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")),
    ]
    family = "Helvetica"
    for normal, bold in fonts:
        if normal.exists() and bold.exists():
            pdfmetrics.registerFont(TTFont("ReportSans", str(normal)))
            pdfmetrics.registerFont(TTFont("ReportSans-Bold", str(bold)))
            pdfmetrics.registerFontFamily(
                "ReportSans", normal="ReportSans", bold="ReportSans-Bold")
            family = "ReportSans"
            break
    styles = getSampleStyleSheet()
    for style in styles.byName.values():
        style.fontName = family
    styles["BodyText"].fontSize = 8.5
    styles["BodyText"].leading = 12
    styles.add(styles["BodyText"].clone("TableCell"))
    styles["TableCell"].fontSize = 7
    styles["TableCell"].leading = 8.5
    styles["Heading1"].spaceBefore = 14
    styles["Heading2"].spaceBefore = 10
    document = SimpleDocTemplate(str(DESTINATION), pagesize=A4, leftMargin=1.6 * cm,
                                 rightMargin=1.6 * cm, topMargin=1.5 * cm,
                                 bottomMargin=1.5 * cm)
    content = []
    lines = SOURCE.read_text(encoding="utf-8").splitlines()
    index = 0
    while index < len(lines):
        line = lines[index].strip()
        if not line:
            content.append(Spacer(1, 0.16 * cm))
        elif line.startswith("# "):
            content.append(Paragraph(rich(line[2:]), styles["Title"]))
        elif line.startswith("## "):
            content.append(Paragraph(rich(line[3:]), styles["Heading1"]))
        elif line.startswith("### "):
            content.append(Paragraph(rich(line[4:]), styles["Heading2"]))
        elif line.startswith("!["):
            match = re.match(r"!\[([^]]*)\]\(([^)]+)\)", line)
            if match:
                path = SOURCE.parent / match.group(2)
                image = Image(str(path))
                ratio = min(17.5 * cm / image.imageWidth, 8 * cm / image.imageHeight)
                image.drawWidth *= ratio
                image.drawHeight *= ratio
                content.append(KeepTogether([image, Spacer(1, 0.1 * cm)]))
        elif line.startswith("| "):
            table_lines = []
            while index < len(lines) and lines[index].strip().startswith("|"):
                row = [entry.strip() for entry in lines[index].strip().strip("|").split("|")]
                if not all(re.fullmatch(r"[-: ]+", entry) for entry in row):
                    table_lines.append([Paragraph(rich(entry), styles["TableCell"])
                                        for entry in row])
                index += 1
            index -= 1
            widths = ([4.2 * cm] + [(17.5 - 4.2) * cm / (len(table_lines[0]) - 1)] *
                      (len(table_lines[0]) - 1))
            table = Table(table_lines, colWidths=widths, repeatRows=1, hAlign="LEFT")
            table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8eef5")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]))
            content.append(table)
        elif line.startswith("- "):
            content.append(Paragraph("• " + rich(line[2:]), styles["BodyText"]))
        else:
            content.append(Paragraph(rich(line), styles["BodyText"]))
        index += 1
    document.build(content)
    return DESTINATION


if __name__ == "__main__":
    print(render())
