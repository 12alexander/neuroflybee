"""Exporta el artículo LaTeX a Word con formato IEEE (v19).

    LaTeX → (latex_flatten) LaTeX simple → pandoc JSON → transformaciones → pandoc docx
          → ajustes finales (anchos de figura por columna)

Resultado: carta, Times New Roman 10 pt, título y autores a una columna (3 + 2 autores),
cuerpo a dos columnas, figuras/tablas anchas (`figure*`, `table*`) a todo el ancho,
ecuaciones nativas de Word, referencias IEEE numeradas y pendientes en rojo.
El fuente oficial sigue siendo LaTeX: el .docx es para revisar y comentar en equipo.
"""
from __future__ import annotations

import copy
import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from xml.sax.saxutils import escape

from .latex_flatten import WIDE_END, WIDE_START, LatexFlattener, roman

# ------------------------------------------------------------------ página IEEE (twips)
PAGE_W, PAGE_H = 12240, 15840                   # carta
MARGIN_LR, MARGIN_T, MARGIN_B, COL_GAP = 900, 1080, 1440, 360
TEXT_W = PAGE_W - 2 * MARGIN_LR                 # 7 in
COL_W = (TEXT_W - COL_GAP) // 2                 # 3.5 in
EMU_PER_TWIP = 635
FONT = "Times New Roman"


def section_break(cols: int) -> str:
    return ("<w:p><w:pPr><w:sectPr><w:type w:val=\"continuous\"/>"
            f"<w:pgSz w:w=\"{PAGE_W}\" w:h=\"{PAGE_H}\"/>"
            f"<w:pgMar w:top=\"{MARGIN_T}\" w:right=\"{MARGIN_LR}\" w:bottom=\"{MARGIN_B}\" w:left=\"{MARGIN_LR}\" "
            "w:header=\"720\" w:footer=\"720\" w:gutter=\"0\"/>"
            f"<w:cols w:num=\"{cols}\" w:space=\"{COL_GAP}\"/></w:sectPr></w:pPr></w:p>")


def _run(text: str, size: int = 20, italic: bool = False, bold: bool = False) -> str:
    props = f"<w:rFonts w:ascii=\"{FONT}\" w:hAnsi=\"{FONT}\" w:cs=\"{FONT}\"/>"
    props += ("<w:b/>" if bold else "") + ("<w:i/>" if italic else "") + f"<w:sz w:val=\"{size}\"/><w:szCs w:val=\"{size}\"/>"
    return f"<w:r><w:rPr>{props}</w:rPr><w:t xml:space=\"preserve\">{escape(text)}</w:t></w:r>"


def _cell(author: dict, width: int) -> str:
    paras = [f"<w:p><w:pPr><w:jc w:val=\"center\"/><w:spacing w:after=\"0\"/></w:pPr>{_run(author['name'], 22)}</w:p>"]
    for i, line in enumerate(author["lines"]):
        italic = i < 2                           # programa y facultad en cursiva (plantilla IEEE)
        paras.append(f"<w:p><w:pPr><w:jc w:val=\"center\"/><w:spacing w:after=\"0\"/></w:pPr>{_run(line, 20, italic)}</w:p>")
    return (f"<w:tc><w:tcPr><w:tcW w:w=\"{width}\" w:type=\"dxa\"/></w:tcPr>{''.join(paras)}</w:tc>")


def authors_xml(authors: list[dict], per_row: int = 3) -> str:
    """Bloque de autores IEEE: filas de 3 centradas (5 autores → 3 + 2)."""
    width = TEXT_W // per_row
    out = []
    for i in range(0, len(authors), per_row):
        row = authors[i:i + per_row]
        grid = "".join(f"<w:gridCol w:w=\"{width}\"/>" for _ in row)
        borders = "".join(f"<w:{b} w:val=\"nil\"/>" for b in ("top", "left", "bottom", "right", "insideH", "insideV"))
        out.append(f"<w:tbl><w:tblPr><w:tblW w:w=\"{width * len(row)}\" w:type=\"dxa\"/><w:jc w:val=\"center\"/>"
                   f"<w:tblBorders>{borders}</w:tblBorders><w:tblLayout w:type=\"fixed\"/></w:tblPr>"
                   f"<w:tblGrid>{grid}</w:tblGrid><w:tr>{''.join(_cell(a, width) for a in row)}</w:tr></w:tbl>")
        out.append("<w:p><w:pPr><w:spacing w:after=\"120\"/></w:pPr></w:p>")
    return "".join(out)


# ------------------------------------------------------------------ AST de pandoc
def _raw(xml: str) -> dict:
    return {"t": "RawBlock", "c": ["openxml", xml]}


def _styled(style: str, blocks: list) -> dict:
    return {"t": "Div", "c": [["", [], [["custom-style", style]]], blocks]}


def _str(text: str) -> list:
    out = []
    for i, w in enumerate(text.split(" ")):
        if i:
            out.append({"t": "Space"})
        out.append({"t": "Str", "c": w})
    return out


def _plain_text(inlines: list) -> str:
    return "".join(i["c"] if i["t"] == "Str" else " " if i["t"] in ("Space", "SoftBreak") else
                   _plain_text(i["c"]) if i["t"] in ("Strong", "Emph") else "" for i in inlines)


def _walk(node, fn):
    """Recorre el AST aplicando `fn` a cada lista (primero los hijos, luego la lista)."""
    if isinstance(node, list):
        for x in node:
            _walk(x, fn)
        node[:] = fn(node)
    elif isinstance(node, dict):
        for v in node.values():
            _walk(v, fn)


def _mark_pending(items: list) -> list:
    out = []
    for it in items:
        if isinstance(it, dict) and it.get("t") == "Strong" and _plain_text(it["c"]).startswith("[pendiente"):
            out.append({"t": "Span", "c": [["", [], [["custom-style", "Pendiente"]]], [it]]})
        else:
            out.append(it)
    return out


def _table_widths(items: list) -> list:
    """Anchos de columna explícitos: la primera (nombres) más ancha, el resto parejo."""
    for it in items:
        if isinstance(it, dict) and it.get("t") == "Table":
            specs = it["c"][2]
            n = len(specs)
            first = 0.30 if n <= 2 else (0.24 if n > 5 else 0.34)
            rest = (1 - first) / max(n - 1, 1)
            for i, spec in enumerate(specs):
                spec[1] = {"t": "ColWidth", "c": first if i == 0 else rest}
            specs[0][0] = {"t": "AlignLeft"}
    return items


def transform(doc: dict, title: str, authors: list[dict]) -> dict:
    blocks, out = doc["blocks"], []
    abstract = doc["meta"].pop("abstract", None)
    out.append(_styled("Title", [{"t": "Para", "c": _str(title)}]))
    out.append(_raw(authors_xml(authors)))
    out.append(_raw(section_break(1)))            # título y autores: una columna
    if abstract:
        paras = abstract["c"] if abstract["t"] == "MetaBlocks" else [{"t": "Para", "c": abstract["c"]}]
        first = paras[0]["c"]
        out.append(_styled("Abstract", [{"t": "Para", "c": [{"t": "Strong", "c": [{"t": "Emph", "c": _str("Resumen—")}]}] + first}]
                           + paras[1:]))
    sec, sub = 0, 0
    for b in blocks:
        if b["t"] == "Div" and "IEEEkeywords" in b["c"][0][1]:
            inl = b["c"][1][0]["c"]
            out.append(_styled("Abstract", [{"t": "Para", "c": [{"t": "Strong", "c": [{"t": "Emph", "c": _str("Palabras clave—")}]}] + inl}]))
            continue
        if b["t"] == "Para" and len(b["c"]) == 1 and b["c"][0].get("c") in (WIDE_START, WIDE_END):
            out.append(_raw(section_break(2 if b["c"][0]["c"] == WIDE_START else 1)))
            continue
        if b["t"] == "Header":
            level, attr, inl = b["c"]
            unnumbered = "unnumbered" in attr[1]
            if level == 1:
                if not unnumbered:
                    sec += 1; sub = 0
                    inl = _str(f"{roman(sec)}.") + [{"t": "Space"}] + inl
            elif level == 2:
                sub += 1
                inl = _str(f"{chr(64 + sub)}.") + [{"t": "Space"}] + inl
            b = {"t": "Header", "c": [level, attr, inl]}
        out.append(b)
    doc["blocks"] = out
    _walk(doc["blocks"], _mark_pending)
    _walk(doc["blocks"], _table_widths)
    return doc


# ------------------------------------------------------------------ documento de referencia (estilos)
def build_reference(path: Path) -> None:
    import docx
    from docx.enum.style import WD_STYLE_TYPE
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.oxml.ns import qn
    from docx.shared import Inches, Pt, RGBColor

    data = subprocess.run(["pandoc", "--print-default-data-file", "reference.docx"], capture_output=True, check=True).stdout
    path.write_bytes(data)
    d = docx.Document(str(path))

    class _Styles:  # búsqueda por nombre visible (python-docx traduce «Heading 1» y falla con pandoc)
        def __init__(self, styles):
            self._by = {x.name: x for x in styles}
            self._styles = styles
        def __getitem__(self, name):
            return self._by[name]
        def __iter__(self):
            return iter(self._styles)
        def add_style(self, *a):
            return self._styles.add_style(*a)
    st = _Styles(d.styles)

    def font(style, size, bold=None, italic=None, small_caps=None, color=RGBColor(0, 0, 0)):
        f = style.font
        f.name, f.size = FONT, Pt(size)
        rpr = style.element.get_or_add_rPr()
        fonts = rpr.find(qn("w:rFonts"))
        if fonts is None:
            fonts = rpr.makeelement(qn("w:rFonts"), {}); rpr.insert(0, fonts)
        for k in ("w:ascii", "w:hAnsi", "w:cs", "w:eastAsia"):
            fonts.set(qn(k), FONT)
        for k in ("w:asciiTheme", "w:hAnsiTheme", "w:cstheme", "w:eastAsiaTheme"):
            fonts.attrib.pop(qn(k), None)
        if bold is not None: f.bold = bold
        if italic is not None: f.italic = italic
        if small_caps is not None: f.small_caps = small_caps
        if color is not None: f.color.rgb = color

    def para(style, align=None, before=0, after=0, first=None, line=None):
        pf = style.paragraph_format
        if align is not None: pf.alignment = align
        pf.space_before, pf.space_after = Pt(before), Pt(after)
        if first is not None: pf.first_line_indent = first
        if line is not None: pf.line_spacing = line

    J, C, L = WD_ALIGN_PARAGRAPH.JUSTIFY, WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.LEFT
    for name in ("Normal", "Body Text", "First Paragraph", "Compact"):
        font(st[name], 10); para(st[name], J, first=Inches(0.14) if name != "Compact" else None)
    font(st["Compact"], 8); para(st["Compact"], C, first=Inches(0))
    font(st["Title"], 24, bold=False); para(st["Title"], C, 0, 10)
    st["Title"].element.get_or_add_pPr()
    for name, size in (("Heading 1", 10), ("Heading 2", 10), ("Heading 3", 10)):
        font(st[name], size, bold=False, italic=name != "Heading 1", small_caps=name == "Heading 1")
        para(st[name], C if name == "Heading 1" else L, 8 if name == "Heading 1" else 6, 4)
    font(st["Abstract"], 9, bold=True, italic=False); para(st["Abstract"], J, 0, 6, first=Inches(0.14))
    for name in ("Image Caption", "Table Caption"):
        font(st[name], 8, italic=False); para(st[name], C if name == "Table Caption" else J, 2, 8)
    font(st["Bibliography"], 8); para(st["Bibliography"], J, 0, 2)
    for name in ("Captioned Figure", "Figure"):
        if name in [s.name for s in st]:
            para(st[name], C, 6, 0)
    mono = st["Verbatim Char"]
    mono.font.name, mono.font.size = "Courier New", Pt(8)
    pend = st.add_style("Pendiente", WD_STYLE_TYPE.CHARACTER)
    pend.font.color.rgb, pend.font.bold = RGBColor(0xC0, 0, 0), True
    s = d.sections[0]
    s.page_width, s.page_height = Inches(8.5), Inches(11)
    s.left_margin = s.right_margin = Inches(MARGIN_LR / 1440)
    s.top_margin, s.bottom_margin = Inches(MARGIN_T / 1440), Inches(MARGIN_B / 1440)
    cols = s._sectPr.find(qn("w:cols"))
    if cols is None:
        cols = s._sectPr.makeelement(qn("w:cols"), {}); s._sectPr.append(cols)
    cols.set(qn("w:num"), "2"); cols.set(qn("w:space"), str(COL_GAP))
    d.save(str(path))


# ------------------------------------------------------------------ ajustes finales
def fit_figures(path: Path) -> int:
    """Ancho de cada figura según la columna en la que cae (secciones de 1 o 2 columnas)."""
    import docx
    from docx.oxml.ns import qn

    d = docx.Document(str(path))
    body = d.element.body
    pending, fixed = [], 0

    def apply(cols: int) -> None:
        nonlocal fixed
        width_emu = int((TEXT_W if cols == 1 else COL_W) * 0.97 * EMU_PER_TWIP)
        for inline in pending:
            ext = inline.find(qn("wp:extent"))
            cx, cy = int(ext.get("cx")), int(ext.get("cy"))
            new_cy = int(cy * width_emu / cx)
            ext.set("cx", str(width_emu)); ext.set("cy", str(new_cy))
            for a in inline.iter(qn("a:ext")):
                a.set("cx", str(width_emu)); a.set("cy", str(new_cy))
            fixed += 1
        pending.clear()

    for child in body.iterchildren():
        pending.extend(child.iter(qn("wp:inline")))
        sect = child.find(f"{qn('w:pPr')}/{qn('w:sectPr')}") if child.tag == qn("w:p") else None
        if child.tag == qn("w:sectPr"):
            sect = child
        if sect is not None:
            cols = sect.find(qn("w:cols"))
            apply(int(cols.get(qn("w:num"), "1")) if cols is not None else 1)
    d.save(str(path))
    return fixed


# ------------------------------------------------------------------ API
def export(paper_dir: Path, figures_dir: Path, out_docx: Path) -> dict:
    if shutil.which("pandoc") is None:
        raise RuntimeError("Se necesita pandoc (https://pandoc.org) para exportar a Word")
    flat = LatexFlattener(paper_dir, figures_dir).flatten()
    macros = flat.preamble_macros.replace("\\textcolor{red}{\\textbf{[pendiente: #1]}}", "\\textbf{[pendiente: #1]}")
    macros = macros.replace("\\textdegree", "°")
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "flat.tex").write_text(macros + "\n\\begin{document}\n" + flat.body + "\n\\end{document}\n", encoding="utf-8")
        ast = json.loads(subprocess.run(["pandoc", "-f", "latex", "-t", "json", str(tmp / "flat.tex")],
                                        capture_output=True, check=True, text=True).stdout)
        ast = transform(ast, flat.title, flat.authors)
        (tmp / "ast.json").write_text(json.dumps(ast), encoding="utf-8")
        ref = tmp / "reference.docx"
        build_reference(ref)
        csl = paper_dir / "word" / "ieee.csl"
        cmd = ["pandoc", "-f", "json", "-t", "docx", str(tmp / "ast.json"), "-o", str(out_docx),
               "--reference-doc", str(ref), "--citeproc", "--bibliography", str(paper_dir / "refs.bib"),
               "-M", "lang=es-PE", "-M", "link-citations=false", "--resource-path", str(figures_dir)]
        if csl.exists():
            cmd += ["--csl", str(csl)]
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    figures = fit_figures(out_docx)
    return {"figuras": figures, "autores": len(flat.authors), "referencias_cruzadas": len(flat.numbers)}
