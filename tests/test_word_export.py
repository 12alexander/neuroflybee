"""v19: exportación del artículo a Word con formato IEEE."""
import shutil
from pathlib import Path

import pytest

from neuroflybee.reporting.latex_flatten import WIDE_END, WIDE_START, LatexFlattener, roman

ROOT = Path(__file__).resolve().parents[1]


def test_roman():
    assert [roman(n) for n in (1, 4, 9, 12)] == ["I", "IV", "IX", "XII"]


def test_flatten_numbers_and_authors():
    f = LatexFlattener(ROOT / "paper", ROOT / "results" / "figures").flatten()
    assert len(f.authors) == 5 and f.authors[0]["lines"][-1] == "correo1@uni.pe"
    assert f.numbers["fig:escenarios"] == "1" and f.numbers["tab:origenes"] == "I" and f.numbers["tab:datos"] == "II"
    assert "\\input" not in f.body and "\\ref{" not in f.body and "\\IfFileExists" not in f.body
    assert f.body.count(WIDE_START) == f.body.count(WIDE_END) > 0
    assert "Fig. 1." in f.body and "TABLA I." in f.body


@pytest.mark.skipif(shutil.which("pandoc") is None, reason="requiere pandoc")
def test_export_docx(tmp_path):
    docx = pytest.importorskip("docx")
    from neuroflybee.reporting.word_export import COL_W, EMU_PER_TWIP, export
    out = tmp_path / "paper.docx"
    info = export(ROOT / "paper", ROOT / "results" / "figures", out)
    assert info["autores"] == 5 and info["figuras"] >= 9
    d = docx.Document(str(out))
    text = "\n".join(p.text for p in d.paragraphs)
    assert "Resumen—" in text and "I. Introducción" in text and "Referencias" in text
    assert "[1]" in text                                   # referencias IEEE numeradas
    widths = [int(e.get("cx")) for e in d.element.body.iter("{http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing}extent")]
    assert max(widths) > COL_W * EMU_PER_TWIP              # hay figuras a todo el ancho
    assert min(widths) <= COL_W * EMU_PER_TWIP             # y figuras a una columna
