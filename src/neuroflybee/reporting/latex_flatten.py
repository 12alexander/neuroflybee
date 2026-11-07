"""Aplana el artículo LaTeX para convertirlo a Word con pandoc (v19).

Resuelve lo que pandoc no entiende del fuente IEEE: `\\input` recursivos, `\\IfFileExists`,
rutas de figuras, `\\resizebox`, numeración de figuras/tablas/ecuaciones (estilo IEEE:
«Fig. 1», «TABLA I») y referencias cruzadas. El resultado es LaTeX simple, sin paquetes.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

WIDE_START, WIDE_END = "@@ANCHO-INICIO@@", "@@ANCHO-FIN@@"


def roman(n: int) -> str:
    vals = [(10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I")]
    out = ""
    for v, s in vals:
        while n >= v:
            out += s; n -= v
    return out


def _strip_comments(text: str) -> str:
    return re.sub(r"(?<!\\)%.*", "", text)


def _balanced(text: str, start: int) -> int:
    """Índice de la llave que cierra la que abre en `start`."""
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{" and text[i - 1] != "\\":
            depth += 1
        elif text[i] == "}" and text[i - 1] != "\\":
            depth -= 1
            if depth == 0:
                return i
    raise ValueError("llaves desbalanceadas")


def _arg(text: str, pos: int) -> tuple[str, int]:
    """Lee un argumento {…} que empieza en `pos` (salta espacios). Devuelve (contenido, fin)."""
    while text[pos] in " \n\t":
        pos += 1
    end = _balanced(text, pos)
    return text[pos + 1:end], end + 1


@dataclass
class Flattened:
    preamble_macros: str
    title: str
    authors: list[dict]
    body: str
    numbers: dict[str, str] = field(default_factory=dict)


class LatexFlattener:
    def __init__(self, paper_dir: Path, figures_dir: Path):
        self.paper = paper_dir
        self.figures = figures_dir

    # ------------------------------------------------------------ \input e \IfFileExists
    def _resolve_inputs(self, text: str, depth: int = 0) -> str:
        text = _strip_comments(text)
        out, pos = [], 0
        pattern = re.compile(r"\\IfFileExists|\\input")
        while (m := pattern.search(text, pos)):
            out.append(text[pos:m.start()])
            if m.group() == "\\input":
                name, end = _arg(text, m.end())
                path = self.paper / (name if name.endswith(".tex") else name + ".tex")
                out.append(self._resolve_inputs(path.read_text(encoding="utf-8"), depth + 1) if path.exists() else "")
            else:
                fname, end = _arg(text, m.end())
                yes, end = _arg(text, end)
                no, end = _arg(text, end)
                exists = (self.paper / fname).exists()
                out.append(self._resolve_inputs(yes if exists else no, depth + 1))
            pos = end
        out.append(text[pos:])
        return "".join(out)

    # ------------------------------------------------------------ autores
    @staticmethod
    def _authors(block: str) -> list[dict]:
        people = []
        for m in re.finditer(r"\\IEEEauthorblockN\{([^}]*)\}\s*\\IEEEauthorblockA", block):
            aff, _ = _arg(block, m.end())
            lines = [re.sub(r"\\textit\{([^}]*)\}", r"\1", l).replace("--", "–").strip()
                     for l in aff.split("\\\\")]
            people.append({"name": m.group(1).strip(), "lines": [l for l in lines if l]})
        return people

    # ------------------------------------------------------------ numeración IEEE
    def _number(self, body: str) -> tuple[str, dict[str, str]]:
        numbers: dict[str, str] = {}
        counters = {"figure": 0, "table": 0, "equation": 0}
        env_re = re.compile(r"\\begin\{(figure\*?|table\*?|equation)\}(.*?)\\end\{\1\}", re.S)

        def repl(m: re.Match) -> str:
            env, content = m.group(1), m.group(2)
            kind = env.rstrip("*")
            counters[kind] += 1
            n = counters[kind]
            label = re.search(r"\\label\{([^}]*)\}", content)
            if kind == "figure":
                tag = str(n)
                content = content.replace("\\caption{", f"\\caption{{Fig. {tag}. ", 1)
            elif kind == "table":
                tag = roman(n)
                content = content.replace("\\caption{", f"\\caption{{TABLA {tag}. ", 1)
            else:
                tag = f"({n})"
                content = re.sub(r"\\label\{[^}]*\}", "", content)
                content = content.replace("\\begin{split}", "\\begin{aligned}").replace("\\end{split}", "\\end{aligned}")
                content = content.rstrip() + f" \\qquad {tag}\n"
            if label:
                numbers[label.group(1)] = tag
            wide = env.endswith("*")
            env_out = env.rstrip("*")
            block = f"\\begin{{{env_out}}}{content}\\end{{{env_out}}}"
            return f"\n\n{WIDE_START}\n\n{block}\n\n{WIDE_END}\n\n" if wide else block

        body = env_re.sub(repl, body)
        body = re.sub(r"\\eqref\{([^}]*)\}", lambda m: numbers.get(m.group(1), "(?)"), body)
        body = re.sub(r"\\ref\{([^}]*)\}", lambda m: numbers.get(m.group(1), "?"), body)
        return body, numbers

    # ------------------------------------------------------------ limpieza
    def _clean(self, body: str) -> str:
        body = re.sub(r"\\includegraphics(\[[^\]]*\])?\{([^}]*)\}",
                      lambda m: f"\\includegraphics{{{self._figure(m.group(2))}}}", body)
        # \resizebox{…}{!}{ tabla } → tabla
        while (m := re.search(r"\\resizebox\{[^}]*\}\{!\}\{%?", body)):
            open_brace = m.end() - (2 if body[m.end() - 1] == "%" else 1)
            close = _balanced(body, open_brace)
            body = body[:m.start()] + body[open_brace + 1:close] + body[close + 1:]
        body = re.sub(r"\\noindent\\begin\{tabular\}\{@\{\}p\{[^}]*\}p\{[^}]*\}@\{\}\}",
                      r"\\begin{tabular}{ll}", body)
        body = body.replace("\\textdegree", "°").replace("\\,", " ")
        return body

    def _figure(self, name: str) -> str:
        for cand in (self.figures / f"{name}.png", self.paper / "figures" / f"{name}.png", self.figures / name):
            if cand.exists():
                return str(cand)
        return name

    # ------------------------------------------------------------ API
    def flatten(self, main: str = "main.tex") -> Flattened:
        text = self._resolve_inputs((self.paper / main).read_text(encoding="utf-8"))
        title, _ = _arg(text, text.index("\\title") + len("\\title"))
        author_block, _ = _arg(text, text.index("\\author") + len("\\author"))
        macros = "\n".join(m.group(0) for m in re.finditer(r"\\newcommand\{\\(nfb\w+|pendiente)\}(\[\d\])?\{.*?\}(?=\s*(\n|$))", text))
        body = text[text.index("\\maketitle") + len("\\maketitle"):text.index("\\end{document}")]
        body = re.sub(r"\\IfFileExists\{IEEEtran\.bst\}.*", "", body)
        body = re.sub(r"\\bibliographystyle\{[^}]*\}", "", body)
        body = body.replace("\\bibliography{refs}", "\\section*{Referencias}\n")
        body, numbers = self._number(body)
        body = self._clean(body)
        return Flattened(macros, " ".join(title.split()), self._authors(author_block), body, numbers)
