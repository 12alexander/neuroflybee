# Artículo (formato IEEE, máx. 12 páginas)

- `main.tex`: esqueleto con la estructura obligatoria del enunciado (§3.1). Cada
  sección indica en un comentario `TODO(vN)` el commit que la completa.
- `refs.bib`: referencias base del curso y del dominio (estilo IEEE, BibTeX).
- `figures/`: figuras propias del artículo; las figuras de resultados se toman de
  `../results/figures/` (se regeneran con los comandos del README principal).

Compilar en Overleaf (subir la carpeta) o localmente:

```bash
cd paper
pdflatex main && bibtex main && pdflatex main && pdflatex main
```
