# Graph Neural DSO-DTA LaTeX document

The manuscript is `gnn_dso_model.tex`; citations are stored in `references.bib`.

To build with a standard TeX Live installation:

```sh
latexmk -pdf gnn_dso_model.tex
```

Or compile manually:

```sh
pdflatex gnn_dso_model.tex
bibtex gnn_dso_model
pdflatex gnn_dso_model.tex
pdflatex gnn_dso_model.tex
```

The document is explicitly framed as a research and implementation blueprint. It does
not claim that proposed performance thresholds have already been achieved.

