# TransPhaser Manuscript Directory

This directory contains the LaTeX source and assets for the TransPhaser manuscript submission to Bioinformatics.

## Directory Structure

- `main.tex`: The main LaTeX manuscript file.
- `bib/`: Bibliography files.
  - `references.bib`: BibTeX entries.
- `figures/`: High-resolution figures used in the manuscript.
- `tables/`: (Optional) TeX files for complex tables if included via `\input`.

## Compiling

To compile the manuscript, use:

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

## Figures to Include

The following figures from `examples/output/comprehensive_training/` are relevant:
1. `transphaser_latent_pca.png`: Visualizes the latent space separation by population.
2. `training_loss_curves.png`: Shows model convergence and KL annealing.
3. `em_likelihood_history.png`: Baseline EM algorithm convergence.
