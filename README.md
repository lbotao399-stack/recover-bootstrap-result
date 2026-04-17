## recover-bootstrap-result

This repository contains the paper source for "Bootstrapping supersymmetric (matrix) quantum mechanics" together with the implementation workspace for reproducing its SUSY/MP bootstrap results.

It is intentionally a standalone project and does not import or depend on the separate BFSS bootstrap repository.

Current layout:

- `Bootstrapping supersymmetric (matrix) quantum mechanics/`: extracted LaTeX source and running log
- `src/susy_mp_bootstrap/`: implementation package
- `tests/`: regression tests for symbolic reduction, PSD assembly, and matrix counting
- `notes_pro/`: theory inputs handed off to GPT Pro
- `runs/`: reproducible run outputs

The paper source was extracted from:

- `/Users/libotao/Desktop/qcohomology_project/tmp/notion_import_bootstrap_src/source/arxiv_source.tar`
