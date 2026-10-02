# Building docs

This directory contains the Sphinx source for the release pipeline documentation.

## Local build

```bash
conda env update -f environment.yml
conda activate odmatrix
sphinx-build -b html docs docs/_build/html
open docs/_build/html/index.html
```
