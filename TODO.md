# Docs TODO

Cleanup from the docs review is done (Bayesian pages, dead references, viewcode, title, new
`paper_outputs` and `reproduce` pages). Remaining:

- [ ] Decide whether to add the `.md` docs (`new_city.md`, `purpleair_*.md`, `baseline_data_extract.md`).
      They are not in the build (need a toctree entry + `myst_parser`) and reference scripts that are not in this
      release (e.g. `scripts/make_person_subsample_das.py`), so they need review first.
- [ ] Add figure image files / plotting scripts to `paper_outputs` if they are released; today only summary tables exist.
