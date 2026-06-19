# Contributing

Thanks for your interest in contributing to `missing_imputation`.

## Development setup

```bash
git clone <repo-url>
cd missing_imputation
pip install -e ".[dev]"
```

Run the test suite and linter before opening a pull request:

```bash
pytest -q
ruff check src/ tests/
```

CI runs the same checks on Python 3.9, 3.11, and 3.12.

## Project layout

```
src/missing_imputation/
  __init__.py        # public API + METHODS registry
  columns.py         # FACT-E item / subscale / visit definitions
  metrics.py         # classification + ordinal evaluation metrics
  synthetic.py       # PHI-free synthetic data generator
  cli.py             # `missing-impute` command-line interface
  methods/           # one module per imputation method
tests/               # pytest suite (runs on synthetic data only)
examples/            # runnable usage examples
notebook/            # original research notebooks (reference only)
```

## Adding or extracting an imputation method

The package is being assembled by extracting methods from the research
notebooks under `notebook/`. To add a method:

1. Create `src/missing_imputation/methods/<name>.py` exposing
   `apply_<name>_imputation(df, columns_to_impute, validation_df=None,
   validation_masks=None, original_values=None, ...)` returning
   `(imputed_df, validation_results)`.
2. **Preserve the research logic verbatim.** This is a packaging effort, not a
   re-implementation. Copy the algorithm exactly; only change imports and
   wiring needed to make it importable. If a dependency's API forced a change
   (as with `miceforest` 6.x), document it in a comment.
3. Register it in `methods/__init__.py` and in the `METHODS` dict in
   `__init__.py`.
4. Add a smoke test in `tests/test_methods.py` (it should fill all missing
   cells on synthetic data and return a well-formed `validation_results`).
5. If the method needs heavy dependencies (torch, transformers, etc.),
   import them lazily inside the function and add them to an appropriate
   optional-dependency extra in `pyproject.toml`.

## Data safety

**Never commit clinical data.** All examples and tests must use synthetic data
from `make_synthetic_facte`. The `data/` directory is git-ignored. Before
committing, verify nothing under `data/` is staged:

```bash
git diff --cached --name-only | grep -i data   # should print nothing
```

## Commit / PR guidelines

- Keep commits focused and message-clear.
- Work on a feature branch; open PRs against `main`.
- Make sure `pytest` and `ruff` pass.
