# Contributing to Vague

## Dev setup

```bash
git clone https://github.com/lorenzopastore/vague.git
cd vague
pip install -e ".[dev]"
```

This installs the package in editable mode along with all test, lint, and benchmark dependencies.

## Run tests

```bash
pytest
```

Coverage is reported automatically. All new code must have tests.

## Run lint

```bash
ruff check vague/ tests/
```

Fix any reported issues before opening a pull request. The CI gate will block on lint failures.

## Benchmark smoke test

```bash
python -c "from benchmarks.needle import run_needle; r = run_needle(1024, 0.5, n_components=8); assert r['found'] is not None"
```

## Branch naming

| Prefix    | Use for                              |
|-----------|--------------------------------------|
| `feat/`   | New features                         |
| `fix/`    | Bug fixes                            |
| `bench/`  | Benchmark additions or improvements  |

Example: `feat/streaming-recall`, `fix/merge-weights-overflow`, `bench/squad-eval`.

## Pull request checklist

- [ ] Tests pass: `pytest`
- [ ] Lint clean: `ruff check vague/ tests/`
- [ ] Benchmark smoke passes (see above)
- [ ] Docstrings updated for any changed public methods
- [ ] No new library logic in `examples/`, `README.md`, or `CONTRIBUTING.md`

## Scope

This repository's library code lives entirely under `vague/`. Do not add ML logic to examples or documentation files.
