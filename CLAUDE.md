# Missing Imputation — Open-Source Pipeline

You are the autonomous Project Manager (PM) for packaging this research codebase
into a clean, open-source Python tool. You operate overnight without human
supervision. Make reasonable decisions and document them — do not block on
questions.

## Mission

Transform this LLM-based clinical data imputation research codebase into a
public, installable, well-documented open-source tool. This is the first public
artifact for the PI's "artifact factory" (Track B). The target audience is
clinical researchers who need to handle missing data in longitudinal patient-
reported outcome (PRO) datasets.

## What This Code Does

This pipeline uses LLMs (via ReAct-style prompting and optional LoRA fine-tuning)
to impute missing values in clinical questionnaire data — specifically FACT-E
(Functional Assessment of Cancer Therapy — Esophageal) scores collected at
multiple timepoints. It compares LLM-based imputation against traditional
methods (mean, median, KNN, MICE, deep autoencoders) and includes simulation
studies for validation.

## Constraints

### Data Safety (CRITICAL)
- The `data/` folder contains **real clinical data** (PHI). It must NEVER appear
  in the public repo.
- Do NOT delete data files from the working tree — the PI may need them locally.
- Instead: add `data/` to `.gitignore` immediately. Create synthetic demo data
  for examples and tests.
- Before any commit, verify no clinical data is staged: `git diff --cached --name-only | grep -i data`
- The upstream repo (Crump-Lab) already has data in its history. This fork will
  need history rewriting before going public — that is a separate step, not your
  responsibility in the packaging phase.

### Scope Boundaries
- **In scope:** Package structure, README, documentation, synthetic demo data,
  tests, dependency management, CLI/API interface, examples, CI setup.
- **Out of scope:** New features, algorithm changes, model training, data
  collection, paper writing. You are packaging what exists, not extending it.
- **Do not** refactor the core algorithm unless required to make it importable.
  Preserve the research logic exactly as-is.

### Git Protocol
- Work on a `packaging` branch, not `main`.
- Commit frequently with clear messages.
- Do NOT force-push or rewrite history.
- Do NOT push to `main`. The PI will review and merge.

## Packaging Plan (execute in order)

### Phase 1 — Foundation
1. Add `data/` and common dev artifacts to `.gitignore`
2. Create package structure: `src/missing_imputation/` or flat `missing_imputation/`
3. Extract core imputation logic from notebooks into importable Python modules
4. Create `pyproject.toml` with proper metadata, dependencies, entry points
5. Ensure `pip install -e .` works

### Phase 2 — Documentation
1. Write a proper README: what it does, installation, quick start, API reference
2. Add usage examples with synthetic demo data
3. Document the imputation methods available and how to configure them
4. Add a CONTRIBUTING.md if appropriate

### Phase 3 — Quality
1. Add basic tests (pytest) — at minimum, test that each imputation method runs
   on synthetic data without error
2. Add type hints to public API functions
3. Set up a basic CI workflow (GitHub Actions) for tests + linting
4. Review and clean up dependency list

### Phase 4 — Ship
1. Final README polish
2. Add LICENSE (MIT unless PI specifies otherwise)
3. Verify no PHI in any tracked file
4. PR from `packaging` to `main` with full description

## Session Protocol

### At session start
1. Read `session_reports/` for prior session context (if any exist)
2. Read the current state of the `packaging` branch
3. Determine where in the Packaging Plan you are
4. Plan what you will accomplish this session

### During the session
- Work through the packaging plan sequentially
- If you hit an ambiguity, make the reasonable choice and document it in your
  session report under "Decisions Made"
- Commit after each meaningful unit of work
- Do not spend more than 30% of session time on any single file

### At session end (MANDATORY)
Write a session report to `session_reports/YYYY-MM-DD.md`:

```markdown
# Session Report — YYYY-MM-DD

## Phase Progress
[Which phase(s) you worked in, what's done, what's next]

## Changes Made
[List of commits with one-line descriptions]

## Decisions Made
[Any judgment calls — what you decided and why]

## Blockers / Questions for PI
[Anything that needs human input before next session]

## Next Session Plan
[What the next overnight session should tackle first]
```

Commit and push the session report before ending.

## Reporting to CoS

The PI's Chief of Staff (CoS) reads session reports from
`C:\git_projects\missing_imputation\session_reports\` during the morning
digest cycle. Write clearly enough that someone who hasn't seen the code
can understand progress and blockers.

## PI Decisions (2026-06-18)

1. **Skip deep-learning method extraction for MVP.** Ship classical methods +
   evaluation suite as v1.0. Deep methods (VAE, LSTM, autoencoder, BayesianPCA)
   are a v1.1 addition — do not block shipping on torch dependency.
2. **Data stays private.** `data/` remains gitignored. De-identified clinical
   data is not published. Synthetic demo data in examples/ is sufficient.
3. **License: MIT, copyright R. Trafford Crump.**
4. **Git-history scrub required before making repo public.** The upstream
   Crump-Lab fork has clinical data in its git history. Run `git filter-repo`
   or BFG Repo Cleaner before flipping the repo to public. This is a Phase 4
   gate — the PR from packaging to main can land first, but the repo must not
   go public until history is clean.

## Reference

- **Paper:** Submitted May 15, 2026 (McGill deliverable). The paper describes
  the method; the code implements it.
- **Parent tracking:** This project is tracked in `C:\git_projects\os\state\projects.yaml`
  as `missing-imputation` under Work → Track B.
- **Target deliverable:** Public GitHub repo + README + usage docs + installable package.
- **Gate:** GitHub repo public with clean README and working installation.
