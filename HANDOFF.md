# Session Handoff

This file is the **current state** — read this first. For the full
chronological story of how the project got here (calibration sessions,
real-data findings, the training experiments, the six-phase follow-up round),
see [HANDOFF_HISTORY.md](HANDOFF_HISTORY.md). See `README.md` for how the tool
itself works and its module layout.

## Repo & environment

**Repo:** `RShiri/legooartlocator` — branch `claude/lego-pdf-part-scanner-lc3aiu`.
**Tests:** run `pytest` (don't hard-code the count here — it drifts every
round). `ruff check .` for lint. CI (`.github/workflows/ci.yml`) runs the main
matrix (Ubuntu 3.11/3.12 + Windows 3.12, plus lint) and a separate job that
installs the `[train]` extra so torch-dependent tests run for real in CI too.
**User's machine:** Windows, Python venv at `.venv`, `run.bat` wrapper (`run
scan ...` / `run debug ...`). No Tesseract binary — not needed: quantities are
read by the built-in dependency-free `DigitOCR`, bag numbers via ordinal
assignment. AMD Radeon RX 6800 GPU — no CUDA, no DirectML (Python version too
new for `torch-directml`'s wheels) — training runs on CPU, tractable for these
dataset sizes (minutes, not hours). Ryzen 5 5600X, 16GB RAM. `torch` is
installed (`.venv` has the `[train]` extra). `REBRICKABLE_API_KEY` is in
`.env` (gitignored — re-provide it if missing in a fresh checkout). **User
does not want the paid Claude vision engine used** — all work has been on the
free local engine (`--engine local`).

## What the tool does

Maps every part in a LEGO instruction-PDF to its numbered bag and pages (a
mapping that exists nowhere else — not in any API, only in the printed
booklet). Two engines:
- **`claude`** — paid Anthropic vision API, two-pass (cheap triage + detailed
  read), cached by page-image hash. Not used for this project's work so far.
- **`local`** (the one actually used) — free: OpenCV detects callout cells +
  bag-start pages (`vision_local.py`), an ensemble identifier matches each
  crop against the set's inventory using Brickognize (free API) + optional
  trained/CLIP embeddings + colour (`identify.py`), constrained to a real
  inventory (Rebrickable key or a free CSV/JSON export).
- `--set NNNN` with no PDF path auto-downloads instructions from lego.com.
- `lpl debug <pdf>` renders detection overlays for threshold calibration,
  zero cost, safe to run repeatedly.

Full module map: `README.md` → **Layout**.

## Current defaults & measured results (most important section)

Four things are shipped as **defaults** on the local engine, each measured
positive on real data before being made the default (full evidence in
HANDOFF_HISTORY.md):

1. **Icon-style training augmentation** (`--augment-style icon`) — flattens
   shading + adds a black edge outline so catalog reference photos look more
   like flat instruction-booklet icons.
2. **Semi-hard negative mining** (`--mining semihard`) — mines hard triplet
   negatives from the current model each epoch instead of sampling randomly.
3. **Capacity/trust-aware reconciliation** (`--capacity-reconcile`) — stops an
   embedding-only match from over-filling a part past its known inventory
   quantity.
4. **Dependency-free quantity OCR** (`DigitOCR`, automatic when Tesseract
   isn't installed) — reads the printed "Nx" label by digit template-matching.

**Measured, reproducible results:**
- **76307** (Iron Man Mech, 47-part inventory, single unnumbered bag):
  baseline (no embeddings) 31/47 identified & 12 count-mismatch warnings →
  **37/47 identified & 7 warnings** with the trained checkpoint
  (`models/lego_embed_76307_v4.pt`, gitignored — regenerate with
  `run train-embedding --set 76307`) and all defaults above. Command:
  `run scan 6559641.pdf --engine local --set 76307 --embeddings
  --embedding-weights models/lego_embed_76307_v4.pt`.
- **76269** (Avengers Tower, 786-line inventory, 3 booklets/644 pages):
  booklet 1 baseline 168 identified & 128 mismatches → **251 identified
  (+49%) & 157 mismatches** — and a *lower* mismatch rate per identified part
  (76% → 62%) — with a trained checkpoint (`models/lego_embed_76269.pt`,
  gitignored). This is the strongest evidence that the approach generalises
  past the one set it was tuned on.

**What was tried and did NOT make the cut** (each a real, measured attempt —
don't re-try these without new data/reasoning; full detail in
HANDOFF_HISTORY.md's "Follow-up round 2" section):
- An open-set embedding-score floor — corroborated and embedding-only match
  scores fully overlap, no threshold works.
- Real-crop self-training (`--real-crops`, ships as opt-in) — more precise
  but less coverage on 76307 (only 49 real crops existed across 31 parts).
- LDraw-rendered training images (`--ldraw-dir`/`--ldraw-only`, ship as
  opt-in) — whole-inventory and targeted-only variants both tried;
  whole-inventory recovered one previously-unreachable printed part but
  regressed two others; targeting specific parts didn't help because
  semi-hard mining couples the whole embedding space through shared
  negatives regardless of which parts' photos changed.

## Codebase hardening pass (latest work)

A full audit (3 parallel Explore agents: core-module bugs, test/tooling gaps,
docs/packaging hygiene) turned into a fix-everything-real pass: retry/backoff
added to the Claude vision client, Rebrickable error handling fixed so a bad
key/outage actually falls back to vision-only instead of crashing, page-range
parsing validated, a corrupt/truncated auto-downloaded PDF now gets
re-downloaded instead of reused forever, inventory-file JSON edge cases
hardened, a `BagSegment` ordering validator added, the `lpl debug` calibration
tool now isolates one bad page instead of losing the whole run, `ruff`
lint added (clean), and test coverage added for every previously-untested
module (`aggregate.py`, `cache.py`, `pdf_render.py`, `rebrickable.py`,
`models.py`, plus real `CliRunner` end-to-end CLI tests where there were none
before). Also fixed a real latent bug this pass's own lint step caught:
`locate.py` used `Tuple[...]` in three type hints without importing `Tuple`
(silent because `from __future__ import annotations` defers evaluation —
would only have surfaced if something ever introspected the annotations). See
the git log for exact commits; nothing here changed any shipped default or
measured result above — it was pure hardening on top of an already-measured,
stable system.

## Next steps (ranked)

1. **76269 residual bag-marker false positives** (~2% of 644 pages, not yet
   individually diagnosed — same "illustration element looks digit-shaped"
   class as 76307's already-fixed cases).
2. **76269's own scan quality beyond booklet 1** — booklet 1 is validated
   with embeddings; booklets 2/3 haven't been.
3. **If pursuing the domain-gap further**: the two real levers left, per the
   "what was tried" list above, are a *larger* real-crop corpus (more scans →
   bigger `--dump-crops` manifests than the 49-crop one tried) or a genuinely
   different training objective — not another LDraw variant (two honest
   attempts there didn't pan out).
4. **No PR opened yet** for this branch — not requested.

## Resume prompt

Paste this into a fresh session to continue:

```
I'm continuing work on legooartlocator (github.com/RShiri/legooartlocator,
branch claude/lego-pdf-part-scanner-lc3aiu). Read HANDOFF.md at the repo root
first for current state, then HANDOFF_HISTORY.md if you need the full
chronological story, then README.md for how the tool works.

It's a LEGO instruction-PDF scanner mapping parts to bag/page. Local engine
only (`--engine local`) -- no paid Claude API. Shipped defaults: icon-style
training augmentation, semi-hard negative mining, capacity-aware
reconciliation, and a dependency-free quantity-OCR reader -- all measured
positive on two real sets (76307: 37/47 identified & 7 warnings; 76269
booklet 1: +49% coverage with a *lower* mismatch rate). Several follow-on
ideas (embedding-score floor, real-crop self-training, LDraw-rendered
training images in two forms) were tried and honestly did not beat the
current defaults -- don't re-attempt them without new data or reasoning; see
HANDOFF.md's "What was tried and did NOT make the cut" for why each failed.

A full codebase hardening pass (bug fixes across pdf_render/rebrickable/
vision/fetcher/inventory/models/debug_overlay/bags, new tests for every
previously-untested module, ruff lint, doc/packaging cleanup) is also done --
see HANDOFF.md's "Codebase hardening pass" section.

Next: see "Next steps" in HANDOFF.md.
```
