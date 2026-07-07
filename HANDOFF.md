# Session Handoff

This file exists so a **new** Claude Code session (with no memory of prior
conversations) can pick this project up cold. Read this first, then README.md
for how the tool itself works.

## Where things stand

**Repo:** `RShiri/legooartlocator` — branch `claude/lego-pdf-part-scanner-lc3aiu`
**Last commit:** `9c94f48` — "Suppress bag-marker detection on dense/parts-list pages"
**Tests:** 94 passing, all offline (`pytest`)
**User's environment:** Windows, Python venv at `.venv`, `run.bat` wrapper
(`run scan ...` / `run debug ...`), no Tesseract binary installed, no paid
API keys purchased yet.

The tool scans a LEGO instruction-PDF and maps every part to the numbered bag
and pages it appears on (that mapping exists nowhere else — not in any API,
only in the printed booklet). Two engines:

- **`claude`** — paid Anthropic vision API, two-pass (cheap triage + detailed
  read), cached by page-image hash.
- **`local`** — free: OpenCV detects callout cells + bag-start pages
  (`vision_local.py`), an ensemble identifier matches each crop against the
  set's inventory using Brickognize (free API) + optional CLIP embeddings +
  colour (`identify.py`), constrained to a real inventory (Rebrickable key or
  a free CSV/JSON export via `--inventory-file`).
- `--set NNNN` with no PDF path **auto-downloads** the instructions from
  lego.com (`fetcher.py`) — confirmed working live on the user's machine.
- `lpl debug <pdf> --pages N-M` renders detection overlays + a threshold mask
  + per-page stats to a folder, **without** calling any paid/free identify
  API — pure visualization, safe to run repeatedly, zero cost. This is the
  calibration tool and it has already earned its keep (see below).

Full module map is in `README.md` → **Layout**. Don't duplicate it here.

## What's proven with real data (this is the important part)

The user has `6559641.pdf` locally — the real booklet for **LEGO set 76307**
(a LEGO Marvel Iron Man mech). They ran:

```
run debug 6559641.pdf --out out\debug --pages 1-10
```

and shared back two real page screenshots (page 1 = cover, page 4 = a "shake
the bag out" illustration) plus `stats.json`. This is the **only** real-world
testing that has happened so far — no full `lpl scan` (the identify pipeline)
has been run yet, only detection-only calibration via `lpl debug`.

That calibration found and fixed **two real bugs**, both confirmed by the user
seeing the corrected overlay:

1. **False bag-marker on illustration art** (page 4): the "shake the bag"
   circle illustration is tall and near-black like a numeral, so it got a
   blue `bag?` box around the whole thing. Fixed by adding upper-bound
   area/width gates to `DetectConfig` (`bag_marker_max_area_frac`,
   `bag_marker_max_width_frac`) — calibrated so real numerals still pass but
   large artwork doesn't.
2. **False bag-marker on the cover page** (page 1): dense with small
   logos/badges (tripping `is_parts_list`) *and* a moderately-sized character
   render that passed gate #1's thresholds. This one mattered more than
   cosmetics — since `assign_ordinal_bag_numbers` assigns "last known + 1" to
   every candidate in page order, an undetected false positive on page 1
   would have become "Bag 1", shifting every real bag number off by one on
   a full scan. Fixed by skipping bag-marker detection entirely on any page
   already flagged `is_parts_list` (a page cluttered enough to look like a
   contents/BOM/cover grid is never also a clean bag-start page).

Both fixes shipped with regression tests reproducing the exact real-page
shapes. User confirmed page 4's box was gone after the fix ("Page 1's bag?
box should be gone now. true").

## Immediate next step

Pick up the calibration loop where it left off:

1. `run debug 6559641.pdf --out out\debugN --pages 1-30` (or a wider range) —
   the goal is to reach an **actual bag-start page** (bag 1, 2, ...) and
   confirm it's detected correctly. Every real page seen so far has been
   front-matter (cover, intro illustration); no genuine bag marker has been
   visually confirmed yet.
2. Once detection looks right, run the **full identify pipeline** for the
   first time on a real set:
   ```
   run scan 6559641.pdf --engine local --inventory-file inv.csv --out out
   ```
   (`inv.csv` = a free BrickLink/Rebrickable parts export for 76307, or use
   `--set 76307` with a `REBRICKABLE_API_KEY` instead). This has never been
   run against a real PDF — accuracy of the Brickognize/colour identification
   ensemble on real crops is completely unverified.
3. Load `out/result.json` into `web/index.html` (drag-drop) and sanity-check
   a few parts against the physical instructions.

## Known open items (not yet started, roughly ranked)

- **No real `lpl scan --engine local` run yet.** Everything below the
  detection layer (Brickognize matching, colour scoring, count reconciliation)
  is unit-tested but has zero real-world mileage.
- **Callout-cell gray-band thresholds unconfirmed.** Only bag-marker false
  positives have been found/fixed so far. `mask_*.png` from `lpl debug` (the
  gray-band threshold visualization) hasn't been reviewed yet — worth a look
  before trusting callout counts.
- **`--embeddings` (local CLIP image matching) never exercised** on a real
  set — needs the `[ml]` extra (`pip install -e ".[ml]"`, pulls in torch).
- **Tesseract not installed** on the user's machine — quantities currently
  default to 1 on the local engine (bag numbers work fine without it, via
  ordinal assignment). Installing Tesseract would unlock real quantity/part-ID
  OCR; not yet requested by the user.
- Cover/BOM-style pages' callout-like false detections (logos/badges counted
  as "cells", tripping `is_parts_list`) are currently harmless — such pages
  are already excluded from parts attribution — but worth a note if it ever
  causes a *real* build-step page to be wrongly treated as front matter.
- **No PR opened** for this branch — user hasn't asked for one yet.

## Full session history (condensed)

1. Built the initial scanner: PDF render → Claude vision (two-pass, cached) →
   bag segmentation → Rebrickable reconciliation → `result.json`/`.csv` → web
   viewer.
2. Added the free local-identification stack: Brickognize client, embedding
   retrieval (CLIP), LEGO colour table, CSV/JSON inventory loader, ensemble
   identifier — all unit-tested with fakes/injection (no network needed to
   test).
3. Wired it end-to-end as `--engine local` (`locate.py` orchestrator), added
   `lego.com` auto-download by set number (`fetcher.py`), graceful
   missing-dependency errors, a `run.bat` Windows wrapper.
4. Ran a **staged, multi-agent improvement round** — a Plan-agent review of
   the codebase found two real bugs and a design insight before any code was
   written, then six independent Sonnet agents built in parallel (verified
   pairwise file-disjoint):
   - **S6** — scan-killing crash fix (one failed identify call used to abort
     the whole scan with zero output) + colour-only false-identification fix
     (colour alone can no longer "identify" a part).
   - **S3** — structural, OCR-free bag-marker detection: bags are always
     numbered 1..N in page order, so a detected-but-unreadable bag-start page
     gets its number assigned ordinally, no Tesseract required. Also fixed a
     latent multi-digit bug (bag "12" used to OCR as two separate digits).
   - **S2** — Brickognize responses cached by crop-image hash, retried with
     backoff, throttled; one long-lived HTTP client instead of one per call.
   - **S1** — `lpl debug` calibration tool (overlays, threshold mask,
     per-page stats) — this is what found the two real bugs below.
   - **S4** — web viewer v2: warnings panel, bag-overview cards, occurrence
     drill-down, count-mismatch highlighting.
   - **S5** — CI (pytest on Ubuntu 3.11/3.12 + Windows 3.12).
   - Integration wave: wired ordinal bag numbering into `locate_local`, added
     the `lpl debug` CLI command, cache/panel-band flags; fixed a bug found
     during integration itself (`lpl debug` crashed on missing Tesseract
     instead of degrading like `scan` does — now shares one helper).
5. **Real-data calibration** (this is genuinely new information, not
   speculative): user ran `lpl debug` against their real 76307 PDF, shared
   two page screenshots, and two real false-positive bugs (described above
   under "What's proven with real data") were found and fixed as a direct
   result — not something unit tests alone would have caught, since they
   were shape/size miscalibrations specific to real illustration artwork.

## Resume prompt

Paste this into a fresh session to continue:

```
I'm continuing work on legooartlocator (github.com/RShiri/legooartlocator,
branch claude/lego-pdf-part-scanner-lc3aiu). Read HANDOFF.md at the repo root
first for full session history and current state, then README.md for how the
tool works.

Short version: it's a LEGO instruction-PDF scanner that maps parts to
bag/page. Two engines: a paid Claude-vision engine and a free local
OpenCV+Brickognize engine (`lpl scan --engine local`). 94 tests pass.

I've been running `lpl debug <pdf> --pages N-M` against my real set 76307 PDF
(6559641.pdf) and sending back page screenshots for calibration -- two real
bag-marker false positives were already found and fixed this way. No full
`lpl scan` (the identify pipeline) has been run yet, only detection
calibration via `lpl debug`.

Next: [paste your latest `lpl debug`/`lpl scan` output or a page screenshot
here, or say "let's run debug on pages X-Y next" to continue calibration].
```
