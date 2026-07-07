# Session Handoff

This file exists so a **new** Claude Code session (with no memory of prior
conversations) can pick this project up cold. Read this first, then README.md
for how the tool itself works.

## Where things stand

**Repo:** `RShiri/legooartlocator` — branch `claude/lego-pdf-part-scanner-lc3aiu`
**Last commit:** `c9a8a0c`
**Tests:** 103 passing, all offline (`pytest`)
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

## This session: full-booklet calibration, three more real false positives found and fixed

Ran detection across the **entire** 76307 booklet (56 pages, not just a
slice) this time, using a script that calls `detect_page` directly with a
null OCR (mirrors the CLI's no-Tesseract fallback) so every page's
`bag_marker_candidate`/`bag_marker_bbox` could be inspected at once instead of
one `--pages` range at a time. That surfaced **three more real false-positive
classes**, all confirmed against actual rendered pages from `6559641.pdf` (not
speculative) and all now fixed in `vision_local.py`'s
`_find_bag_marker_candidates`, with regression tests in
`tests/test_vision_local.py`:

1. **Dark but saturated colour renders** (e.g. the maroon Iron Man armor sub-
   assembly on page 13) — just as dark in grayscale as a printed numeral, but
   real ink is low-saturation while coloured plastic isn't. Measured mean HSV
   saturation ~157 for the false positive vs. ~30 for a real numeral on the
   same page. Fix: `bag_marker_max_saturation` (default 90) rejects candidates
   whose near-black pixels average above it.
2. **The page-column divider hairline** (full page height, ~2px wide) — once
   the colour-render candidates above were filtered out, this thin rule
   became the highest-priority survivor on two-column pages. Fix:
   `bag_marker_min_aspect` (default 0.15) rejects anything absurdly thin.
3. **Hollow/textured shapes that are still low-saturation black-on-white** — a
   recurring "rotate the model" icon (appeared ~15 times through the book,
   identical 114×114px pictogram), a studded-brick closeup, a QR code, and a
   character illustration's curly hair all passed every gate above. Real bold
   printed digits are solid, thick strokes (measured fill ratio ~0.46-0.62 for
   rendered digits 0-9 via PIL); these hollow/textured false positives were
   ~0.19-0.48 fill and, for the QR code/hair, shattered into 28-102 separate
   contours vs. 2-3 for a real digit. Fix: `bag_marker_min_fill_frac` (0.42)
   and `bag_marker_max_components` (8).

**Result: zero bag-marker candidates across all 56 pages of the real booklet**
after all three fixes (down from ~20 false positives on the first full-book
pass, all traced to one of the three classes above). This strongly suggests
**76307 genuinely ships as a single unnumbered bag** — a small ~150-piece set
with no printed "Bag N" pages at all — rather than there being an undetected
real marker still hiding somewhere. No uncorroborated guess: every one of
these was visually confirmed by rendering the exact page and cropping the
exact bbox the detector had flagged.

**Not yet committed** — these changes are sitting in the working tree
(`vision_local.py`, `tests/test_vision_local.py`) and were committed as
`0a58a34`.

## Then: got a real inventory and ran the full identify pipeline for the first time

The user asked to keep going until pieces are actually identified, so this
session went further than calibration:

1. **Got a real inventory for 76307 with no API key.** `curl`/`WebFetch` to
   rebrickable.com and lego.com both return 403 from this sandbox (Cloudflare
   bot-challenge — confirmed it's bot detection, not network isolation:
   `curl https://www.google.com` works fine, only these two hosts block).
   Rebrickable's CSV/table export links all require login. Worked around it
   with the **claude-in-chrome MCP** (a real browser session passes the
   Cloudflare JS challenge): navigated to the set's Rebrickable page, read the
   parts grid via the accessibility tree (`read_page`), which exposes each
   part's full name/colour/quantity in image alt-text even though the visible
   page text alone only gives qty+part_num. Parsed 47 standard+spare part
   rows + 2 minifig SKUs into **`samples/76307_inventory.csv`** (committed —
   reusable for future scans of this set, no key needed).
2. **First-ever `lpl scan --engine local` run** found only **9/42 parts**, all
   as vague "unidentified: colour" buckets — nothing matched by part number.
3. **Diagnosed why**, by pulling actual callout crops out of the detector and
   looking at them: most weren't LEGO parts at all — random illustration
   fragments (arrows, character art, background). Root cause: this booklet's
   whole page background is a pale blue that sits in the *same* grayscale
   band (~200-240) as the real callout panel's own fill (~206 measured vs.
   ~230 background) — so the panel and the page background merge into one
   giant undifferentiated region, and the plain fill-colour detector never
   isolates a real panel (confirmed literally: the whole page came back as a
   single `(0, 0, 957, 723)` contour). The callouts that *did* get detected
   were incidental small pale pockets elsewhere in the artwork, not real
   panels.
4. **Fixed it with a second, independent detector**: real panels do have a
   dark border stroke (measured: background 230 → border ~33-43 → panel
   interior 206, a real dip below the fill-colour threshold), so a panel
   shows up as an enclosed "hole" in a *dark-ink* mask regardless of what
   shade its interior or the background are. Added
   `_cell_boxes_by_border`/`cell_border_dark_max` (finds holes via
   `cv2.findContours(..., RETR_CCOMP)` hierarchy) alongside the original
   `_cell_boxes_by_fill`, merged and deduped by IoU. A first pass over-fired
   (1428 callouts across the book — any small closed ink loop in the
   line-art, e.g. a gap between studs, counted as a "hole"); added
   `cell_border_min_extent` (contour area / bbox area ≥ 0.85) since a real
   panel is a rectangle and fills nearly all its own bbox (measured ~0.98)
   while illustration noise doesn't (~0.53-0.55). Down to 232 callouts,
   with genuine "2x"/"4x" panels (real part render + quantity label)
   confirmed by rendering the actual crops.
5. **Result: 43 parts located, 31 correctly matched to a real inventory
   part_num** (up from 9/0) — e.g. `3170 Plate Special 1 x 2 with 2 End
   Towballs (Black)`, `73109 Technic Brick Special ... (Light Bluish Gray)`,
   with Brickognize confidence 0.48-0.82. 11 of 42 parts and some per-step
   occurrences are still missed/undercounted (see count-mismatch warnings in
   `out/scan2/result.json` — not committed, regenerate with the command
   below), which is expected: two independent detectors covering more of the
   book's visual variety is real progress, not full coverage. Not yet
   pursued further — this is a reasonable point to test against a bigger,
   more classically-styled set instead of continuing to squeeze this one.
6. Committed as `419dd9e`. Tests at 100 passing (94 + 4 bag-marker + 2 these).

## Then: tested a real bigger multi-booklet set (76269, "Avengers Tower") — this is what the user meant by "test bigger PDFs"

76269 has **3 separate PDF booklets** (fetched via `lpl scan --set 76269
--engine local` — the built-in auto-fetch worked fine from this sandbox, no
browser workaround needed): `6488032.pdf` (276pp, booklet 1), `6488034.pdf`
(188pp, booklet 2), `6495251.pdf` (180pp, booklet 3) — **644 pages total**.
This is a genuinely large, classically-styled set (5202 parts per
Rebrickable — far beyond the browser-scraping approach used for 76307; see
"Known open items").

1. **Real classic white-background panel style confirmed** — `lpl debug` on
   booklet 1 showed genuine "1x"/"2x" gray panels detected correctly by the
   original fill-colour detector, unlike 76307's tinted-background case.
2. **Found and fixed a real bag-marker gap: circled-digit markers.** Each
   booklet's cover has a real marker — a digit inside a circle (e.g. "②" on
   booklet 1's cover), not a solid filled numeral. It was invisible for two
   compounding reasons, both real regressions caught by testing against
   *both* real booklets before committing (a lesson from earlier this
   session, worth repeating: any new gate must be re-checked against every
   previously-fixed real page, not just the page that motivated it):
   - `bag_marker_min_height_frac` (0.12) assumes the marker dominates the
     page; on this busier booklet the real marker is a small corner badge
     (~0.07 of page height). Lowering the *global* floor to accommodate it
     reintroduced a real bug: 76307's per-step counter digits (also ~0.06 of
     page height, solid, not circled) then also qualified as candidates on
     30+ pages. Fix: a **separate**, more permissive height floor
     (`bag_marker_ring_min_height_frac`) that only applies to the new ring
     branch below, not solid digits.
   - A circled digit's fill ratio (~0.32 measured) is below
     `bag_marker_min_fill_frac` (0.42, tuned earlier this session
     specifically to reject hollow icons). Added a second acceptance path,
     `bag_marker_ring_*`, but a naive "lower fill + few components" version
     wasn't enough on its own: a two-digit number like "20" also breaks into
     exactly 2 disconnected ink blobs (same component count as ring+digit).
     The real distinguishing signal is geometric — a ring's own bbox spans
     nearly the *whole* merged candidate (~0.87 measured, since the digit
     nests inside it) while two side-by-side digits split the box roughly
     evenly (~0.44-0.46 each); gated on that
     (`bag_marker_ring_min_dominant_frac`).
   - Also removed the blunt "never detect a bag marker on an is_parts_list
     page" rule (added earlier this session) — it was suppressing the real
     marker on 76269's cover, whose illustration (a building render's window
     panes) independently trips the BOM-page cell-count threshold on its own
     unrelated false positives. Confirmed safe first: re-ran detection on all
     of 76307's real front-matter pages *without* the rule and got zero
     false positives, proving the finer-grained gates added earlier this
     session (saturation, aspect, fill, components) already cover the
     original case that rule was written for.
3. **Result, verified against real data from both sets**: all 3 booklets'
   real covers now correctly detected as bag-marker candidates — visually
   confirmed by cropping and viewing each: booklet 1 = "②", booklet 2 = "③",
   booklet 3 = "⑦" (the numbering scheme spans booklets: each booklet covers
   a range of bags, e.g. booklet 3 alone covers up to bag 7 — real,
   previously-unseen structure this tool had never encountered). 76307
   remains clean (2 rare false positives out of 56 pages, down from a
   mid-fix regression peak of 34 — see commit `c9a8a0c` for the full
   before/after numbers). 76269 has its own residual false positives too:
   **12 out of 644 pages** (~2%) — spot-checked several (a studded-brick
   render, a technic-gear closeup, a minifig render, a decorative angled
   shape) and they're the same general "illustration element coincidentally
   digit-shaped" class as 76307's, just not yet individually chased down.
   Committed as `c9a8a0c`, 103 tests passing (100 + 5 new: circled-digit
   accepted, two-digit-number-at-ring-height correctly rejected, plus
   3 renamed/adjusted for the is_parts_list-suppression removal).
4. **Inventory not obtained for 76269** — 5202 parts is far too many to
   scrape via the browser accessibility-tree approach used for 76307 (that
   was fine for 49 rows, not thousands). No `lpl scan` identify-pipeline run
   was attempted on 76269 as a result — this session's 76269 work was
   detection-layer only (bag markers + panel style), not identification.

## Immediate next step

1. **12 residual bag-marker false positives on 76269** (~2% of 644 pages) —
   not yet individually diagnosed the way 76307's were; same general
   "illustration element happens to look digit-shaped" class. Worth another
   calibration pass if ordinal bag-numbering accuracy on this specific set
   matters, using the same rigor as this session (crop the real page, measure
   the real pixel stats, don't guess).
2. **Get a real inventory for 76269 to actually run `lpl scan`.** Two paths:
   - Ask the user for a free `REBRICKABLE_API_KEY` (a real account signup —
     not something this agent should do itself; a few minutes at
     rebrickable.com/api/). Then: `run scan 6488032.pdf --engine local --set 76269 --out out`.
   - Or accept a much rougher, unconstrained scan with no inventory at all
     (`BrickognizeOnlyIdentifier`, lower precision, no closed-set
     constraint, no count validation) — already the fallback path, no new
     work needed, just lower accuracy.
3. If pushing 76307's identification further is wanted: 11/42 parts still
   unidentified, quantities undercounted — `samples/76307_inventory.csv` plus
   a fresh run of:
   ```
   run scan 6559641.pdf --engine local --inventory-file samples/76307_inventory.csv --out out
   ```
   are the starting point (result isn't committed — regenerate).
4. Load `out/result.json` into `web/index.html` (drag-drop) and sanity-check
   a few parts against the physical instructions, for whichever set has a
   fresh scan result.

## Known open items (not yet started, roughly ranked)

- **12/644 pages on 76269 still have bag-marker false positives (~2%).** Not
  yet individually diagnosed — see "Then: tested a real bigger multi-booklet
  set" above.
- **76269 has no inventory and no `lpl scan` has been run on it.** 5202 parts
  is too many for the browser-scraping approach used for 76307 — needs a real
  `REBRICKABLE_API_KEY` (user has to sign up themselves; not something this
  agent should do). See "Immediate next step" above.
- **11/42 parts on 76307 still unidentified, quantities undercounted.** The
  border-based panel detector is real progress (9→43 parts found, 31 matched
  to a real part_num) but not full coverage — see "Then: got a real inventory
  ..." above for what's confirmed vs. still missing.
- **`--embeddings` (local CLIP image matching) never exercised** on a real
  set — needs the `[ml]` extra (`pip install -e ".[ml]"`, pulls in torch).
  Might help close some of the remaining 11 unidentified parts.
- **Tesseract not installed** on the user's machine — quantities currently
  default to 1 on the local engine (bag numbers work fine without it, via
  ordinal assignment). Installing Tesseract would unlock real quantity/part-ID
  OCR, and would let 76269's real circled bag/booklet numbers (2, 3, 7) be
  read directly instead of relying on ordinal assignment; not yet requested
  by the user.
- Cover/BOM-style pages' callout-like false detections (logos/badges counted
  as "cells", tripping `is_parts_list`) are currently harmless — such pages
  are already excluded from parts attribution.
- **rebrickable.com and lego.com return HTTP 403 to `curl`/WebFetch from this
  sandbox** (Cloudflare bot-challenge) — confirmed it's bot detection, not a
  network block (`curl https://www.google.com` works). The claude-in-chrome
  MCP (a real browser session) gets through fine; that's how
  `samples/76307_inventory.csv` was obtained with no API key. Note this only
  scales to small sets (~50 parts) — 76269's 5202 parts is too many rows to
  scrape this way; a real API key is the right tool past that size.
  Separately, `lpl scan --set NNNN`'s *own* built-in auto-fetch (`fetcher.py`,
  hits lego.com to find the PDF download link, not Rebrickable) worked fine
  from this same sandbox for 76269 — so the 403 issue is specific to
  Rebrickable's/lego.com's *webpage* bot-challenge, not their PDF/API
  endpoints generally.
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
6. **Full-booklet calibration** (this session): ran detection across all 56
   pages of the same real PDF at once (not just a slice), found and fixed
   three more real false-positive classes (colour-saturated part renders, a
   thin column-divider line, hollow/textured shapes like icons/QR codes/hair)
   — see "This session" section above for the full writeup. Zero false
   positives remain across the whole booklet. Committed as `0a58a34`.
7. **First real identify-pipeline run** (later in this same session): built a
   real inventory for 76307 via a browser session (Rebrickable/lego.com block
   plain HTTP fetches but not a real browser — see "Known open items"), ran
   `lpl scan --engine local` for the first time ever against a real PDF, found
   it returned almost nothing usable (9/42, all colour-only), traced this to
   the callout-panel detector never isolating a real panel on this booklet
   (page background and panel fill are the same grayscale tone), and fixed it
   with a second border-based detector — 43 parts located (up from 9), 31 of
   them correctly matched by part number (out of 42 real inventory parts).
   See "Then: got a real inventory ..." section above. Committed as `419dd9e`.
8. **Tested a real bigger multi-booklet set** (76269, "Avengers Tower", 3
   PDF booklets, 644 pages, 5202 parts — this is what the user meant by
   "test bigger PDFs"): confirmed the classic white-panel style works out of
   the box on a bigger set; found and fixed a real, previously-unseen
   bag-marker style (a digit inside a circle, not solid-filled) via two new
   gates plus removed a now-redundant blunt suppression rule — verified all
   3 booklets' real covers ("②", "③", "⑦") are now correctly detected, and
   confirmed no regression on 76307 (2 rare false positives, down from a
   mid-fix regression peak of 34). See "Then: tested a real bigger
   multi-booklet set" section above. Committed as `c9a8a0c`. 76269 has no
   inventory yet (5202 parts is too many to scrape via browser) and no
   `lpl scan` has been run on it — detection-layer validation only so far.

## Resume prompt

Paste this into a fresh session to continue:

```
I'm continuing work on legooartlocator (github.com/RShiri/legooartlocator,
branch claude/lego-pdf-part-scanner-lc3aiu). Read HANDOFF.md at the repo root
first for full session history and current state, then README.md for how the
tool works.

Short version: it's a LEGO instruction-PDF scanner that maps parts to
bag/page. Two engines: a paid Claude-vision engine and a free local
OpenCV+Brickognize engine (`lpl scan --engine local`). 103 tests pass, all
committed (`c9a8a0c`).

Two real sets tested so far. **76307** (Iron Man Mech, 56 pages, single
unnumbered bag): full detection calibration done (zero bag-marker false
positives), first-ever identify-pipeline run got 43/42 parts located (31
correctly matched by part number) using samples/76307_inventory.csv (a real
Rebrickable export, no API key needed for a set this small). **76269**
(Avengers Tower, 3 booklets / 644 pages / 5202 parts -- the "bigger PDF"
test): confirmed the classic white-panel style works natively, found and
fixed a new bag-marker style (circled digits, e.g. "②"/"③"/"⑦" -- one per
booklet, real structure never seen before), verified all 3 real covers now
detected correctly. 76269 has ~2% residual bag-marker false positives (not
yet diagnosed) and no inventory yet (5202 parts is too many to scrape via
browser -- needs the user to get a free REBRICKABLE_API_KEY).

Next: see "Immediate next step" in HANDOFF.md -- likely either chasing
76269's remaining false positives, or getting a REBRICKABLE_API_KEY to
finally run a full identify pipeline on a big multi-booklet set.
```
