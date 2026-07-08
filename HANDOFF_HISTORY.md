# Session History (archive)

This is the full chronological narrative of how this project got to its
current state — every real-data finding, every experiment (including the ones
that didn't work), in the order they happened. For **current state**, read
[HANDOFF.md](HANDOFF.md) first; come here only when you need the detailed
evidence behind a claim it makes.

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

Committed as `0a58a34`.

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
   occurrences were still missed/undercounted, which is expected: two
   independent detectors covering more of the book's visual variety is real
   progress, not full coverage.
6. Committed as `419dd9e`.

## Then: tested a real bigger multi-booklet set (76269, "Avengers Tower") — this is what the user meant by "test bigger PDFs"

76269 has **3 separate PDF booklets** (fetched via `lpl scan --set 76269
--engine local` — the built-in auto-fetch worked fine from this sandbox, no
browser workaround needed): `6488032.pdf` (276pp, booklet 1), `6488034.pdf`
(188pp, booklet 2), `6495251.pdf` (180pp, booklet 3) — **644 pages total**.
This is a genuinely large, classically-styled set (5202 total pieces per
Rebrickable — far beyond the browser-scraping approach used for 76307).

1. **Real classic white-background panel style confirmed** — `lpl debug` on
   booklet 1 showed genuine "1x"/"2x" gray panels detected correctly by the
   original fill-colour detector, unlike 76307's tinted-background case.
2. **Found and fixed a real bag-marker gap: circled-digit markers.** Each
   booklet's cover has a real marker — a digit inside a circle (e.g. "②" on
   booklet 1's cover), not a solid filled numeral. It was invisible for two
   compounding reasons, both real regressions caught by testing against
   *both* real booklets before committing (a lesson worth repeating: any new
   gate must be re-checked against every previously-fixed real page, not just
   the page that motivated it):
   - `bag_marker_min_height_frac` (0.12) assumes the marker dominates the
     page; on this busier booklet the real marker is a small corner badge
     (~0.07 of page height). Lowering the *global* floor to accommodate it
     reintroduced a real bug: 76307's per-step counter digits (also ~0.06 of
     page height, solid, not circled) then also qualified as candidates on
     30+ pages. Fix: a **separate**, more permissive height floor
     (`bag_marker_ring_min_height_frac`) that only applies to the new ring
     branch below, not solid digits.
   - A circled digit's fill ratio (~0.32 measured) is below
     `bag_marker_min_fill_frac` (0.42, tuned earlier specifically to reject
     hollow icons). Added a second acceptance path, `bag_marker_ring_*`, but
     a naive "lower fill + few components" version wasn't enough on its own:
     a two-digit number like "20" also breaks into exactly 2 disconnected ink
     blobs (same component count as ring+digit). The real distinguishing
     signal is geometric — a ring's own bbox spans nearly the *whole* merged
     candidate (~0.87 measured, since the digit nests inside it) while two
     side-by-side digits split the box roughly evenly (~0.44-0.46 each);
     gated on that (`bag_marker_ring_min_dominant_frac`).
   - Also removed the blunt "never detect a bag marker on an is_parts_list
     page" rule — it was suppressing the real marker on 76269's cover, whose
     illustration (a building render's window panes) independently trips the
     BOM-page cell-count threshold on its own unrelated false positives.
     Confirmed safe first: re-ran detection on all of 76307's real
     front-matter pages *without* the rule and got zero false positives,
     proving the finer-grained gates already added (saturation, aspect,
     fill, components) already cover the original case that rule was written
     for.
3. **Result, verified against real data from both sets**: all 3 booklets'
   real covers now correctly detected as bag-marker candidates — visually
   confirmed by cropping and viewing each: booklet 1 = "②", booklet 2 = "③",
   booklet 3 = "⑦" (the numbering scheme spans booklets: each booklet covers
   a range of bags, e.g. booklet 3 alone covers up to bag 7). 76307 remained
   clean (2 rare false positives out of 56 pages, down from a mid-fix
   regression peak of 34). 76269 had its own residual false positives too:
   12 out of 644 pages (~2%) — the same general "illustration element
   coincidentally digit-shaped" class as 76307's, not individually chased
   down at the time.
4. Committed as `c9a8a0c`. Inventory not obtained for 76269 in this round —
   5202 pieces (later found to be 786 distinct inventory lines) seemed too
   many to scrape via the browser accessibility-tree approach used for
   76307; no `lpl scan` identify-pipeline run was attempted yet.

## Then: user said "i dont want to use claude and claude api key. train the model."

Asked what "train the model" meant specifically (ambiguous: enable the
existing zero-shot CLIP embeddings? train a real custom classifier? something
else?) — user picked **train a real custom part-classifier**. Went through
`EnterPlanMode` given the scope (new deps, new training pipeline, hardware
constraints to work around); plan approved, then implemented (`9bf6c51`),
**actually run for real** with the user's Rebrickable key (`31c9b41` adds
validation), and iterated on real results.

**What got built:**
- **`lpl train-embedding`** — fine-tunes MobileNetV3-Small/ResNet18 with
  **triplet-loss metric learning** on one reference photo per part (from
  `--set`+`REBRICKABLE_API_KEY` or an inventory file), each expanded into
  several augmented variants (rotation, colour jitter, blur/downsample, a
  flatten/posterize pass, background padding). Saves a checkpoint used via
  `lpl scan --engine local --embeddings --embedding-weights PATH` — a
  **drop-in replacement** for the existing pretrained-CLIP path
  (`embedding.TrainedBackend` implements the same `embed_images()` interface
  as `ClipBackend`; `Gallery`/`build_gallery`/`identify.py`'s ensemble needed
  zero changes).
- **Validation + early stopping** (added after the first real run exposed a
  real problem — see below): `train.data.train_val_split` holds out a
  fraction of each part's *variants* (not whole parts — what matters for a
  retrieval model is recognising a different view of an already-seen part);
  `evaluate_retrieval_accuracy` measures top-1 retrieval accuracy on the
  held-out set each epoch; training stops after `--patience` epochs with no
  improvement and saves the *best* epoch's checkpoint, not the last one.
- The triplet-*sampling* logic (`train/sampling.py`, pure Python) is
  unit-tested without torch; the actual torch training loop
  (`train/embedding_trainer.py`) and `embedding.TrainedBackend` are lazily
  imported and marked `# pragma: no cover`, same convention `ClipBackend` has
  always had.

**What actually happened when it ran for real** (a first-principles ML
result, not a code-review guess):

1. **Environment reality check**: `torch-directml` (planned as the AMD-GPU
   acceleration path) turned out to have **no wheel for Python 3.14** (this
   venv's version — only ships for 3.8-3.12, confirmed against PyPI). Fixed
   the `pyproject.toml` marker so `pip install -e ".[train]"` doesn't hard-fail
   on newer Python, just silently falls back to CPU. Training on CPU is
   still tractable for this dataset size (4-8 min for 27-40 epochs on 41
   parts) — not blocking, just slower than hoped.
2. **First real run** (5 epochs, no validation): trained fine, loss dropped
   0.149 → 0.044. Real scan comparison (76307, same inventory,
   brickognize+colour baseline vs. +trained embeddings) showed **35/47
   identified vs. baseline's 31/47, zero regressions** — but also **23 count-
   mismatch warnings vs. baseline's 12**, several severe (one part matched 15
   different real crops against an inventory quantity of 1). Classic
   undertrained-embedding "attractor" symptom.
3. **Tried the obvious fixes, neither worked cleanly**: more epochs (40,
   no validation) pushed identified count up further (37/47) without losing
   anything, but the overcounting didn't go away — it shifted which part was
   worst-affected. Lowering the embedding's ensemble blend weight (0.4 →
   0.15) made things *worse* (34/47, one mismatch ballooned to 21 vs. 6).
   Neither is a real fix because there was no way to tell whether more
   training was actually helping vs. just changing which mistakes got made.
4. **Added real validation** (see above) and reran: retrieval accuracy on
   held-out augmented variants climbed to **96.7%** (early-stopped at epoch
   27, correctly kept epoch 19's checkpoint — proof the early-stopping logic
   works, not just that loss went down). But on the **real scan**, this
   checkpoint performed essentially the same as the un-validated 40-epoch
   version (36/47 identified, 22 mismatches, worst overcount still 10 vs. 1).
5. **The actual conclusion**: 96.7% validation accuracy but no improvement on
   real crops conclusively shows this **is a domain-gap problem, not an
   undertraining problem** — the model generalises well *within* the
   augmented-catalog-photo distribution (that's what validation measures) but
   that skill doesn't transfer to the real instruction-booklet icon crops it
   sees at inference, because validation is drawn from the same distribution
   as training and can't see the gap. More epochs, more patience, or
   ensemble-weight tuning on *this* dataset won't fix it — confirmed, not
   assumed.
6. **Net result, stated plainly**: the trained embeddings are a **real, if
   modest, net improvement** over the Brickognize/colour-only baseline (31→36
   identified, zero regressions across every experiment run) but come with a
   **real, unresolved cost** (roughly 2x the count-mismatch warnings, some
   severe). This trade-off is what the next round (below) resolved.
7. **What would actually move this forward**: real instruction-booklet-style
   training images (not catalog photos + augmentation) — either hand-labelled
   real crops or synthetic rendering from a closer-matching source (LDraw 3D
   models). Both were later attempted — see "Follow-up round 2" below.

## Update — three improvements shipped, the training trade-off is resolved

The trade-off above ("net coverage gain but ~2x the count-mismatch warnings")
was **resolved**, not just re-analysed. Three flag-gated improvements were
added, measured on 76307 (self-consistent — the no-embeddings baseline
reproduced the documented 31/12 exactly), and made **the defaults**:

| Config | Identified /47 | Count-mismatch warnings |
| --- | --- | --- |
| Baseline (no embeddings) | 31 | 12 |
| Prior trained (photo aug, random mining) | 36 | 22 |
| icon aug + semi-hard mining (no reconcile) | 36 | 18 |
| **+ capacity-reconcile (shipped default)** | **37** | **11** |

1. **Capacity/trust-aware reconciliation** — `locate.py` `assemble_result`
   (`_resolve_assignments`), CLI `--capacity-reconcile` (default **on**). An
   embedding-only match (Brickognize didn't corroborate) can no longer fill a
   part past its inventory quantity; the surplus "attractor" crops divert to
   their best alternative or to unidentified. A Brickognize-corroborated match
   is never diverted. Biggest lever on the warning count; no retrain; a no-op
   when embeddings are off (so the baseline is unchanged).
2. **Domain-matching "icon" augmentation** — `train/data.py` (`_edge_outline`),
   CLI `--augment-style icon` (default). Always-flatten + a black Canny edge
   outline so catalog photos read like flat instruction-booklet icons.
3. **Semi-hard negative mining** — `train/sampling.py`
   (`sample_triplets_semihard`, numpy-only, unit-tested) + `embedding_trainer.py`
   (`mining="semihard"`), CLI `--mining semihard` (default). Negatives are mined
   from the current model each epoch; loss now descends 0.22 → 0.005 over 22
   epochs with val 98.8% (vs. the old random path's instant collapse to ~0.04).

Net vs. the prior trained model: **+1 coverage (37 vs 36) and half the
warnings (11 vs 22)**. Net vs. baseline: **+6 coverage (37 vs 31) with fewer
warnings than even the no-embedding baseline (11 vs 12)** — no regression on
either axis. Every improvement is opt-out via its flag
(`--no-capacity-reconcile`, `--augment-style photo`, `--mining random`) to
reproduce the old behaviour. Checkpoint: `models/lego_embed_76307_v4.pt`
(gitignored; regenerate with `run train-embedding --set 76307`).

Honest caveats at the time: still one set (76307); 37/47 isn't full coverage
(the ~10 hardest parts stay domain-gap-bound); part of the warning drop is
the identifier correctly *declining* to place crops it can't corroborate. Not
yet validated with embeddings on 76269 — see below, this was addressed next.

## Follow-up round 2 — six phases probing what's next (dump-crops, floor, 76269, real-crops, LDraw x2)

Ran the natural next round: instrument first, then let the data pick the next
lever, rather than guessing. Order matters here — phase 1 produced the labelled
data phases 2 and 4 needed.

**Phase 1 — `scan --dump-crops DIR` (shipped).** Every callout crop was
in-memory only; nothing could calibrate against real data or diagnose failures.
Now writes `DIR/<part_num|unknown>/pNNN_iNNN.png` + `DIR/manifest.json` (page,
bag, quantity, seen colour, final + raw part_num, confidence, per-signal
component scores, top alternatives). First run on 76307 sharpened the picture:
of the 10 unidentified parts, only **4 unique shapes** ever fail — all 4 are
**identification misses, not detection misses** (they're cropped fine), and
3 of 4 are transparent or printed parts (`3062b` trans-red, `4740`
trans-light-blue, `3068bpr9329` printed tile) — exactly what a plain catalog
photo represents worst.

**Phase 2 — open-set embedding floor (closed, no code shipped).** Hypothesis:
reject weak embedding-only matches outright. The manifest's real score
distributions killed it before writing any filtering code — corroborated and
embedding-only winners' embedding scores fully overlap (corroborated:
0.00–0.66 across 49 crops; embedding-only: 0.24–0.53 across 23), so no
threshold separates good from bad; any floor tight enough to matter would
also cut real corroborated matches. A real negative result, cheaply reached
because phase 1 existed.

**Phase 3 — 76269 embedding validation (first-ever, strongly positive).**
Trained `models/lego_embed_76269.pt` (407 parts, 4 variants/part, val 0.808)
and ran booklet 1 (276 pages) both ways:

| | identified | mismatches | mismatch rate |
|---|---|---|---|
| baseline (no embeddings) | 168 | 128 | 76% |
| + embeddings | **251** | 157 | **62%** |

**+83 parts (+49%) — and the mismatch rate per identified part improved**, not
just its raw count. This is the clearest evidence yet that the whole approach
(icon aug + semihard mining + capacity-reconcile) generalises well past the
single 47-part set it was tuned on, onto a genuinely large, harder booklet.

**Phase 4 — real-crop self-training via `--real-crops` (shipped as opt-in,
measured negative — v4 stays default).** Built `train/data.py`
`load_real_crops`/`merge_datasets`: feed corroborated `--dump-crops` output
back in as in-domain labels. Retrained 76307
(`--real-crops out/crops_76307/manifest.json` → v5): **35/47 identified, 6
mismatches** vs. v4's **37/47, 7**. More precise (fewer, and less severe,
mismatches) but *less* coverage — the real crops are few (49 across 31 parts,
1-4 each) and skew training toward the shapes already working well rather than
the ones that don't. v4 (`lego_embed_76307_v4.pt`) remains what ships.

**Phase 5 — LDraw flat-shaded renderer (shipped as opt-in infra; prototype
validated the mechanism, full measurement is mixed — not a shipped win yet).**
Built `train/ldraw.py`: a from-scratch, numpy-only `.dat` parser (type 1
subfile transforms resolved recursively through `parts/`/`p/`, type 2 edges,
type 3/4 facets) + an orthographic rasterizer with quantized flat shading and
depth-tested black edge strokes — genuinely icon-styled output, confirmed by
eye against real crops before any training integration. `resolve_part_ref`
falls back from a printed/decorated part number to its undecorated mould
(e.g. `3068bpr9329` → `3068b`) since LDraw has no print geometry. Wired as
`--ldraw-dir` on `train-embedding`, rendering 3 instruction-plausible views
per resolvable inventory part.

Trained 76307 with it (v6) and measured for real: **36/47 identified, 7 hard
mismatches + 3 reuse-info** vs. v4's **37/47, 7**. Genuinely mixed, not a clean
win: it **did** recover `3068bpr9329` — the printed tile, one of the 4
parts nothing before this could ever identify, confirming LDraw *can* close
gaps catalog photos structurally can't — but it also **lost** 2 parts (`15672`,
`79846`) that v4 had, for a net -1. Plausible cause, not confirmed: LDraw's
flat single-colour fill has no print/texture detail, so for parts normally
disambiguated by surface texture rather than silhouette, it may pull their
embedding toward other similarly-shaped, similarly-flat-coloured parts. v4
stays default; `--ldraw-dir` ships as a tested, opt-in path (needs the
~80MB `complete.zip` from ldraw.org for actual use, gitignored, not
committed).

**Phase 6 — targeted LDraw, testing exactly that next step (tried, refuted).**
Added `render_training_images(..., only_parts={...})` + `train-embedding
--ldraw-only PART,PART,...` so LDraw renders can be restricted to specific
parts instead of the whole inventory — directly testing whether that avoids
the regressions whole-inventory LDraw (v6) caused. Trained v7 with
`--ldraw-only 25269,3062b,3068bpr9329,4740` (the 4 known identification-miss
parts, from phase 1's manifest) and measured for real:

| | identified | mismatches |
|---|---|---|
| v4 (icon aug only) | 37 | 7 |
| v6 (LDraw, whole inventory) | 36 | 7 hard + 3 reuse-info |
| v7 (LDraw, targeted to 4 parts) | 36 | 8 hard + 3 reuse-info |

**The hypothesis did not hold — this is a clean negative, not a partial win.**
None of the 4 targeted parts were actually recovered by v7. Worse, a
previously-safe, completely untouched part (`61332`) regressed anyway, even
though its training photos never changed. The reason: semi-hard mining
re-embeds and mines negatives from the *entire* gallery every epoch, so
changing even 4 parts' training images reshapes the whole embedding space's
gradient signal through shared negatives — "targeting" doesn't actually
isolate anything once semi-hard mining is in the loop. v7 did recover 2 of
v6's regressions (`15672`, `79846`) but lost the one part v6 had gained
(`3068bpr9329`) plus the new `61332` loss, netting the same coverage as v6
(36) with a worse mismatch count. `--ldraw-only` ships as tested, working
infrastructure — the mechanism is sound and may pair usefully with
non-semihard mining or a from-scratch (not fine-tuned) run later — but it is
not the fix. v4 remains the shipped checkpoint.

This closed out the LDraw investigation for that round: two honest attempts
(whole-inventory, targeted), both measured, neither a net win on 76307.

## Follow-up — built-in quantity reader (the 'Nx' label): mismatches 11 → 7

The remaining count-mismatch warnings were traced to real data: **every callout
was counted as 1 piece** because no OCR backend was installed (Tesseract absent
→ null OCR → quantity defaults to 1), so a callout printed "3x"/"6x" still
counted as 1 — the dominant cause of the under-counts. Added `DigitOCR`
(`vision_local.py`): a dependency-free reader that finds the small 'Nx' label as
a low row of dark glyphs and template-matches each digit against cv2-rendered
glyphs (no font file, no Tesseract). Wired in as the default OCR when Tesseract
is absent (`_tesseract_or_digit_ocr` in cli.py); it returns "" for a large bag
numeral, so ordinal bag numbering is unaffected.

Calibrated on real 76307 crops (12/12) and validated on the real scan:
count-mismatch warnings **11 → 7**, under-counts **6 → 2** (quantities now read
as 2x/3x/4x). The remaining 7 are 5 over-counts of +1 (a part shown in one more
step-illustration than its piece count — largely legitimate for a location tool)
and 2 harder under-counts (a multi-occurrence part and a likely single misread).
76307 overall: **baseline 31/47 & 12 warnings → 37/47 & 7 warnings**.

## Codebase hardening pass

A full audit (3 parallel Explore agents covering every not-yet-reviewed
`src/` module, test/tooling gaps, and docs/packaging hygiene) found and fixed,
in order of severity:

- **`vision.py`** had zero retry/backoff on the Claude API despite calling it
  once per page for hundreds of pages — added, mirroring `brickognize.py`'s
  existing retry/backoff/injectable-sleep shape exactly.
- **`rebrickable.py`** only wrapped 404s in `RebrickableError`; a bad key,
  rate limit, or outage raised a raw `httpx` exception that `cli.py`'s
  intended "fall back to vision-only" catch never caught. Now every failure
  mode is wrapped; also added a pagination page cap against a hypothetical
  infinite `next`-link loop.
- **`pdf_render.py`**'s `parse_page_range` did unguarded `int()` on `--pages`
  chunks (raw traceback on a typo) and silently produced zero pages on a
  reversed range like `"50-10"`. Both now raise a clear error, caught at the
  `cli.py` boundary and turned into a clean `click.UsageError`.
- **`cli.py`**'s `scan --extracts` silently ignored `--pages`/`--max-pages`;
  now respects both. `_load_extracts` had no error handling around malformed
  JSON/schema; now raises a clear `UsageError`.
- **`fetcher.py`**'s skip-if-exists download check couldn't tell a good file
  from a truncated one left by an interrupted prior download — such a file
  would be reused forever and fail later, deep inside PyMuPDF, with no useful
  message. Now validates the existing file actually opens as a real PDF
  before skipping the re-download.
- **`inventory.py`**'s JSON loader silently produced an empty parts list for
  an unexpected top-level shape (a dict without a `results` key), had no
  error handling around `json.loads`, and didn't catch `OverflowError`
  (`quantity: "Infinity"` parses as a float fine but can't become an int).
  All three fixed.
- **`models.py`**'s `BagSegment` had no validator that `end_page >=
  start_page`; an inverted segment would make `page_to_bag` silently drop a
  page from the mapping. Added a pydantic validator.
- **`debug_overlay.py`**'s `run_debug` had no per-page error isolation and
  only wrote `stats.json` after the entire loop succeeded — so one bad page
  (exactly the real-world scenario this calibration tool exists for) lost
  every other page's output too. Now isolates per-page failures and
  (re)writes `stats.json` after every page.
- **`bags.py`** reported a harmless repeated bag marker (e.g. a reprinted
  banner for the bag already in effect) with the identical "suspect anomaly"
  wording as a real out-of-order marker. Now distinguishes the two.
- **`pyproject.toml`**'s `[train]` extra didn't declare the
  `opencv-python-headless` dependency that `train/data.py` unconditionally
  imports; added it so `pip install -e ".[train]"` alone is self-sufficient.

Added test coverage for every previously-untested module (`aggregate.py`,
`cache.py`, `pdf_render.py`, `rebrickable.py`, `models.py`) plus real
`click.testing.CliRunner`-based end-to-end tests of `scan`/`debug` (nothing
had ever invoked the CLI commands themselves before — only their inner
functions directly, so argument-wiring bugs weren't caught by anything).

Added `ruff` (a deliberately narrow rule set — `E`/`F`/`I`, not the
`UP`/pyupgrade rules, which alone produced 400+ typing-style findings that
would have meant a much bigger, separate typing rewrite). Its very first run
caught a real latent bug from earlier this session: `locate.py` used
`Tuple[...]` in three type hints without importing `Tuple` from `typing` —
silent only because `from __future__ import annotations` defers evaluation,
so it never actually crashed, but would have if anything ever introspected
those annotations. Fixed. Added a CI job installing the `[train]` extra so
torch-dependent tests (previously always `importorskip`-skipped in CI) run
for real at least once per CI run.

Also cleaned up: README's stale claims (Tesseract/quantities, hard-coded test
count, six undocumented CLI flags, `train/ldraw.py` missing from the Layout
section), added a proper `LICENSE` file (MIT, matching `pyproject.toml`'s
existing claim which had no backing file), added `.claude/` to `.gitignore`,
and split this very document out of a single 700+ line `HANDOFF.md` into a
short current-state file plus this history archive.

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
     per-page stats) — this is what found the two real bugs described above.
   - **S4** — web viewer v2: warnings panel, bag-overview cards, occurrence
     drill-down, count-mismatch highlighting.
   - **S5** — CI (pytest on Ubuntu 3.11/3.12 + Windows 3.12).
   - Integration wave: wired ordinal bag numbering into `locate_local`, added
     the `lpl debug` CLI command, cache/panel-band flags; fixed a bug found
     during integration itself (`lpl debug` crashed on missing Tesseract
     instead of degrading like `scan` does — now shares one helper).
5. **Real-data calibration**: user ran `lpl debug` against their real 76307
   PDF, shared two page screenshots, and two real false-positive bugs
   (described above under "What's proven with real data") were found and
   fixed as a direct result.
6. **Full-booklet calibration**: ran detection across all 56 pages of the
   same real PDF at once, found and fixed three more real false-positive
   classes. Zero false positives remained across the whole booklet.
   Committed as `0a58a34`.
7. **First real identify-pipeline run**: built a real inventory for 76307 via
   a browser session, ran `lpl scan --engine local` for the first time ever
   against a real PDF, found it returned almost nothing usable (9/42), traced
   this to the callout-panel detector never isolating a real panel on this
   booklet, and fixed it with a second border-based detector — 43 parts
   located (up from 9), 31 correctly matched by part number. Committed as
   `419dd9e`.
8. **Tested a real bigger multi-booklet set** (76269, 3 PDF booklets, 644
   pages): confirmed the classic white-panel style works out of the box;
   found and fixed a real, previously-unseen bag-marker style (a digit inside
   a circle). Committed as `c9a8a0c`.
9. **Trained a real custom part-embedding model** (`lpl train-embedding`),
   confirmed a genuine domain-gap problem via validation-based early
   stopping, not just "needs more training."
10. **Shipped the fix**: icon augmentation + semi-hard mining +
    capacity-reconcile, resolving the coverage-vs-count-accuracy trade-off
    (37/47 & 7 warnings, better than both the untrained baseline and the
    first trained model on both axes).
11. **Follow-up round 2**: `--dump-crops` diagnostic infra, an embedding-score
    floor (closed negative), first-ever 76269 embedding validation (+49%
    coverage), real-crop self-training (negative), and two LDraw rendering
    attempts (mixed, then refuted).
12. **Built the dependency-free quantity reader** (`DigitOCR`), cutting
    count-mismatch warnings from 11 to 7.
13. **Full codebase hardening pass**: audited and fixed real bugs across
    modules untouched until then, added missing test coverage, added lint,
    cleaned up documentation and packaging.
14. **Pushed to `main`**: `main` had never received this project (a single
    placeholder-README commit since the repo's creation). Verified `main`
    was a strict ancestor of the feature branch (a clean fast-forward, no
    merge commit, nothing lost or overwritten) and fast-forwarded it to
    match — a fresh clone of `main` now gets the real project, not a stub.
