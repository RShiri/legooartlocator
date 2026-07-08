# legooartlocator

Scan a LEGO set's building-instruction PDF and produce a **part → bag / page**
map: given a part, find which numbered bag it's in and on which pages it
appears. That per-bag mapping isn't available from any API — it lives only in
the printed instructions — so the PDF has to be scanned.

## How it works

A hybrid pipeline:

1. **Render** — each PDF page is rasterised to an image (PyMuPDF).
2. **Vision (two-pass)** — a cheap **triage** pass (fast model, low DPI)
   classifies every page: bag marker? BOM? any callouts to read? Then a
   **detailed** pass (capable model, high DPI) reads callouts only on the pages
   triage flagged — covers, story, "completed", and BOM pages skip the
   expensive pass entirely. Output is schema-forced (no text parsing) and cached
   per page-image hash, so re-runs are free. Use `--single-pass` to run the
   detailed model on every page instead.
3. **Segment** — sparse bag markers become a dense `page → bag` map (bag
   membership is a step function over page order).
4. **Reconcile** — the set's canonical inventory is pulled from Rebrickable and
   each callout is matched to it. Because the inventory is a closed, known set
   of parts, this constrains the fuzzy image match; per-bag quantities are then
   checked against inventory totals and mismatches flagged.
5. **Aggregate** — emit `result.json` + `result.csv` (part → bags/pages/qty,
   canonical name/image, confidence).
6. **View** — a static `web/index.html` loads `result.json` and lets you search
   by part name / colour / element id and browse by bag.

## Install

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"           # core + tests
pip install -e ".[local]"         # + local engine (OpenCV/numpy) — needed for --engine local
```

> After every `git pull`, re-run `pip install -e ".[local]"` — it's a no-op when
> nothing changed and picks up any newly added dependencies (otherwise you may
> hit `ModuleNotFoundError`).

## Usage

You can pass a local PDF, **or just a set number** — with `--set NNNN` and no PDF
path the tool auto-downloads the official instructions from lego.com (scrapes the
building-instructions page, grabs the CDN PDF link, downloads it). For sets with
several booklets, pick one with `--booklet N`; change region with `--locale en-us`.

```bash
# Auto-download the PDF for a set, then scan it
lpl scan --set 76307 --engine local          # zero-account: auto-fetch + Brickognize
lpl scan --set 76307                          # paid vision engine (needs ANTHROPIC_API_KEY)

# Full pipeline (needs ANTHROPIC_API_KEY; REBRICKABLE_API_KEY enables reconcile)
lpl scan path/to/instructions.pdf --set 76307 --out out

# Vision-only (no inventory reconciliation)
lpl scan instructions.pdf --no-rebrickable

# Cost control while developing
lpl scan instructions.pdf --pages 1-40 --max-pages 20 --dpi 120

# Two-pass is on by default; tune it or turn it off
lpl scan instructions.pdf --triage-dpi 110 --dpi 180   # default two-pass
lpl scan instructions.pdf --single-pass                # detailed model on every page

# Fully local engine — no paid API (OpenCV detect + Brickognize/embedding ID)
lpl scan --set 76307 --engine local                              # zero accounts (Brickognize only)
lpl scan instructions.pdf --engine local --set 76307             # inventory via Rebrickable key
lpl scan instructions.pdf --engine local --inventory-file inv.csv # inventory from a free export, no key
lpl scan instructions.pdf --engine local --inventory-file inv.csv --embeddings  # add local image matching
```

Then open `web/index.html` and drop in `out/result.json` (or serve the folder
and it auto-loads a sibling `result.json`).

**Windows shortcut:** a `run.bat` wrapper is included so you can skip the long
venv path — from the repo folder just type:

```bat
run scan 76307.pdf --set 76307 --out out
```

(equivalent to `.venv\Scripts\python.exe -m legopartlocator.cli scan ...`).

Keys are read from the environment or a `.env` file (see `.env.example`):
`ANTHROPIC_API_KEY`, and optionally `REBRICKABLE_API_KEY`
(free key at <https://rebrickable.com/api/>).

### Offline / blocked-network mode

If the vision pass can't run (no key, or the Anthropic/LEGO/Rebrickable hosts
are blocked by a network policy), you can feed pre-extracted page JSON directly:

```bash
lpl scan --extracts samples/demo_extracts.json --no-rebrickable --out out
```

`samples/demo_extracts.json` is a small worked example; `samples/demo.json` is
the `result.json` it produces — load that in the viewer to see the UI without
any network or keys.

## Free / local part identification (no paid API)

As an alternative to Claude vision for *identifying* parts, there's an ensemble
identifier that combines two independent, free signals — both constrained to the
set's inventory (the answer must be a part the set actually contains, which is
what makes identification tractable):

- **Brickognize** (`brickognize.py`) — a free, purpose-built LEGO-part model;
  POST a callout crop, get ranked candidate parts. No key.
- **Embedding retrieval** (`embedding.py`) — embed the set's inventory reference
  images once with a pretrained encoder (open_clip, optional `[ml]` extra) and
  take the nearest neighbour to the crop. No training.
- **Colour agreement** — the crop's colour vs. the inventory line's colour.

`identify.py` blends these with configurable weights, renormalising over
whichever signals are present, and returns the best inventory match plus
alternatives and a per-signal breakdown. The inventory itself is free:

```python
from legopartlocator.inventory import load_inventory_file      # CSV/JSON, no key
# or legopartlocator.rebrickable.RebrickableClient (free key)
```

Install the optional embedding backend with `pip install -e ".[ml]"` (pulls in
torch + open_clip). The blending, nearest-neighbour math, inventory loading, and
API parsing are all unit-tested offline; the live Brickognize HTTP call and the
torch encoder are injected so they're swappable.

This is wired end-to-end as the **`--engine local`** path:

```
PDF -> vision_local (OpenCV: callout crops + bag numerals + quantity OCR)
    -> locate.assemble_result (identify each crop, constrained to inventory)
    -> result.json / result.csv
```

Run it with `lpl scan instructions.pdf --engine local --inventory-file inv.csv`
(see Usage). It needs the `[local]` extra (`pip install -e ".[local]"`). The
**Tesseract binary** is optional: without it, a built-in, dependency-free digit
reader (`DigitOCR`) still reads each callout's printed "Nx" quantity by digit
template-matching, and **bag numbers work either way** — real LEGO bags are
always numbered 1..N in page order, so a detected bag-start page with no
readable digit is assigned the next number ordinally. Install Tesseract only if
you also want printed part/element ids read (`DigitOCR` doesn't attempt those).
`--embeddings` additionally turns on the image-matcher (needs the `[ml]`
extra + network to fetch reference images).

Brickognize calls are cached (by crop-image hash, reusing `--cache-dir`/
`--no-cache`) and retried with backoff, so a re-scan or a flaky connection
doesn't re-pay for or crash on every callout; one failed identify no longer
aborts the whole scan — it's logged as a warning and that callout falls into
the "unidentified" bucket.

A few more `--engine local` flags worth knowing about:
- `--capacity-reconcile` (on by default; `--no-capacity-reconcile` to disable)
  stops an embedding-only match from over-filling a part past its known
  inventory quantity — the surplus crop is diverted to its next-best
  candidate instead of inflating a count-mismatch warning.
- `--dump-crops DIR` writes every callout crop plus a `manifest.json`
  (final + raw assignment, per-signal confidence, alternatives) — useful for
  debugging misidentifications, or as input to `--real-crops` below.

### Training your own embedding model

`--embeddings` normally uses pretrained CLIP zero-shot (no training). You can
instead fine-tune a small CNN (MobileNetV3-Small/ResNet18) on a set's own
reference images with triplet-loss metric learning — no paid API, no Claude:

```bash
pip install -e ".[train]"          # torch + torchvision (+ torch-directml on Windows)
lpl train-embedding --inventory-file samples/76307_inventory.csv --out models/lego_embed.pt
lpl scan instructions.pdf --engine local --embeddings --embedding-weights models/lego_embed.pt
```

Each part's single reference photo is expanded into several augmented
variants (rotation, colour jitter, blur/downsample, a flat-shading pass, and
background padding — narrowing the gap between glossy catalog photos and the
flatter instruction-booklet icon crops actually scanned) and trained so
same-part crops embed close together. On Windows with a non-CUDA GPU (AMD/
Intel), `torch-directml` gives real acceleration; otherwise it falls back to
CPU. This is a retrieval model for *this project's* known parts, not a
general-purpose LEGO classifier — realistic expectation is an incremental
accuracy boost over pretrained CLIP on your own inventories, not a solved
problem. The triplet-sampling logic (`train/sampling.py`) is unit-tested
offline; the actual torch training loop (`train/embedding_trainer.py`) and
`embedding.TrainedBackend` need the `[train]` extra to run, same as
`ClipBackend` already does for `[ml]`.

Tuning knobs worth knowing about (`lpl train-embedding --help` for the full
list):
- `--augment-style icon` (default) flattens shading and adds a black edge
  outline so catalog photos look more like flat instruction-booklet icons;
  `--augment-style photo` is the original, unmodified pipeline.
- `--mining semihard` (default) mines hard negatives from the current model
  each epoch instead of sampling them at random; `--mining random` is the
  original uniform sampling.
- `--real-crops MANIFEST` (repeatable) feeds corroborated crops from a prior
  `scan --dump-crops` back in as real, in-domain training examples.
- `--ldraw-dir LIBRARY_ROOT` (point it at the `ldraw/` folder from
  ldraw.org's free `complete.zip`) renders flat-shaded, icon-style training
  images straight from LDraw's part geometry — the closest structural match
  to how instruction booklets actually draw parts. `--ldraw-only
  PART,PART,...` restricts this to specific parts instead of the whole
  inventory.

### Calibrating detection on a real page

The OpenCV thresholds (`DetectConfig`: gray band, cell-area/aspect gates) are
calibrated to the standard LEGO callout style but real DPI/print variation may
need adjustment. `lpl debug` visualises exactly what the detector caught:

```bash
lpl debug instructions.pdf --out out/debug --pages 1-10
```

This writes, per page: `page_NNN.png` (detected callout boxes + bag marker
overlaid), `mask_NNN.png` (the gray-band threshold mask — the fastest way to
see if `panel_gray_low`/`panel_gray_high` need adjusting), and `stats.json`
(per-page counts and the area-fraction/aspect numbers the `DetectConfig` gates
are tuned against). Override the band with `--panel-low`/`--panel-high` (also
available on `lpl scan --engine local`) once you know what to change.

## Development

```bash
pytest         # offline unit tests: bags, reconcile, local engine, identify, training, etc.
ruff check .   # lint
```

CI (`.github/workflows/ci.yml`) runs the suite on Ubuntu (3.11, 3.12) and
Windows (3.12), plus lint, on every push/PR — no network, keys, or Tesseract
needed. A separate job installs the `[train]` extra so the torch-dependent
tests (skipped in the main matrix) run for real at least once per CI run.

## Layout

```
src/legopartlocator/
  cli.py           # `lpl scan` / `lpl debug` entrypoints
  pdf_render.py    # PyMuPDF: page -> PNG, text, cover set-number detection
  vision.py        # Claude vision engine: schema-forced extraction + caching
  vision_local.py  # OpenCV local engine: callout/bag detection, quantity OCR
  detection.py     # PageDetection/DetectedCallout contract (crop bytes + bbox)
  debug_overlay.py # visualise local detection: overlays, mask, per-page stats
  identify.py      # ensemble part identifier (Brickognize + embedding + colour)
  brickognize.py   # free part-ID API client (cached, retried, throttled)
  embedding.py      # cosine-NN gallery over inventory reference images; ClipBackend/TrainedBackend
  train/            # `lpl train-embedding`: augmentation (photo/icon + LDraw renders), triplet sampling/mining, training loop
  train/ldraw.py    #   from-scratch LDraw .dat parser + flat-shaded icon renderer (no cv2/torch needed to parse/render)
  colors.py         # LEGO colour table + dominant-colour extraction
  inventory.py      # local CSV/JSON inventory loader (no API key)
  fetcher.py        # auto-download instruction PDFs from lego.com by set number
  locate.py         # local-engine orchestration: detect -> identify -> ScanResult
  bags.py           # marker -> page->bag step-function segmentation
  rebrickable.py    # inventory client
  reconcile.py      # callout <-> inventory matching + count validation (claude engine)
  aggregate.py      # ScanResult -> JSON/CSV
  models.py         # pydantic schemas
  cache.py          # page/crop-hash JSON cache
web/index.html      # static searchable viewer (warnings, bag cards, occurrences)
```

## Notes & limits

- **Cost** scales with page count. The two-pass flow (cheap triage over all
  pages, detailed reads only where needed) plus per-page caching and
  `--pages`/`--max-pages` keep it in check. Models are configurable via
  `ANTHROPIC_MODEL` (detail) and `ANTHROPIC_TRIAGE_MODEL` (triage).
- **Part identification** from small renders is inherently fuzzy. Reconciliation
  against the known inventory + count validation is what makes results
  trustworthy; the `confidence` field and count-mismatch warnings tell you where
  to double-check. Vision-only mode groups by colour+shape and is lower
  confidence.
