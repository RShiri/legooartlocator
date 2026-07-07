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
pip install -e ".[dev]"
```

## Usage

```bash
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
lpl scan instructions.pdf --engine local --set 76307              # inventory via Rebrickable key
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
(see Usage). It needs the `[local]` extra (`pip install -e ".[local]"`) and the
**Tesseract binary** for reading quantities/bag numbers — without Tesseract it
still runs but quantities default to 1 and bag numbers aren't read. `--embeddings`
additionally turns on the image-matcher (needs the `[ml]` extra + network to
fetch reference images).

> **Status:** fully wired and unit-tested (blending, NN, inventory, detection
> mechanics, and the `assemble_result` orchestrator). The one thing that still
> needs a real instruction page is **tuning the OpenCV `DetectConfig`
> thresholds** (gray band, cell-area gates) — they're calibrated to the standard
> callout style but real DPI/print variation will want adjustment.

## Development

```bash
pytest         # offline unit tests: bag segmentation + reconciliation
```

## Layout

```
src/legopartlocator/
  cli.py         # `lpl scan` entrypoint
  pdf_render.py  # PyMuPDF: page -> PNG, text, cover set-number detection
  vision.py      # Claude vision, schema-forced extraction + caching
  bags.py        # marker -> page->bag step-function segmentation
  rebrickable.py # inventory client
  reconcile.py   # callout <-> inventory matching + count validation
  aggregate.py   # ScanResult -> JSON/CSV
  models.py      # pydantic schemas
  cache.py       # page-hash JSON cache
web/index.html   # static searchable viewer
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
