# legooartlocator

Scan a LEGO set's building-instruction PDF and produce a **part → bag / page**
map: given a part, find which numbered bag it's in and on which pages it
appears. That per-bag mapping isn't available from any API — it lives only in
the printed instructions — so the PDF has to be scanned.

## How it works

A hybrid pipeline:

1. **Render** — each PDF page is rasterised to an image (PyMuPDF).
2. **Vision** — Claude's vision model reads each page into structured JSON:
   bag markers, parts callouts (quantity + colour + shape + any printed id),
   and whether the page is a parts-list (BOM). Output is schema-forced (no
   text parsing) and cached per page-image hash, so re-runs are free.
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
```

Then open `web/index.html` and drop in `out/result.json` (or serve the folder
and it auto-loads a sibling `result.json`).

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

- **Cost** scales with page count; caching + `--pages`/`--max-pages` keep it in
  check. A two-pass optimisation (cheap marker/callout detection first, detailed
  extraction only on callout pages) is a natural next step.
- **Part identification** from small renders is inherently fuzzy. Reconciliation
  against the known inventory + count validation is what makes results
  trustworthy; the `confidence` field and count-mismatch warnings tell you where
  to double-check. Vision-only mode groups by colour+shape and is lower
  confidence.
