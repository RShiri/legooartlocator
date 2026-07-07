"""Command-line entrypoint: `lpl scan <pdf>`.

Ties the pipeline together: render -> vision -> segment -> (reconcile) ->
aggregate -> write result.json / result.csv.

For offline work (or environments where the LEGO/Anthropic/Rebrickable hosts
are blocked) you can skip the vision pass entirely with
``--extracts page_extracts.json``, a JSON list of per-page PageExtract dicts.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import List, Optional

import click

from . import __version__
from .aggregate import build_result, write_outputs
from .bags import page_to_bag, segment_bags
from .cache import JSONCache
from .models import PageExtract


def _load_dotenv(path: str = ".env") -> None:
    """Minimal .env loader (no dependency). Does not overwrite existing env vars."""
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip().strip('"').strip("'")
        os.environ.setdefault(key, val)


def _load_extracts(path: str) -> List[PageExtract]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [PageExtract(**d) for d in data]


@click.group()
@click.version_option(__version__, prog_name="lpl")
def main() -> None:
    """LEGO Part Locator — map parts to the bag/page they appear in."""


@main.command()
@click.argument("pdf", type=click.Path(exists=True, dir_okay=False), required=False)
@click.option("--set", "set_num", default=None, help="LEGO set number (e.g. 76307). Auto-detected if omitted.")
@click.option("--pages", "page_spec", default=None, help="1-based page range, e.g. '1-40,55'.")
@click.option("--max-pages", type=int, default=None, help="Cap number of pages scanned (dev/cost control).")
@click.option("--dpi", type=int, default=180, show_default=True, help="Render DPI for the detailed vision pass.")
@click.option("--triage-dpi", type=int, default=110, show_default=True, help="Render DPI for the cheap triage pass.")
@click.option("--single-pass", is_flag=True, help="Disable two-pass; run the detailed model on every page.")
@click.option("--no-rebrickable", is_flag=True, help="Skip Rebrickable reconciliation (vision-only).")
@click.option("--engine", type=click.Choice(["claude", "local"]), default="claude", show_default=True,
              help="Extraction+ID engine: 'claude' (vision API) or 'local' (OpenCV + Brickognize/embeddings, no paid API).")
@click.option("--inventory-file", type=click.Path(exists=True, dir_okay=False), default=None,
              help="Local set inventory CSV/JSON (BrickLink/Rebrickable export). Enables the local engine with no key.")
@click.option("--no-brickognize", is_flag=True, help="[local engine] Disable the Brickognize signal.")
@click.option("--embeddings", is_flag=True, help="[local engine] Enable embedding retrieval (needs the [ml] extra + network).")
@click.option("--locale", default="en-gb", show_default=True, help="LEGO site locale for auto-download (e.g. en-us, de-de).")
@click.option("--booklet", type=int, default=1, show_default=True, help="Which booklet to scan when a set has several.")
@click.option("--download-dir", default=".", show_default=True, help="Where to save auto-downloaded PDFs.")
@click.option("--extracts", type=click.Path(exists=True, dir_okay=False), default=None,
              help="Load pre-extracted page JSON instead of running the vision pass.")
@click.option("--out", "out_dir", default="out", show_default=True, help="Output directory.")
@click.option("--cache-dir", default=".lpl_cache", show_default=True, help="Vision cache directory.")
@click.option("--no-cache", is_flag=True, help="Ignore the vision cache.")
def scan(
    pdf: Optional[str],
    set_num: Optional[str],
    page_spec: Optional[str],
    max_pages: Optional[int],
    dpi: int,
    triage_dpi: int,
    single_pass: bool,
    no_rebrickable: bool,
    engine: str,
    inventory_file: Optional[str],
    no_brickognize: bool,
    embeddings: bool,
    locale: str,
    booklet: int,
    download_dir: str,
    extracts: Optional[str],
    out_dir: str,
    cache_dir: str,
    no_cache: bool,
) -> None:
    """Scan a LEGO instruction PDF and write result.json / result.csv.

    Give a PDF path, or just --set NNNN to auto-download the instructions from
    lego.com, or --extracts for offline mode.
    """
    _load_dotenv()

    # Auto-download the PDF from lego.com when only a set number was given.
    if not pdf and not extracts and set_num:
        pdf = _autofetch_pdf(set_num, locale, booklet, download_dir)

    # Local engine: OpenCV detection + Brickognize/embedding identification (no paid API).
    if engine == "local":
        if not pdf:
            raise click.UsageError("--engine local requires a PDF path or --set to auto-download.")
        _run_local(pdf, set_num, page_spec, max_pages, dpi, inventory_file,
                   no_brickognize, embeddings, no_rebrickable, out_dir)
        return

    if not pdf and not extracts:
        raise click.UsageError(
            "Provide a PDF path, or --set NNNN to auto-download it, or --extracts for offline mode."
        )

    # 1. Page extracts: either from a file (offline) or via render + vision.
    if extracts:
        page_extracts = _load_extracts(extracts)
        num_pages = max((p.page_index for p in page_extracts), default=-1) + 1
        click.echo(f"Loaded {len(page_extracts)} page extracts from {extracts}.")
    else:
        page_extracts, num_pages = _render_and_extract(
            pdf, page_spec, max_pages, dpi, triage_dpi, single_pass, cache_dir, no_cache
        )

    # 2. Resolve set number (needed for reconciliation).
    if not set_num and pdf:
        from .pdf_render import detect_set_number

        set_num = detect_set_number(pdf)
        if set_num:
            click.echo(f"Auto-detected set number: {set_num}")

    # 3. Bag segmentation.
    segments, seg_warnings = segment_bags(page_extracts, num_pages)
    p2b = page_to_bag(segments)
    click.echo(f"Segmented into {len({s.bag for s in segments})} bag(s).")

    # 4. Optional inventory + reconciliation.
    inventory = None
    set_name = None
    reconciled = False
    rec_warnings: List[str] = []
    if not no_rebrickable and set_num:
        inventory, set_name = _fetch_inventory(set_num)
        reconciled = inventory is not None

    from .reconcile import reconcile

    parts, rec_warnings = reconcile(page_extracts, p2b, inventory=inventory)

    # 5. Aggregate + write.
    result = build_result(
        parts=parts,
        segments=segments,
        num_pages=num_pages,
        set_num=set_num,
        set_name=set_name,
        source_pdf=pdf,
        reconciled=reconciled,
        warnings=seg_warnings + rec_warnings,
    )
    paths = write_outputs(result, out_dir)

    click.echo(f"\nParts located: {len(parts)}")
    if result.warnings:
        click.echo(f"Warnings: {len(result.warnings)} (see result.json)")
    click.echo(f"Wrote {paths['json']} and {paths['csv']}.")
    click.echo("Open web/index.html and load result.json to search.")


def _render_and_extract(pdf, page_spec, max_pages, dpi, triage_dpi, single_pass, cache_dir, no_cache):
    from .pdf_render import PageRenderer, parse_page_range, page_count, render_pages
    from .vision import VisionExtractor, extract_pages, extract_pages_two_pass

    total = page_count(pdf)
    indices = parse_page_range(page_spec, total)
    if max_pages is not None:
        indices = indices[:max_pages]
    extractor = VisionExtractor(cache=JSONCache(Path(cache_dir)))

    def progress(done, tot):
        click.echo(f"  vision {done}/{tot}   \r", nl=False)

    if single_pass:
        click.echo(f"Single-pass: rendering {len(indices)}/{total} pages at {dpi} DPI...")
        rendered = list(render_pages(pdf, dpi=dpi, page_indices=indices))
        page_extracts = extract_pages(rendered, extractor, use_cache=not no_cache, progress=progress)
        click.echo("")
        return page_extracts, total

    # Two-pass: cheap triage over all pages, detailed read only where needed.
    click.echo(f"Pass 1/2 (triage): rendering {len(indices)}/{total} pages at {triage_dpi} DPI...")
    triage_renders = list(render_pages(pdf, dpi=triage_dpi, page_indices=indices))
    click.echo("Pass 2/2 (detail): reading callouts on flagged pages...")
    with PageRenderer(pdf, dpi=dpi) as detail_renderer:
        page_extracts, stats = extract_pages_two_pass(
            triage_renders, detail_renderer.render, extractor,
            use_cache=not no_cache, progress=progress,
        )
    click.echo("")
    click.echo(
        f"Two-pass: triaged {stats['triaged']} pages, detailed {stats['detailed']}, "
        f"skipped {stats['skipped']} (no callouts)."
    )
    return page_extracts, total


def _autofetch_pdf(set_num, locale, booklet, download_dir):
    """Download the set's instruction PDF from lego.com and return the chosen booklet's path."""
    from .fetcher import FetchError, InstructionFetcher, page_url

    click.echo(f"No PDF given — fetching instructions for set {set_num} from {page_url(set_num, locale)} ...")
    try:
        paths = InstructionFetcher().fetch(set_num, dest_dir=download_dir, locale=locale)
    except FetchError as exc:
        raise click.UsageError(str(exc))
    except Exception as exc:  # network/HTTP errors
        raise click.UsageError(f"Failed to fetch instructions for set {set_num}: {exc}")

    if booklet < 1 or booklet > len(paths):
        raise click.UsageError(f"--booklet {booklet} out of range (set has {len(paths)} booklet(s)).")
    if len(paths) > 1:
        listing = ", ".join(f"{i + 1}:{p.name}" for i, p in enumerate(paths))
        click.echo(f"Found {len(paths)} booklets [{listing}].")
        click.echo(f"Scanning booklet {booklet}: {paths[booklet - 1].name} (use --booklet N for others).")
    else:
        click.echo(f"Downloaded {paths[0].name}.")
    return str(paths[booklet - 1])


def _run_local(pdf, set_num, page_spec, max_pages, dpi, inventory_file,
               no_brickognize, embeddings, no_rebrickable, out_dir):
    """Local engine: detect callouts with OpenCV, identify with the free ensemble."""
    from .brickognize import BrickognizeClient
    from .inventory import load_inventory_file

    # The local engine pulls in the optional [local] deps (numpy via colors,
    # OpenCV via vision_local). Turn a missing-dependency crash into a clear hint.
    try:
        from .locate import locate_local, make_identifier
    except ImportError as exc:
        raise click.UsageError(
            f"The local engine needs the optional dependencies ({exc.name} is missing). "
            'Install them with:  pip install -e ".[local]"'
        )

    if not set_num:
        from .pdf_render import detect_set_number

        set_num = detect_set_number(pdf)
        if set_num:
            click.echo(f"Auto-detected set number: {set_num}")

    # An inventory lets us constrain identification and reconcile counts; it's
    # optional — without it we fall back to unconstrained Brickognize-only.
    inventory = None
    set_name = None
    if inventory_file:
        inventory = load_inventory_file(inventory_file)
        click.echo(f"Loaded {len(inventory)} inventory lines from {inventory_file}.")
    elif set_num and not no_rebrickable:
        inventory, set_name = _fetch_inventory(set_num)

    brickognize = None if no_brickognize else BrickognizeClient()
    gallery = None  # set below only when embeddings are enabled with an inventory

    if not inventory:
        if brickognize is None:
            raise click.UsageError(
                "The local engine needs either a set inventory (--inventory-file PATH, a free "
                "BrickLink/Rebrickable export; or --set with REBRICKABLE_API_KEY) or Brickognize "
                "(don't pass --no-brickognize)."
            )
        click.echo("No inventory: identifying with Brickognize only (unconstrained, lower precision).")
        from .identify import BrickognizeOnlyIdentifier

        identifier = BrickognizeOnlyIdentifier(brickognize)
        reconcile_counts = False
    else:
        gallery = backend = None
        if embeddings:
            try:
                from .embedding import ClipBackend, build_gallery, download_reference_images

                click.echo("Building embedding gallery (downloading reference images)...")
                backend = ClipBackend()
                ref_images = download_reference_images(inventory)
                gallery = build_gallery(inventory, ref_images, backend)
                click.echo(f"Gallery: {len(gallery)} parts embedded.")
            except Exception as exc:  # torch/network/etc. — degrade gracefully
                click.echo(f"Embeddings disabled ({exc}).", err=True)
                gallery = backend = None

        identifier = make_identifier(
            inventory, brickognize=brickognize, gallery=gallery, backend=backend, use_color=True
        )
        reconcile_counts = True

    # OCR (quantities + bag numerals) needs the Tesseract binary; degrade if absent.
    import shutil

    from .vision_local import LocalDetector

    detector = None
    if shutil.which("tesseract") is None:
        click.echo(
            "Tesseract binary not found: quantities default to 1 and bag numbers "
            "won't be read. Install Tesseract for full local accuracy.",
            err=True,
        )

        class _NullOCR:
            def read_text(self, image_bgr):
                return ""

        detector = LocalDetector(ocr=_NullOCR())

    signals = ["colour"]
    if brickognize:
        signals.insert(0, "brickognize")
    if gallery:
        signals.append("embedding")
    click.echo(f"Local engine: detecting + identifying at {dpi} DPI (signals: {', '.join(signals)})...")

    def progress(done, tot):
        click.echo(f"  page {done}/{tot}   \r", nl=False)

    result = locate_local(
        pdf, inventory or [], identifier, dpi=dpi, page_spec=page_spec, max_pages=max_pages,
        detector=detector, use_color=True, reconcile_counts=reconcile_counts,
        set_num=set_num, set_name=set_name, progress=progress,
    )
    click.echo("")
    paths = write_outputs(result, out_dir)
    click.echo(f"\nParts located: {len(result.parts)}")
    if result.warnings:
        click.echo(f"Warnings: {len(result.warnings)} (see result.json)")
    click.echo(f"Wrote {paths['json']} and {paths['csv']}.")
    click.echo("Open web/index.html and load result.json to search.")


def _fetch_inventory(set_num: str):
    from .rebrickable import RebrickableClient, RebrickableError

    if not os.environ.get("REBRICKABLE_API_KEY"):
        click.echo("REBRICKABLE_API_KEY not set; running vision-only.", err=True)
        return None, None
    try:
        client = RebrickableClient()
        info = client.get_set_info(set_num)
        inventory = client.get_set_parts(set_num)
        click.echo(f"Fetched {len(inventory)} inventory lines for set {set_num}.")
        return inventory, info.get("name")
    except RebrickableError as exc:
        click.echo(f"Rebrickable error: {exc}. Running vision-only.", err=True)
        return None, None


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
