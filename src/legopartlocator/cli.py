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
    extracts: Optional[str],
    out_dir: str,
    cache_dir: str,
    no_cache: bool,
) -> None:
    """Scan a LEGO instruction PDF and write result.json / result.csv."""
    _load_dotenv()

    if not pdf and not extracts:
        raise click.UsageError("Provide a PDF path, or --extracts for offline mode.")

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
