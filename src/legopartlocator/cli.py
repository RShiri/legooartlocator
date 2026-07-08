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


def _tesseract_or_digit_ocr():
    """Return None (use the default TesseractOCR) if the binary is present, else
    a dependency-free ``DigitOCR`` that reads the 'Nx' quantity label by digit
    template-matching. Bag numbers still work either way — they're assigned
    ordinally (see locate.py); DigitOCR returns "" for the large bag numeral so
    that path is unchanged. Printed part ids still need Tesseract."""
    import shutil

    if shutil.which("tesseract") is not None:
        return None

    from .vision_local import DigitOCR

    click.echo(
        "Tesseract binary not found: reading quantities with the built-in digit "
        "reader (the 'Nx' label). Bag numbers are assigned ordinally; install "
        "Tesseract to also read printed part ids.",
        err=True,
    )
    return DigitOCR()


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
@click.option("--embedding-weights", type=click.Path(exists=True, dir_okay=False), default=None,
              help="[local engine] Use a checkpoint from `lpl train-embedding` instead of pretrained CLIP (implies --embeddings).")
@click.option("--capacity-reconcile/--no-capacity-reconcile", default=True, show_default=True,
              help="[local engine] Stop an embedding-only match from over-filling a part past its inventory quantity "
                   "(diverts the attractor's spurious extra crops); reduces count-mismatch warnings.")
@click.option("--dump-crops", "dump_crops_dir", type=click.Path(file_okay=False), default=None,
              help="[local engine] Write every callout crop + a manifest.json (assignment, per-signal scores) "
                   "to this directory — a labelled dataset for calibration, diagnosis, and --real-crops training.")
@click.option("--locale", default="en-gb", show_default=True, help="LEGO site locale for auto-download (e.g. en-us, de-de).")
@click.option("--booklet", type=int, default=1, show_default=True, help="Which booklet to scan when a set has several.")
@click.option("--download-dir", default=".", show_default=True, help="Where to save auto-downloaded PDFs.")
@click.option("--panel-low", type=int, default=None, help="[local engine] Override DetectConfig.panel_gray_low (callout background band).")
@click.option("--panel-high", type=int, default=None, help="[local engine] Override DetectConfig.panel_gray_high.")
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
    embedding_weights: Optional[str],
    capacity_reconcile: bool,
    dump_crops_dir: Optional[str],
    locale: str,
    booklet: int,
    download_dir: str,
    panel_low: Optional[int],
    panel_high: Optional[int],
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
                   no_brickognize, embeddings, embedding_weights, capacity_reconcile,
                   dump_crops_dir, no_rebrickable, out_dir, cache_dir, no_cache,
                   panel_low, panel_high)
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


@main.command()
@click.argument("pdf", type=click.Path(exists=True, dir_okay=False))
@click.option("--out", "out_dir", default="out/debug", show_default=True, help="Directory for overlay PNGs + stats.json.")
@click.option("--pages", "page_spec", default=None, help="1-based page range, e.g. '1-10'.")
@click.option("--dpi", type=int, default=180, show_default=True, help="Render DPI.")
@click.option("--panel-low", type=int, default=None, help="Override DetectConfig.panel_gray_low.")
@click.option("--panel-high", type=int, default=None, help="Override DetectConfig.panel_gray_high.")
def debug(pdf: str, out_dir: str, page_spec: Optional[str], dpi: int,
          panel_low: Optional[int], panel_high: Optional[int]) -> None:
    """Visualise local (--engine local) detection on a PDF to tune thresholds.

    Writes page_NNN.png (detected callout/bag boxes overlaid), mask_NNN.png
    (the gray-band threshold mask), and stats.json (per-page counts and the
    area-fraction/aspect numbers DetectConfig's gates are tuned against).
    """
    try:
        from .debug_overlay import run_debug
        from .vision_local import DetectConfig
    except ImportError as exc:
        raise click.UsageError(
            f"The debug command needs the optional dependencies ({exc.name} is missing). "
            'Install them with:  pip install -e ".[local]"'
        )

    config = DetectConfig()
    if panel_low is not None:
        config.panel_gray_low = panel_low
    if panel_high is not None:
        config.panel_gray_high = panel_high

    ocr = _tesseract_or_digit_ocr()

    click.echo(f"Detecting on {pdf} at {dpi} DPI -> {out_dir} ...")
    stats = run_debug(pdf, out_dir, dpi=dpi, page_spec=page_spec, config=config, ocr=ocr)
    n_pages = len(stats.get("pages", []))
    total_callouts = sum(p.get("n_callouts", 0) for p in stats.get("pages", []))
    click.echo(f"Wrote {n_pages} page(s) of overlays/masks + stats.json to {out_dir}.")
    click.echo(f"Total callouts detected: {total_callouts}.")
    click.echo("Open the page_*.png files to see what was caught; tune --panel-low/--panel-high from mask_*.png.")


@main.command("train-embedding")
@click.option("--inventory-file", type=click.Path(exists=True, dir_okay=False), default=None,
              help="Local set inventory CSV/JSON (needs an image_url column).")
@click.option("--set", "set_num", default=None, help="LEGO set number; fetches the inventory via REBRICKABLE_API_KEY.")
@click.option("--out", "out_path", default="models/lego_embed.pt", show_default=True, help="Checkpoint output path.")
@click.option("--backbone", type=click.Choice(["mobilenet_v3_small", "resnet18"]), default="mobilenet_v3_small",
              show_default=True)
@click.option("--embedding-dim", type=int, default=256, show_default=True)
@click.option("--epochs", type=int, default=40, show_default=True, help="Max epochs (early stopping usually stops sooner).")
@click.option("--variants-per-part", type=int, default=8, show_default=True,
              help="Augmented crops generated per reference image.")
@click.option("--augment-style", type=click.Choice(["photo", "icon"]), default="icon", show_default=True,
              help="'icon' flattens shading and adds a black edge outline so catalog photos look "
                   "more like flat instruction-booklet icons (narrows the domain gap). 'photo' is the "
                   "original catalog-photo pipeline.")
@click.option("--triplets-per-epoch", type=int, default=200, show_default=True)
@click.option("--mining", type=click.Choice(["random", "semihard"]), default="semihard", show_default=True,
              help="'semihard' mines hard negatives from the current model each epoch instead of "
                   "sampling them at random (keeps gradient alive; ~2x epoch time on CPU). 'random' is "
                   "the original uniform sampling.")
@click.option("--real-crops", "real_crops", multiple=True,
              type=click.Path(exists=True, dir_okay=False),
              help="manifest.json from `lpl scan --dump-crops` — corroborated real callout crops are "
                   "added as in-domain training examples (repeatable).")
@click.option("--ldraw-dir", "ldraw_dir", type=click.Path(exists=True, file_okay=False), default=None,
              help="LDraw library root (the folder holding parts/ and p/, from ldraw.org complete.zip). "
                   "Adds flat-shaded icon-style renders of each part as training images — the closest "
                   "match to how instruction booklets actually draw parts.")
@click.option("--ldraw-only", "ldraw_only", default=None,
              help="Comma-separated part numbers to restrict --ldraw-dir rendering to (e.g. the "
                   "unidentified parts from a prior scan). Every other part's training data is left "
                   "completely untouched. Omit to render for the whole inventory.")
@click.option("--batch-size", type=int, default=16, show_default=True)
@click.option("--val-frac", type=float, default=0.25, show_default=True,
              help="Fraction of each part's variants held out to measure retrieval accuracy each epoch.")
@click.option("--patience", type=int, default=6, show_default=True,
              help="Stop after this many epochs with no val-accuracy improvement; saves the best epoch's checkpoint.")
@click.option("--seed", type=int, default=0, show_default=True)
def train_embedding(
    inventory_file: Optional[str], set_num: Optional[str], out_path: str, backbone: str,
    embedding_dim: int, epochs: int, variants_per_part: int, augment_style: str,
    triplets_per_epoch: int, mining: str, real_crops: tuple, ldraw_dir: Optional[str],
    ldraw_only: Optional[str], batch_size: int, val_frac: float, patience: int, seed: int,
) -> None:
    """Fine-tune a local part-embedding model on a set's reference images.

    No paid API involved: pulls one reference photo per part (from an
    inventory file, or --set with a free REBRICKABLE_API_KEY), augments it
    into several synthetic variants, and trains with triplet loss so same-part
    crops embed close together. Held-out variants are used to measure
    retrieval accuracy each epoch and stop early once it stops improving --
    training loss alone can't tell learning from memorising the augmented
    training images. The resulting checkpoint is a drop-in replacement for
    pretrained CLIP: `lpl scan --engine local --embeddings --embedding-weights
    <out_path>`.
    """
    _load_dotenv()
    try:
        from .train.data import build_augmented_dataset, collect_training_images
        from .train.embedding_trainer import train_embedding_model
    except ImportError as exc:
        raise click.UsageError(
            f"train-embedding needs the optional training dependencies ({exc.name} is missing). "
            'Install them with:  pip install -e ".[train]"'
        )

    from .inventory import load_inventory_file

    if inventory_file:
        inventory = load_inventory_file(inventory_file)
        click.echo(f"Loaded {len(inventory)} inventory lines from {inventory_file}.")
    elif set_num:
        inventory, _set_name = _fetch_inventory(set_num)
        if not inventory:
            raise click.UsageError(f"Could not fetch an inventory for set {set_num}.")
    else:
        raise click.UsageError("Provide --inventory-file or --set.")

    click.echo("Downloading reference images...")
    ref_images = collect_training_images(inventory)
    click.echo(f"Got {len(ref_images)}/{len(inventory)} reference images (missing image_url or fetch failures are skipped).")
    if len(ref_images) < 2:
        raise click.UsageError("Need reference images for at least 2 distinct parts to train (need positive/negative pairs).")

    click.echo(f"Augmenting into {variants_per_part} variants per part ({augment_style} style)...")
    dataset = build_augmented_dataset(ref_images, variants_per_part=variants_per_part, seed=seed, style=augment_style)

    if real_crops:
        from .train.data import load_real_crops, merge_datasets

        for manifest in real_crops:
            real = load_real_crops(manifest)
            n_crops = sum(len(v) for v in real.values())
            click.echo(f"Real crops from {manifest}: {n_crops} corroborated crops across {len(real)} parts.")
            dataset = merge_datasets(dataset, real)

    if ldraw_dir:
        from .train.data import merge_datasets
        from .train.ldraw import render_training_images

        only_parts = None
        if ldraw_only:
            only_parts = {p.strip() for p in ldraw_only.split(",") if p.strip()}
            click.echo(f"Rendering LDraw icon-style views (restricted to {len(only_parts)} part(s))...")
        else:
            click.echo("Rendering LDraw icon-style views...")
        renders = render_training_images(inventory, ldraw_dir, only_parts=only_parts)
        n_imgs = sum(len(v) for v in renders.values())
        click.echo(f"LDraw renders: {n_imgs} views across {len(renders)}/{len(inventory)} inventory lines.")
        dataset = merge_datasets(dataset, renders)

    def progress(epoch, total, loss, val_accuracy):
        if val_accuracy is None:
            click.echo(f"  epoch {epoch + 1}/{total}: loss={loss:.4f}")
        else:
            click.echo(f"  epoch {epoch + 1}/{total}: loss={loss:.4f} val_accuracy={val_accuracy:.3f}")

    click.echo(f"Training ({backbone}, up to {epochs} epochs, {triplets_per_epoch} triplets/epoch, "
               f"{mining} mining, patience={patience})...")
    result = train_embedding_model(
        dataset, out_path, backbone=backbone, embedding_dim=embedding_dim, epochs=epochs,
        triplets_per_epoch=triplets_per_epoch, mining=mining, batch_size=batch_size, val_frac=val_frac,
        patience=patience, seed=seed, progress_callback=progress,
    )
    acc_msg = f"{result.best_val_accuracy:.3f}" if result.best_val_accuracy is not None else "n/a (too little data for a val split)"
    click.echo(f"\nWrote {result.out_path} (ran {result.epochs_run} epoch(s), best epoch {result.best_epoch + 1}, "
               f"best val_accuracy={acc_msg}, final loss {result.final_loss:.4f}, "
               f"{result.n_parts} parts, {result.n_triplets_seen} triplets seen).")


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
               no_brickognize, embeddings, embedding_weights, capacity_reconcile,
               dump_crops_dir, no_rebrickable, out_dir, cache_dir, no_cache,
               panel_low, panel_high):
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

    brickognize = None
    if not no_brickognize:
        bk_cache = None if no_cache else JSONCache(Path(cache_dir))
        brickognize = BrickognizeClient(cache=bk_cache)
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
        if embeddings or embedding_weights:
            try:
                from .embedding import build_gallery, download_reference_images

                if embedding_weights:
                    from .embedding import TrainedBackend

                    click.echo(f"Building embedding gallery (trained checkpoint: {embedding_weights})...")
                    backend = TrainedBackend(embedding_weights)
                else:
                    from .embedding import ClipBackend

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

    from .vision_local import DetectConfig, LocalDetector

    config = DetectConfig()
    if panel_low is not None:
        config.panel_gray_low = panel_low
    if panel_high is not None:
        config.panel_gray_high = panel_high

    ocr = _tesseract_or_digit_ocr()
    detector = LocalDetector(config=config, ocr=ocr) if (ocr or panel_low or panel_high) else None

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
        capacity_reconcile=capacity_reconcile, dump_crops_dir=dump_crops_dir,
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
