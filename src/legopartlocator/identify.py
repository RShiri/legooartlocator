"""Ensemble part identification, constrained to the set inventory.

Combines up to three independent signals for a single callout crop:
  * Brickognize predictions (free LEGO-part model),
  * embedding retrieval against the inventory reference gallery,
  * colour agreement between the crop's stated colour and the inventory line.

Every signal is filtered to the set inventory (the answer must be a part the
set actually contains), then blended with configurable weights. Weights are
renormalised over whichever signals are present, so the identifier degrades
gracefully when one is unavailable (e.g. Brickognize offline, or a part with
no reference image).

The blending is pure and fully unit-tested; the signal producers
(Brickognize HTTP, the embedding backend) are injected.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .brickognize import BrickognizeClient
from .embedding import EmbeddingBackend, Gallery
from .models import InventoryPart
from .reconcile import _tokens  # shared tokeniser

DEFAULT_WEIGHTS = {"brickognize": 0.5, "embedding": 0.4, "color": 0.1}


def blend_scores(
    signals: Dict[str, Dict[int, float]], weights: Dict[str, float]
) -> Dict[int, float]:
    """Weighted blend of per-candidate signal scores.

    ``signals`` maps signal name -> {candidate_index: score in [0, 1]}. Only
    signals that are present and non-empty contribute; their weights are
    renormalised to sum to 1 so a missing signal doesn't deflate every score.
    """
    present = {name: s for name, s in signals.items() if s}
    total_w = sum(weights.get(name, 0.0) for name in present)
    if total_w <= 0:
        return {}
    blended: Dict[int, float] = {}
    all_keys = {k for s in present.values() for k in s}
    for key in all_keys:
        blended[key] = sum(
            (weights.get(name, 0.0) / total_w) * present[name].get(key, 0.0)
            for name in present
        )
    return blended


def color_score(seen_color: Optional[str], part: InventoryPart) -> float:
    """Jaccard overlap of colour tokens between the crop and an inventory line."""
    a = _tokens(seen_color)
    b = _tokens(part.color_name)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


@dataclass
class IdentificationResult:
    part: Optional[InventoryPart]
    confidence: float
    components: Dict[str, float] = field(default_factory=dict)
    alternatives: List[Tuple[InventoryPart, float]] = field(default_factory=list)


class PartIdentifier:
    def __init__(
        self,
        inventory: Sequence[InventoryPart],
        brickognize: Optional[BrickognizeClient] = None,
        gallery: Optional[Gallery] = None,
        backend: Optional[EmbeddingBackend] = None,
        weights: Optional[Dict[str, float]] = None,
        top_k: int = 5,
        color_scorer: Callable[[Optional[str], InventoryPart], float] = color_score,
    ):
        self.inventory = list(inventory)
        self.brickognize = brickognize
        self.gallery = gallery
        self.backend = backend
        self.weights = weights or DEFAULT_WEIGHTS
        self.top_k = top_k
        self.color_scorer = color_scorer
        # part_num -> inventory indices (a part_num may appear in several colours).
        self._by_partnum: Dict[str, List[int]] = {}
        for i, p in enumerate(self.inventory):
            self._by_partnum.setdefault(p.part_num, []).append(i)

    def identify(
        self, crop_bytes: bytes, seen_color: Optional[str] = None
    ) -> IdentificationResult:
        signals: Dict[str, Dict[int, float]] = {}

        # Signal 1: Brickognize (part_num -> best score), spread over that
        # part_num's inventory colour lines.
        if self.brickognize is not None:
            bscore: Dict[int, float] = {}
            for cand in self.brickognize.predict(crop_bytes):
                for idx in self._by_partnum.get(cand.part_num, []):
                    bscore[idx] = max(bscore.get(idx, 0.0), max(0.0, cand.score))
            if bscore:
                signals["brickognize"] = bscore

        # Signal 2: embedding retrieval against the inventory gallery.
        if self.gallery is not None and self.backend is not None and len(self.gallery):
            query_vec = self.backend.embed_images([crop_bytes])[0]
            escore: Dict[int, float] = {}
            index_of = {id(p): i for i, p in enumerate(self.inventory)}
            for hit in self.gallery.query(query_vec, top_k=self.top_k):
                idx = index_of.get(id(hit.part))
                if idx is not None:
                    escore[idx] = max(0.0, hit.score)  # clamp negative cosine
            if escore:
                signals["embedding"] = escore

        # Signal 3: colour agreement (only meaningful when a colour was read).
        if seen_color:
            cscore = {
                i: self.color_scorer(seen_color, p) for i, p in enumerate(self.inventory)
            }
            cscore = {i: v for i, v in cscore.items() if v > 0}
            if cscore:
                signals["color"] = cscore

        # Colour is corroborative only: without a Brickognize or embedding hit
        # to anchor the answer, colour alone can't identify a specific part —
        # it would just pick an arbitrary inventory line of the seen colour.
        if not signals.get("brickognize") and not signals.get("embedding"):
            return IdentificationResult(part=None, confidence=0.0)

        blended = blend_scores(signals, self.weights)
        if not blended:
            return IdentificationResult(part=None, confidence=0.0)

        ranked = sorted(blended.items(), key=lambda kv: kv[1], reverse=True)
        best_idx, best_score = ranked[0]
        components = {name: signals[name].get(best_idx, 0.0) for name in signals}
        alternatives = [(self.inventory[i], s) for i, s in ranked[1 : self.top_k]]
        return IdentificationResult(
            part=self.inventory[best_idx],
            confidence=float(best_score),
            components=components,
            alternatives=alternatives,
        )


class BrickognizeOnlyIdentifier:
    """Unconstrained identifier: returns the top Brickognize candidate directly.

    Used when no set inventory is available — the zero-account path. Lower
    precision than the inventory-constrained ensemble (no closed-set filtering,
    no colour disambiguation of the same part across colours), but needs nothing
    but the free Brickognize service.
    """

    def __init__(self, brickognize: BrickognizeClient, top_k: int = 5):
        self.brickognize = brickognize
        self.top_k = top_k

    @staticmethod
    def _to_part(cand, color: Optional[str] = None) -> InventoryPart:
        return InventoryPart(
            part_num=cand.part_num, name=cand.name, quantity=0,
            color_name=color, image_url=cand.img_url,
        )

    def identify(self, crop_bytes: bytes, seen_color: Optional[str] = None) -> IdentificationResult:
        cands = self.brickognize.predict(crop_bytes)
        if not cands:
            return IdentificationResult(part=None, confidence=0.0)
        top = cands[0]
        return IdentificationResult(
            part=self._to_part(top, seen_color),
            confidence=max(0.0, top.score),
            components={"brickognize": top.score},
            alternatives=[(self._to_part(c), c.score) for c in cands[1 : self.top_k]],
        )
