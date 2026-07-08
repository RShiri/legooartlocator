"""Fine-tune a small pretrained CNN backbone into a LEGO-part embedding model
using triplet loss, on our own augmented reference-image dataset.

Not part of the base install or the ``[ml]`` extra (which is CLIP-only,
inference-only) -- needs the new ``[train]`` extra (torch + torchvision, plus
torch-directml on Windows for real GPU use on non-CUDA cards). Lazily
imported, exactly like ``embedding.ClipBackend``, so nothing here affects the
base install or its tests.

The triplet-sampling logic itself (which crops pair up as anchor/positive/
negative) lives in ``sampling.py`` as plain, torch-free Python so it can be
unit-tested without torch installed; this module is the torch-dependent
glue around it and is intentionally *not* unit-tested directly (mirrors
``ClipBackend``, marked ``# pragma: no cover - requires torch``).
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np

from .data import train_val_split
from .sampling import flatten_dataset, sample_triplets, sample_triplets_semihard

CHECKPOINT_FORMAT_VERSION = 1


@dataclass
class TrainingResult:
    out_path: Path
    epochs_run: int
    best_epoch: int
    final_loss: float
    best_val_accuracy: Optional[float]
    n_parts: int
    n_triplets_seen: int


def _select_device(torch):  # pragma: no cover - requires torch
    """Prefer DirectML (AMD/Intel GPUs on Windows), then CUDA, then CPU."""
    try:
        import torch_directml  # type: ignore

        return torch_directml.device()
    except ImportError:
        pass
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _build_backbone(
    torch, backbone_name: str, embedding_dim: int, pretrained: bool = True
):  # pragma: no cover - requires torch
    """``pretrained=False`` skips the ImageNet-weights download -- randomly
    initialised, only useful for fast offline tests of the architecture/
    checkpoint plumbing, never for an actual training run."""
    import torch.nn as nn
    import torchvision.models as tv_models

    if backbone_name == "mobilenet_v3_small":
        weights = tv_models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        net = tv_models.mobilenet_v3_small(weights=weights)
        in_features = net.classifier[-1].in_features
        net.classifier[-1] = nn.Linear(in_features, embedding_dim)
    elif backbone_name == "resnet18":
        weights = tv_models.ResNet18_Weights.DEFAULT if pretrained else None
        net = tv_models.resnet18(weights=weights)
        net.fc = nn.Linear(net.fc.in_features, embedding_dim)
    else:
        raise ValueError(f"Unknown backbone: {backbone_name!r}")
    return net


def _preprocess_transform():  # pragma: no cover - requires torch
    import torchvision.transforms as T

    return T.Compose(
        [
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def _decode_to_tensor(transform, image_bytes: bytes):  # pragma: no cover - requires torch
    from PIL import Image  # type: ignore

    pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return transform(pil)


def _embed_crops(torch, model, crops, device, transform, batch_size=32):  # pragma: no cover - requires torch
    """L2-normalised embeddings for a flat list of crop bytes, batched."""
    import torch.nn.functional as F

    out = []
    for start in range(0, len(crops), batch_size):
        batch = crops[start : start + batch_size]
        tensors = torch.stack([_decode_to_tensor(transform, c) for c in batch]).to(device)
        with torch.no_grad():
            out.append(F.normalize(model(tensors), p=2, dim=-1))
    return torch.cat(out, dim=0) if out else torch.zeros((0, 0))


def _embed_tensors(torch, model, tensors, device, batch_size=32):  # pragma: no cover - requires torch
    """L2-normalised embeddings for pre-decoded tensors, returned as a numpy
    array. Runs in eval mode with grads off -- used to mine hard negatives at
    the start of each epoch -- and restores the model's training mode after."""
    import torch.nn.functional as F

    was_training = model.training
    model.eval()
    out = []
    try:
        for start in range(0, len(tensors), batch_size):
            batch = torch.stack(tensors[start : start + batch_size]).to(device)
            with torch.no_grad():
                out.append(F.normalize(model(batch), p=2, dim=-1).cpu())
    finally:
        if was_training:
            model.train()
    return torch.cat(out, dim=0).numpy() if out else np.zeros((0, 0), dtype=np.float32)


def evaluate_retrieval_accuracy(
    torch, model, train_dataset: Dict[str, Sequence[bytes]], val_dataset: Dict[str, Sequence[bytes]],
    device, transform,
) -> Optional[float]:  # pragma: no cover - requires torch
    """Top-1 retrieval accuracy: embed every val crop, find its nearest
    neighbour among *all* train crops (across every part, not just its own),
    check the retrieved part_num matches. This is the thing that actually
    matters for a retrieval model -- does a different view of an already-seen
    part still retrieve correctly -- as opposed to training loss, which only
    says the model separates the specific triplets it was shown.

    Returns ``None`` if there's nothing to evaluate (empty val set).
    """
    was_training = model.training
    model.eval()
    try:
        gallery_parts = []
        gallery_crops = []
        for part_num, crops in train_dataset.items():
            gallery_parts.extend([part_num] * len(crops))
            gallery_crops.extend(crops)
        val_parts = []
        val_crops = []
        for part_num, crops in val_dataset.items():
            val_parts.extend([part_num] * len(crops))
            val_crops.extend(crops)
        if not gallery_crops or not val_crops:
            return None

        gallery = _embed_crops(torch, model, gallery_crops, device, transform)
        queries = _embed_crops(torch, model, val_crops, device, transform)
        sims = queries @ gallery.T
        nearest = sims.argmax(dim=1).cpu().tolist()
        correct = sum(1 for i, idx in enumerate(nearest) if gallery_parts[idx] == val_parts[i])
        return correct / len(val_parts)
    finally:
        if was_training:
            model.train()


def train_embedding_model(
    dataset: Dict[str, Sequence[bytes]],
    out_path: str,
    *,
    backbone: str = "mobilenet_v3_small",
    embedding_dim: int = 256,
    epochs: int = 40,
    triplets_per_epoch: int = 200,
    batch_size: int = 16,
    margin: float = 0.3,
    mining: str = "semihard",
    lr: float = 1e-4,
    seed: int = 0,
    val_frac: float = 0.25,
    patience: int = 6,
    device_override: Optional[str] = None,
    progress_callback=None,
) -> TrainingResult:  # pragma: no cover - requires torch
    """Fine-tune ``backbone`` with triplet loss on an augmented dataset, with
    validation-based early stopping.

    ``dataset`` is ``{part_num: [augmented crop bytes, ...]}`` -- see
    ``train.data.build_augmented_dataset``. Each part's variants are split
    into train/val (``train.data.train_val_split``); triplets are sampled
    only from train, and after every epoch retrieval accuracy is measured on
    the held-out val crops (``evaluate_retrieval_accuracy``). Training loss
    alone can't tell "learned to generalise" from "memorised the augmented
    training images" -- val accuracy is what actually answers that. Stops
    after ``patience`` epochs with no val-accuracy improvement and saves the
    *best* checkpoint seen, not necessarily the last epoch's. If there's too
    little data to hold out a val set at all, falls back to training the
    full ``epochs`` and saving the final state (a warning-worthy but valid
    degraded mode, e.g. for a very small inventory).

    Saves a checkpoint to ``out_path`` loadable by ``embedding.TrainedBackend``.
    """
    try:
        import torch
        import torch.nn as nn
    except ImportError as exc:
        raise ImportError(
            "Training needs the optional ML deps. Install with: pip install -e \".[train]\""
        ) from exc

    if mining not in ("random", "semihard"):
        raise ValueError(f"mining must be 'random' or 'semihard', got {mining!r}")

    device = torch.device(device_override) if device_override else _select_device(torch)
    model = _build_backbone(torch, backbone, embedding_dim).to(device)
    model.train()

    transform = _preprocess_transform()
    train_dataset, val_dataset = train_val_split(dataset, val_frac=val_frac, seed=seed)
    has_val = any(val_dataset.values())

    flat = flatten_dataset(train_dataset)
    # Pre-decode every training crop once; the datasets this trains on are
    # small enough (augmented reference images for one set's inventory) to
    # fit in memory as tensors.
    tensors = [_decode_to_tensor(transform, crop) for _part_num, crop in flat]

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.TripletMarginLoss(margin=margin)

    final_loss = 0.0
    n_triplets_seen = 0
    best_val_accuracy: Optional[float] = None
    best_epoch = -1
    best_state_dict = None
    epochs_without_improvement = 0
    epochs_run = 0

    for epoch in range(epochs):
        epochs_run = epoch + 1
        if mining == "semihard":
            # Re-embed the training crops with the current model, then mine
            # informative negatives from that geometry (one extra forward pass
            # over the dataset per epoch -- the ~2x epoch cost noted in the CLI).
            current_emb = _embed_tensors(torch, model, tensors, device)
            triplets = sample_triplets_semihard(
                train_dataset, current_emb, n=triplets_per_epoch,
                seed=seed * 1_000_003 + epoch, margin=margin,
            )
        else:
            triplets = sample_triplets(
                train_dataset, n=triplets_per_epoch, seed=seed * 1_000_003 + epoch
            )
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, len(triplets), batch_size):
            batch = triplets[start : start + batch_size]
            if not batch:
                continue
            anchors = torch.stack([tensors[t.anchor] for t in batch]).to(device)
            positives = torch.stack([tensors[t.positive] for t in batch]).to(device)
            negatives = torch.stack([tensors[t.negative] for t in batch]).to(device)

            optimizer.zero_grad()
            a_emb = nn.functional.normalize(model(anchors), p=2, dim=-1)
            p_emb = nn.functional.normalize(model(positives), p=2, dim=-1)
            n_emb = nn.functional.normalize(model(negatives), p=2, dim=-1)
            loss = loss_fn(a_emb, p_emb, n_emb)
            loss.backward()
            optimizer.step()

            epoch_loss += float(loss.item())
            n_batches += 1
            n_triplets_seen += len(batch)

        final_loss = epoch_loss / max(1, n_batches)

        val_accuracy = None
        if has_val:
            val_accuracy = evaluate_retrieval_accuracy(
                torch, model, train_dataset, val_dataset, device, transform
            )

        if progress_callback is not None:
            progress_callback(epoch, epochs, final_loss, val_accuracy)

        if not has_val:
            continue  # no early-stopping signal available; train the full budget

        if best_val_accuracy is None or val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_epoch = epoch
            best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    final_state_dict = best_state_dict if best_state_dict is not None else model.state_dict()

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "backbone": backbone,
            "embedding_dim": embedding_dim,
            "state_dict": {k: v.cpu() for k, v in final_state_dict.items()},
        },
        out,
    )
    return TrainingResult(
        out_path=out,
        epochs_run=epochs_run,
        best_epoch=best_epoch if best_state_dict is not None else epochs_run - 1,
        final_loss=final_loss,
        best_val_accuracy=best_val_accuracy,
        n_parts=len(dataset),
        n_triplets_seen=n_triplets_seen,
    )
