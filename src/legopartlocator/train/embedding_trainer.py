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

from .sampling import flatten_dataset, sample_triplets

CHECKPOINT_FORMAT_VERSION = 1


@dataclass
class TrainingResult:
    out_path: Path
    epochs: int
    final_loss: float
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


def train_embedding_model(
    dataset: Dict[str, Sequence[bytes]],
    out_path: str,
    *,
    backbone: str = "mobilenet_v3_small",
    embedding_dim: int = 256,
    epochs: int = 5,
    triplets_per_epoch: int = 200,
    batch_size: int = 16,
    margin: float = 0.3,
    lr: float = 1e-4,
    seed: int = 0,
    device_override: Optional[str] = None,
    progress_callback=None,
) -> TrainingResult:  # pragma: no cover - requires torch
    """Fine-tune ``backbone`` with triplet loss on an augmented dataset.

    ``dataset`` is ``{part_num: [augmented crop bytes, ...]}`` -- see
    ``train.data.build_augmented_dataset``. Saves a checkpoint to
    ``out_path`` loadable by ``embedding.TrainedBackend``.
    """
    try:
        import torch
        import torch.nn as nn
    except ImportError as exc:
        raise ImportError(
            "Training needs the optional ML deps. Install with: pip install -e \".[train]\""
        ) from exc

    device = torch.device(device_override) if device_override else _select_device(torch)
    model = _build_backbone(torch, backbone, embedding_dim).to(device)
    model.train()

    transform = _preprocess_transform()
    flat = flatten_dataset(dataset)
    # Pre-decode every crop once; the datasets this trains on are small
    # enough (augmented reference images for one set's inventory) to fit in
    # memory as tensors.
    tensors = [_decode_to_tensor(transform, crop) for _part_num, crop in flat]

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.TripletMarginLoss(margin=margin)

    final_loss = 0.0
    n_triplets_seen = 0
    for epoch in range(epochs):
        triplets = sample_triplets(dataset, n=triplets_per_epoch, seed=seed * 1_000_003 + epoch)
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
        if progress_callback is not None:
            progress_callback(epoch, epochs, final_loss)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "backbone": backbone,
            "embedding_dim": embedding_dim,
            "state_dict": model.cpu().state_dict(),
        },
        out,
    )
    return TrainingResult(
        out_path=out,
        epochs=epochs,
        final_loss=final_loss,
        n_parts=len(dataset),
        n_triplets_seen=n_triplets_seen,
    )
