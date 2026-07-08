"""Contract test for TrainedBackend / the training checkpoint format.

Skipped when torch isn't installed (it's an optional [train] extra, not part
of the base test environment) -- mirrors this project's existing convention
of not unit-testing ClipBackend directly. Runs for real wherever torch *is*
available, using pretrained=False so it needs no network/ImageNet download.
"""

import cv2
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from legopartlocator.embedding import TrainedBackend  # noqa: E402
from legopartlocator.train.embedding_trainer import _build_backbone  # noqa: E402


def _fake_checkpoint(tmp_path, backbone="mobilenet_v3_small", embedding_dim=8):
    model = _build_backbone(torch, backbone, embedding_dim, pretrained=False)
    path = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "format_version": 1,
            "backbone": backbone,
            "embedding_dim": embedding_dim,
            "state_dict": model.state_dict(),
        },
        path,
    )
    return path


def _png_bytes(color=(10, 20, 30)):
    img = np.full((64, 64, 3), color, dtype=np.uint8)
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return buf.tobytes()


def test_trained_backend_loads_checkpoint_and_embeds_images(tmp_path):
    path = _fake_checkpoint(tmp_path, embedding_dim=8)
    backend = TrainedBackend(str(path), device="cpu")

    out = backend.embed_images([_png_bytes(), _png_bytes((200, 100, 5))])

    assert out.shape == (2, 8)


def test_trained_backend_output_is_l2_normalized(tmp_path):
    path = _fake_checkpoint(tmp_path, embedding_dim=8)
    backend = TrainedBackend(str(path), device="cpu")

    out = backend.embed_images([_png_bytes()])

    assert np.allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-5)


def test_trained_backend_respects_embedding_dim_from_checkpoint(tmp_path):
    path = _fake_checkpoint(tmp_path, embedding_dim=32)
    backend = TrainedBackend(str(path), device="cpu")

    out = backend.embed_images([_png_bytes()])

    assert out.shape == (1, 32)
