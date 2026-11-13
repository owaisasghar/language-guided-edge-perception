"""The CLIP embedder with real weights. Skipped unless the pinned checkpoint is in the local HF cache
(it is downloaded on first use, ~600 MB; see docs/milestone5.md). Never downloads anything."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from lgep.reid.embedder import ClipEmbedder, EmbedderConfig

IMAGE = Path(__file__).resolve().parents[1] / "data/samples/milestone1/000000401862.jpg"
pytestmark = [pytest.mark.weights, pytest.mark.skipif(not IMAGE.is_file(), reason="run scripts/download_samples.py")]


@pytest.fixture(scope="module")
def embedder():
    try:
        return ClipEmbedder(EmbedderConfig(local_files_only=True))
    except OSError:
        pytest.skip("CLIP weights not in the local cache")


@pytest.fixture(scope="module")
def crops():
    im = np.asarray(Image.open(IMAGE).convert("RGB"))
    return [im[200:460, 190:280], im[160:480, 380:450], im[100:300, 100:300]]  # two people, one square patch


def test_embeddings_are_normalised_deterministic_and_batch_independent(embedder, crops):
    a = embedder.embed_images(crops)
    assert a.shape == (3, embedder.dim) and np.allclose(np.linalg.norm(a, axis=1), 1, atol=1e-5)
    assert np.allclose(a, embedder.embed_images(crops), atol=1e-6)
    singles = np.concatenate([embedder.embed_images([c]) for c in crops])
    # Batched vs single: GPU kernels differ with batch size (and cuDNN uses TF32 convolutions by default),
    # measured max |diff| 2.6e-4; the embeddings are the same to cosine > 0.99999.
    assert np.all((a * singles).sum(axis=1) > 0.99999)


def test_letterbox_only_differs_for_non_square_crops(embedder, crops):
    def variant(mode):
        e = ClipEmbedder.__new__(ClipEmbedder)
        e.__dict__.update(embedder.__dict__)
        e.cfg = EmbedderConfig(crop_mode=mode, local_files_only=True)
        return e

    lb, st = variant("letterbox").embed_images(crops), variant("stretch").embed_images(crops)
    assert np.allclose(lb[2], st[2], atol=1e-5)  # square crop: no padding, identical
    assert float(lb[0] @ st[0]) < 0.999  # tall crop: padding changes the input


def test_text_embeddings(embedder):
    t = embedder.embed_texts(["a person wearing a blue shirt", "a white truck"])
    assert t.shape == (2, embedder.dim) and np.allclose(np.linalg.norm(t, axis=1), 1, atol=1e-5)
    assert embedder.embed_texts([]).shape == (0, embedder.dim)
