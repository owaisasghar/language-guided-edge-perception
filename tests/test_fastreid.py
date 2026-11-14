"""FastReID embedder (Milestone 9): checkpoint key mapping, configuration, and (with weights) the loaded model."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from lgep.reid.embedder import EmbedderConfig, build_embedder
from lgep.reid.fastreid_embedder import remap_state_dict

REPO = Path(__file__).resolve().parents[1]
MODELS = {
    "vehicle": ("configs/VeRi/sbs_R50-ibn.yml", "models/fastreid/veri_sbs_R50-ibn.pth", (256, 256)),
    "person": ("configs/MSMT17/sbs_R50-ibn.yml", "models/fastreid/msmt_sbs_R50-ibn.pth", (384, 128)),
}


def test_old_bn_neck_names_are_mapped():
    sd = {"heads.bnneck.weight": 1, "heads.bnneck.running_var": 2, "heads.bottleneck.0.bias": 3,
          "backbone.conv1.weight": 4, "heads.classifier.weight": 5}
    assert remap_state_dict(sd) == {"heads.bottleneck.0.weight": 1, "heads.bottleneck.0.running_var": 2,
                                    "heads.bottleneck.0.bias": 3, "backbone.conv1.weight": 4,
                                    "heads.classifier.weight": 5}


def test_config_validation():
    assert EmbedderConfig.from_dict({"backend": "fastreid", "model_id": "configs/VeRi/sbs_R50-ibn.yml",
                                     "weights": "x.pth"}).backend == "fastreid"
    with pytest.raises(ValueError, match="weights"):
        EmbedderConfig.from_dict({"backend": "fastreid", "model_id": "configs/VeRi/sbs_R50-ibn.yml"})
    with pytest.raises(ValueError, match="backend"):
        EmbedderConfig.from_dict({"backend": "osnet"})
    with pytest.raises(FileNotFoundError, match="weights"):
        build_embedder({"backend": "fastreid", "model_id": "configs/VeRi/sbs_R50-ibn.yml",
                        "weights": "models/fastreid/missing.pth", "device": "cpu"})


@pytest.mark.weights
@pytest.mark.parametrize("kind", ["vehicle", "person"])
def test_model_loads_every_inference_weight(kind):
    model_id, weights, hw = MODELS[kind]
    if not (REPO / weights).is_file() or not (REPO / "third_party/fast-reid/fastreid").is_dir():
        pytest.skip("FastReID checkout or weights missing (docs/milestone9.md)")
    e = build_embedder({"backend": "fastreid", "model_id": model_id, "weights": weights, "device": "cpu",
                        "batch_size": 4})
    assert e.input_hw == hw and e.dim == 2048
    # the BN neck (old name heads.bnneck in the person checkpoint) holds the checkpoint's statistics
    ck = remap_state_dict(torch.load(REPO / weights, map_location="cpu", weights_only=True)["model"])
    own = e.model.state_dict()
    for k in ("heads.bottleneck.0.running_mean", "heads.bottleneck.0.weight", "backbone.conv1.weight"):
        assert torch.equal(own[k].float(), ck[k].float()), k
    rng = np.random.default_rng(0)
    crops = [rng.integers(0, 255, (int(rng.integers(40, 300)), int(rng.integers(40, 300)), 3), dtype=np.uint8)
             for _ in range(6)]
    f = e.embed_images(crops)
    assert f.shape == (6, 2048) and np.allclose(np.linalg.norm(f, axis=1), 1, atol=1e-5)
    # eval mode: a crop's embedding does not depend on the other crops of the batch
    assert np.allclose(e.embed_images(crops[:2]), f[:2], atol=1e-4)
    assert e.embed_images([]).shape == (0, 2048)
    with pytest.raises(NotImplementedError):
        e.embed_texts(["a red car"])
