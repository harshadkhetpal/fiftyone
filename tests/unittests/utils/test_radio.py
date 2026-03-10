"""Comprehensive test suite for C-RADIOv4 wrapper - extended coverage."""

import logging
import os
import sys
import tempfile

import fiftyone as fo
import fiftyone.brain as fob
import fiftyone.core.labels as fol
import fiftyone.core.models as fom
import fiftyone.utils.torch as fout
import fiftyone.zoo as foz
import fiftyone.zoo.models as fozm
import numpy as np
from PIL import Image
import pytest
import torch

pytest.importorskip("transformers")

from fiftyone.utils.radio import (
    CRadioV4Model,
    CRadioV4ModelConfig,
    DEFAULT_CRADIO_MODEL,
    RadioOutputProcessor,
    SpatialHeatmapOutputProcessor,
)


# =============================================================================
# CONFIG TESTS - EXTENDED
# =============================================================================

def test_config_default_repo():
    config = CRadioV4ModelConfig({})
    assert config.hf_repo == DEFAULT_CRADIO_MODEL
    assert config.hf_repo == "nvidia/C-RADIOv4-H"

def test_config_output_summary():
    config = CRadioV4ModelConfig({"output_type": "summary"})
    assert config.output_type == "summary"
    assert config.as_feature_extractor

def test_config_output_spatial():
    config = CRadioV4ModelConfig({"output_type": "spatial"})
    assert config.output_type == "spatial"
    assert not getattr(config, 'as_feature_extractor', False)

def test_config_mixed_precision_default():
    config = CRadioV4ModelConfig({})
    assert config.use_mixed_precision

def test_config_mixed_precision_false():
    config = CRadioV4ModelConfig({"use_mixed_precision": False})
    assert not config.use_mixed_precision

def test_config_smoothing_default():
    config = CRadioV4ModelConfig({})
    assert config.apply_smoothing

def test_config_smoothing_false():
    config = CRadioV4ModelConfig({"apply_smoothing": False})
    assert not config.apply_smoothing

def test_config_sigma_default():
    config = CRadioV4ModelConfig({})
    assert config.smoothing_sigma == 1.51

def test_config_sigma_custom():
    config = CRadioV4ModelConfig({"smoothing_sigma": 3.0})
    assert config.smoothing_sigma == 3.0

def test_config_sigma_zero():
    config = CRadioV4ModelConfig({"smoothing_sigma": 0.0})
    assert config.smoothing_sigma == 0.0

def test_config_sigma_small():
    config = CRadioV4ModelConfig({"smoothing_sigma": 0.1})
    assert config.smoothing_sigma == 0.1

def test_config_sigma_large():
    config = CRadioV4ModelConfig({"smoothing_sigma": 10.0})
    assert config.smoothing_sigma == 10.0

def test_config_so400m():
    config = CRadioV4ModelConfig({"hf_repo": "nvidia/C-RADIOv4-SO400M"})
    assert config.hf_repo == "nvidia/C-RADIOv4-SO400M"

def test_config_hf_revision():
    config = CRadioV4ModelConfig({"hf_revision": "deadbeef"})
    assert config.hf_revision == "deadbeef"

def test_config_inheritance():
    config = CRadioV4ModelConfig({})
    assert isinstance(config, fout.TorchImageModelConfig)

def test_config_has_zoo_model():
    config = CRadioV4ModelConfig({})
    assert isinstance(config, fozm.HasZooModel)

def test_config_combined():
    config = CRadioV4ModelConfig({
        "hf_repo": "nvidia/C-RADIOv4-SO400M",
        "hf_revision": "deadbeef",
        "output_type": "spatial",
        "use_mixed_precision": False,
        "apply_smoothing": True,
        "smoothing_sigma": 2.0,
    })
    assert config.hf_repo == "nvidia/C-RADIOv4-SO400M"
    assert config.hf_revision == "deadbeef"
    assert config.output_type == "spatial"
    assert not config.use_mixed_precision
    assert config.apply_smoothing
    assert config.smoothing_sigma == 2.0


def test_load_model_uses_hf_revision(monkeypatch):
    class _StubModel:
        def __init__(self):
            self.device = None
            self.eval_called = False

        def to(self, device):
            self.device = device
            return self

        def eval(self):
            self.eval_called = True

    class _StubWrapper:
        _device = "cpu"

    calls = {}

    def _fake_from_pretrained(repo, **kwargs):
        calls["repo"] = repo
        calls["kwargs"] = kwargs
        return _StubModel()

    monkeypatch.setattr(
        "transformers.AutoModel.from_pretrained", _fake_from_pretrained
    )

    config = CRadioV4ModelConfig({"hf_revision": "deadbeef"})
    model = CRadioV4Model._load_model(_StubWrapper(), config)

    assert calls["repo"] == config.hf_repo
    assert calls["kwargs"]["trust_remote_code"] is True
    assert calls["kwargs"]["revision"] == "deadbeef"
    assert model.device == "cpu"
    assert model.eval_called


def test_load_image_processor_uses_hf_revision(monkeypatch):
    calls = {}

    def _fake_from_pretrained(repo, **kwargs):
        calls["repo"] = repo
        calls["kwargs"] = kwargs
        return object()

    monkeypatch.setattr(
        "transformers.CLIPImageProcessor.from_pretrained",
        _fake_from_pretrained,
    )

    config = CRadioV4ModelConfig({"hf_revision": "deadbeef"})
    CRadioV4Model._load_image_processor(None, config)

    assert calls["repo"] == config.hf_repo
    assert calls["kwargs"]["revision"] == "deadbeef"


def test_check_mixed_precision_support_handles_runtime_error(
    monkeypatch, caplog
):
    class _StubWrapper:
        _device = "cuda:0"
        _using_gpu = True

    def _raise_runtime_error(_device):
        raise RuntimeError("device query failed")

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        torch.cuda, "get_device_capability", _raise_runtime_error
    )

    with caplog.at_level(logging.WARNING, logger="fiftyone.utils.radio"):
        supported = CRadioV4Model._check_mixed_precision_support(
            _StubWrapper()
        )

    assert not supported
    assert "Could not determine mixed precision support" in caplog.text


# =============================================================================
# OUTPUT PROCESSOR TESTS - EXTENDED
# =============================================================================

def test_radio_proc_batch1():
    proc = RadioOutputProcessor()
    tensor = torch.randn(1, 2560)
    result = proc(tensor, (640, 480))
    assert len(result) == 1
    assert result[0].shape == (2560,)

def test_radio_proc_batch2():
    proc = RadioOutputProcessor()
    tensor = torch.randn(2, 2560)
    result = proc(tensor, [(640, 480), (800, 600)])
    assert len(result) == 2

def test_radio_proc_batch8():
    proc = RadioOutputProcessor()
    tensor = torch.randn(8, 2560)
    result = proc(tensor, [(100, 100)] * 8)
    assert len(result) == 8

def test_radio_proc_batch16():
    proc = RadioOutputProcessor()
    tensor = torch.randn(16, 2560)
    result = proc(tensor, [(100, 100)] * 16)
    assert len(result) == 16

def test_radio_proc_dim():
    proc = RadioOutputProcessor()
    for dim in [512, 1024, 2560, 3072]:
        tensor = torch.randn(1, dim)
        result = proc(tensor, (100, 100))
        assert result[0].shape == (dim,)

def test_radio_proc_dtype_float32():
    proc = RadioOutputProcessor()
    tensor = torch.randn(1, 2560).float()
    result = proc(tensor, (100, 100))
    assert result[0].dtype == np.float32

def test_radio_proc_dtype_float16():
    proc = RadioOutputProcessor()
    tensor = torch.randn(1, 2560).half()
    result = proc(tensor, (100, 100))
    # numpy converts to float32 or float16 depending on version
    assert result[0].dtype in [np.float32, np.float16]

def test_radio_proc_dtype_bfloat16():
    proc = RadioOutputProcessor()
    tensor = torch.randn(1, 2560).bfloat16()
    result = proc(tensor, (100, 100))
    assert result[0].dtype == np.float32  # bfloat16 converts to float32

def test_radio_proc_gpu():
    if not torch.cuda.is_available():
        return  # Skip if no GPU
    proc = RadioOutputProcessor()
    tensor = torch.randn(1, 2560).cuda()
    result = proc(tensor, (100, 100))
    assert isinstance(result[0], np.ndarray)

def test_radio_proc_numpy():
    proc = RadioOutputProcessor()
    arr = np.random.randn(3, 2560).astype(np.float32)
    result = proc(arr, [(100, 100)] * 3)
    assert len(result) == 3

def test_spatial_proc_nchw():
    proc = SpatialHeatmapOutputProcessor()
    tensor = torch.randn(1, 1280, 32, 32)
    result = proc(tensor, [(640, 480)])
    assert result[0].map.shape == (480, 640)

def test_spatial_proc_nchw_small():
    proc = SpatialHeatmapOutputProcessor()
    tensor = torch.randn(1, 512, 16, 16)
    result = proc(tensor, [(320, 240)])
    assert result[0].map.shape == (240, 320)

def test_spatial_proc_nchw_large():
    proc = SpatialHeatmapOutputProcessor()
    tensor = torch.randn(1, 2048, 64, 64)
    result = proc(tensor, [(1024, 768)])
    assert result[0].map.shape == (768, 1024)

def test_spatial_proc_nlc_256():
    proc = SpatialHeatmapOutputProcessor()
    tensor = torch.randn(1, 256, 1280)  # 16x16 patches
    result = proc(tensor, [(640, 480)])
    assert result[0].map.shape == (480, 640)

def test_spatial_proc_nlc_1024():
    proc = SpatialHeatmapOutputProcessor()
    tensor = torch.randn(1, 1024, 1280)  # 32x32 patches
    result = proc(tensor, [(640, 480)])
    assert result[0].map.shape == (480, 640)

def test_spatial_proc_nlc_nonsquare():
    proc = SpatialHeatmapOutputProcessor()
    tensor = torch.randn(1, 512, 1280)  # 32x16 or 16x32
    result = proc(tensor, [(640, 480)])
    assert result[0].map.shape == (480, 640)


def test_spatial_proc_prime_tokens_warns(caplog):
    proc = SpatialHeatmapOutputProcessor()
    tensor = torch.randn(1, 509, 1280)

    with caplog.at_level(logging.WARNING, logger="fiftyone.utils.radio"):
        result = proc(tensor, [(640, 480)])

    assert result[0].map.shape == (480, 640)
    assert "Prime token count 509 produced a 1x509 spatial layout" in caplog.text

def test_spatial_proc_dtype():
    proc = SpatialHeatmapOutputProcessor()
    tensor = torch.randn(1, 512, 16, 16)
    result = proc(tensor, [(320, 240)])
    assert result[0].map.dtype == np.uint8

def test_spatial_proc_range():
    proc = SpatialHeatmapOutputProcessor()
    tensor = torch.randn(1, 512, 16, 16)
    result = proc(tensor, [(320, 240)])
    assert result[0].map.min() >= 0
    assert result[0].map.max() <= 255

def test_spatial_proc_range_attr():
    proc = SpatialHeatmapOutputProcessor()
    tensor = torch.randn(1, 512, 16, 16)
    result = proc(tensor, [(320, 240)])
    assert result[0].range == [0, 255]

def test_spatial_proc_smoothing_effect():
    tensor = torch.randn(1, 512, 16, 16)

    proc_smooth = SpatialHeatmapOutputProcessor(apply_smoothing=True, smoothing_sigma=2.0)
    proc_no_smooth = SpatialHeatmapOutputProcessor(apply_smoothing=False)

    result_smooth = proc_smooth(tensor.clone(), [(320, 240)])
    result_no_smooth = proc_no_smooth(tensor.clone(), [(320, 240)])

    # Smoothed version should have lower variance in local regions
    # (just check they're different)
    assert not np.array_equal(result_smooth[0].map, result_no_smooth[0].map)

def test_spatial_proc_all_nan():
    proc = SpatialHeatmapOutputProcessor()
    tensor = torch.full((1, 512, 8, 8), float('nan'))
    result = proc(tensor, [(100, 100)])
    assert not np.isnan(result[0].map).any()

def test_spatial_proc_all_inf():
    proc = SpatialHeatmapOutputProcessor()
    tensor = torch.full((1, 512, 8, 8), float('inf'))
    result = proc(tensor, [(100, 100)])
    assert not np.isinf(result[0].map).any()

def test_spatial_proc_mixed_nan_inf():
    proc = SpatialHeatmapOutputProcessor()
    tensor = torch.randn(1, 512, 8, 8)
    tensor[0, :10, 0, 0] = float('nan')
    tensor[0, 10:20, 1, 1] = float('inf')
    tensor[0, 20:30, 2, 2] = float('-inf')
    result = proc(tensor, [(100, 100)])
    assert not np.isnan(result[0].map).any()
    assert not np.isinf(result[0].map).any()

def test_spatial_proc_constant():
    proc = SpatialHeatmapOutputProcessor()
    tensor = torch.ones(1, 512, 8, 8)
    result = proc(tensor, [(100, 100)])
    # Constant input should produce zeros (no variation to visualize)
    assert result[0].map.shape == (100, 100)

def test_spatial_proc_batch_diff_sizes():
    proc = SpatialHeatmapOutputProcessor()
    tensor = torch.randn(4, 512, 16, 16)
    sizes = [(640, 480), (800, 600), (1024, 768), (320, 240)]
    result = proc(tensor, sizes)
    assert len(result) == 4
    assert result[0].map.shape == (480, 640)
    assert result[1].map.shape == (600, 800)
    assert result[2].map.shape == (768, 1024)
    assert result[3].map.shape == (240, 320)


# =============================================================================
# INFERENCE TESTS - EXTENDED
# =============================================================================

def test_infer_rgb():
    config = CRadioV4ModelConfig({"output_type": "summary"})
    model = CRadioV4Model(config)
    img = Image.new("RGB", (512, 512), color=(128, 128, 128))
    with model:
        result = model._predict_all([img])
    assert result[0].shape == (2560,)

def test_infer_random():
    config = CRadioV4ModelConfig({"output_type": "summary"})
    model = CRadioV4Model(config)
    arr = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
    img = Image.fromarray(arr)
    with model:
        result = model._predict_all([img])
    assert result[0].shape == (2560,)

@pytest.mark.parametrize(
    "color",
    [(0, 0, 0), (255, 255, 255), (255, 0, 0)],
    ids=["black", "white", "red"],
)
def test_infer_uniform_color(color):
    config = CRadioV4ModelConfig({"output_type": "summary"})
    model = CRadioV4Model(config)
    img = Image.new("RGB", (512, 512), color=color)
    with model:
        result = model._predict_all([img])
    assert result[0].shape == (2560,)

def test_infer_gradient():
    config = CRadioV4ModelConfig({"output_type": "summary"})
    model = CRadioV4Model(config)
    arr = np.zeros((512, 512, 3), dtype=np.uint8)
    for i in range(512):
        arr[i, :, :] = int(i / 2)
    img = Image.fromarray(arr)
    with model:
        result = model._predict_all([img])
    assert result[0].shape == (2560,)

@pytest.mark.parametrize(
    "size",
    [(128, 128), (256, 256), (512, 512), (1024, 1024), (1920, 1080), (3840, 2160)],
    ids=["128", "256", "512", "1024", "1080p", "4k"],
)
def test_infer_resolution(size):
    config = CRadioV4ModelConfig({"output_type": "summary"})
    model = CRadioV4Model(config)
    img = Image.new("RGB", size)
    with model:
        result = model._predict_all([img])
    assert result[0].shape == (2560,)

@pytest.mark.parametrize(
    "size",
    [(480, 640), (640, 480), (100, 1000), (1000, 100)],
    ids=["portrait", "landscape", "extreme-portrait", "extreme-landscape"],
)
def test_infer_aspect_ratio(size):
    config = CRadioV4ModelConfig({"output_type": "summary"})
    model = CRadioV4Model(config)
    img = Image.new("RGB", size)
    with model:
        result = model._predict_all([img])
    assert result[0].shape == (2560,)

@pytest.mark.parametrize(
    ("count", "image_size"),
    [(1, (512, 512)), (2, (512, 512)), (4, (512, 512)), (8, (256, 256))],
    ids=["1", "2", "4", "8"],
)
def test_infer_batch_size(count, image_size):
    config = CRadioV4ModelConfig({"output_type": "summary"})
    model = CRadioV4Model(config)
    imgs = [Image.new("RGB", image_size) for _ in range(count)]
    with model:
        result = model._predict_all(imgs)
    assert len(result) == count

def test_infer_batch_mixed():
    config = CRadioV4ModelConfig({"output_type": "summary"})
    model = CRadioV4Model(config)
    imgs = [
        Image.new("RGB", (256, 256)),
        Image.new("RGB", (512, 512)),
        Image.new("RGB", (640, 480)),
        Image.new("RGB", (1024, 768)),
    ]
    with model:
        result = model._predict_all(imgs)
    assert len(result) == 4
    for r in result:
        assert r.shape == (2560,)

def test_infer_spatial_single():
    config = CRadioV4ModelConfig({"output_type": "spatial"})
    model = CRadioV4Model(config)
    img = Image.new("RGB", (640, 480))
    with model:
        result = model._predict_all([img])
    assert isinstance(result[0], fol.Heatmap)
    assert result[0].map.shape == (480, 640)

def test_infer_spatial_batch():
    config = CRadioV4ModelConfig({"output_type": "spatial"})
    model = CRadioV4Model(config)
    imgs = [
        Image.new("RGB", (320, 240)),
        Image.new("RGB", (640, 480)),
        Image.new("RGB", (800, 600)),
    ]
    with model:
        result = model._predict_all(imgs)
    assert len(result) == 3
    assert result[0].map.shape == (240, 320)
    assert result[1].map.shape == (480, 640)
    assert result[2].map.shape == (600, 800)

def test_infer_spatial_aspect():
    config = CRadioV4ModelConfig({"output_type": "spatial"})
    model = CRadioV4Model(config)
    # 16:9 aspect
    img = Image.new("RGB", (1920, 1080))
    with model:
        result = model._predict_all([img])
    assert result[0].map.shape == (1080, 1920)

def test_infer_different_images():
    config = CRadioV4ModelConfig({"output_type": "summary"})
    model = CRadioV4Model(config)
    img1 = Image.new("RGB", (512, 512), color=(255, 0, 0))
    img2 = Image.new("RGB", (512, 512), color=(0, 255, 0))
    with model:
        result = model._predict_all([img1, img2])
    # Different images should produce different embeddings
    assert not np.allclose(result[0], result[1])

def test_infer_same_image():
    config = CRadioV4ModelConfig({"output_type": "summary"})
    model = CRadioV4Model(config)
    img = Image.new("RGB", (512, 512), color=(128, 128, 128))
    with model:
        result1 = model._predict_all([img])
        result2 = model._predict_all([img])
    assert np.allclose(result1[0], result2[0], rtol=1e-4, atol=1e-6)


# =============================================================================
# FIFTYONE INTEGRATION TESTS - EXTENDED
# =============================================================================

def test_fo_model_type():
    config = CRadioV4ModelConfig({})
    model = CRadioV4Model(config)
    assert isinstance(model, fom.Model)

def test_fo_embeddings_field():
    dataset = foz.load_zoo_dataset("quickstart", max_samples=3)
    try:
        config = CRadioV4ModelConfig({"output_type": "summary"})
        model = CRadioV4Model(config)

        dataset.compute_embeddings(model, embeddings_field="my_embeddings")

        assert "my_embeddings" in dataset.first().field_names
    finally:
        fo.delete_dataset(dataset.name)

def test_fo_embeddings_type():
    dataset = foz.load_zoo_dataset("quickstart", max_samples=3)
    try:
        config = CRadioV4ModelConfig({"output_type": "summary"})
        model = CRadioV4Model(config)

        dataset.compute_embeddings(model, embeddings_field="test_emb")

        sample = dataset.first()
        emb = np.array(sample.test_emb)
        assert isinstance(emb, np.ndarray)
    finally:
        fo.delete_dataset(dataset.name)

def test_fo_embeddings_dim():
    dataset = foz.load_zoo_dataset("quickstart", max_samples=3)
    try:
        config = CRadioV4ModelConfig({"output_type": "summary"})
        model = CRadioV4Model(config)

        dataset.compute_embeddings(model, embeddings_field="test_emb")

        for sample in dataset:
            emb = np.array(sample.test_emb)
            assert emb.shape == (2560,)

    finally:
        fo.delete_dataset(dataset.name)

def test_fo_heatmap_type():
    dataset = foz.load_zoo_dataset("quickstart", max_samples=3)
    try:
        config = CRadioV4ModelConfig({"output_type": "spatial"})
        model = CRadioV4Model(config)

        dataset.apply_model(model, label_field="test_heat")

        for sample in dataset:
            assert isinstance(sample.test_heat, fol.Heatmap)

    finally:
        fo.delete_dataset(dataset.name)

def test_fo_heatmap_dims():
    dataset = foz.load_zoo_dataset("quickstart", max_samples=5)
    try:
        config = CRadioV4ModelConfig({"output_type": "spatial"})
        model = CRadioV4Model(config)

        dataset.apply_model(model, label_field="test_heat")

        for sample in dataset:
            img = Image.open(sample.filepath)
            w, h = img.size
            heat_h, heat_w = sample.test_heat.map.shape
            assert heat_w == w
            assert heat_h == h

    finally:
        fo.delete_dataset(dataset.name)

def test_fo_embeddings_view():
    dataset = foz.load_zoo_dataset("quickstart", max_samples=10)
    try:
        view = dataset.take(5)

        config = CRadioV4ModelConfig({"output_type": "summary"})
        model = CRadioV4Model(config)

        view.compute_embeddings(model, embeddings_field="test_emb")

        # Only view samples should have embeddings
        count_with_emb = len([s for s in dataset if s.test_emb is not None])
        assert count_with_emb == 5

    finally:
        fo.delete_dataset(dataset.name)

def test_fo_heatmap_view():
    dataset = foz.load_zoo_dataset("quickstart", max_samples=10)
    try:
        view = dataset.skip(3).take(4)

        config = CRadioV4ModelConfig({"output_type": "spatial"})
        model = CRadioV4Model(config)

        view.apply_model(model, label_field="test_heat")

        count_with_heat = len([s for s in dataset if s.test_heat is not None])
        assert count_with_heat == 4

    finally:
        fo.delete_dataset(dataset.name)

def test_fo_recompute_embeddings():
    dataset = foz.load_zoo_dataset("quickstart", max_samples=3)
    try:
        config = CRadioV4ModelConfig({"output_type": "summary"})
        model = CRadioV4Model(config)

        # Compute twice
        dataset.compute_embeddings(model, embeddings_field="test_emb")
        first_emb = np.array(dataset.first().test_emb).copy()

        dataset.compute_embeddings(model, embeddings_field="test_emb")
        second_emb = np.array(dataset.first().test_emb)

        # Should be the same (deterministic)
        assert np.allclose(first_emb, second_emb, rtol=1e-4, atol=1e-6)

    finally:
        fo.delete_dataset(dataset.name)

def test_fo_sort_by_similarity():
    dataset = foz.load_zoo_dataset("quickstart", max_samples=15)
    try:
        config = CRadioV4ModelConfig({"output_type": "summary"})
        model = CRadioV4Model(config)

        dataset.compute_embeddings(model, embeddings_field="test_emb")
        fob.compute_similarity(dataset, embeddings="test_emb", brain_key="test_sim")

        # Sort by similarity to first sample
        query_id = dataset.first().id
        similar = dataset.sort_by_similarity(query_id, brain_key="test_sim", k=5)

        assert len(similar) == 5
        # First result should be the query itself
        assert similar.first().id == query_id

    finally:
        fo.delete_dataset(dataset.name)

def test_fo_uniqueness():
    dataset = foz.load_zoo_dataset("quickstart", max_samples=10)
    try:
        config = CRadioV4ModelConfig({"output_type": "summary"})
        model = CRadioV4Model(config)

        dataset.compute_embeddings(model, embeddings_field="test_emb")
        fob.compute_uniqueness(dataset, embeddings="test_emb")

        # Check uniqueness field exists
        for sample in dataset:
            assert hasattr(sample, 'uniqueness')
            assert 0 <= sample.uniqueness <= 1

    finally:
        fo.delete_dataset(dataset.name)

def test_fo_multiple_fields():
    dataset = foz.load_zoo_dataset("quickstart", max_samples=3)
    try:

        # Compute embeddings
        config_emb = CRadioV4ModelConfig({"output_type": "summary"})
        model_emb = CRadioV4Model(config_emb)
        dataset.compute_embeddings(model_emb, embeddings_field="radio_emb")

        # Compute heatmaps
        config_heat = CRadioV4ModelConfig({"output_type": "spatial"})
        model_heat = CRadioV4Model(config_heat)
        dataset.apply_model(model_heat, label_field="radio_heat")

        # Both should exist
        sample = dataset.first()
        assert sample.radio_emb is not None
        assert sample.radio_heat is not None

    finally:
        fo.delete_dataset(dataset.name)

def test_fo_shuffled():
    dataset = foz.load_zoo_dataset("quickstart", max_samples=10, shuffle=True, seed=42)
    try:
        config = CRadioV4ModelConfig({"output_type": "summary"})
        model = CRadioV4Model(config)

        dataset.compute_embeddings(model, embeddings_field="test_emb")

        for sample in dataset:
            assert sample.test_emb is not None
            assert np.array(sample.test_emb).shape == (2560,)

    finally:
        fo.delete_dataset(dataset.name)


# =============================================================================
# EDGE CASE TESTS - EXTENDED
# =============================================================================

def test_edge_gray_to_rgb():
    config = CRadioV4ModelConfig({"output_type": "summary"})
    model = CRadioV4Model(config)
    gray = Image.new("L", (512, 512), color=128)
    rgb = gray.convert("RGB")
    with model:
        result = model._predict_all([rgb])
    assert result[0].shape == (2560,)

def test_edge_rgba_to_rgb():
    config = CRadioV4ModelConfig({"output_type": "summary"})
    model = CRadioV4Model(config)
    rgba = Image.new("RGBA", (512, 512), color=(128, 64, 192, 128))
    rgb = rgba.convert("RGB")
    with model:
        result = model._predict_all([rgb])
    assert result[0].shape == (2560,)

def test_edge_palette_to_rgb():
    config = CRadioV4ModelConfig({"output_type": "summary"})
    model = CRadioV4Model(config)
    p = Image.new("P", (512, 512))
    rgb = p.convert("RGB")
    with model:
        result = model._predict_all([rgb])
    assert result[0].shape == (2560,)

def test_edge_1bit_to_rgb():
    config = CRadioV4ModelConfig({"output_type": "summary"})
    model = CRadioV4Model(config)
    bw = Image.new("1", (512, 512))
    rgb = bw.convert("RGB")
    with model:
        result = model._predict_all([rgb])
    assert result[0].shape == (2560,)

def test_edge_cmyk_to_rgb():
    config = CRadioV4ModelConfig({"output_type": "summary"})
    model = CRadioV4Model(config)
    cmyk = Image.new("CMYK", (512, 512))
    rgb = cmyk.convert("RGB")
    with model:
        result = model._predict_all([rgb])
    assert result[0].shape == (2560,)

def test_edge_min_size():
    config = CRadioV4ModelConfig({"output_type": "summary"})
    model = CRadioV4Model(config)
    img = Image.new("RGB", (32, 32))
    with model:
        result = model._predict_all([img])
    assert result[0].shape == (2560,)

def test_edge_odd_dims():
    config = CRadioV4ModelConfig({"output_type": "summary"})
    model = CRadioV4Model(config)
    img = Image.new("RGB", (511, 513))
    with model:
        result = model._predict_all([img])
    assert result[0].shape == (2560,)

def test_edge_prime_dims():
    config = CRadioV4ModelConfig({"output_type": "summary"})
    model = CRadioV4Model(config)
    img = Image.new("RGB", (509, 503))
    with model:
        result = model._predict_all([img])
    assert result[0].shape == (2560,)

def test_edge_power2_dims():
    config = CRadioV4ModelConfig({"output_type": "summary"})
    model = CRadioV4Model(config)
    with model:
        for size in [64, 128, 256, 512, 1024]:
            img = Image.new("RGB", (size, size))
            result = model._predict_all([img])
            assert result[0].shape == (2560,)

def test_edge_from_file():
    config = CRadioV4ModelConfig({"output_type": "summary"})
    model = CRadioV4Model(config)

    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    temp_path = tmp.name
    tmp.close()
    try:
        img = Image.new("RGB", (512, 512), color=(100, 150, 200))
        img.save(temp_path)

        # Load and process
        loaded = Image.open(temp_path).convert("RGB")
        with model:
            result = model._predict_all([loaded])
        loaded.close()

        assert result[0].shape == (2560,)
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)

def test_edge_jpeg():
    config = CRadioV4ModelConfig({"output_type": "summary"})
    model = CRadioV4Model(config)

    tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    temp_path = tmp.name
    tmp.close()
    try:
        img = Image.new("RGB", (512, 512), color=(100, 150, 200))
        img.save(temp_path, quality=85)

        loaded = Image.open(temp_path).convert("RGB")
        with model:
            result = model._predict_all([loaded])
        loaded.close()

        assert result[0].shape == (2560,)
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)

def test_edge_model_reuse():
    config = CRadioV4ModelConfig({"output_type": "summary"})
    model = CRadioV4Model(config)

    with model:
        for i in range(5):
            img = Image.new("RGB", (256, 256), color=(i*50, i*50, i*50))
            result = model._predict_all([img])
            assert result[0].shape == (2560,)

def test_edge_spatial_tiny():
    config = CRadioV4ModelConfig({"output_type": "spatial"})
    model = CRadioV4Model(config)
    img = Image.new("RGB", (64, 64))
    with model:
        result = model._predict_all([img])
    assert result[0].map.shape == (64, 64)

def test_edge_spatial_large():
    config = CRadioV4ModelConfig({"output_type": "spatial"})
    model = CRadioV4Model(config)
    img = Image.new("RGB", (2048, 1536))
    with model:
        result = model._predict_all([img])
    assert result[0].map.shape == (1536, 2048)


# =============================================================================
# RUN ALL TESTS
# =============================================================================

if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
