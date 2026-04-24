import numpy as np
import torch
from PIL import Image
from FACE_JOB.shared.datasets.emotion_combined import (
    EmotionSample,
    CombinedEmotionDS,
    BalancedSourceSampler,
)


def test_combined_ds_preserves_source_tag():
    im = Image.new("RGB", (64, 64), color=(128, 128, 128))
    samples = [
        EmotionSample(image=im, label=0, source="ferplus"),
        EmotionSample(image=im, label=1, source="rafdb"),
        EmotionSample(image=im, label=2, source="expw"),
    ]
    ds = CombinedEmotionDS(samples, img_size=64, mode="eval")
    assert len(ds) == 3
    tensor, label, source_idx = ds[1]
    assert tensor.shape == (3, 64, 64)
    assert label == 1
    # source index 1 = rafdb (ordering: ferplus=0, rafdb=1, expw=2)
    assert source_idx == 1


def test_balanced_sampler_draws_from_all_sources():
    samples = (
        [EmotionSample(None, 0, "ferplus")] * 100
        + [EmotionSample(None, 0, "rafdb")] * 10
        + [EmotionSample(None, 0, "expw")] * 500
    )
    sampler = BalancedSourceSampler(samples, batch_size=30, num_batches=50)
    indices = list(sampler)
    assert len(indices) == 30 * 50
    # For each batch of 30 draws, expect 10 from each source (balanced within batch).
    batch = indices[:30]
    sources = [samples[i].source for i in batch]
    assert sources.count("ferplus") == 10
    assert sources.count("rafdb") == 10
    assert sources.count("expw") == 10
