from __future__ import annotations

import torch

from spAlignDE.alignment._hipt_feature_extractor import (
    _forward_all256,
    _forward_all4k,
)


class _TokenModel:
    def __init__(self, channels: int, patches: int):
        self.channels = channels
        self.patches = patches

    def get_intermediate_layers(self, batch, n=1):
        assert n == 1
        shape = (batch.shape[0], self.patches + 1, self.channels)
        return [torch.arange(torch.tensor(shape).prod()).reshape(shape).float()]


class _OfficialHiptShape:
    device256 = "cpu"
    device4k = "cpu"
    model256 = _TokenModel(channels=384, patches=256)
    model4k = _TokenModel(channels=192, patches=2)

    @staticmethod
    def prepare_img_tensor(image):
        return image, 1, 2


def test_official_hipt_token_api_produces_dense_feature_fields():
    model = _OfficialHiptShape()
    image = torch.zeros((1, 3, 256, 512))

    cls_field, subpatch_field = _forward_all256(model, image)
    assert cls_field.shape == (1, 384, 1, 2)
    assert subpatch_field.shape == (1, 384, 1, 2, 16, 16)

    _, context_field = _forward_all4k(model, cls_field)
    assert context_field.shape == (1, 192, 1, 2)
