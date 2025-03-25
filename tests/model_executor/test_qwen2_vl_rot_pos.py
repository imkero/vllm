from unittest.mock import Mock

import torch
import pytest

from vllm.model_executor.models.qwen2_vl import Qwen2VisionTransformer
from vllm.model_executor.models.qwen2_5_vl import Qwen2_5_VisionTransformer

@pytest.mark.parametrize("spatial_merge_size", [2, 3])
def test_qwen2_vl_rot_pos_correctness(dist_init, spatial_merge_size):
    vision_config = Mock(**{
        "depth": 32,
        "embed_dim": 1280,
        "mlp_ratio": 4,
        "num_heads": 16,
        "in_channels": 3,
        "hidden_size": 3584,
        "patch_size": 14,
        "spatial_merge_size": spatial_merge_size,
        "spatial_patch_size": 14,
        "temporal_patch_size": 2
    })

    vit = Qwen2VisionTransformer(
        vision_config=vision_config
    )

    for t in range(1, 3):
        for h in range(1, 100):
            for w in range(1, 100):
                grid_thw = torch.tensor([[t, h * spatial_merge_size, w * spatial_merge_size]], dtype=torch.int64)
                rot_pos_torch = vit.compute_rot_pos_torch(grid_thw)
                rot_pos_numba = vit.compute_rot_pos_numba(grid_thw.numpy(), spatial_merge_size)
                rot_pos_numba = torch.from_numpy(rot_pos_numba)
                assert rot_pos_torch.dtype == rot_pos_numba.dtype
                assert rot_pos_torch.shape == rot_pos_numba.shape
                assert torch.equal(rot_pos_torch, rot_pos_numba)

@pytest.mark.parametrize("spatial_merge_size", [2, 3])
def test_qwen2_5_vl_rot_pos_correctness(dist_init, spatial_merge_size):
    vision_config = Mock(**{
        "depth": 32,
        "hidden_act": "silu",
        "hidden_size": 1280,
        "intermediate_size": 3420,
        "num_heads": 16,
        "in_channels": 3,
        "out_hidden_size": 3584,
        "patch_size": 14,
        "spatial_merge_size": spatial_merge_size,
        "spatial_patch_size": 14,
        "window_size": 112,
        "fullatt_block_indexes": [
            7,
            15,
            23,
            31
        ],
        "tokens_per_second": 2,
        "temporal_patch_size": 2
    })

    vit = Qwen2_5_VisionTransformer(
        vision_config=vision_config
    )

    for t in range(1, 3):
        for h in range(1, 100):
            for w in range(1, 100):
                grid_thw = torch.tensor([[t, h * spatial_merge_size, w * spatial_merge_size]], dtype=torch.int64)
                rot_pos_torch = vit.compute_rot_pos_torch(grid_thw)
                rot_pos_numba = vit.compute_rot_pos_numba(grid_thw.numpy(), spatial_merge_size)
                rot_pos_numba = torch.from_numpy(rot_pos_numba)
                assert rot_pos_torch.dtype == rot_pos_numba.dtype
                assert rot_pos_torch.shape == rot_pos_numba.shape
                assert torch.equal(rot_pos_torch, rot_pos_numba)


@pytest.mark.parametrize("window_size, patch_size, spatial_merge_size", [
    (112, 14, 2),
    (128, 16, 2),
])
def test_qwen2_5_vl_get_window_index_correctness(dist_init, window_size, patch_size, spatial_merge_size):
    vision_config = Mock(**{
        "depth": 32,
        "hidden_act": "silu",
        "hidden_size": 1280,
        "intermediate_size": 3420,
        "num_heads": 16,
        "in_channels": 3,
        "out_hidden_size": 3584,
        "patch_size": patch_size,
        "spatial_merge_size": spatial_merge_size,
        "spatial_patch_size": patch_size,
        "window_size": window_size,
        "fullatt_block_indexes": [
            7,
            15,
            23,
            31
        ],
        "tokens_per_second": 2,
        "temporal_patch_size": 2
    })

    vit = Qwen2_5_VisionTransformer(
        vision_config=vision_config
    )

    for t in range(1, 3):
        for h in range(1, 100):
            for w in range(1, 100):
                grid_thw = torch.tensor([[t, h * spatial_merge_size, w * spatial_merge_size]], dtype=torch.int64)
                
                # torch impl
                window_index_torch, cu_window_seqlens_torch = vit.get_window_index_torch(
                    grid_thw=grid_thw,
                )
                cu_window_seqlens_torch = torch.tensor(cu_window_seqlens_torch, dtype=torch.int64)
                cu_window_seqlens_torch = torch.unique_consecutive(cu_window_seqlens_torch)
                window_seqlens_torch = cu_window_seqlens_torch[1:] - cu_window_seqlens_torch[:-1]
                cu_window_seqlens_torch = cu_window_seqlens_torch.to(torch.int32)
                reverse_indices_torch = torch.argsort(window_index_torch)

                # numba impl
                window_index_numba, reverse_indices_numba, window_seqlens_numba, cu_window_seqlens_numba = vit.get_window_index_numba(
                    grid_thw=grid_thw.numpy(),
                    window_size=window_size,
                    spatial_merge_size=spatial_merge_size,
                    patch_size=patch_size,
                )
                window_index_numba = torch.from_numpy(window_index_numba)
                reverse_indices_numba = torch.from_numpy(reverse_indices_numba)
                window_seqlens_numba = torch.from_numpy(window_seqlens_numba)
                cu_window_seqlens_numba = torch.from_numpy(cu_window_seqlens_numba)

                assert window_index_torch.dtype == window_index_numba.dtype
                assert cu_window_seqlens_torch.dtype == cu_window_seqlens_numba.dtype
                assert reverse_indices_torch.dtype == reverse_indices_numba.dtype
                assert window_seqlens_torch.dtype == window_seqlens_numba.dtype

                assert window_index_torch.shape == window_index_numba.shape
                assert cu_window_seqlens_torch.shape == cu_window_seqlens_numba.shape
                assert reverse_indices_torch.shape == reverse_indices_numba.shape
                assert window_seqlens_torch.shape == window_seqlens_numba.shape

                assert torch.equal(window_index_torch, window_index_numba)
                assert torch.equal(cu_window_seqlens_torch, cu_window_seqlens_numba)
                assert torch.equal(reverse_indices_torch, reverse_indices_numba)
                assert torch.equal(window_seqlens_torch, window_seqlens_numba)
