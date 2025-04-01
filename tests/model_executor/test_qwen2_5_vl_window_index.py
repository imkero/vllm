import torch
import pytest

from vllm.model_executor.models.qwen2_5_vl import Qwen2_5_VisionAttentionScheduler

@pytest.mark.parametrize("window_size, patch_size, spatial_merge_size", [
    (112, 14, 2),
    (128, 16, 2),
])
def test_qwen2_5_vl_get_window_indices_correctness(window_size, patch_size, spatial_merge_size):
    scheduler = Qwen2_5_VisionAttentionScheduler(
        spatial_merge_size=spatial_merge_size,
        window_size=window_size,
        patch_size=patch_size,
        max_position_embeddings=32768,
        device=torch.device("cpu"),
    )

    for t in range(1, 3):
        for h in range(1, 50):
            for w in range(1, 50):
                grid_thw = torch.tensor(
                    [[t, h * spatial_merge_size, w * spatial_merge_size]],
                    dtype=torch.int64,
                )
                
                (
                    window_indices_torch, reverse_indices_torch,
                    seqlens_full_torch, seqlens_window_torch,
                    cu_seqlens_full_torch, cu_seqlens_window_torch,
                ) = scheduler.generate_by_torch(grid_thw)

                (
                    window_indices_numba, reverse_indices_numba,
                    seqlens_full_numba, seqlens_window_numba, 
                    cu_seqlens_full_numba, cu_seqlens_window_numba,
                ) = scheduler.generate_by_numba(grid_thw)

                get_assertion_msg = lambda: f"mismatch at grid_thw={grid_thw}"

                assert window_indices_torch.dtype == window_indices_numba.dtype, get_assertion_msg()
                assert reverse_indices_torch.dtype == reverse_indices_numba.dtype, get_assertion_msg()
                assert seqlens_full_torch.dtype == seqlens_full_numba.dtype, get_assertion_msg()
                assert seqlens_window_torch.dtype == seqlens_window_numba.dtype, get_assertion_msg()
                assert cu_seqlens_full_torch.dtype == cu_seqlens_full_numba.dtype, get_assertion_msg()
                assert cu_seqlens_window_torch.dtype == cu_seqlens_window_numba.dtype, get_assertion_msg()

                assert window_indices_torch.shape == window_indices_numba.shape, get_assertion_msg()
                assert reverse_indices_torch.shape == reverse_indices_numba.shape, get_assertion_msg()
                assert seqlens_full_torch.shape == seqlens_full_numba.shape, get_assertion_msg()
                assert seqlens_window_torch.shape == seqlens_window_numba.shape, get_assertion_msg()
                assert cu_seqlens_full_torch.shape == cu_seqlens_full_numba.shape, get_assertion_msg()
                assert cu_seqlens_window_torch.shape == cu_seqlens_window_numba.shape, get_assertion_msg()

                assert torch.equal(window_indices_torch, window_indices_numba), get_assertion_msg()
                assert torch.equal(reverse_indices_torch, reverse_indices_numba), get_assertion_msg()
                assert torch.equal(seqlens_full_torch, seqlens_full_numba), get_assertion_msg()
                assert torch.equal(seqlens_window_torch, seqlens_window_numba), get_assertion_msg()
                assert torch.equal(cu_seqlens_full_torch, cu_seqlens_full_numba), get_assertion_msg()
                assert torch.equal(cu_seqlens_window_torch, cu_seqlens_window_numba), get_assertion_msg()
