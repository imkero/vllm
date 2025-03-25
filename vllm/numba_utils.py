# SPDX-License-Identifier: Apache-2.0

from numba import types
from numba.extending import intrinsic
from numba.core.cgutils import raw_memcpy

@intrinsic
def numba_memcpy(typingctx, dst, dst_offset, src, src_offset, elem_count):
    assert dst.dtype == src.dtype

    def codegen(context, builder, signature, args):
        dst, dst_offset, src, src_offset, elem_count = args

        dst_ptr = builder.gep(dst, [dst_offset])
        src_ptr = builder.gep(src, [src_offset])
        item_size = context.get_abi_sizeof(dst_ptr.type)

        raw_memcpy(builder, dst_ptr, src_ptr, elem_count, item_size)

        return context.get_dummy_value()
    
    sig = types.void(
        types.CPointer(dst.dtype),
        dst_offset,
        types.CPointer(dst.dtype),
        src_offset,
        elem_count,
    )
    
    return sig, codegen

from numba import jit

@jit(nopython=True, inline="always")
def numba_replicate_exponential(arr, start_pos, repeat_elem_num, repeat_times):
    """
    repeat by exponential

    e.g.:
    1. arr[start_pos : start_pos + r * 1] = arr[start_pos - r : start_pos]
    2. arr[start_pos + r * 1 : start_pos + r * 3] = arr[start_pos - r : start_pos + r * 1]
    3. arr[start_pos + r * 3 : start_pos + r * 7] = arr[start_pos - r : start_pos + r * 3]
    ...
    """
    num_elem_left = repeat_times * repeat_elem_num
    num_step_elem = repeat_elem_num
    
    while num_step_elem <= num_elem_left:
        numba_memcpy(arr, start_pos, arr, start_pos - num_step_elem, num_step_elem)
        start_pos += num_step_elem
        num_elem_left -= num_step_elem
        num_step_elem <<= 1 # num_step_elem *= 2
    
    if num_elem_left > 0:
        numba_memcpy(arr, start_pos, arr, start_pos - num_elem_left, num_elem_left)
