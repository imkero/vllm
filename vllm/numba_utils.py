# SPDX-License-Identifier: Apache-2.0

from ctypes import c_void_p
from typing import overload

from numba import types, jit
from numba.extending import intrinsic
from numba.core.cgutils import raw_memcpy


@overload
def numba_array_memcpy(dst_arr: c_void_p, dst_offset: int, src_arr: c_void_p, src_offset: int, elem_count: int) -> None:
    ...

@intrinsic(inline=True)
def numba_array_memcpy(typingctx, dst_arr, dst_offset, src_arr, src_offset, elem_count):
    """calling C memcpy for numpy array in numba no-python code"""

    assert dst_arr.dtype == src_arr.dtype, \
        f"dst_arr and src_arr must have the same dtype"

    def codegen(context, builder, signature, args):
        dst, dst_offset, src, src_offset, elem_count = args

        dst_ptr = builder.gep(dst, [dst_offset])
        src_ptr = builder.gep(src, [src_offset])
        item_size = context.get_abi_sizeof(dst_ptr.type)

        raw_memcpy(builder, dst_ptr, src_ptr, elem_count, item_size)

        return context.get_dummy_value()
    
    sig = types.void(
        types.CPointer(dst_arr.dtype),
        dst_offset,
        types.CPointer(dst_arr.dtype),
        src_offset,
        elem_count,
    )
    
    return sig, codegen

@jit(nopython=True, inline="always")
def numba_cdiv(a, b):
    """ceiling division"""
    return -(-a // b)
