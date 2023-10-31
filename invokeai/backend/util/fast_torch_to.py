from typing import Optional, Union

import torch

DTYPE_SIZES = {
    torch.float16: 16,
    torch.bfloat16: 16,
    torch.float32: 32,
}


def fast_torch_to(
    tensor: torch.Tensor,
    device: Union[torch.device, str],
    dtype: torch.dtype,
    copy: bool = False,
):
    """A helper function that is equivalent to calling torch's `Tensor.to(...)`, but offers a speed improvement in some
    cases.

    When a tensor is both being moved to a different device and cast to a new dtype, there is an opportunity to
    improve performance by specifying the order of these operations. This behaviour is discussed on this pytorch issue:
    https://github.com/pytorch/pytorch/issues/58812. This improvement has not been applied upstream in pytorch, because
    the optimal ordering depends on several factors. Anecdotally, it has been observed that for many (all?) InvokeAI
    use cases it is faster to perform dtype casts on CUDA devices than on the CPU.
    """
    dst_device = torch.device(device)
    src_tensor_size = DTYPE_SIZES[tensor.dtype]
    dst_tensor_size = DTYPE_SIZES[dtype]

    if tensor.device.type == "cpu" and dst_device.type == "gpu" and src_tensor_size < dst_tensor_size:
        return tensor.to(device=dst_device, copy=copy).to(dtype=dtype)

    if tensor.device.type == "cuda" and dst_device.type == "cpu" and src_tensor_size > dst_tensor_size:
        return tensor.to(dtype=dtype, copy=copy).to(device=dst_device)

    # if (
    #     dst_device.type != tensor.device.type
    #     and dtype != tensor.dtype
    #     and (tensor.device.type == "cuda" or dst_device.type == "cuda")
    # ):
    #     # The optimizations below are based around the following timings:
    #     # cast_on_gpu < cast_on_cpu << transfer_to_or_from_gpu
    #     #
    #     # This leads to the following optimizations:
    #     # 1. Tranferring data to/from a cuda device is significantly slower than casting a tensor's dtype. So, when
    #     #   we  are doing both, choose the operation ordering that minimizes the data bandwidth being transferred.
    #     # 2. Casting a tensor's dtype is slightly faster on a cuda device than on the cpu. So, when casting between
    #     #    dtypes of the same size (e.g. float16 and bloat16), prefer to do this on the cuda device.

    #     src_tensor_size = DTYPE_SIZES[tensor.dtype]
    #     dst_tensor_size = DTYPE_SIZES[dtype]

    #     if src_tensor_size < dst_tensor_size:
    #         # The dtype size is increasing.
    #         # Transfer first, then cast.
    #         if copy and
    #         return tensor.to(device=dst_device, copy=copy).to(dtype=dtype)
    #     elif src_tensor_size > dst_tensor_size:
    #         # The dtype size is decreasing. Cast first, then transfer.
    #         return tensor.to(dtype=dtype, copy=copy).to(device=dst_device)

    # # This outer condition is applied to maximize the fallback speed when either device or dtype is not being changed.
    # if dtype is not None and dst_device is not None:
    #     if (
    #         dst_device.type != tensor.device.type
    #         and dtype != tensor.dtype
    #         and (tensor.device.type == "cuda" or dst_device.type == "cuda")
    #     ):
    #         # The optimizations below are based around the following timings:
    #         # cast_on_gpu < cast_on_cpu << transfer_to_or_from_gpu
    #         #
    #         # This leads to the following optimizations:
    #         # 1. Tranferring data to/from a cuda device is significantly slower than casting a tensor's dtype. So, when
    #         #   we  are doing both, choose the operation ordering that minimizes the data bandwidth being transferred.
    #         # 2. Casting a tensor's dtype is slightly faster on a cuda device than on the cpu. So, when casting between
    #         #    dtypes of the same size (e.g. float16 and bloat16), prefer to do this on the cuda device.

    #         src_tensor_size = DTYPE_SIZES[tensor.dtype]
    #         dst_tensor_size = DTYPE_SIZES[dtype]

    #         if src_tensor_size < dst_tensor_size:
    #             # The dtype size is increasing. Transfer first, then cast.
    #             return tensor.to(device=dst_device, copy=copy).to(dtype=dtype)
    #         elif src_tensor_size > dst_tensor_size:
    #             # The dtype size is decreasing. Cast first, then transfer.
    #             return tensor.to(dtype=dtype, copy=copy).to(device=dst_device)

    # # If moving from "cuda" to "cpu".
    # if tensor.device.type == "cuda" and dst_device.type == "cpu":
    #     if src_tensor_size < dst_tensor_size:
    #         # The dtype size is increasing. Transfer first, then cast.
    #         return tensor.to(device=device, copy=copy).to(dtype=dtype)
    #     elif src_tensor_size > dst_tensor_size:
    #         # The dtype size is decreasing. Cast first, then transfer.
    #         return tensor.to(dtype=dtype, copy=copy).to(device=device)
    #     else:
    #         # The dtype size is not changing. Cast on the GPU, then transfer.
    #         return tensor.to(dtype=dtype, copy=copy).to(device=device)
    # # If moving from "cpu" to "cuda".
    # elif tensor.device.type == "cpu" and dst_device.type == "cuda":
    #     if src_tensor_size < dst_tensor_size:
    #         # The dtype size is increasing. Transfer first, then cast.
    #         return tensor.to(device=device, copy=copy).to(dtype=dtype)
    #     elif src_tensor_size > dst_tensor_size:
    #         # The dtype size is decreasing. Cast first, then transfer.
    #         return tensor.to(dtype=dtype, copy=copy).to(device=device)
    #     else:
    #         # The dtype size is not changing. Transfer, then cast on the GPU.
    #         return tensor.to(device=device, copy=copy).to(dtype=dtype)

    # dst_device = torch.device(device) if device is not None else None

    # device_type_is_changing = dst_device is not None and dst_device.type != tensor.device.type
    # dtype_is_changing = dtype is not None and dtype != tensor.dtype

    # # If both the device type and dtype are changing, then there's an opportunity to improve speed by deliberately
    # # ordering these operations.
    # if device_type_is_changing and dtype_is_changing:
    #     # Note that if the tensor is being copied, this must happen on the first operation.
    #     if tensor.device.type == "cuda":
    #         return tensor.to(dtype=dtype, copy=copy).to(device=device)
    #     elif dst_device.type == "cuda":
    #         return tensor.to(device=device, copy=copy).to(dtype=dtype)

    #     # TODO: We currently only apply this optimization for "cuda" device types. The same optimization may also be
    #     # beneficial for "mps" device types, but this hasn't been tested yet. Also, keep in mind that MPS does not
    #     # support bfloat16.

    return tensor.to(device=dst_device, dtype=dtype, copy=copy)
