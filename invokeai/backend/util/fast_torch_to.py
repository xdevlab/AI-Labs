from typing import Optional, Union

import torch


def fast_torch_to(
    tensor: torch.Tensor,
    device: Optional[Union[torch.device, str]] = None,
    dtype: Optional[torch.dtype] = None,
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
    dst_device = torch.device(device) if device is not None else None

    device_type_is_changing = dst_device is not None and dst_device.type != tensor.device.type
    dtype_is_changing = dtype is not None and dtype != tensor.device

    # If both the device type and dtype are changing, then there's an opportunity to improve speed by deliberately
    # ordering these operations.
    if device_type_is_changing and dtype_is_changing:
        # Note that if the tensor is being copied, this must happen on the first operation.
        if tensor.device.type == "cuda":
            return tensor.to(dtype=dtype, copy=copy).to(device=device)
        elif dst_device.type == "cuda":
            return tensor.to(device=device, copy=copy).to(dtype=dtype)

        # TODO: We currently only apply this optimization for "cuda" device types. The same optimization may also be
        # beneficial for "mps" device types, but this hasn't been tested yet. Also, keep in mind that MPS does not
        # support bfloat16.

    return tensor.to(device=device, dtype=dtype, copy=copy)
