import itertools
import time

import numpy as np
import pytest
import torch

from invokeai.backend.util.fast_torch_to import fast_torch_to


@pytest.mark.parametrize("src_device", ["cpu", "cuda"])
@pytest.mark.parametrize("src_dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("dst_device", ["cpu", "cuda", None])
@pytest.mark.parametrize("dst_dtype", [torch.float32, torch.float16, torch.bfloat16, None])
@pytest.mark.parametrize("copy", [True, False])
def test_fast_torch_to_correctness(src_device, src_dtype, dst_device, dst_dtype, copy):
    """Test that fast_torch_to(...) produces the same result as directly calling Tensor.to(...).

    Note that this test function is run with the cross-product of all parameter values
    (i.e. 2 * 3 * 3 * 4 * 2 = 144 variations).
    """
    if not torch.cuda.is_available() and (src_device == "cuda" or dst_device == "cuda"):
        pytest.skip("Requires CUDA device.")

    dim_1 = 5
    dim_2 = 6

    # Prepare matching inputs for both the 'fast' and 'slow' implementations.
    fast_input = torch.randn((dim_1, dim_2), device=src_device, dtype=src_dtype)
    slow_input = fast_input.detach().clone()
    torch.testing.assert_close(fast_input, slow_input, check_device=True, check_dtype=True)
    assert fast_input is not slow_input

    # Run fast_torch_to(...) and Tensor.to(...) on the same inputs.
    fast_output = fast_torch_to(fast_input, device=dst_device, dtype=dst_dtype, copy=copy)
    slow_output = slow_input.to(device=dst_device, dtype=dst_dtype, copy=copy)

    # Verify that fast_torch_to(...) and Tensor.to(...) produce the same results.
    torch.testing.assert_close(fast_input, slow_input, check_device=True, check_dtype=True)
    torch.testing.assert_close(fast_output, slow_output, check_device=True, check_dtype=True)
    assert (fast_input is fast_output) == (slow_input is slow_output)


@pytest.mark.parametrize("src_device", ["cpu", "cuda"])
@pytest.mark.parametrize("src_dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("dst_device", ["cpu", "cuda", None])
@pytest.mark.parametrize("dst_dtype", [torch.float32, torch.float16, torch.bfloat16, None])
@pytest.mark.parametrize("copy", [True, False])
def test_fast_torch_to_no_speed_regression(src_device, src_dtype, dst_device, dst_dtype, copy):
    """Test that fast_torch_to(...) is as fast or faster than Tensor.to(...) for a suite of test parameters that we care
    about.

    Note that this test function is run with the cross-product of all parameter values
    (i.e. 2 * 3 * 3 * 4 * 2 = 144 variations).
    """

    dim = 5000
    fast_times = []
    slow_times = []
    for _ in range(4):
        # Prepare matching inputs for fast_torch_to(...) and Tensor.to(...).
        slow_input = torch.randn((dim, dim), device=src_device, dtype=src_dtype)
        fast_input = slow_input.detach().clone()

        # Run Tensor.to(...) and record the time taken.
        torch.cuda.synchronize()
        slow_start = time.time()
        slow_input.to(device=dst_device, dtype=dst_dtype, copy=copy)
        torch.cuda.synchronize()
        slow_times.append(time.time() - slow_start)

        # Run fast_torch_to(...) and record the time taken.
        torch.cuda.synchronize()
        fast_start = time.time()
        fast_torch_to(fast_input, device=dst_device, dtype=dst_dtype, copy=copy)
        torch.cuda.synchronize()
        fast_times.append(time.time() - fast_start)

    # Drop first run as a CUDA warm-up.
    mean_fast_time = np.mean(fast_times[1:])
    mean_slow_time = np.mean(slow_times[1:])
    change = mean_fast_time - mean_slow_time

    # We use an absolute tolerance in addition to a relative tolerance, because some conversions are very fast and
    # result in flaky relative comparisons.
    max_allowed_change = max(0.0001, 0.1 * mean_slow_time)
    assert change < max_allowed_change


def test_fast_torch_to_speed_improvement():
    dim = 2000
    total_slow = 0
    total_fast = 0
    for src_device, src_dtype, dst_device, dst_dtype, copy in itertools.product(
        ["cpu", "cuda"],
        [torch.float32, torch.float16, torch.bfloat16],
        ["cpu", "cuda"],
        [torch.float32, torch.float16, torch.bfloat16],
        [True, False],
    ):
        if (
            src_device == "cuda"
            and src_dtype == torch.float16
            and dst_device == "cpu"
            and dst_dtype == torch.bfloat16
            and copy == True
        ):
            pass
        slow_times = []
        fast_times = []
        for _ in range(6):
            # Prepare matching inputs for fast_torch_to(...) and Tensor.to(...).
            slow_input = torch.randn((dim, dim), device=src_device, dtype=src_dtype)
            fast_input = slow_input.detach().clone()

            # Run Tensor.to(...) and record the time taken.
            torch.cuda.synchronize()
            slow_start = time.time()
            slow_input.to(device=dst_device, dtype=dst_dtype, copy=copy)
            torch.cuda.synchronize()
            slow_times.append(time.time() - slow_start)

            # Run fast_torch_to(...) and record the time taken.
            torch.cuda.synchronize()
            fast_start = time.time()
            fast_torch_to(fast_input, device=dst_device, dtype=dst_dtype, copy=copy)
            torch.cuda.synchronize()
            fast_times.append(time.time() - fast_start)

        # Log all results for development purposes.
        slow_mean = np.mean(slow_times[1:])  # Skip first result as warm-up.
        fast_mean = np.mean(fast_times[1:])  # Skip first result as warm-up.
        print(
            f"src=({str(src_device): <5}, {str(src_dtype): <14}), dst=({str(dst_device): <5}, {str(dst_dtype): <16}),"
            f" copy={str(copy): <5}:   Tensor.to()={slow_mean:.5f}s,   fast_torch_to()={fast_mean:.5f}s,   change="
            f" {(fast_mean-slow_mean) / slow_mean * 100:6.2f}%"
        )
        total_slow += slow_mean
        total_fast += fast_mean

    # We expect a 20% reduction in total execution time.
    change = total_fast - total_slow
    assert change < (-0.5 * total_slow)
