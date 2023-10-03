import ctypes

from invokeai.backend.model_management.model_cache import MemorySnapshot, get_pretty_snapshot_diff
from invokeai.backend.model_management.models.base import BaseModelType, ModelType, SubModelType
from invokeai.backend.util.test_utils import install_and_load_model, model_installer, slow, torch_device


class MALLINFO2(ctypes.Structure):
    """
    https://man7.org/linux/man-pages/man3/mallinfo.3.html

    struct mallinfo2 {
        size_t arena;     /* Non-mmapped space allocated (bytes) */
        size_t ordblks;   /* Number of free chunks */
        size_t smblks;    /* Number of free fastbin blocks */
        size_t hblks;     /* Number of mmapped regions */
        size_t hblkhd;    /* Space allocated in mmapped regions (bytes) */
        size_t usmblks;   /* See below */
        size_t fsmblks;   /* Space in freed fastbin blocks (bytes) */
        size_t uordblks;  /* Total allocated space (bytes) */
        size_t fordblks;  /* Total free space (bytes) */
        size_t keepcost;  /* Top-most, releasable space (bytes) */
    };
    """

    _fields_ = [
        ("arena", ctypes.c_size_t),
        ("ordblks", ctypes.c_size_t),
        ("smblks", ctypes.c_size_t),
        ("hblks", ctypes.c_size_t),
        ("hblkhd", ctypes.c_size_t),
        ("usmblks", ctypes.c_size_t),
        ("fsmblks", ctypes.c_size_t),
        ("uordblks", ctypes.c_size_t),
        ("fordblks", ctypes.c_size_t),
        ("keepcost", ctypes.c_size_t),
    ]


def print_mallinfo2(libc):
    mallinfo2 = libc.mallinfo2
    mallinfo2.restype = MALLINFO2
    info = mallinfo2()

    msg = ""
    msg += (
        f"{'arena': <10}= {(info.arena/2**30):15.5f}   /* Non-mmapped space allocated (GB). (uordblks + fordblks) */\n"
    )
    msg += f"{'ordblks': <10}= {(info.ordblks): >15}   /* Number of free chunks */\n"
    msg += f"{'smblks': <10}= {(info.smblks): >15}   /* Number of free fastbin blocks */\n"
    msg += f"{'hblks': <10}= {(info.hblks): >15}   /* Number of mmapped regions */\n"
    msg += f"{'hblkhd': <10}= {(info.hblkhd/2**30):15.5f}   /* Space allocated in mmapped regions (GB) */\n"
    msg += f"{'usmblks': <10}= {(info.usmblks): >15}   /* Unused */\n"
    msg += f"{'fsmblks': <10}= {(info.fsmblks/2**30):15.5f}   /* Space in freed fastbin blocks (GB) */\n"
    msg += f"{'uordblks': <10}= {(info.uordblks/2**30):15.5f}   /* Total allocated space (GB) */\n"
    msg += f"{'fordblks': <10}= {(info.fordblks/2**30):15.5f}   /* Total free space (GB) */\n"
    msg += f"{'keepcost': <10}= {(info.keepcost/2**30):15.5f}   /* Top-most, releasable space (GB) */\n"
    print(msg)


@slow
def test_unet_memory_leak(model_installer, torch_device):
    libc = ctypes.cdll.LoadLibrary("libc.so.6")

    print("Start --------")
    # libc.malloc_stats()
    print_mallinfo2(libc)
    model_info = install_and_load_model(
        model_installer=model_installer,
        model_path_id_or_url="TODO",
        model_name="stable-diffusion-xl-base-1-0",
        base_model=BaseModelType.StableDiffusionXL,
        model_type=ModelType.Main,
        submodel_type=SubModelType.UNet,
    )
    print("Load model into CPU --------")
    # libc.malloc_stats()
    print_mallinfo2(libc)

    with model_info as model:
        print("First load into GPU --------")
        # libc.malloc_stats()
        print_mallinfo2(libc)
        for i in range(10):
            print("***************************")
            print("Iteration 0")
            print("***************************")
            snapshot_before = MemorySnapshot.capture()
            model.to("cuda")
            snapshot_after = MemorySnapshot.capture()
            print(f"To CUDA {i}: {get_pretty_snapshot_diff(snapshot_before, snapshot_after)}.")
            print(f"{i}: CUDA --------")
            # libc.malloc_stats()
            print_mallinfo2(libc)

            snapshot_before = MemorySnapshot.capture()
            model.to("cpu")
            snapshot_after = MemorySnapshot.capture()
            print(f"To CPU {i}: {get_pretty_snapshot_diff(snapshot_before, snapshot_after)}.")
            print(f"{i}: CPU --------")
            # libc.malloc_stats()
            print_mallinfo2(libc)

    assert False
