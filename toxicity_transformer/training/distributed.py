from __future__ import annotations

import os
import torch
import torch.distributed as dist

def is_dist_avail_and_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()

def get_rank() -> int:
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()

def get_world_size() -> int:
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()

def is_main_process() -> bool:
    return get_rank() == 0

def init_distributed() -> bool:
    # torchrun sets these environment variables
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
        torch.distributed.barrier()
        return True
    return False

def cleanup_distributed():
    if is_dist_avail_and_initialized():
        dist.destroy_process_group()
