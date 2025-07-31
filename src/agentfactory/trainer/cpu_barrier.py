# cpu_barrier.py
import torch
import torch.distributed as dist
from datetime import timedelta

class CpuBarrier:
    """
    Host-side barrier for single-node multi-GPU training.
    As long as each rank enters and exits the context once, heavy CUDA
    initialization will only be executed on rank-0.
    """
    _gloo_group = None          # class-level cache

    def __init__(self, accelerator, timeout_sec: int = 1800):
        self.acc     = accelerator
        self.timeout = timeout_sec
        # Build the gloo group on every rank beforehand
        self.group   = self._get_or_create_gloo_group()

    # ---------- helpers ----------
    def _need_sync(self) -> bool:
        return self.acc.num_processes > 1 and dist.is_initialized()

    def _get_or_create_gloo_group(self):
        if not self._need_sync():
            return None

        # If the main backend is already gloo, use the WORLD group directly.
        if dist.get_backend() == "gloo":
            return dist.group.WORLD

        # Otherwise, create a dedicated gloo CPU group when using another
        # GPU backend (e.g. NCCL).
        if CpuBarrier._gloo_group is None:
            CpuBarrier._gloo_group = dist.new_group(backend="gloo")
        return CpuBarrier._gloo_group

    # ---------- context-manager ----------
    def __enter__(self):
        if not self._need_sync():
            return self

        if not self.acc.is_main_process:          # Only non-main ranks enter first
            dist.monitored_barrier(
                group   = self.group,
                timeout = timedelta(seconds=self.timeout),
            )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self._need_sync():
            return False

        if self.acc.is_main_process:              # Only the main rank calls once here
            dist.monitored_barrier(
                group   = self.group,
                timeout = timedelta(seconds=self.timeout),
            )
        # Let exceptions propagate normally
        return False
