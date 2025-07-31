# rl_packed_dataloader_v3_improved.py  ·  Deterministic packing + sentinel‑flag scheme (v3‑rev)

"""A slimmer, *Accelerate*‑friendly bin‑packing helper.

This revision reflects three new constraints (2025‑07‑04):
1. **No empty buckets** – After the initial FFD pass we *redistribute* the   
   smallest movable samples so every bucket holds ≥ 1 element.
2. **`world_size` is caller‑supplied** – no hidden autodetection.
3. **Rank handling is left to Accelerate** – sampler produces the *global*   
   plan; `DataLoaderShard` (inside ``accelerator.prepare``) takes care of   
   slicing batches per rank, so the sampler itself never queries rank.

Interface sketch
----------------
```python
raw_dataset = …
wrapper = SentinelWrapper(raw_dataset)
batch_sampler = BinPackBatchSampler(
    dataset=wrapper,
    max_capacity=128,
    num_big_batches=4,
    world_size=accelerator.state.num_processes,
)
batch_sampler.set_epoch(epoch)
data_loader = DataLoader(
    wrapper,
    batch_sampler=batch_sampler,
    collate_fn=wrap_sentinel_collate(base_collate),
)
data_loader = accelerator.prepare(data_loader)  # shard per rank
```
"""

from __future__ import annotations
import random
from typing import List, Sequence, Tuple, Optional, Callable, Any, NamedTuple
import torch
from torch.utils.data import Dataset, BatchSampler

# ---------------------------------------------------------------------------
# 0. Sample information structure
# ---------------------------------------------------------------------------

class SampleInfo(NamedTuple):
    """Encapsulates sample index and its sequence length."""
    index: int
    length: int

# ---------------------------------------------------------------------------
# 1.  Deterministic, *non‑empty* bin‑packing
# ---------------------------------------------------------------------------

def _redistribute_to_fill_empty_buckets(
    packed_buckets: List[List[SampleInfo]],
    max_capacity: int,
) -> None:
    """Post‑process *packed_buckets* so none are empty.
    
    Strategy – for each empty bucket pick the **shortest** sample from the
    **heaviest** donor bucket that can spare it (donor remains non‑empty and
    does not overflow capacity).
    """
    def bucket_load(bucket: List[SampleInfo]) -> int:
        return sum(sample.length for sample in bucket)
    
    empty_bucket_ids = [i for i, bucket in enumerate(packed_buckets) if not bucket]
    donor_bucket_ids = [
        i for i, bucket in sorted(enumerate(packed_buckets), key=lambda p: -len(p[1])) 
        if bucket
    ]
    
    for empty_id in empty_bucket_ids:
        sample_placed = False
        empty_bucket_current_load = bucket_load(packed_buckets[empty_id])
        
        for donor_id in donor_bucket_ids:
            if len(packed_buckets[donor_id]) == 1:
                continue  # donor would become empty
                        
            # pick shortest sample in donor bucket
            shortest_sample = min(packed_buckets[donor_id], key=lambda s: s.length)
            donor_bucket_current_load = bucket_load(packed_buckets[donor_id])
                        
            # check if transfer is feasible
            can_move = (
                shortest_sample.length + empty_bucket_current_load <= max_capacity and 
                donor_bucket_current_load - shortest_sample.length > 0
            )
                        
            if can_move:
                # transfer sample from donor to empty bucket
                packed_buckets[donor_id].remove(shortest_sample)
                packed_buckets[empty_id].append(shortest_sample)
                sample_placed = True
                break
                        
        if not sample_placed:
            raise RuntimeError("Unable to redistribute samples; data too small/skewed")


def deterministic_pack(
    samples: List[SampleInfo],
    max_capacity: int,
    world_size: int,
) -> List[List[List[SampleInfo]]]:
    """Deterministically pack *samples* into groups of ``world_size`` buckets.
    
    Returns a list of *groups* (super‑steps) containing SampleInfo objects.
    *Every* bucket is non‑empty.
    Raises ``ValueError`` if any sample > max_capacity.
    """
    # --- validation -----------------------------------------------------
    for sample in samples:
        if sample.length > max_capacity:
            raise ValueError(
                f"Sample {sample.index} length {sample.length} > capacity {max_capacity}"
            )
    
    # --- First‑Fit‑Decreasing (deterministic) ---------------------------
    # sort by length descending, then by index for deterministic tie-breaking
    sorted_samples = sorted(samples, key=lambda s: (-s.length, s.index))
        
    packed_buckets: List[List[SampleInfo]] = []
    
    def bucket_load(bucket: List[SampleInfo]) -> int:
        return sum(sample.length for sample in bucket)
    
    for sample in sorted_samples:
        target_bucket_idx = None
        best_load = max_capacity + 1
                
        # find best-fit bucket (lowest load that can accommodate the sample)
        for bucket_idx, bucket in enumerate(packed_buckets):
            current_load = bucket_load(bucket)
            if current_load + sample.length <= max_capacity and current_load < best_load:
                target_bucket_idx, best_load = bucket_idx, current_load
                        
        if target_bucket_idx is None:
            # spawn *world_size* fresh buckets in one go (may be empty for now)
            start_idx = len(packed_buckets)
            packed_buckets.extend([] for _ in range(world_size))
            target_bucket_idx = start_idx
                    
        packed_buckets[target_bucket_idx].append(sample)

    # --- validate bucket count (should never fail with correct logic) ---
    if len(packed_buckets) % world_size != 0:
        raise RuntimeError(
            f"Internal error: bucket count {len(packed_buckets)} not divisible by world_size {world_size}"
        )

    # --- move samples so no bucket empty --------------------------------
    _redistribute_to_fill_empty_buckets(packed_buckets, max_capacity)

    # --- sort buckets by load descending before grouping ---------------
    bucket_with_loads = [(bucket, bucket_load(bucket)) for bucket in packed_buckets]
    sorted_bucket_with_loads = sorted(bucket_with_loads, key=lambda x: -x[1])
    sorted_buckets = [bucket for bucket, _ in sorted_bucket_with_loads]
        
    # --- group sorted buckets -------------------------------------------
    bucket_groups: List[List[List[SampleInfo]]] = [
        sorted_buckets[i : i + world_size] for i in range(0, len(sorted_buckets), world_size)
    ]
        
    return bucket_groups

# ---------------------------------------------------------------------------
# 2.  Dataset → SentinelWrapper
# ---------------------------------------------------------------------------

class SentinelWrapper(Dataset):
    """Wrap *raw_dataset* and expose a single **extra index** for the sentinel.
    
    * ``len(wrapper) == len(raw_dataset)`` – sentinel not counted.
    * ``SENTINEL_IDX`` equals ``len(raw_dataset)``.
    """
    
    def __init__(self, raw_dataset: Dataset):
        self.raw_dataset = raw_dataset
                
        if hasattr(raw_dataset, "get_sequence_lengths"):
            self._cached_lengths: Optional[List[int]] = raw_dataset.get_sequence_lengths()
            if not isinstance(self._cached_lengths, list):
                raise TypeError("get_sequence_lengths() must return List[int]")
        else:
            self._cached_lengths = None
                    
        self.SENTINEL_IDX = len(raw_dataset)

    # ----- Dataset interface -------------------------------------------
    
    def __len__(self):
        return len(self.raw_dataset)

    def __getitem__(self, sample_idx):  # type: ignore[override]
        if sample_idx == self.SENTINEL_IDX:
            return {"is_last_marker": True}
        return self.raw_dataset[sample_idx]

    # ----- helpers ------------------------------------------------------
    
    def get_sample_length(self, sample_idx: int) -> int:
        """Get the sequence length for a specific sample index."""
        if sample_idx == self.SENTINEL_IDX:
            raise RuntimeError("get_sample_length() called on sentinel index")
                    
        if self._cached_lengths is not None:
            return self._cached_lengths[sample_idx]
                    
        sample_data = self.raw_dataset[sample_idx]
        if not hasattr(sample_data, "length"):
            raise AttributeError("Raw sample missing .length attribute")
        return sample_data.length

# ---------------------------------------------------------------------------
# 3.  BatchSampler (global plan; rank‑agnostic)
# ---------------------------------------------------------------------------

class BinPackBatchSampler(BatchSampler):
    """Deterministic bin‑packing *without* per‑rank logic.
    
    The sampler yields **one bucket per iteration** (list of indices). After
    ``accelerator.prepare``, each rank receives its shard automatically.
    """
    
    def __init__(
        self,
        *,
        dataset: SentinelWrapper,
        max_capacity: int,
        num_big_batches: int,
        world_size: int,
    ) -> None:
        self.dataset = dataset
        self.max_capacity = max_capacity
        self.num_big_batches = num_big_batches
        self.world_size = world_size
        self._global_plan: Optional[List[List[int]]] = None

    # ------------------------------------------------------------------
    
    def _create_epoch_plan(self, epoch: int):
        """Create the global batching plan for a specific epoch."""
        random.seed(epoch)  # Accelerate sets same seed on all ranks
                
        total_samples = len(self.dataset.raw_dataset)
        
        # create all samples with their lengths upfront
        all_samples = [
            SampleInfo(index=i, length=self.dataset.get_sample_length(i))
            for i in range(total_samples)
        ]
        
        # shuffle SampleInfo objects (preserves index-length binding)
        random.shuffle(all_samples)

        samples_per_big_batch = (total_samples + self.num_big_batches - 1) // self.num_big_batches
        sentinel_idx = self.dataset.SENTINEL_IDX

        all_packed_buckets: List[List[int]] = []

        for big_batch_idx in range(self.num_big_batches):
            start_idx = big_batch_idx * samples_per_big_batch
            end_idx = (big_batch_idx + 1) * samples_per_big_batch
            batch_samples = all_samples[start_idx:end_idx]
                        
            if not batch_samples:
                continue
            
            bucket_groups = deterministic_pack(
                batch_samples, self.max_capacity, self.world_size
            )

            # tag *last* group of this big‑batch with sentinel marker
            # convert SampleInfo buckets to index buckets
            for group_idx, group in enumerate(bucket_groups):
                index_group = []
                for bucket in group:
                    index_bucket = [sample.index for sample in bucket]
                    # add sentinel to last group
                    if group_idx == len(bucket_groups) - 1:
                        index_bucket.insert(0, sentinel_idx)
                    index_group.append(index_bucket)
                all_packed_buckets.extend(index_group)

        self._global_plan = all_packed_buckets

    # ------------------------------------------------------------------
    
    def set_epoch(self, epoch: int):
        """Set the epoch and regenerate the batching plan."""
        self._create_epoch_plan(epoch)

    # ------------------------ required API -----------------------------
    
    def __iter__(self):
        if self._global_plan is None:
            raise RuntimeError("set_epoch() must be called before iterating")
        yield from self._global_plan

    def __len__(self):
        if self._global_plan is None:
            raise RuntimeError("set_epoch() must be called before __len__()")
        return len(self._global_plan)

# ---------------------------------------------------------------------------
# 4.  Collate wrapper (sentinel handling)
# ---------------------------------------------------------------------------

def wrap_sentinel_collate(base_collate_fn: Callable[[List[Any]], Any]):
    """Wrap *base_collate_fn* so it recognises the leading sentinel marker."""
    SENTINEL_MARKER_KEY = "is_last_marker"

    def sentinel_aware_collate(batch_samples: List[Any]):
        is_last_batch = False
                
        # check if first sample is a sentinel marker
        if (batch_samples and 
            isinstance(batch_samples[0], dict) and 
            batch_samples[0].get(SENTINEL_MARKER_KEY)):
            is_last_batch = True
            batch_samples = batch_samples[1:]  # remove sentinel from actual batch
                    
        collated_output = base_collate_fn(batch_samples)
                
        # add is_last flag to output
        if isinstance(collated_output, dict):
            collated_output["is_last"] = is_last_batch
            return collated_output
        else:
            return collated_output, is_last_batch

    return sentinel_aware_collate