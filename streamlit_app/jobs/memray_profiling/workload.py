"""A synthetic pipeline whose call tree is shaped to make a flamegraph useful.

Pure Python: no Ray, no memray. Both the local example
(``examples/memray_example.py``) and the Ray job (``profile_job.py``) profile
this same code, so their flamegraphs are directly comparable.

The call tree, as it appears in a flamegraph::

    run_workload
    ├── ingest_batches        retained  — the bulk of peak memory
    │   ├── RecordBuffer.append
    │   │   └── _decode
    │   └── RecordBuffer.compact
    ├── build_index           retained  — a recursion tower
    │   └── _index_node
    │       └── _index_node  (… index_depth levels deep)
    ├── transform_rows        live at peak, freed on return
    │   ├── _decode           ← the same leaf, under a second parent
    │   └── _widen
    └── summarize_rows        churn only
        └── _fold

The stages deliberately fall into three tiers, so each of memray's views shows
something the others do not:

* :func:`ingest_batches` and :func:`build_index` **retain** what they allocate,
  so they appear in the default view and are all that is left under
  ``--leaks``.
* :func:`transform_rows` materializes its output, so it is live at the high
  watermark and appears in the default view -- but it frees everything on
  return and vanishes under ``--leaks``.
* :func:`summarize_rows` never holds more than one copy at a time, so it is
  absent from both and only appears under ``--temporary-allocations``.

Two more details worth noticing: :func:`_index_node` recurses, drawing a tall
narrow tower that is unmistakable on sight, and :func:`_decode` is reached from
two different parents, so it appears twice -- flamegraphs merge by *stack*, not
by function.
"""

from typing import Any

# ~262 MiB retained by the ingest stage at the defaults: big enough to be
# unmistakable in the flamegraph, small enough to leave room on a Ray worker.
DEFAULT_BATCHES = 512
DEFAULT_FRAMES_PER_BATCH = 64
DEFAULT_FRAME_SIZE = 8192

# The index is about depth, not size: a fanout-3 tree six levels deep gives the
# recursion tower enough leaves (3**6 = 729) to be visible without being heavy.
DEFAULT_INDEX_DEPTH = 6
INDEX_FANOUT = 3
INDEX_LEAF_MULTIPLIER = 4

# The transform stage materializes its whole output at once, so this sets how
# big that block is: 64 rows x 8 KiB x 128 comes to ~64 MiB, a clear band in
# the flamegraph against the ~262 MiB the retained stages hold.
TRANSIENT_SIZE_MULTIPLIER = 128

# Deliberately module-level and never evicted: this is the "leak" the examples
# exist to make visible.
CACHE: dict[str, Any] = {}


def reset_cache() -> None:
    """Empty the retained store so repeated runs start from zero."""
    CACHE.clear()


def _decode(payload: bytes) -> bytes:
    """Copy one record.

    Called from both :meth:`RecordBuffer.append` and :func:`transform_rows`,
    which is what puts the same leaf under two different parents in the graph.

    Args:
        payload: The raw record.

    Returns:
        A copy of ``payload``.
    """
    return bytes(payload)


def _widen(row: bytes, multiplier: int) -> bytes:
    """Expand a record into a larger buffer."""
    return row * multiplier


def _fold(rows: list[bytes]) -> int:
    """Total the rows, copying each one along the way and freeing it again."""
    return sum(len(bytes(row)) for row in rows)


class RecordBuffer:
    """Accumulates decoded records and consolidates them on demand.

    A class rather than a function so the flamegraph shows bound methods, the
    way it would for real pipeline code.
    """

    def __init__(self) -> None:
        """Start with no rows."""
        self._rows: list[bytes] = []

    def append(self, row: bytes) -> None:
        """Decode ``row`` and keep the copy."""
        self._rows.append(_decode(row))

    def compact(self) -> bytes:
        """Join every row into one buffer.

        Returns:
            The concatenated rows. The per-row copies are dropped when the
            buffer goes out of scope, so this stage churns as well as retains.
        """
        return b"".join(self._rows)


def ingest_batches(
    batches: int = DEFAULT_BATCHES,
    frames_per_batch: int = DEFAULT_FRAMES_PER_BATCH,
    frame_size: int = DEFAULT_FRAME_SIZE,
) -> int:
    """Build one compacted buffer per batch and retain every one of them.

    This is the stage that dominates peak memory and survives to the end of the
    capture, so it is what ``--leaks`` reports.

    Args:
        batches: Number of batches to ingest.
        frames_per_batch: Records per batch.
        frame_size: Size of each record in bytes.

    Returns:
        Total bytes retained by this stage.
    """
    total = 0
    for index in range(batches):
        buffer = RecordBuffer()
        for _ in range(frames_per_batch):
            buffer.append(b"x" * frame_size)
        compacted = buffer.compact()
        CACHE[f"batch-{index}"] = compacted
        total += len(compacted)
    return total


def _index_node(depth: int, fanout: int, leaf_size: int) -> list[Any]:
    """Build one level of the index tree, recursing until ``depth`` is zero.

    Args:
        depth: Levels still to build below this one.
        fanout: Children per internal node.
        leaf_size: Size of each leaf's payload in bytes.

    Returns:
        A leaf payload list at depth zero, otherwise a list of child nodes.
    """
    if depth <= 0:
        return [b"i" * leaf_size]
    return [_index_node(depth - 1, fanout, leaf_size) for _ in range(fanout)]


def build_index(
    depth: int = DEFAULT_INDEX_DEPTH,
    fanout: int = INDEX_FANOUT,
    frame_size: int = DEFAULT_FRAME_SIZE,
) -> int:
    """Build a nested index and retain it.

    The work happens in :func:`_index_node`, which calls itself, so this stage
    draws a tower ``depth`` frames tall in the flamegraph.

    Args:
        depth: Levels of nesting to build.
        fanout: Children per internal node.
        frame_size: Base record size; leaves are a multiple of it.

    Returns:
        Total bytes retained by the index.
    """
    leaf_size = frame_size * INDEX_LEAF_MULTIPLIER
    CACHE["index"] = _index_node(depth, fanout, leaf_size)
    return fanout**depth * leaf_size


def transform_rows(rows: list[bytes], multiplier: int) -> int:
    """Decode and widen every row, holding the whole result in memory at once.

    This is the middle tier: the materialized list is live when the process
    hits its high watermark, so the stage shows up in the default flamegraph --
    but it is freed on return, so it is gone again under ``--leaks``. Real
    pipelines produce this shape constantly.

    Args:
        rows: Records to transform.
        multiplier: How much larger each widened buffer is than the record.

    Returns:
        Total bytes allocated, none of which is still live on return.
    """
    widened = [_widen(_decode(row), multiplier) for row in rows]
    return sum(len(row) for row in widened)


def summarize_rows(rows: list[bytes]) -> int:
    """Fold the rows down to a byte count, retaining nothing."""
    return _fold(rows)


def run_workload(
    batches: int = DEFAULT_BATCHES,
    frames_per_batch: int = DEFAULT_FRAMES_PER_BATCH,
    frame_size: int = DEFAULT_FRAME_SIZE,
    index_depth: int = DEFAULT_INDEX_DEPTH,
) -> tuple[int, int]:
    """Run all four pipeline stages once.

    Args:
        batches: Number of batches to ingest.
        frames_per_batch: Records per batch.
        frame_size: Size of each record in bytes.
        index_depth: Levels of nesting in the index tree.

    Returns:
        ``(retained_bytes, transient_bytes)``.
    """
    retained = ingest_batches(batches, frames_per_batch, frame_size)
    retained += build_index(index_depth, INDEX_FANOUT, frame_size)

    sample = [b"s" * frame_size for _ in range(frames_per_batch)]
    transient = transform_rows(sample, TRANSIENT_SIZE_MULTIPLIER)
    transient += summarize_rows(sample)
    return retained, transient
