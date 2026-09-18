"""Tests for the shared synthetic pipeline both memray examples profile."""

import pytest

from streamlit_app.jobs.memray_profiling import workload


@pytest.fixture(autouse=True)
def clean_cache():
    workload.reset_cache()
    yield
    workload.reset_cache()


def test_decode_copies_without_aliasing():
    payload = b"abc"
    assert workload._decode(payload) == payload


def test_widen_multiplies_the_row():
    assert workload._widen(b"ab", 3) == b"ababab"


def test_fold_totals_the_rows():
    assert workload._fold([b"ab", b"cde"]) == 5


def test_record_buffer_compacts_appended_rows():
    buffer = workload.RecordBuffer()
    buffer.append(b"ab")
    buffer.append(b"cd")

    assert buffer.compact() == b"abcd"


def test_record_buffer_starts_empty():
    assert workload.RecordBuffer().compact() == b""


def test_ingest_batches_retains_one_entry_per_batch():
    total = workload.ingest_batches(batches=4, frames_per_batch=8, frame_size=16)

    assert total == 4 * 8 * 16
    assert len(workload.CACHE) == 4
    assert len(workload.CACHE["batch-0"]) == 8 * 16


def test_ingest_batches_accumulates_across_calls():
    """The cache is never evicted, which is the leak the examples demonstrate."""
    workload.ingest_batches(batches=2, frames_per_batch=4, frame_size=16)
    workload.ingest_batches(batches=4, frames_per_batch=4, frame_size=16)

    assert len(workload.CACHE) == 4


def test_reset_cache_empties_it():
    workload.ingest_batches(batches=2, frames_per_batch=2, frame_size=8)
    workload.reset_cache()

    assert workload.CACHE == {}


def test_index_node_returns_a_leaf_at_depth_zero():
    node = workload._index_node(0, 3, 8)

    assert node == [b"i" * 8]


def test_index_node_recurses_to_the_requested_depth():
    node = workload._index_node(2, 2, 4)

    assert len(node) == 2
    assert len(node[0]) == 2
    assert node[0][0] == [b"i" * 4]


def test_build_index_retains_the_tree_and_totals_the_leaves():
    total = workload.build_index(depth=2, fanout=3, frame_size=4)

    leaf_size = 4 * workload.INDEX_LEAF_MULTIPLIER
    assert total == 3**2 * leaf_size
    assert len(workload.CACHE["index"]) == 3


def test_transform_rows_retains_nothing():
    total = workload.transform_rows([b"ab", b"cd"], multiplier=4)

    assert total == 2 * 2 * 4
    assert workload.CACHE == {}


def test_summarize_rows_totals_without_retaining():
    assert workload.summarize_rows([b"abc", b"de"]) == 5
    assert workload.CACHE == {}


def test_run_workload_reports_retained_and_transient():
    retained, transient = workload.run_workload(
        batches=2, frames_per_batch=4, frame_size=64, index_depth=1
    )

    leaf_size = 64 * workload.INDEX_LEAF_MULTIPLIER
    assert retained == (2 * 4 * 64) + (workload.INDEX_FANOUT * leaf_size)
    assert transient == (4 * 64 * workload.TRANSIENT_SIZE_MULTIPLIER + 4 * 64)


def test_run_workload_leaves_only_retained_stages_in_the_cache():
    workload.run_workload(batches=2, frames_per_batch=2, frame_size=32, index_depth=1)

    assert set(workload.CACHE) == {"batch-0", "batch-1", "index"}
