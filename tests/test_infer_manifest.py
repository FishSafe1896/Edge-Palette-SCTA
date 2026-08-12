import pytest

from scripts.infer_manifest import batched_rows


def test_batched_rows_splits_rows_without_dropping_tail():
    rows = [{"pair_id": str(i)} for i in range(5)]

    batches = list(batched_rows(rows, batch_size=2))

    assert [[row["pair_id"] for row in batch] for batch in batches] == [
        ["0", "1"],
        ["2", "3"],
        ["4"],
    ]


def test_batched_rows_rejects_non_positive_batch_size():
    with pytest.raises(ValueError, match="batch_size"):
        list(batched_rows([{"pair_id": "0"}], batch_size=0))
