import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "src" / "ChinesePaperCutting" / "ChinesePaperCutting_Transfer"
sys.path.insert(0, str(MODEL_DIR))

from dataset_sampler import InfiniteSamplerWrapper


class DummyDataset:
    def __len__(self):
        return 7


def test_infinite_sampler_wrapper_repeats_sequence_for_same_seed():
    first = iter(InfiniteSamplerWrapper(DummyDataset(), seed=2026))
    second = iter(InfiniteSamplerWrapper(DummyDataset(), seed=2026))

    first_indices = [next(first) for _ in range(20)]
    second_indices = [next(second) for _ in range(20)]

    assert first_indices == second_indices
