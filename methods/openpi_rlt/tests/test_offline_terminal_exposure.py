from pathlib import Path
import importlib.util
import copy
import numpy as np
import pytest

path = Path(__file__).resolve().parents[3] / "scripts/offline_terminal_exposure.py"
spec = importlib.util.spec_from_file_location("offline_terminal_exposure", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

@pytest.mark.parametrize("suffix", ["", "checkpoints", "checkpoints/deeper"])
def test_output_rejects_asset_directory_and_descendants(tmp_path, suffix):
    assets = tmp_path / "assets"
    with pytest.raises(ValueError, match="outside"):
        module.validate_output_path(assets / suffix, assets)
    assert not assets.exists()

def test_output_accepts_separate_analysis_subdirectory(tmp_path):
    output = tmp_path / "analysis" / "matched"
    assert module.validate_output_path(output, tmp_path / "assets") == output.resolve()

@pytest.mark.parametrize("pool,quota", [([], 4), ([1000], 0)])
def test_unavailable_or_disabled_quota_keeps_exact_batch_and_rng(pool, quota):
    base = np.arange(128)
    rng = np.random.default_rng(42)
    before = copy.deepcopy(rng.bit_generator.state)
    actual = module.quota_indices(base, pool, quota, rng)
    np.testing.assert_array_equal(actual, base)
    assert rng.bit_generator.state == before
    assert not np.shares_memory(actual, base)

def test_quota_replaces_exact_slots_without_mutating_original():
    base = np.arange(128)
    before = base.copy()
    actual = module.quota_indices(base, [1000, 1001], 4, np.random.default_rng(42))
    np.testing.assert_array_equal(base, before)
    assert np.count_nonzero(actual != base) == 4
    assert np.isin(actual[actual != base], [1000, 1001]).all()
    assert len(actual) == 128
