import json

import pytest

import train


@pytest.mark.parametrize("method", ["lejepa", "simclr", "simsiam"])
def test_one_epoch_end_to_end(method, tmp_path):
    result = train.main(["--method", method, "--config", "configs/smoke.yaml",
                         "--runs-dir", str(tmp_path / "runs"), "--results-dir", str(tmp_path / "results")])
    for key in ["knn_top1", "effective_rank", "linear_top1_100pct", "linear_top1_10pct"]:
        assert key in result
    saved = tmp_path / "results" / "smoke" / method / "fake" / "s0.json"
    assert json.loads(saved.read_text())["method"] == method
    assert (tmp_path / "runs" / "smoke" / method / "fake" / "s0" / "last.pt").exists()
