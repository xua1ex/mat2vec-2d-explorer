import numpy as np
import pytest

from explorer.build_data import chemical_family, group_mc2d, pca


def rec(f, gap):
    return {"attributes": {"chemical_formula_reduced": f, "_mcloud_band_gap": gap}}


def test_group_mc2d_merges_polymorphs_and_reports_drops():
    records = [rec("MoS2", 1.6), rec("MoS2", 0.9), rec("WS2", 1.8), rec("Graphene0", None),
               rec("XyZ", 1.0), rec("Fe2O3", 0.0)]
    df, cov = group_mc2d(records, {"MoS2": 5000, "S2W": 12})
    mos2 = df.set_index("token").loc["MoS2"]
    assert mos2.n_entries == 2 and mos2.gap_min == 0.9 and mos2.gap_max == 1.6 and mos2.polymorphs_merged
    assert df.set_index("token").loc["S2W"].low_count
    assert cov["gap_positive"] == 4 and cov["in_vocab"] == 2 and cov["dropped_not_in_vocab"] == 1


def test_pca_orders_components_by_variance():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(500, 5)) * np.array([10, 5, 1, 0.1, 0.1])
    coords, ratio = pca(X, n=3)
    assert coords.shape == (500, 3) and ratio[0] > ratio[1] > ratio[2] and ratio.sum() <= 1
    assert np.isclose(coords.mean(axis=0), 0, atol=1e-8).all()


def test_chemical_family_priority():
    assert chemical_family("MoS2") == "sulfide"
    assert chemical_family("Se2W") == "selenide"
    assert chemical_family("BiClO") == "halide"     # halide wins over oxide by convention
    assert chemical_family("AsGa") == "pnictide"
    assert chemical_family("GaSi") == "other"


def test_build_paths_do_not_depend_on_working_directory():
    from pathlib import Path
    from explorer import build_data
    root = Path(build_data.__file__).resolve().parents[1]
    assert Path(build_data.OUT_DIR) == root / "artifacts"
    assert Path(build_data.RAW_CACHE) == root / "data" / "mc2d_raw.json"


@pytest.mark.artifacts
def test_exported_vectors_stay_aligned_with_candidate_metadata():
    """Regression: row i of cand_in/cand_out/coords must belong to row i of candidates.parquet."""
    from pathlib import Path
    import pandas as pd
    art = Path(__file__).resolve().parents[1] / "artifacts"
    a = np.load(art / "vectors.npz")
    cand = pd.read_parquet(art / "candidates.parquet")
    coords = pd.read_parquet(art / "coords.parquet")
    assert list(cand.token) == list(a["cand_tokens"]) == list(coords.token)
    vocab_row = {t: i for i, t in enumerate(a["vocab_tokens"])}
    for i, t in enumerate(a["cand_tokens"]):
        assert np.array_equal(a["cand_in"][i], a["vocab_in"][vocab_row[t]]), t
    counts = dict(zip(a["vocab_tokens"], a["counts"]))
    assert all(counts[t] == c for t, c in zip(cand.token, cand["count"]))


def test_build_explains_a_missing_model(monkeypatch, tmp_path):
    import pytest
    from explorer import build_data
    monkeypatch.setattr(build_data, "MODEL_DIR", str(tmp_path))
    monkeypatch.setattr("sys.argv", ["build_data"])
    with pytest.raises(SystemExit) as exc:
        build_data.main()
    msg = str(exc.value)
    assert "MAT2VEC_MODEL_DIR" in msg and str(tmp_path) in msg and "storage.googleapis.com/mat2vec/" in msg


def test_build_explains_missing_build_dependencies(monkeypatch):
    import sys
    import pytest
    from explorer import build_data
    monkeypatch.setitem(sys.modules, "gensim", None)
    monkeypatch.setitem(sys.modules, "gensim.models", None)
    monkeypatch.setattr("sys.argv", ["build_data"])
    with pytest.raises(SystemExit) as exc:
        build_data.main()
    assert "pip install -r requirements-build.txt" in str(exc.value)
