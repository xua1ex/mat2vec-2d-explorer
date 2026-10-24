import numpy as np

from explorer import ranking as r


def test_unit_zero_row_is_zero_not_nan():
    out = r.unit(np.array([[3.0, 4.0], [0.0, 0.0]]))
    assert np.allclose(out[0], [0.6, 0.8]) and np.all(out[1] == 0)


def test_score_and_ranks():
    cand = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    s = r.score(cand, np.array([1.0, 0.0]))
    assert list(r.ranks(s)) == [1, 3, 2]


def test_resolve_target_order():
    idx = {"MoS2": 0, "thermoelectric": 1, "chemical_vapor_deposition": 2}
    assert r.resolve_target("MoS2", idx) == "MoS2"
    assert r.resolve_target(" Thermoelectric ", idx) == "thermoelectric"
    assert r.resolve_target("Chemical  vapor deposition", idx) == "chemical_vapor_deposition"
    assert r.resolve_target("", idx) is None
    assert r.resolve_target("unobtainium", idx) is None


def test_context_words_top_k():
    vocab = np.array([[1.0, 0.0], [0.0, 1.0], [0.9, 0.1]])
    df = r.context_words(np.array([1.0, 0.0]), vocab, ["a", "b", "c"], np.array([0.0, 1.0]), k=2)
    assert list(df.token) == ["a", "c"] and list(df.columns) == ["token", "assoc", "target_cos"]


def test_popularity_bias_detects_frequency_ranking():
    counts = np.array([1, 10, 100, 1000])
    out = r.popularity_bias(np.log(counts), counts, top=2)
    assert np.isclose(out["spearman"], 1.0) and out["median_count_top"] == 550.0


def test_positive_only_metrics():
    s = np.array([0.9, 0.8, 0.1, 0.05])
    pos = np.array([True, False, True, False])
    assert r.recall_at_k(s, pos, k=2) == 0.5
    assert r.median_percentile(s, pos) == 50.0   # ranks 1 and 3 of 4 -> 25%, 75% -> median 50


def test_match_known_reports_missing():
    mask, missing = r.match_known(["MoS2", "S2W", "XyZ9"], ["S2W", "MoS2", "Se2W"])
    assert list(mask) == [True, True, False] and missing == ["XyZ9"]
