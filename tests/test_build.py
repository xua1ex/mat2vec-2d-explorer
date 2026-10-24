import numpy as np

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
