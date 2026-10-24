"""Integrity of the curated literature set data/known_methods.csv."""
import pandas as pd

from explorer.formula import normalized_formula

K = pd.read_csv("data/known_methods.csv", keep_default_na=False)
METHODS = {"chemical_vapor_deposition", "ALD", "molecular_beam_epitaxy", "magnetron_sputtering",
           "pulsed_laser_deposition"}


def test_one_row_per_material_method_pair():
    assert not K.duplicated(["method", "token"]).any()
    assert len(K) == 92


def test_fields_are_valid():
    assert set(K.method) <= METHODS
    assert set(K.growth_type) <= {"direct_2d", "thin_film", "conversion", "hybrid", "bulk"}
    assert (K.doi.str.len() > 0).all() and (K.evidence.str.len() > 0).all()
    assert all(normalized_formula(f) == t for f, t in zip(K.formula, K.token))


def test_polymorph_mismatches_are_flagged_explicitly():
    flagged = set(zip(K.method[K.phase_mismatch], K.formula[K.phase_mismatch]))
    assert flagged == {("chemical_vapor_deposition", "MoTe2"), ("chemical_vapor_deposition", "WTe2"),
                       ("molecular_beam_epitaxy", "WTe2"), ("molecular_beam_epitaxy", "SnSe"),
                       ("magnetron_sputtering", "BN")}


def test_training_window_overlap_count():
    assert int((K.year.astype(int) <= 2018).sum()) == 63


def test_evidence_is_the_title_or_a_marked_paraphrase():
    other = K[K.evidence != K.title]
    assert len(other) == 5
    assert other.evidence.str.startswith("Abstract (paraphrased): ").all()
