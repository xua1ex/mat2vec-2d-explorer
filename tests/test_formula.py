from explorer.formula import normalized_formula, is_simple_formula


def test_normalized_formula_matches_mat2vec():
    assert normalized_formula("LiCoO2") == "CoLiO2"
    assert normalized_formula("WS2") == "S2W"
    assert normalized_formula("Ni0.5Fe0.5") == "FeNi"
    assert normalized_formula("GaAs") == "AsGa"


def test_is_simple_formula_matches_mat2vec():
    assert is_simple_formula("LiFePO4")
    assert is_simple_formula("O2")
    assert not is_simple_formula("Fe")    # single element
    assert not is_simple_formula("IV")    # roman numeral
    assert not is_simple_formula("BN")    # all-caps: known mat2vec limitation, kept on purpose
