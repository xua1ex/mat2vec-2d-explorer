"""Chemical-formula normalization, vendored from mat2vec.

Source: mat2vec/processing/process.py (MaterialsTextProcessor), commit 7ca8961,
https://github.com/materialsintelligence/mat2vec
MIT License, Copyright (c) 2019 Authors of Tshitoyan et al. Nature (2019).
See THIRD_PARTY_NOTICES.md.

The logic is unchanged so that formulas map to the exact tokens in the pretrained
Mat2vec vocabulary (e.g. WS2 -> S2W). The only edits: module-level functions instead
of class methods, and stdlib `re` instead of `regex` (the pattern is compatible).
"""
import re

from monty.fractions import gcd_float
from pymatgen.core.composition import Composition, CompositionError
from pymatgen.core.periodic_table import Element

# Exactly IV, VI or has 2 consecutive II, or roman in parenthesis: is not a simple formula.
VALENCE_INFO = re.compile(r"(II+|^IV$|^VI$|\(IV\)|\(V?I{0,3}\))")


def is_element(txt):
    """True if the string is a chemical symbol, e.g. Hg, Fe, V."""
    try:
        Element(txt)
        return True
    except ValueError:
        return False


def is_simple_formula(text):
    """True if the string is a simple chemical formula, e.g. IrMn, LiFePO4.

    Excludes some roman numbers (IV) and all-caps abbreviations without digits,
    which also excludes a few real materials such as BN (a known mat2vec limitation).
    """
    if VALENCE_INFO.search(text) is not None:
        # 2 consecutive II, IV or VI should not be parsed as formula.
        return False
    elif any(char.isdigit() or char.islower() for char in text):
        try:
            if text in ["O2", "N2", "Cl2", "F2", "H2"]:
                # Diatomic at room temperature, despite having a single element.
                return True
            composition = Composition(text)
            # Has to contain more than one element; single elements are handled differently.
            if len(composition.keys()) < 2 or any([not is_element(key) for key in composition.keys()]):
                return False
            return True
        except (CompositionError, ValueError, OverflowError):
            return False
    else:
        return False


def get_ordered_integer_formula(el_amt, max_denominator=1000):
    """{element: amount} -> alphabetically ordered formula with smallest integer stoichiometry."""
    g = gcd_float(list(el_amt.values()), 1 / max_denominator)
    d = {k: round(v / g) for k, v in el_amt.items()}
    formula = ""
    for k in sorted(d):
        if d[k] > 1:
            formula += k + str(d[k])
        elif d[k] != 0:
            formula += k
    return formula


def normalized_formula(formula, max_denominator=1000):
    """Normalize a formula to the Mat2vec vocabulary form, e.g. Ni0.5Fe0.5 -> FeNi."""
    try:
        formula_dict = Composition(formula).get_el_amt_dict()
        return get_ordered_integer_formula(formula_dict, max_denominator)
    except (CompositionError, ValueError):
        return formula
