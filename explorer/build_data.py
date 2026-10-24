"""Build step: pretrained Mat2vec + MC2D band gaps -> slim local artifacts for the app.

    python -m explorer.build_data [--refresh]

Reads the ~880 MB pretrained model once (MAT2VEC_MODEL_DIR) and writes artifacts/. Only the four artifact files are
committed (MIT License; redistribution of the pretrained weights confirmed by the first author); the raw model is not.
"""
import argparse
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from pymatgen.core.composition import Composition

from explorer.formula import is_simple_formula, normalized_formula
from explorer.ranking import context_words, ranks, score

MC2D_URL = "https://optimade.materialscloud.org/main/mc2d/v1/structures"
ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = os.environ.get("MAT2VEC_MODEL_DIR", str(ROOT.parent / "mat2vec" / "mat2vec" / "training" / "models"))
RAW_CACHE = str(ROOT / "data" / "mc2d_raw.json")
OUT_DIR = str(ROOT / "artifacts")
MODEL_FILES = ["pretrained_embeddings", "pretrained_embeddings.wv.vectors.npy",
               "pretrained_embeddings.trainables.syn1neg.npy"]
MODEL_URL = "https://storage.googleapis.com/mat2vec/"  # official source used by the original mat2vec setup.py
LOW_COUNT = 20
VOCAB_SIZE = 50_000
CURATED_TARGETS = [
    "thermoelectric", "photovoltaic", "photocatalytic", "ferroelectric", "piezoelectric",
    "superconducting", "topological_insulator", "magnetic",
    # "atomic_layer_deposition" never became a phrase token; "ALD" (count 5,072) is the ALD token.
    "chemical_vapor_deposition", "ALD", "molecular_beam_epitaxy",
    "magnetron_sputtering", "sputtering", "pulsed_laser_deposition", "exfoliation",
]
# Display convention for colouring only, first match wins (BiOCl -> halide).
FAMILIES = [
    ("halide", {"F", "Cl", "Br", "I"}), ("telluride", {"Te"}), ("selenide", {"Se"}),
    ("sulfide", {"S"}), ("oxide", {"O"}), ("nitride", {"N"}),
    ("pnictide", {"P", "As", "Sb", "Bi"}), ("carbide", {"C"}),
]


def group_mc2d(records, vocab_counts):
    """MC2D records -> one row per Mat2vec token (polymorphs merged) + coverage counts."""
    groups = {}
    positive = 0
    for rec in records:
        attrs = rec["attributes"]
        gap = attrs.get("_mcloud_band_gap")
        if gap is None or gap <= 0:
            continue
        positive += 1
        formula = attrs["chemical_formula_reduced"]
        g = groups.setdefault(normalized_formula(formula), {"formulas": set(), "gaps": []})
        g["formulas"].add(formula)
        g["gaps"].append(gap)

    rows, dropped = [], []
    for token, g in sorted(groups.items()):
        if token not in vocab_counts:
            dropped.append(token)
            continue
        count = vocab_counts[token]
        rows.append({
            "token": token,
            "formula_examples": ", ".join(sorted(g["formulas"])[:3]),
            "n_entries": len(g["gaps"]),
            "gap_min": min(g["gaps"]),
            "gap_max": max(g["gaps"]),
            "count": count,
            "polymorphs_merged": len(g["gaps"]) > 1,
            "low_count": count < LOW_COUNT,
        })
    coverage = {
        "mc2d_total": len(records),
        "gap_positive": positive,
        "unique_formulas": len(groups),
        "in_vocab": len(rows),
        "dropped_not_in_vocab": len(dropped),
        "dropped_examples": dropped[:10],
    }
    return pd.DataFrame(rows), coverage


def pca(X, n=3):
    """PCA via SVD of the centered data: (coords, explained-variance ratio)."""
    Xc = np.asarray(X, dtype=np.float64)
    Xc = Xc - Xc.mean(axis=0)
    _, s, vt = np.linalg.svd(Xc, full_matrices=False)
    return Xc @ vt[:n].T, s[:n] ** 2 / (s ** 2).sum()


def chemical_family(token):
    try:
        elements = {str(e) for e in Composition(token).elements}
    except Exception:
        return "other"
    return next((name for name, els in FAMILIES if elements & els), "other")


def fetch_mc2d(refresh=False):
    import requests

    if os.path.exists(RAW_CACHE) and not refresh:
        with open(RAW_CACHE) as f:
            return json.load(f)
    records, url = [], MC2D_URL
    params = {"filter": "_mcloud_band_gap>0", "page_limit": 500,
              "response_fields": "chemical_formula_reduced,_mcloud_band_gap"}
    while url:
        resp = requests.get(url, params=params, timeout=60)
        resp.raise_for_status()
        page = resp.json()
        records += page["data"]
        nxt = (page.get("links") or {}).get("next")
        url = nxt["href"] if isinstance(nxt, dict) else nxt
        params = None  # the next link already carries the query
    os.makedirs(os.path.dirname(RAW_CACHE), exist_ok=True)
    with open(RAW_CACHE, "w") as f:
        json.dump(records, f)
    return records


def paper_check(wv, syn1neg):
    """Reproduce Tshitoyan et al. Fig. 2a ranks over all vocabulary formulas with count > 3."""
    formulas = [t for t in wv.index_to_key
                if wv.get_vecattr(t, "count") > 3 and re.match(r"^[A-Z][A-Za-z0-9.]*$", t)
                and is_simple_formula(t)]
    idx = [wv.key_to_index[t] for t in formulas]
    r = dict(zip(formulas, ranks(score(syn1neg[idx], wv["thermoelectric"]))))
    out = {"n_formulas": len(formulas), "CuLi2Sb": int(r["CuLi2Sb"]), "Cu3Nb2O8": int(r["Cu3Nb2O8"])}
    out["passed"] = (out["CuLi2Sb"], out["Cu3Nb2O8"]) == (326, 345)
    return out


def artifact_check(path, wv, syn1neg, cand_tokens, target="chemical_vapor_deposition"):
    """Both ranking modes and context words must work from the exported float16 arrays."""
    a = np.load(path)
    vocab_index = {t: i for i, t in enumerate(a["vocab_tokens"])}
    t16 = a["vocab_in"][vocab_index[target]]
    cidx = [wv.key_to_index[t] for t in cand_tokens]
    out = {}
    for mode, c16, c32 in (("in_in", a["cand_in"], wv.vectors[cidx]), ("out_in", a["cand_out"], syn1neg[cidx])):
        rho = pd.Series(score(c16, t16)).corr(pd.Series(score(c32, wv[target])), method="spearman")
        out[f"{mode}_spearman_vs_float32"] = round(float(rho), 6)
    top = int(np.argmax(score(a["cand_out"], t16)))
    out["context_rows"] = len(context_words(a["cand_out"][top], a["vocab_in"], list(a["vocab_tokens"]), t16))
    out["passed"] = min(out["in_in_spearman_vs_float32"], out["out_in_spearman_vs_float32"]) > 0.999 \
        and out["context_rows"] == 10
    return out


def main():
    try:
        from gensim.models import Word2Vec
    except ImportError:
        raise SystemExit("The build step needs gensim and pymatgen:  pip install -r requirements-build.txt")

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--refresh", action="store_true", help="re-download MC2D instead of using the cache")
    args = ap.parse_args()
    missing = [f for f in MODEL_FILES if not os.path.exists(os.path.join(MODEL_DIR, f))]
    if missing:
        raise SystemExit(
            f"Pretrained Mat2vec model not found in {MODEL_DIR} (missing: {', '.join(missing)}).\n"
            f"Download the three files from the original authors' storage:\n"
            + "".join(f"  {MODEL_URL}{f}\n" for f in MODEL_FILES)
            + "then set MAT2VEC_MODEL_DIR to their folder. They are by Tshitoyan et al. and distributed under the "
              "mat2vec MIT License. You only need them to rebuild artifacts/, which the repository already includes.")
    t0 = time.time()

    records = fetch_mc2d(args.refresh)
    model = Word2Vec.load(os.path.join(MODEL_DIR, "pretrained_embeddings"))
    wv, syn1neg = model.wv, model.syn1neg
    counts = {t: wv.get_vecattr(t, "count") for t in wv.index_to_key}

    cand, coverage = group_mc2d(records, counts)
    cand_tokens = list(cand.token)
    targets = [t for t in CURATED_TARGETS if t in wv.key_to_index]

    # index_to_key is frequency-sorted, so its head is the most frequent tokens.
    vocab_tokens = list(wv.index_to_key[:VOCAB_SIZE])
    seen = set(vocab_tokens)
    vocab_tokens += [t for t in cand_tokens + targets if t not in seen]
    vidx = [wv.key_to_index[t] for t in vocab_tokens]
    cidx = [wv.key_to_index[t] for t in cand_tokens]

    os.makedirs(OUT_DIR, exist_ok=True)
    vec_path = os.path.join(OUT_DIR, "vectors.npz")
    np.savez(vec_path,
             vocab_tokens=np.array(vocab_tokens), vocab_in=wv.vectors[vidx].astype(np.float16),
             counts=np.array([counts[t] for t in vocab_tokens]),
             cand_tokens=np.array(cand_tokens), cand_in=wv.vectors[cidx].astype(np.float16),
             cand_out=syn1neg[cidx].astype(np.float16), targets=np.array(targets))
    cand.to_parquet(os.path.join(OUT_DIR, "candidates.parquet"), index=False)

    coords, explained = pca(wv.vectors[cidx], n=3)
    pd.DataFrame({"token": cand_tokens, "pc1": coords[:, 0], "pc2": coords[:, 1], "pc3": coords[:, 2],
                  "family": [chemical_family(t) for t in cand_tokens]}
                 ).to_parquet(os.path.join(OUT_DIR, "coords.parquet"), index=False)

    meta = {
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "coverage": coverage,
        "paper_check": paper_check(wv, syn1neg),
        "artifact_check": artifact_check(vec_path, wv, syn1neg, cand_tokens),
        "pca_explained": [round(float(x), 4) for x in explained],
        "missing_targets": [t for t in CURATED_TARGETS if t not in targets],
        "low_count_threshold": LOW_COUNT,
    }
    with open(os.path.join(OUT_DIR, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta, indent=2))
    print(f"done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
