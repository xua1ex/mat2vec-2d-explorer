"""The results-only public bundle: precomputed outputs for the hosted app, never embedding vectors."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "public_artifacts"
EXPECTED = {"candidates.csv", "coords.csv", "scores.csv", "context.csv", "meta.json", "README.md"}
CURATED = 15


def test_bundle_has_only_expected_small_files():
    files = {p.name for p in BUNDLE.iterdir()}
    assert files == EXPECTED
    assert sum(p.stat().st_size for p in BUNDLE.iterdir()) < 3_000_000


def test_bundle_contains_no_embedding_vectors():
    """No vector arrays, no wide numeric blocks, no model-array names anywhere in the bundle."""
    for name in EXPECTED - {"meta.json", "README.md"}:
        df = pd.read_csv(BUNDLE / name)
        numeric = df.select_dtypes("number").shape[1]
        assert numeric <= CURATED + 5, (name, numeric)          # a 200-d vector block would be ~200 columns
        assert not any(k in c for c in df.columns for k in ("vocab_in", "cand_in", "cand_out", "vector")), name
    meta = json.loads((BUNDLE / "meta.json").read_text())
    assert not {"vocab_in", "cand_in", "cand_out"} & set(json.dumps(meta).split('"'))


def test_bundle_covers_all_curated_targets_for_every_candidate():
    scores = pd.read_csv(BUNDLE / "scores.csv")
    cand = pd.read_csv(BUNDLE / "candidates.csv")
    meta = json.loads((BUNDLE / "meta.json").read_text())
    assert len(meta["bundle"]["targets"]) == CURATED and set(scores.target) == set(meta["bundle"]["targets"])
    assert len(scores) == len(cand) * CURATED and set(scores.columns) == {"token", "target", "out_in", "in_in"}
    for _, block in scores.groupby("target"):
        assert list(block.token) == list(cand.token)            # one score per candidate, in candidate order
    ctx = pd.read_csv(BUNDLE / "context.csv")
    assert ctx.groupby("token").size().max() <= 10 and set(ctx.token) <= set(cand.token)
    assert {f"cos_{t}" for t in meta["bundle"]["targets"]} <= set(ctx.columns)


@pytest.mark.artifacts
def test_bundle_is_up_to_date_and_matches_full_mode(tmp_path):
    from explorer import public_bundle, ranking as r
    public_bundle.export(ROOT / "artifacts", tmp_path)
    assert (tmp_path / "README.md").read_text() == (BUNDLE / "README.md").read_text()
    for name in EXPECTED - {"meta.json", "README.md"}:
        pd.testing.assert_frame_equal(pd.read_csv(tmp_path / name), pd.read_csv(BUNDLE / name), check_exact=True)
    a = np.load(ROOT / "artifacts" / "vectors.npz")
    idx = {t: i for i, t in enumerate(a["vocab_tokens"])}
    scores = pd.read_csv(BUNDLE / "scores.csv")
    for target in ("chemical_vapor_deposition", "ALD", "thermoelectric"):
        block = scores[scores.target == target]
        full = r.score(a["cand_out"], a["vocab_in"][idx[target]])
        assert np.array_equal(block.out_in.to_numpy(np.float32), full)   # identical, so identical rankings


def test_bundle_is_attributed_to_mat2vec_with_licensing_note():
    readme = (BUNDLE / "README.md").read_text()
    for phrase in ("Mat2vec", "Tshitoyan", "10.1038/s41586-019-1335-8", "github.com/materialsintelligence/mat2vec",
                   "MC2D", "CC BY 4.0", "MIT License", "Vahe Tshitoyan", "not the pretrained embedding vectors"):
        assert phrase in readme, phrase
    bundle = json.loads((BUNDLE / "meta.json").read_text())["bundle"]
    assert "10.1038/s41586-019-1335-8" in bundle["derived_from"] and "MIT License" in bundle["licence_note"]


def test_bundle_has_no_secrets_paths_or_personal_data():
    import re
    pattern = re.compile("/" + "Users/|/home/|/private/|" + r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+\.[a-z]{2,}|ghp_|github_pat_|"
                         r"BEGIN [A-Z ]*PRIVATE KEY|api[_-]?key|password|secret", re.I)
    for f in BUNDLE.iterdir():
        hits = pattern.findall(f.read_text(errors="replace"))
        assert not hits, (f.name, hits[:3])
