"""Results-only public bundle for the hosted app: precomputed outputs, no embedding vectors.

    python -m explorer.public_bundle      # reads artifacts/, writes public_artifacts/ (no model needed)

The bundle holds precomputed, limited-scope outputs derived from the Mat2vec embeddings: candidate metadata, PCA
coordinates, both scores for the 15 curated targets, and each candidate's top-10 context words with their
similarity to every target. It does not contain the 200-d vectors (vocab_in, cand_in, cand_out), which stay in the
git-ignored artifacts/. The licensing of the Mat2vec weights, and of outputs derived from them, is unresolved.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from explorer import ranking as r

ROOT = Path(__file__).resolve().parents[1]
FLOAT = "%.9g"  # 9 significant digits round-trip float32 exactly, so rankings match the full app
MAT2VEC = ("Tshitoyan, V. et al. Unsupervised word embeddings capture latent knowledge from materials science "
           "literature. Nature 571, 95-98 (2019). doi:10.1038/s41586-019-1335-8; "
           "https://github.com/materialsintelligence/mat2vec")
LICENCE_NOTE = ("Derived from the pretrained Mat2vec embeddings, whose redistribution terms are unresolved; "
                "this bundle's publication status depends on that answer.")
README = f"""# public_artifacts: results-only bundle

Precomputed, limited-scope model-derived outputs for the hosted version of mat2vec-2d-explorer.
This is **not the pretrained embedding vectors** and contains no model files.

| File | Contents |
|---|---|
| `scores.csv` | co-occurrence (output·input) and similarity (input·input) scores, 591 materials × 15 predefined targets |
| `context.csv` | each material's top-10 context words, their co-occurrence value and similarity to each target |
| `candidates.csv` | MC2D formulas, band gaps and Mat2vec token counts |
| `coords.csv` | PCA coordinates (PC1–PC3) of the candidate embeddings and a chemical-family label |
| `meta.json` | coverage, build checks, explained variance, list of targets |

**Derived from Mat2vec.** All scores, context words and PCA coordinates were computed with the pretrained Mat2vec
embeddings: {MAT2VEC}. The embeddings were used unchanged and are not included here.

**Licensing.** The redistribution terms of the Mat2vec model weights are unresolved, so the licensing status of
outputs derived from them is also unresolved. This project's MIT License does not cover the weights.

**MC2D.** Formulas and band gaps come from the Materials Cloud MC2D database (CC BY 4.0;
Mounet et al., Nat. Nanotechnol. 13, 246 (2018), doi:10.1038/s41565-017-0035-5), filtered and merged as described
in THIRD_PARTY_NOTICES.md.

Regenerate from local artifacts with `python -m explorer.public_bundle`.
"""


def export(artifacts=ROOT / "artifacts", out=ROOT / "public_artifacts"):
    artifacts, out = Path(artifacts), Path(out)
    a = np.load(artifacts / "vectors.npz")
    cand = pd.read_parquet(artifacts / "candidates.parquet")
    coords = pd.read_parquet(artifacts / "coords.parquet")
    meta = json.loads((artifacts / "meta.json").read_text())
    tokens, vocab_tokens = list(a["cand_tokens"]), list(a["vocab_tokens"])
    if not list(cand.token) == tokens == list(coords.token):
        raise SystemExit("artifacts/ are out of sync; rebuild them with python -m explorer.build_data")
    vocab = {t: i for i, t in enumerate(vocab_tokens)}
    targets = list(a["targets"])
    target_in = {t: a["vocab_in"][vocab[t]] for t in targets}

    scores = pd.concat(pd.DataFrame({"token": tokens, "target": t, "out_in": r.score(a["cand_out"], v),
                                     "in_in": r.score(a["cand_in"], v)}) for t, v in target_in.items())
    rows = []
    for i, token in enumerate(tokens):  # same top-10 words the full app shows; they do not depend on the target
        words = r.context_words(a["cand_out"][i], a["vocab_in"], vocab_tokens, target_in[targets[0]])
        word_in = a["vocab_in"][[vocab[w] for w in words.token]]
        block = pd.DataFrame({"token": token, "word": words.token, "assoc": words.assoc})
        for t, v in target_in.items():
            block[f"cos_{t}"] = r.unit(word_in) @ r.unit(v)
        rows.append(block)

    out.mkdir(exist_ok=True)
    cand.to_csv(out / "candidates.csv", index=False, float_format=FLOAT)
    coords.to_csv(out / "coords.csv", index=False, float_format=FLOAT)
    scores.to_csv(out / "scores.csv", index=False, float_format=FLOAT)
    pd.concat(rows).to_csv(out / "context.csv", index=False, float_format=FLOAT)
    meta["bundle"] = {"targets": targets, "contents": "precomputed, limited-scope model-derived outputs (scores, PCA "
                      "coordinates, candidate metadata, top-10 context words); not the pretrained embedding vectors",
                      "derived_from": MAT2VEC, "licence_note": LICENCE_NOTE, "built_from_artifacts": meta["built_at"]}
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    (out / "README.md").write_text(README)
    return out


if __name__ == "__main__":
    out = export()
    for p in sorted(out.iterdir()):
        print(f"{p.stat().st_size / 1024:8.1f} KB  {p.relative_to(ROOT)}")
