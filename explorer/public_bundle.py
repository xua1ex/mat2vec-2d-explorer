"""Results-only public bundle for the hosted app: precomputed outputs, no embedding vectors.

    python -m explorer.public_bundle      # reads artifacts/, writes public_artifacts/ (no model needed)

The bundle holds candidate metadata, PCA coordinates, both scores for the curated targets, and each candidate's
top-10 context words with their similarity to every target. The 200-d vectors (vocab_in, cand_in, cand_out)
stay in the git-ignored artifacts/: 15 target directions cannot reconstruct a 200-d embedding.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from explorer import ranking as r

ROOT = Path(__file__).resolve().parents[1]
FLOAT = "%.9g"  # 9 significant digits round-trip float32 exactly, so rankings match the full app


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
    meta["bundle"] = {"targets": targets, "contents": "precomputed scores, PCA coordinates, candidate metadata and "
                      "top-10 context words; no embedding vectors", "built_from_artifacts": meta["built_at"]}
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    return out


if __name__ == "__main__":
    out = export()
    for p in sorted(out.iterdir()):
        print(f"{p.stat().st_size / 1024:8.1f} KB  {p.relative_to(ROOT)}")
