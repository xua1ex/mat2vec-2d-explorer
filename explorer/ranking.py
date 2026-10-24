"""Scoring and evaluation on Mat2vec vectors. numpy + pandas only (no model, no scipy).

Two ranking modes share one function:
  in_in  = score(candidate INPUT vectors,  target input vector)  -> plain cosine similarity
  out_in = score(candidate OUTPUT vectors, target input vector)  -> Tshitoyan et al. (2019),
           "likelihood of co-occurring with the target word"
All scores use the full 200-d vectors; PCA is display-only.
"""
import numpy as np
import pandas as pd


def unit(X):
    """Row-wise L2 normalization in float32 (artifacts are float16). Zero rows stay zero."""
    X = np.asarray(X, dtype=np.float32)
    norm = np.linalg.norm(X, axis=-1, keepdims=True)
    return np.divide(X, norm, out=np.zeros_like(X), where=norm > 0)


def score(cand, target):
    return unit(cand) @ unit(target)


def ranks(scores):
    """1-based ranks, 1 = highest score."""
    out = np.empty(len(scores), dtype=int)
    out[np.argsort(-np.asarray(scores), kind="stable")] = np.arange(1, len(scores) + 1)
    return out


def resolve_target(text, token_index):
    """Map user text to a vocabulary token: exact (formulas are case-sensitive), lowercase, then phrase."""
    text = text.strip()
    for cand in (text, text.lower(), "_".join(text.lower().split())):
        if cand and cand in token_index:
            return cand
    return None


def context_words(mat_out, vocab_in, vocab_tokens, target_in, k=10):
    """Words the material is most likely to co-occur with (out·in), and each word's cosine to the target.

    This is the paper's Fig. 2c idea: the bridge words that connect a material to a target.
    """
    assoc = unit(vocab_in) @ unit(mat_out)
    top = np.argsort(-assoc, kind="stable")[:k]
    return pd.DataFrame({
        "token": [vocab_tokens[i] for i in top],
        "assoc": assoc[top],
        "target_cos": unit(vocab_in[top]) @ unit(target_in),
    })


def popularity_bias(scores, counts, top=20):
    """How much a ranking just tracks how often a material is mentioned."""
    counts = np.asarray(counts, dtype=float)
    order = np.argsort(-np.asarray(scores), kind="stable")[:top]
    return {
        "spearman": pd.Series(scores).corr(pd.Series(np.log(counts)), method="spearman"),
        "median_count_top": float(np.median(counts[order])),
        "median_count_all": float(np.median(counts)),
    }


def recall_at_k(scores, positive, k=20):
    """Share of known positives ranked in the top k. Positive-only: unlabeled items are not negatives."""
    positive = np.asarray(positive, dtype=bool)
    return float(positive[np.argsort(-np.asarray(scores), kind="stable")[:k]].sum() / positive.sum())


def median_percentile(scores, positive):
    """Median percentile rank of known positives (lower is better, 50 = chance). Positive-only."""
    positive = np.asarray(positive, dtype=bool)
    return float(np.median(ranks(scores)[positive] / len(scores) * 100))


def match_known(known, candidates):
    """Boolean mask over candidates for known tokens, plus known tokens absent from the candidates."""
    known_set, cand_set = set(known), set(candidates)
    return np.array([c in known_set for c in candidates]), [k for k in known if k not in cand_set]
