# public_artifacts: results-only bundle

Precomputed, limited-scope model-derived outputs: the fallback used by mat2vec-2d-explorer when `artifacts/` is unavailable.
This is **not the pretrained embedding vectors** and contains no model files.

| File | Contents |
|---|---|
| `scores.csv` | co-occurrence (output·input) and similarity (input·input) scores, 591 materials × 15 predefined targets |
| `context.csv` | each material's top-10 context words, their co-occurrence value and similarity to each target |
| `candidates.csv` | MC2D formulas, band gaps and Mat2vec token counts |
| `coords.csv` | PCA coordinates (PC1–PC3) of the candidate embeddings and a chemical-family label |
| `meta.json` | coverage, build checks, explained variance, list of targets |

**Derived from Mat2vec.** All scores, context words and PCA coordinates were computed with the pretrained Mat2vec
embeddings: Tshitoyan, V. et al. Unsupervised word embeddings capture latent knowledge from materials science literature. Nature 571, 95-98 (2019). doi:10.1038/s41586-019-1335-8; https://github.com/materialsintelligence/mat2vec. The embeddings were used unchanged and are not included here.

**Licensing.** The Mat2vec embeddings, and these outputs derived from them, are distributed under the mat2vec
MIT License (Copyright (c) 2019 Authors of Tshitoyan et al. Nature (2019)). Redistribution of the pretrained weights under the MIT License was
confirmed by first author Vahe Tshitoyan via email in October 2026.

**MC2D.** Formulas and band gaps come from the Materials Cloud MC2D database (CC BY 4.0;
Mounet et al., Nat. Nanotechnol. 13, 246 (2018), doi:10.1038/s41565-017-0035-5), filtered and merged as described
in THIRD_PARTY_NOTICES.md.

Regenerate from local artifacts with `python -m explorer.public_bundle`.
