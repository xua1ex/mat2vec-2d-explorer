# public_artifacts: results-only bundle

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
embeddings: Tshitoyan, V. et al. Unsupervised word embeddings capture latent knowledge from materials science literature. Nature 571, 95-98 (2019). doi:10.1038/s41586-019-1335-8; https://github.com/materialsintelligence/mat2vec. The embeddings were used unchanged and are not included here.

**Licensing.** The redistribution terms of the Mat2vec model weights are unresolved, so the licensing status of
outputs derived from them is also unresolved. This project's MIT License does not cover the weights.

**MC2D.** Formulas and band gaps come from the Materials Cloud MC2D database (CC BY 4.0;
Mounet et al., Nat. Nanotechnol. 13, 246 (2018), doi:10.1038/s41565-017-0035-5), filtered and merged as described
in THIRD_PARTY_NOTICES.md.

Regenerate from local artifacts with `python -m explorer.public_bundle`.
