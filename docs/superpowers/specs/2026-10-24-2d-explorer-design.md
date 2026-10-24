# mat2vec-2d-explorer — Design (v1)

## Intent

A standalone, portfolio-grade materials-informatics project for recruiters and professors. In the first 30 seconds a visitor:
1. picks a target property or deposition method;
2. sees ranked 2D semiconductor candidates with real MC2D band-gap data;
3. explores how candidates relate in an interactive embedding map.

It shows what literature embeddings capture and where they fail. It never presents literature similarity as proof that a material can be deposited by a method.

Built on the pretrained Mat2vec embeddings (Tshitoyan et al., *Nature* 571, 95, 2019, doi:10.1038/s41586-019-1335-8) and MC2D (Mounet et al., *Nat. Nanotechnol.* 13, 246, 2018, doi:10.1038/s41565-017-0035-5). Target: polished v1 in about one week.

## Goals (v1)

- Pick a target from a curated dropdown (properties and deposition methods) or type a word, and get a ranked candidate table.
- Rank candidates two ways, side by side:
  - **input·input cosine**;
  - **normalized output·input**, the paper's method.
- An interactive PCA map of candidate embeddings with a **2D / 3D toggle**: the first 2 or 3 principal components of the 200-d input vectors, rendered in Plotly with rotate, zoom and hover.
  - Colour by chemical family (default), by score or by band gap.
  - Hover shows the MC2D formula, the Mat2vec token and the band-gap range.
  - Axis titles show each component's explained-variance percentage.
  - PCA is display only. All rankings use the full 200-d vectors.
- MC2D band gap for every candidate.
- Per-candidate explanation:
  - token count and rank in both modes;
  - warning badges;
  - top context words.
- One evaluation page: what the embeddings capture and where they fail.
- Reproducible: one command builds the data, one command runs the app.

## Non-goals (v1)

- Stability data of any kind. Bulk Materials Project E_hull is **not** a proxy for 2D stability.
- Retraining Mat2vec, new ML models, LLMs or a chatbot.
- Free-text tokenization of sentences (ChemDataExtractor).
- Deploying the 800 MB pretrained model.
- Hand-built feasibility or precursor tables.

## Anti-goals (failure even if it works)

- Any wording implying a high score means "this material can be grown by CVD/ALD". Every method ranking carries the caveat "literature association before 2018, not feasibility".
- Hiding failure modes. Low-count tokens, merged polymorphs and dropped formulas must be visible.
- Modifying the original `mat2vec` repository.

## Decisions made by the user

- **Separate repo** `mat2vec-2d-explorer`, not a fork. The original `mat2vec` checkout stays unchanged.
- Reuse only the Mat2vec functionality that is needed, with license and attribution.
- No stability in v1.
- **Python 3.11** in a separate environment.
- v1 scope:
  - pretrained Mat2vec rankings (both modes);
  - MC2D band gaps;
  - PCA with a 2D/3D toggle (UMAP optional, dropped);
  - a simple, meaningful evaluation;
  - a polished Streamlit dashboard.
- Minimize features and dependencies.
- Don't commit or redistribute the pretrained weights or derived vector artifacts until permission is verified.
- No GitHub repo, push or deploy without approval.
- Evaluation must not treat unreported material–method pairs as negatives. Popularity bias takes priority if the known-CVD labels are unreliable.

## Builder's calls (made by Claude; check while reading)

1. **Reuse by vendoring, not importing.** Copy `is_simple_formula`, `get_ordered_integer_formula`, `normalized_formula` and the element list from `mat2vec/processing/process.py` (MIT, © 2019 Authors of Tshitoyan et al.) into `explorer/formula.py`.
   - Keep their logic unchanged. Add a header comment naming the source file and the commit (`7ca8961`).
   - Ship `LICENSE` (MIT, the project's own) plus `THIRD_PARTY_NOTICES.md` containing the mat2vec MIT notice and the MC2D CC BY 4.0 attribution.
   - Consequence: there is no dependency on the `mat2vec` package or on ChemDataExtractor, and no edits to the original repo.
2. **Model location.** The build step reads the pretrained files from `MAT2VEC_MODEL_DIR`, which defaults to `../mat2vec/mat2vec/training/models`. The README documents the three Google Cloud Storage download URLs for anyone without the original repo.
3. **Dependencies (minimal).**
   - Runtime: `streamlit`, `pandas`, `numpy`, `plotly`, `pyarrow`.
   - Build step only: `gensim>=4.3,<5`, `pymatgen`, `requests`.
   - PCA is computed with `numpy.linalg.svd`: center the data, take the top singular vectors, explained variance = s²/Σs². That means no scikit-learn. UMAP is dropped from v1 (user: optional; it would add numba and umap-learn).
   - Tests: `pytest`.
   - Two files: `requirements.txt` (app) and `requirements-build.txt` (build). This keeps a future Streamlit Cloud deploy light.
4. **Day-1 compatibility check.** In the Python 3.11 env:
   - the pretrained model loads under gensim 4.3+ (it already loads under 4.2);
   - the vendored `normalized_formula` gives `LiCoO2`→`CoLiO2`, `WS2`→`S2W`, `Ni0.5Fe0.5`→`FeNi` with current pymatgen/monty.

   Fallback if loading fails: a ~15-line exporter run once in the old `mat2vec` env that dumps raw vectors and counts to `.npy`. `build_data.py` then reads those. `build_data.py` is not made to work under both gensim 3 and 4.
5. **Build step plus slim artifacts.** `explorer/build_data.py` loads the full model once and writes these files to `artifacts/`:
   - `candidates.parquet`;
   - `vectors.npz` (float16):
     - input vectors for the ~50k most frequent tokens plus all candidates;
     - output vectors for candidates;
     - token counts;
   - `coords.parquet` (`pc1, pc2, pc3`, `family`);
   - `meta.json` (build date, coverage numbers, paper-check result, PCA explained-variance ratios for PC1–PC3).

   The app reads only these files, so it starts fast.
   - **Artifacts stay local and git-ignored** (user decision): `vectors.npz` contains part of the pretrained weights, whose redistribution permission is unverified.
   - The README documents how to rebuild them. Revisit only after the permission is confirmed.
6. **Candidate set.** MC2D via its public OPTIMADE API (`https://optimade.materialscloud.org/main/mc2d/v1/structures`; no key; 2,683 entries, 1,458 with `_mcloud_band_gap > 0`, checked 2026-10-24). Keep band gap > 0.
   - Map `chemical_formula_reduced` through `normalized_formula()`, then keep those in the vocabulary.
   - Group entries that share a formula (polymorphs). Show the band-gap range and an entry count, and add a "polymorphs merged" badge.
   - Cache the raw response to `data/mc2d_raw.json` so rebuilds work offline.
7. **Scoring recipe (verified in the source session against the paper).**
   - Output·input score = (output vector of material, L2-normalized) · (input vector of target, L2-normalized).
   - Input·input score = cosine of the two input vectors.
   - Paper check: rank over all vocabulary formulas passing `is_simple_formula` with count > 3 (30,769 tokens). Target `thermoelectric` gives `CuLi2Sb` (Li2CuSb) rank 326 and `Cu3Nb2O8` rank 345.
   - `build_data.py` runs this check once against the full model and records pass/fail in `meta.json`.
   - Unit tests use small synthetic vectors, so the tests need no model.
   - App ranks are over MC2D candidates only, so they are not comparable to paper ranks.
7b. **Chemical family (for colouring only).** Assign the family from the elements of the normalized formula, by the first rule that matches:
    1. halide: F, Cl, Br, I;
    2. telluride: Te;
    3. selenide: Se;
    4. sulfide: S;
    5. oxide: O;
    6. nitride: N;
    7. pnictide: P, As, Sb, Bi;
    8. carbide: C;
    9. other.

    The order is a display convention, not chemistry. For example, BiOCl is labelled halide. The About page says so.
8. **Targets.**
   - Curated dropdown, checked against the vocabulary at build time:
     - properties: thermoelectric, photovoltaic, photocatalytic, ferroelectric, piezoelectric, superconducting, topological_insulator, magnetic;
     - methods: chemical_vapor_deposition, ALD, molecular_beam_epitaxy, sputtering, pulsed_laser_deposition, exfoliation. (`atomic_layer_deposition` is not a vocabulary token; `ALD` is, with count 5,072.)
   - Free-text box: lowercase, join spaces with `_`, look up in the 50k table. Show "not in vocabulary" otherwise.
9. **Explanations.** For a selected candidate, show:
   - token count and rank in both modes;
   - badges: "low count (<20)", "polymorphs merged";
   - top 10 context words = tokens w maximizing out(material)·in(w), each with its cosine to the target (the paper's Fig. 2c idea).
10. **Evaluation (one page, two parts).**
    - **Popularity bias (label-free).** For each target and mode:
      - Spearman correlation between score and log token count;
      - median token count of the top-20 vs all candidates.
    This is the **primary** evaluation.
    - **Known-association recovery (secondary, positive-only).** `data/known_cvd.csv` lists ~20–30 MC2D 2D semiconductors with well-documented CVD growth, one citation each. The user reviews this list.
      - Unreported pairs are **unlabeled, not negative**, so use only metrics that need positives alone:
        - recall@20 (share of known positives in the top 20);
        - median percentile rank of known positives.
      - Both are computed for the two modes and for a log-count baseline, also with candidates of count < 20 removed. No AUPRC or precision, because those assume the unlabeled pairs are negatives.
      - If fewer than 10 known positives map to candidates, show this part as "insufficient labels" and keep only the popularity-bias analysis.
      - Stated on the page: this measures recovery of pre-2018 literature association, is partly circular, and says nothing about feasibility for unstudied materials.
    - **Coverage, on the data panel.** MC2D formulas → found in vocabulary → dropped, with reasons: not in vocabulary, all-caps formulas like BN that the formula detector misses, merged polymorphs.
11. **Layout.**

    ```
    mat2vec-2d-explorer/
      explorer/  formula.py  build_data.py  ranking.py  app.py
      tests/     test_formula.py  test_ranking.py
      data/      known_cvd.csv  (mc2d_raw.json cached, git-ignored)
      artifacts/ (generated)
      docs/superpowers/{specs,plans}/
      README.md  LICENSE  THIRD_PARTY_NOTICES.md  requirements*.txt
    ```

## Constraints

- About one week, solo.
- Must run on a laptop. Build step < 10 min. App start from artifacts < 5 s.
- Original `mat2vec` repo untouched.

## Left to the builder

Streamlit styling, plot details (Plotly), the exact 50k cutoff, test granularity, and About-page wording. The About page must state the caveats above and cite the paper and MC2D.

## Unverified (check during build)

- That gensim 4.3+ loads the pretrained model.
- Redistribution terms for the pretrained weights. The repo code is MIT; the weights' licence is not stated separately. Until this is verified, artifacts stay local and nothing is pushed or deployed.
- Whether MC2D's OPTIMADE band gap is null for some semiconductors (filtering on gap > 0 drops those).

## Amendment (user, 2026-10-24): multi-method evaluation

The evaluation covers five deposition methods. Target tokens were checked in the vocabulary:

| Method | Token | Count |
|---|---|---|
| CVD | `chemical_vapor_deposition` | 11,680 |
| ALD | `ALD` | 5,072 |
| MBE | `molecular_beam_epitaxy` | 5,616 |
| Magnetron sputtering | `magnetron_sputtering` | 7,948 |
| PLD | `pulsed_laser_deposition` | 6,520 |

`magnetron_sputtering` is added to the Explore dropdown. The generic `sputtering` stays.

- **Literature set** `data/known_methods.csv` (replaces `known_cvd.csv`). Columns: `method, formula, token, phase, growth_type, doi, year, title, evidence`.
  - `growth_type` is one of `direct_2d`, `thin_film`, `conversion` (deposit a precursor, then sulfurize, selenize or otherwise convert), `hybrid`, `bulk`.
  - `evidence` quotes the title or abstract sentence that reports the process.
  - Each row is checked by reading the title and abstract.
  - A material may appear under several methods.
  - Aim for 10–20 pairs per method. Never weaken the evidence to reach that number.
  - Candidate papers are found with `tools/find_literature.py` (OpenAlex search, kept for provenance). Rows are kept only after review.
- **Evaluation:** for each method, both modes plus a log-count baseline, all under the shared min-count filter:
  - Recall@20 and median percentile rank (positive-only);
  - Spearman of score vs log count.

  A "direct 2D growth only" toggle restricts the positives. Fewer than 10 positives in the filtered set → recovery is labelled "insufficient data", while the ranking and bias analysis are still shown.
- **Evaluation page:**
  - a method selector;
  - a cross-method comparison table;
  - a bias scatter per method;
  - a references table;
  - sample sizes on every metric;
  - the limitations list (literature relationships, not feasibility; unlabeled ≠ negative; bulk, conversion and polymorph caveats; small positive sets; similarity ≠ deposition quality or optimal conditions).
