"""2D Semiconductor Materials Explorer — Streamlit app.  Run: streamlit run explorer/app.py

Reads only the local artifacts written by `python -m explorer.build_data` (no gensim, no model).
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# `streamlit run explorer/app.py` puts explorer/ on sys.path, not the repo root.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from explorer import ranking as r  # noqa: E402

ARTIFACTS = os.environ.get("EXPLORER_ARTIFACTS", str(ROOT / "artifacts"))
# Hosted version: results-only bundle from `python -m explorer.public_bundle` (no embedding vectors).
PUBLIC = os.environ.get("EXPLORER_PUBLIC", str(ROOT / "public_artifacts"))
PUBLIC_FILES = ["candidates.csv", "coords.csv", "scores.csv", "context.csv", "meta.json"]
HOSTED_ABOUT = ("\n- **This results-only version** runs on the fallback bundle (`public_artifacts/`): precomputed, "
                "limited-scope outputs derived from the Mat2vec embeddings (scores for the 15 predefined targets, PCA "
                "coordinates, candidate metadata and top-10 context words). It is not the embedding vectors, so "
                "free-text queries need the full version with `artifacts/`.")
HOSTED_NOTE = ("Free-text queries are unavailable in the results-only version: they need the embedding vectors in "
               "`artifacts/`. Run the full version to query any word.")
CAVEAT = "Literature association before 2018, not feasibility."
DEFAULT_TARGET = "chemical_vapor_deposition"
MODES = {"out_in": "Co-occurrence score (output·input)", "in_in": "Similarity score (input·input)"}
TOP = 20
MIN_POS = 10
KNOWN = ROOT / "data" / "known_methods.csv"
METHODS = {
    "Chemical vapor deposition (CVD)": "chemical_vapor_deposition",
    "Atomic layer deposition (ALD)": "ALD",
    "Molecular beam epitaxy (MBE)": "molecular_beam_epitaxy",
    "Magnetron sputtering": "magnetron_sputtering",
    "Pulsed laser deposition (PLD)": "pulsed_laser_deposition",
}
MODE_COLORS = {"Co-occurrence (out·in)": "#4C78A8", "Similarity (in·in)": "#F58518"}
RANKINGS = {"out_in": "Co-occurrence (out·in)", "in_in": "Similarity (in·in)", "log_count": "Token-count baseline"}
LIMITATIONS = """**Read these results with care.**
- Mat2vec captures relationships in the literature up to 2018, not proven deposition feasibility.
- Unreported material–method pairs are unlabeled, not confirmed negatives.
- Some verified papers report thin films, bulk crystals, post-deposition conversion (e.g. selenizing a metal film)
  or a different polymorph than the MC2D entry; see *Growth type* and *Phase* below.
- Positive sets are small (13–26 per method; 0–14 for direct 2D growth), so differences between methods are
  indicative, not conclusive.
- A higher similarity score does not imply better deposition quality or optimal processing conditions."""

st.set_page_config(page_title="2D Semiconductor Explorer", layout="wide")


FILES = ["vectors.npz", "candidates.parquet", "coords.parquet", "meta.json"]


@st.cache_resource  # once per server and path: no 42 MB copy on every widget change
def load_artifacts(path):
    vec = dict(np.load(os.path.join(path, "vectors.npz")))
    cand = pd.read_parquet(os.path.join(path, "candidates.parquet"))
    coords = pd.read_parquet(os.path.join(path, "coords.parquet"))
    # Row i of the candidate vectors must describe row i of the metadata; a mismatch would silently mislabel scores.
    aligned = list(cand.token) == list(vec["cand_tokens"]) == list(coords.token)
    cand = cand.merge(coords, on="token")
    with open(os.path.join(path, "meta.json")) as f:
        meta = json.load(f)
    vec["vocab_index"] = {t: i for i, t in enumerate(vec["vocab_tokens"])}
    return {"mode": "full", "vec": vec, "cand": cand, "meta": meta, "aligned": aligned, "targets": list(vec["targets"])}


@st.cache_resource
def load_public(path):
    cand = pd.read_csv(os.path.join(path, "candidates.csv"), keep_default_na=False)
    coords = pd.read_csv(os.path.join(path, "coords.csv"), keep_default_na=False)
    scores = pd.read_csv(os.path.join(path, "scores.csv"))
    meta = json.loads(Path(path, "meta.json").read_text())
    targets = meta["bundle"]["targets"]
    blocks = {t: b for t, b in scores.groupby("target", sort=False)}
    aligned = list(cand.token) == list(coords.token) and all(list(blocks[t].token) == list(cand.token) for t in targets)
    return {"mode": "public", "cand": cand.merge(coords, on="token"), "meta": meta, "aligned": aligned,
            "targets": targets, "context": pd.read_csv(os.path.join(path, "context.csv"), keep_default_na=False),
            "scores": {t: {m: blocks[t][m].to_numpy(np.float32) for m in ("out_in", "in_in")} for t in targets}}


def target_scores(data, target):
    """Both scores for every candidate: computed from the vectors (full) or looked up (results-only bundle)."""
    if data["mode"] == "public":
        return data["scores"][target]
    vec = data["vec"]
    t_in = vec["vocab_in"][vec["vocab_index"][target]]
    return {"out_in": r.score(vec["cand_out"], t_in), "in_in": r.score(vec["cand_in"], t_in)}


def candidate_context(data, token, target):
    if data["mode"] == "public":
        rows = data["context"][data["context"].token == token]
        return pd.DataFrame({"token": rows.word.to_numpy(), "assoc": rows.assoc.to_numpy(),
                             "target_cos": rows[f"cos_{target}"].to_numpy()})
    vec = data["vec"]
    i = int(np.where(vec["cand_tokens"] == token)[0][0])
    t_in = vec["vocab_in"][vec["vocab_index"][target]]
    return r.context_words(vec["cand_out"][i], vec["vocab_in"], list(vec["vocab_tokens"]), t_in)


def badges(row):
    notes = []
    if row["low_count"]:
        notes.append("low count")
    if row["polymorphs_merged"]:
        notes.append(f"{row['n_entries']} polymorphs")
    return ", ".join(notes)


def gap_label(row):
    lo, hi = row["gap_min"], row["gap_max"]
    return f"{lo:.2f}" if np.isclose(lo, hi) else f"{lo:.2f}–{hi:.2f}"


def ranking_table(df, mode):
    out = df.sort_values(f"rank_{mode}")
    return pd.DataFrame({
        "Rank": out[f"rank_{mode}"],
        "Formula (MC2D)": out["formula_examples"],
        "Score": out[mode].round(3),
        "Band gap (eV)": out.apply(gap_label, axis=1),
        "Token count": out["count"],
        "Notes": out.apply(badges, axis=1),
    })


def pick_target(data):
    targets = data["targets"]
    chosen = st.sidebar.selectbox(
        "Target property or deposition method", targets, key="target",
        index=targets.index(DEFAULT_TARGET) if DEFAULT_TARGET in targets else 0,
        format_func=lambda t: t.replace("_", " "))
    if data["mode"] == "public":
        st.sidebar.caption(HOSTED_NOTE)
        return chosen
    vec = data["vec"]
    free = st.sidebar.text_input("…or type any word (e.g. photodetector, MoS2)", key="free")
    if free.strip():
        resolved = r.resolve_target(free, vec["vocab_index"])
        if resolved is None:
            st.warning(f"“{free}” is not in vocabulary (top 50,000 Mat2vec tokens; formulas are stored "
                       "normalized, e.g. WS2 → S2W). "
                       f"Showing “{chosen.replace('_', ' ')}” instead.")
        else:
            chosen = resolved
    return chosen


def explore(data, min_count):
    cand, meta = data["cand"].copy(), data["meta"]
    cov = meta["coverage"]

    st.title("2D Semiconductor Materials Explorer")
    st.markdown(
        "Rank 2D semiconductors from the **MC2D** database by how the pretrained **Mat2vec** word "
        "embeddings (Tshitoyan et al., *Nature* 2019; trained on ~1.5 million materials-science "
        "abstracts up to 2018) associate them with a property or deposition method.")
    st.caption(f"Coverage: {cov['in_vocab']} of {cov['unique_formulas']:,} normalized MC2D semiconductor "
               f"formulas are in the Mat2vec vocabulary; the other {cov['dropped_not_in_vocab']} "
               "(mostly complex multi-element compounds) have no embedding and cannot be ranked.")

    target = pick_target(data)
    gmin, gmax = float(cand.gap_min.min()), float(cand.gap_max.max())
    gap_lo, gap_hi = st.sidebar.slider("Band gap range (eV)", gmin, gmax, (gmin, gmax), key="gap")

    # Score all candidates on the full 200-d vectors, then apply ONE filter to both modes.
    scored = target_scores(data, target)
    cand["out_in"], cand["in_in"] = scored["out_in"], scored["in_in"]
    shown = cand[(cand["count"] >= min_count) & (cand.gap_max >= gap_lo) & (cand.gap_min <= gap_hi)].copy()
    st.subheader(f"Target: {target.replace('_', ' ')}")
    st.markdown(f"**{len(shown)} of {len(cand)} candidates shown** "
                f"(token count ≥ {min_count}, band gap {gap_lo:.2f}–{gap_hi:.2f} eV).")
    if shown.empty:
        st.warning("No candidates match these filters. Lower the minimum token count or widen the band-gap range.")
        return
    for mode in MODES:
        shown[f"rank_{mode}"] = r.ranks(shown[mode].to_numpy())

    bias = {m: r.popularity_bias(shown[m].to_numpy(), shown["count"].to_numpy(), top=TOP) for m in MODES}
    st.info(
        "**Two ways to score a material, two different biases.** Mat2vec learns two vectors per word. "
        "The **co-occurrence score** asks *how likely is this material's name to appear next to the target "
        "word in an abstract?* It favours frequently discussed materials: the median token count of its "
        f"top {TOP} is **{bias['out_in']['median_count_top']:,.0f}**. The **similarity score** asks *is this "
        "material written about in the same kind of context as the target word?* It tends to put rare tokens "
        f"on top (median token count of its top {TOP}: **{bias['in_in']['median_count_top']:,.0f}**), whose "
        "vectors were learned from only a few sentences and are unreliable. "
        f"For comparison, the median over all {len(shown)} shown candidates is "
        f"**{bias['out_in']['median_count_all']:,.0f}**. The two scores are on different scales "
        "(co-occurrence scores can be negative), so compare ranks, not raw values.")

    cols = st.columns(2)
    for col, mode in zip(cols, MODES):
        col.markdown(f"**{MODES[mode]}**")
        col.dataframe(ranking_table(shown, mode), hide_index=True, height=420)
    st.caption(CAVEAT + " Scores reflect how abstracts up to 2018 discuss these materials, "
               "not whether they can actually be grown by a given method.")

    pca_view(shown, meta)
    detail(shown, data, target)


def pca_view(shown, meta):
    ev = [100 * x for x in meta["pca_explained"]]
    st.subheader("Embedding map")
    c1, c2 = st.columns(2)
    dims = c1.radio("View", ["2D", "3D"], horizontal=True, key="dims",
                    index=1 if st.query_params.get("view") == "3D" else 0)
    color_by = c2.radio("Colour by", ["Chemical family", "Co-occurrence score", "Band gap"],
                        horizontal=True, key="color")
    color = {"Chemical family": "family", "Co-occurrence score": "out_in", "Band gap": "gap_max"}[color_by]
    labels = {f"pc{i + 1}": f"PC{i + 1} ({ev[i]:.1f}%)" for i in range(3)}
    labels.update({"family": "Family", "out_in": "Co-occurrence", "gap_max": "Band gap max (eV)"})
    hover = {"token": True, "gap_min": ":.2f", "gap_max": ":.2f", "count": True,
             "pc1": False, "pc2": False, "pc3": False}
    common = dict(color=color, hover_name="formula_examples", hover_data=hover, labels=labels, height=560,
                  color_discrete_sequence=px.colors.qualitative.Bold)
    if dims == "2D":
        fig = px.scatter(shown, x="pc1", y="pc2", **common)
    else:
        fig = px.scatter_3d(shown, x="pc1", y="pc2", z="pc3", **common)
        fig.update_traces(marker_size=4)
    st.plotly_chart(fig, width="stretch")
    shown_ev = sum(ev[:2] if dims == "2D" else ev)
    st.warning(f"Simplified projection: the {dims} view shows {shown_ev:.0f}% of the variance "
               f"(PC1–PC3 together explain {sum(ev):.0f}%). Nearby points are not necessarily similar, and "
               "all rankings use the original 200-dimensional embeddings, not this map.")


def detail(shown, data, target):
    st.subheader("Why is a material ranked where it is?")
    order = shown.sort_values("rank_out_in")
    token = st.selectbox("Candidate", list(order.token), key="cand",
                         format_func=lambda t: f"{t}  ({order.set_index('token').formula_examples[t]})")
    row = shown.set_index("token").loc[token]
    m = st.columns(4)
    m[0].metric("Co-occurrence rank", f"{row.rank_out_in} / {len(shown)}")
    m[1].metric("Similarity rank", f"{row.rank_in_in} / {len(shown)}")
    m[2].metric("Token count", f"{row['count']:,}")
    m[3].metric("Band gap (eV)", gap_label(row))
    if row.low_count:
        st.warning(f"Low count: this token occurs only {row['count']} times in the training corpus, "
                   "so its embedding is learned from very little text.")
    words = candidate_context(data, token, target)
    st.markdown("**Words this material most often appears with** (co-occurrence), and how similar each "
                "word is to the target. Bridge words with high values in both columns explain a high rank.")
    st.dataframe(words.rename(columns={"token": "Context word", "assoc": "Co-occurrence with material",
                                       "target_cos": "Similarity to target"}).round(3), hide_index=True)


@st.cache_data
def load_known(path=KNOWN):
    return pd.read_csv(path, keep_default_na=False)


def fmt(value, ok, digits=2):
    return f"{value:.{digits}f}" if ok else "insufficient data"


def evaluation(data, min_count):
    cand = data["cand"]
    full = load_known()
    n_pre = int((full.year.astype(int) <= 2018).sum())
    st.title("Evaluation: what the embeddings capture, and where they fail")
    st.warning(LIMITATIONS)

    direct = st.toggle("Direct 2D growth only", key="direct2d",
                       help="Count only papers that grow the 2D material directly in the MC2D phase; exclude thin "
                            "films, precursor conversion, hybrid routes, bulk crystals and polymorph mismatches.")
    known = full[(full.growth_type == "direct_2d") & ~full.phase_mismatch] if direct else full
    keep = (cand["count"] >= min_count).to_numpy()
    sub, idx = cand[keep], np.flatnonzero(keep)
    tokens = list(sub.token)
    st.markdown(f"All analyses use **n = {len(sub)}** MC2D candidates (token count ≥ {min_count}).")
    label = st.selectbox("Deposition method", list(METHODS), key="eval_method")

    # --- Main finding: needs no literature labels at all.
    scored = {lab: target_scores(data, tok) for lab, tok in METHODS.items()}
    bias = pd.DataFrame([{"Method": lab, "Ranking": RANKINGS[m],
                          "Spearman": r.popularity_bias(scored[lab][m][idx], sub["count"])["spearman"]}
                         for lab in scored for m in ("out_in", "in_in")])
    out_pos = int((bias[bias.Ranking == RANKINGS["out_in"]].Spearman > 0).sum())
    in_neg = int((bias[bias.Ranking == RANKINGS["in_in"]].Spearman < 0).sum())
    st.subheader("Main finding: the two scores have opposite popularity biases")
    st.markdown(
        "For each method we correlate every candidate's score with how often its token occurs in the training "
        f"corpus (Spearman). With the current filter, **co-occurrence (out·in)** correlates positively for "
        f"**{out_pos} of 5 methods**: it rewards much-discussed materials. **Similarity (in·in)** correlates "
        f"negatively for **{in_neg} of 5 methods**: it rewards rare tokens whose vectors come from very few "
        "sentences. At the default filter (count > 3) both patterns hold for 5 of 5 methods; raising the minimum "
        "count removes rare tokens and weakens them, which shows how much of the effect rare tokens drive.")
    st.plotly_chart(px.bar(bias, x="Method", y="Spearman", color="Ranking", barmode="group", height=380,
                           color_discrete_map=MODE_COLORS,
                           labels={"Spearman": "Spearman: score vs log token count"}), width="stretch")
    sel = bias[bias.Method == label].set_index("Ranking").Spearman
    m = st.columns(2)
    m[0].metric(f"Spearman, co-occurrence vs log count ({label})", f"{sel[RANKINGS['out_in']]:.2f}")
    m[1].metric(f"Spearman, similarity vs log count ({label})", f"{sel[RANKINGS['in_in']]:.2f}")
    mode = st.radio("Score shown", ["out_in", "in_in"], horizontal=True, key="eval_mode",
                    format_func=lambda k: RANKINGS[k])
    positive, _ = r.match_known(list(known.token[known.method == METHODS[label]]), tokens)
    plot = pd.DataFrame({"Formula": sub.formula_examples, "Token count (log scale)": sub["count"],
                         "Score": scored[label][mode][idx],
                         "Literature": np.where(positive, "verified pair", "unlabeled")})
    st.plotly_chart(px.scatter(plot, x="Token count (log scale)", y="Score", color="Literature", log_x=True,
                               hover_name="Formula", height=420,
                               color_discrete_map={"verified pair": "#E45756", "unlabeled": "#9AA0A6"},
                               category_orders={"Literature": ["unlabeled", "verified pair"]}), width="stretch")

    # --- Supporting evidence: small, mostly in-sample literature set.
    st.subheader("Supporting evidence: exploratory literature-association recovery")
    st.info(f"**{n_pre} of {len(full)}** supporting papers were published by 2018 and may overlap Mat2vec's "
            "training corpus, so this recovery is largely in-sample: it shows the embeddings encode known literature "
            "associations, not that they predict deposition feasibility or new material–process pairs.")
    st.markdown(
        "This is an exploratory literature-association recovery test on unique material–method pairs. "
        "Known pairs are positives; every other material is *unlabeled*, never counted as wrong. "
        f"Recovery metrics need at least {MIN_POS} eligible positives; otherwise they read *insufficient data*. "
        + ("**Direct 2D growth only:** thin films, conversion, hybrid routes and bulk crystals are left out, and "
           "polymorph-mismatched pairs are excluded because a report on another phase does not support the MC2D phase."
           if direct else
           "**All verified pairs:** formula-level associations, including thin films, conversion and hybrid routes; "
           "polymorph-mismatched pairs are kept here but flagged in the references."))
    rows = []
    for lab, token in METHODS.items():
        listed = known[known.method == token]
        pos, _ = r.match_known(list(listed.token), tokens)
        rep = r.report_from_scores({m: scored[lab][m][idx] for m in ("out_in", "in_in")}, sub["count"].to_numpy(),
                                   pos, k=TOP, min_pos=MIN_POS)
        for mode_, row in rep.iterrows():
            rows.append({"Method": lab, "Ranking": RANKINGS[mode_], "Listed pairs": len(listed),
                         "Eligible positives (n)": int(row.n_positive),
                         f"Recall@{TOP}": fmt(row.recall_at_k, row.sufficient),
                         "Median percentile rank (lower = better, 50 = chance)":
                             fmt(row.median_percentile, row.sufficient, 1)})
    st.dataframe(pd.DataFrame(rows), hide_index=True, height=35 * (len(rows) + 1) + 3)
    st.caption("Listed pairs = verified pairs in the current selection; eligible positives = those whose material "
               "passes the token-count filter and is used in the metric. The token-count baseline ranks by "
               "popularity alone.")

    st.markdown("**How each pair was obtained** (all verified pairs, by growth route)")
    routes = pd.crosstab(full.method.map({v: k for k, v in METHODS.items()}), full.growth_type)
    routes = routes.reindex(columns=["direct_2d", "thin_film", "conversion", "hybrid", "bulk"], fill_value=0)
    routes["polymorph mismatch"] = full.groupby(full.method.map({v: k for k, v in METHODS.items()})).phase_mismatch.sum()
    st.dataframe(routes.reset_index(names="Method"), hide_index=True)

    st.subheader(f"Literature references: {label}")
    refs = known[known.method == METHODS[label]]
    st.caption(f"n = {len(refs)} verified material–process pairs in the current selection. Each was checked against "
               "the paper's title or abstract.")
    references_table(refs)


def references_table(refs, with_method=False):
    cols = {"Method": refs.method.map({v: k for k, v in METHODS.items()})} if with_method else {}
    cols.update({"Formula": refs.formula, "Phase": refs.phase, "Growth type": refs.growth_type,
                 "Polymorph mismatch": refs.phase_mismatch, "Year": refs.year, "DOI": "https://doi.org/" + refs.doi,
                 "Note": refs.note, "Title": refs.title})
    st.dataframe(pd.DataFrame(cols), hide_index=True, height=35 * (len(refs) + 1) + 3,
                 column_config={"DOI": st.column_config.LinkColumn("DOI", display_text=r"doi\.org/(.*)")})


def literature():
    """Limited mode: the verified literature set needs no model data."""
    st.title("Literature dataset: verified material–method pairs")
    st.markdown("Each pair links a 2D semiconductor to a deposition method and was checked against the paper's title "
                "or abstract. Unreported pairs are unlabeled, not negatives; *Growth type* separates direct 2D growth "
                "from thin films, precursor conversion and hybrid routes.")
    known = load_known()
    c1, c2 = st.columns(2)
    method = c1.selectbox("Deposition method", ["All methods", *METHODS], key="lit_method")
    growth_types = sorted(known.growth_type.unique())
    growth = c2.multiselect("Growth type", growth_types, default=growth_types, key="lit_growth")
    shown = known[known.growth_type.isin(growth)]
    if method != "All methods":
        shown = shown[shown.method == METHODS[method]]
    st.caption(f"n = {len(shown)} of {len(known)} verified pairs shown.")
    references_table(shown, with_method=True)


def about(meta=None, hosted=False):
    if meta:
        pc, cov = meta["paper_check"], meta["coverage"]
        check = (f"The build step reproduces the paper's Fig. 2a ranks (Li2CuSb rank {pc['CuLi2Sb']}, Cu3Nb2O8 rank "
                 f"{pc['Cu3Nb2O8']}: {'passed' if pc['passed'] else 'FAILED'}).")
        coverage = f"{cov['in_vocab']} of {cov['unique_formulas']:,} normalized formulas are in the Mat2vec vocabulary;"
        variance = f"~{100 * sum(meta['pca_explained']):.0f}%"
        built = f"Artifacts built {meta['built_at']}."
    else:  # limited mode: no artifacts on this machine
        check = "The build step checks that it reproduces the paper's Fig. 2a thermoelectric ranks."
        coverage = "only formulas in the Mat2vec vocabulary can be ranked;"
        variance = "a small fraction"
        built = "No artifacts are present here, so the app runs in limited mode."
    st.title("About this project")
    st.markdown(f"""
This dashboard asks a narrow question: **how does the materials-science literature, as captured by a 2019 word-embedding
model, associate known 2D semiconductors with properties and thin-film deposition methods?**

### Mat2vec
- Word2Vec (skip-gram, 200 dimensions) trained by Tshitoyan *et al.* on ~1.5 million materials-science abstracts
  published up to 2018: *Nature* 571, 95–98 (2019), doi:10.1038/s41586-019-1335-8.
- Every word, phrase (e.g. `chemical_vapor_deposition`) and normalized chemical formula (e.g. WS2 → `S2W`) has
  two vectors. **Similarity (input·input)** compares the contexts two words appear in; **co-occurrence
  (output·input)** estimates how likely a material is to be mentioned next to a word, the score the paper used to
  predict thermoelectrics.
- The pretrained model is used as-is, never retrained. {check}

### MC2D
- Materials Cloud two-dimensional crystals database: Mounet *et al.*, *Nat. Nanotechnol.* 13, 246 (2018),
  doi:10.1038/s41565-017-0035-5; Campi *et al.*, *ACS Nano* 17, 11268 (2023). Retrieved through its public
  OPTIMADE API; licensed CC BY 4.0.
- Candidates are MC2D entries with a DFT band gap > 0; {coverage} polymorphs sharing a formula are merged
  (the band gap is shown as a range).

### Scientific limitations
- Mat2vec captures relationships in the literature up to 2018, **not proven deposition feasibility**, film quality
  or optimal processing conditions.
- Rankings are biased by how often a material is written about: co-occurrence favours famous materials,
  similarity favours rare tokens with unreliable vectors (see the Evaluation page).
- Unreported material–method pairs are unlabeled, not negatives. The verified literature set is small, and most of
  its papers predate 2018, so recovery results are exploratory and largely in-sample.
- A formula token mixes all polymorphs and phases; DFT band gaps are approximate.
- The PCA map shows {variance} of the embedding variance; rankings always use the full
  200-dimensional vectors. Chemical-family colours follow a display convention (first match of halide, telluride,
  selenide, sulfide, oxide, nitride, pnictide, carbide), not a chemical classification.

### Licences and provenance
- This project's code: MIT. `explorer/formula.py` is adapted from mat2vec (MIT, © 2019 Authors of Tshitoyan
  *et al.*). MC2D data: CC BY 4.0. Full notices: `THIRD_PARTY_NOTICES.md`.
- The pretrained Mat2vec embeddings, and the `artifacts/` derived from them, are distributed under the mat2vec
  MIT License (Copyright (c) 2019 Authors of Tshitoyan et al. Nature (2019)); redistribution of the pretrained weights under that licence was confirmed by first author Vahe Tshitoyan via email in October 2026.
  The original ~880 MB model files are not included. {built}{HOSTED_ABOUT if hosted else ""}
""")


def limited_mode():
    st.warning(
        # Folder name only: on a hosted demo the absolute path is the server's, which means nothing to visitors.
        f"**Limited mode:** no model artifacts found in '{Path(ARTIFACTS).name}'. The About page and the literature dataset work "
        "without them. Unavailable until the pretrained Mat2vec model is installed: rankings (co-occurrence and "
        "similarity), the 2D/3D embedding map, context words, free-text targets, and the popularity-bias analysis "
        "and recovery metrics. To enable them, download the pretrained Mat2vec files (see README) and build the "
        "artifacts first:  python -m explorer.build_data")
    pages = ["Literature", "About"]
    requested = st.query_params.get("page")
    page = st.sidebar.radio("Page", pages, key="page", index=pages.index(requested) if requested in pages else 0)
    literature() if page == "Literature" else about()


def main():
    # Checked outside the cache so a build made while the app runs is picked up on the next rerun.
    # Full local artifacts first (unchanged behaviour), then the results-only bundle, then limited mode.
    if all(os.path.exists(os.path.join(ARTIFACTS, f)) for f in FILES):
        data, pages = load_artifacts(ARTIFACTS), ["Explore", "Evaluation", "About"]
        source, fix = ARTIFACTS, "python -m explorer.build_data"
    elif all(os.path.exists(os.path.join(PUBLIC, f)) for f in PUBLIC_FILES):
        data, pages = load_public(PUBLIC), ["Explore", "Evaluation", "Literature", "About"]
        source, fix = PUBLIC, "python -m explorer.public_bundle"
    else:
        limited_mode()
        return
    if not data["aligned"]:
        st.error(f"Artifacts in '{source}' are out of sync (candidate metadata and scores are in a different "
                 f"order). Rebuild them:  {fix}")
        st.stop()
    requested = st.query_params.get("page")  # ?page=Evaluation gives a shareable link
    page = st.sidebar.radio("Page", pages, key="page", index=pages.index(requested) if requested in pages else 0)
    # One filter for every page and both ranking modes.
    min_count = st.sidebar.slider(
        "Minimum token count", 1, 100, 4, key="min_count",
        help="How many times the material's token occurs in the Mat2vec training corpus. "
             "Default 4 (count > 3) matches the paper's filter. Lower it to include rare materials.")
    st.sidebar.caption("Token count = occurrences of the material's token in the Mat2vec training corpus, "
                       "not the number of publications about it.")
    if page == "Explore":
        explore(data, min_count)
    elif page == "Evaluation":
        evaluation(data, min_count)
    elif page == "Literature":
        literature()
    else:
        about(data["meta"], hosted=data["mode"] == "public")


main()
