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
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from explorer import ranking as r  # noqa: E402

ARTIFACTS = os.environ.get("EXPLORER_ARTIFACTS", "artifacts")
CAVEAT = "Literature association before 2018, not feasibility."
DEFAULT_TARGET = "chemical_vapor_deposition"
MODES = {"out_in": "Co-occurrence score (output·input)", "in_in": "Similarity score (input·input)"}
TOP = 20
MIN_POS = 10
KNOWN = Path(__file__).resolve().parents[1] / "data" / "known_methods.csv"
METHODS = {
    "Chemical vapor deposition (CVD)": "chemical_vapor_deposition",
    "Atomic layer deposition (ALD)": "ALD",
    "Molecular beam epitaxy (MBE)": "molecular_beam_epitaxy",
    "Magnetron sputtering": "magnetron_sputtering",
    "Pulsed laser deposition (PLD)": "pulsed_laser_deposition",
}
RANKINGS = {"out_in": "Co-occurrence (out·in)", "in_in": "Similarity (in·in)", "log_count": "Token-count baseline"}
LIMITATIONS = """**Read these results with care.**
- Mat2vec captures relationships in the literature up to 2018, not proven deposition feasibility.
- Unreported material–method pairs are unlabeled, not confirmed negatives.
- Some verified papers report thin films, bulk crystals, post-deposition conversion (e.g. selenizing a metal film)
  or a different polymorph than the MC2D entry; see *Growth type* and *Phase* below.
- Positive sets are small (≈10–25 per method), so differences between methods are indicative, not conclusive.
- A higher similarity score does not imply better deposition quality or optimal processing conditions."""

st.set_page_config(page_title="2D Semiconductor Explorer", layout="wide")


FILES = ["vectors.npz", "candidates.parquet", "coords.parquet", "meta.json"]


@st.cache_resource  # once per server and path: no 42 MB copy on every widget change
def load_artifacts(path):
    vec = dict(np.load(os.path.join(path, "vectors.npz")))
    cand = pd.read_parquet(os.path.join(path, "candidates.parquet"))
    cand = cand.merge(pd.read_parquet(os.path.join(path, "coords.parquet")), on="token")
    with open(os.path.join(path, "meta.json")) as f:
        meta = json.load(f)
    vec["vocab_index"] = {t: i for i, t in enumerate(vec["vocab_tokens"])}
    return {"vec": vec, "cand": cand, "meta": meta}


def badges(row):
    notes = []
    if row["low_count"]:
        notes.append("low count")
    if row["polymorphs_merged"]:
        notes.append(f"{row['n_entries']} polymorphs merged")
    return ", ".join(notes)


def gap_label(row):
    lo, hi = row["gap_min"], row["gap_max"]
    return f"{lo:.2f}" if np.isclose(lo, hi) else f"{lo:.2f}–{hi:.2f}"


def ranking_table(df, mode):
    out = df.sort_values(f"rank_{mode}")
    return pd.DataFrame({
        "Rank": out[f"rank_{mode}"],
        "Formula (MC2D)": out["formula_examples"],
        "Mat2vec token": out["token"],
        "Score": out[mode].round(3),
        "Band gap (eV)": out.apply(gap_label, axis=1),
        "Token count": out["count"],
        "Notes": out.apply(badges, axis=1),
    })


def pick_target(vec):
    targets = list(vec["targets"])
    chosen = st.sidebar.selectbox(
        "Target property or deposition method", targets, key="target",
        index=targets.index(DEFAULT_TARGET) if DEFAULT_TARGET in targets else 0,
        format_func=lambda t: t.replace("_", " "))
    free = st.sidebar.text_input("…or type any word (e.g. photodetector, MoS2)", key="free")
    if free.strip():
        resolved = r.resolve_target(free, vec["vocab_index"])
        if resolved is None:
            st.warning(f"“{free}” is not in vocabulary (top 50,000 Mat2vec tokens). "
                       f"Showing “{chosen.replace('_', ' ')}” instead.")
        else:
            chosen = resolved
    return chosen


def explore(data, min_count):
    vec, cand, meta = data["vec"], data["cand"].copy(), data["meta"]
    cov = meta["coverage"]

    st.title("2D Semiconductor Materials Explorer")
    st.markdown(
        "Rank 2D semiconductors from the **MC2D** database by how the pretrained **Mat2vec** word "
        "embeddings (Tshitoyan et al., *Nature* 2019; trained on ~1.5 million materials-science "
        "abstracts up to 2018) associate them with a property or deposition method.")
    st.caption(f"Coverage: {cov['in_vocab']} of {cov['unique_formulas']:,} normalized MC2D semiconductor "
               f"formulas are in the Mat2vec vocabulary; the other {cov['dropped_not_in_vocab']} "
               "(mostly complex multi-element compounds) have no embedding and cannot be ranked.")

    target = pick_target(vec)
    gmin, gmax = float(cand.gap_min.min()), float(cand.gap_max.max())
    gap_lo, gap_hi = st.sidebar.slider("Band gap range (eV)", gmin, gmax, (gmin, gmax), key="gap")

    # Score all candidates on the full 200-d vectors, then apply ONE filter to both modes.
    t_in = vec["vocab_in"][vec["vocab_index"][target]]
    cand["out_in"] = r.score(vec["cand_out"], t_in)
    cand["in_in"] = r.score(vec["cand_in"], t_in)
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
    detail(shown, vec, t_in)


def pca_view(shown, meta):
    ev = [100 * x for x in meta["pca_explained"]]
    st.subheader("Embedding map")
    c1, c2 = st.columns(2)
    dims = c1.radio("View", ["2D", "3D"], horizontal=True, key="dims")
    color_by = c2.radio("Colour by", ["Chemical family", "Co-occurrence score", "Band gap"],
                        horizontal=True, key="color")
    color = {"Chemical family": "family", "Co-occurrence score": "out_in", "Band gap": "gap_max"}[color_by]
    labels = {f"pc{i + 1}": f"PC{i + 1} ({ev[i]:.1f}%)" for i in range(3)}
    labels.update({"family": "Family", "out_in": "Co-occurrence", "gap_max": "Band gap max (eV)"})
    hover = {"token": True, "gap_min": ":.2f", "gap_max": ":.2f", "count": True,
             "pc1": False, "pc2": False, "pc3": False}
    common = dict(color=color, hover_name="formula_examples", hover_data=hover, labels=labels, height=560)
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


def detail(shown, vec, t_in):
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
    i = int(np.where(vec["cand_tokens"] == token)[0][0])
    words = r.context_words(vec["cand_out"][i], vec["vocab_in"], list(vec["vocab_tokens"]), t_in)
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
    vec, cand = data["vec"], data["cand"]
    st.title("Evaluation: what the embeddings capture, and where they fail")
    st.warning(LIMITATIONS)

    direct = st.toggle("Direct 2D growth only", key="direct2d",
                       help="Count only papers that grow the 2D material directly; exclude thin films, "
                            "precursor conversion, hybrid processes and bulk crystals.")
    known = load_known()
    if direct:
        known = known[known.growth_type == "direct_2d"]
    keep = (cand["count"] >= min_count).to_numpy()
    sub, idx = cand[keep], np.flatnonzero(keep)
    st.markdown(f"Evaluated on **n = {len(sub)}** MC2D candidates (token count ≥ {min_count}). "
                f"Known positives: verified literature pairs from `data/known_methods.csv`"
                f"{' (direct 2D growth only)' if direct else ''}. Recovery metrics need at least "
                f"{MIN_POS} positives; otherwise they read *insufficient data*.")
    n_pre = int((pd.to_numeric(known.year, errors="coerce") <= 2018).sum())
    st.info(f"**Recovery is largely in-sample.** {n_pre} of {len(known)} verified papers were published in "
            "2018 or earlier, inside Mat2vec's training window, so the model may have read these very abstracts. "
            "High recovery shows the embeddings encode known literature associations; it is not a test of "
            "predicting new material–process pairs.")

    reports, rows = {}, []
    for label, token in METHODS.items():
        t_in = vec["vocab_in"][vec["vocab_index"][token]]
        positive, _ = r.match_known(list(known.token[known.method == token]), list(sub.token))
        rep = r.method_report(vec["cand_in"][idx], vec["cand_out"][idx], t_in, sub["count"].to_numpy(),
                              positive, k=TOP, min_pos=MIN_POS)
        reports[label] = (rep, positive, t_in)
        for mode, row in rep.iterrows():
            rows.append({"Method": label, "Ranking": RANKINGS[mode], "Known positives (n)": int(row.n_positive),
                         f"Recall@{TOP}": fmt(row.recall_at_k, row.sufficient),
                         "Median percentile rank (lower = better, 50 = chance)":
                             fmt(row.median_percentile, row.sufficient, 1),
                         "Spearman vs log token count": f"{row.spearman_log_count:.2f}"})
    st.subheader("Ranking performance across deposition methods")
    st.dataframe(pd.DataFrame(rows), hide_index=True)
    st.caption("Positive-only metrics: unlabeled materials are never counted as wrong. The token-count "
               "baseline ranks materials by popularity alone; Mat2vec adds value only where it beats it.")

    bars = pd.DataFrame([{"Method": label, "Ranking": RANKINGS[m], "Spearman": rep.loc[m, "spearman_log_count"]}
                         for label, (rep, _, _) in reports.items() for m in ("out_in", "in_in")])
    st.subheader("Popularity bias by method")
    st.plotly_chart(px.bar(bars, x="Method", y="Spearman", color="Ranking", barmode="group", height=380,
                           labels={"Spearman": "Spearman: score vs log token count"}), width="stretch")
    st.caption("Above 0: the ranking favours frequently mentioned materials. Below 0: it favours rare "
               "tokens, whose embeddings are learned from very little text.")

    label = st.selectbox("Deposition method", list(METHODS), key="eval_method")
    rep, positive, t_in = reports[label]
    m = st.columns(3)
    m[0].metric("Known positives (n)", f"{int(rep.n_positive.iloc[0])} of {len(sub)}")
    m[1].metric("Spearman, co-occurrence vs log count", f"{rep.loc['out_in', 'spearman_log_count']:.2f}")
    m[2].metric("Spearman, similarity vs log count", f"{rep.loc['in_in', 'spearman_log_count']:.2f}")
    mode = st.radio("Score shown", ["out_in", "in_in"], horizontal=True, key="eval_mode",
                    format_func=lambda k: RANKINGS[k])
    plot = pd.DataFrame({"Formula": sub.formula_examples, "Token count (log scale)": sub["count"],
                         "Score": r.score(vec["cand_out" if mode == "out_in" else "cand_in"][idx], t_in),
                         "Literature": np.where(positive, "verified pair", "unlabeled")})
    st.plotly_chart(px.scatter(plot, x="Token count (log scale)", y="Score", color="Literature", log_x=True,
                               hover_name="Formula", height=420,
                               category_orders={"Literature": ["verified pair", "unlabeled"]}), width="stretch")

    st.subheader(f"Literature references: {label}")
    refs = known[known.method == METHODS[label]]
    st.caption(f"n = {len(refs)} verified material–process pairs. Each was checked against the paper's title "
               "or abstract; growth type and phase are recorded because many reports are thin films, "
               "precursor conversion or a different polymorph than the MC2D entry.")
    st.dataframe(pd.DataFrame({
        "Formula": refs.formula, "Phase": refs.phase, "Growth type": refs.growth_type, "Year": refs.year,
        "DOI": "https://doi.org/" + refs.doi, "Note": refs.note, "Title": refs.title}),
        hide_index=True, column_config={"DOI": st.column_config.LinkColumn("DOI", display_text=r"doi\.org/(.*)")})


def main():
    # Checked outside the cache so a build made while the app runs is picked up on the next rerun.
    if not all(os.path.exists(os.path.join(ARTIFACTS, f)) for f in FILES):
        st.error(f"No artifacts found in '{ARTIFACTS}'. Build them first:  python -m explorer.build_data")
        st.stop()
    data = load_artifacts(ARTIFACTS)
    page = st.sidebar.radio("Page", ["Explore", "Evaluation", "About"], key="page")
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
    else:
        st.info("The About page is coming in the next build step.")


main()
