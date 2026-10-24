"""Propose literature evidence for material–deposition-method pairs (provenance helper, not app code).

    python tools/find_literature.py [openalex|crossref] [method ...]
        -> data/literature_candidates_<source>.csv  (git-ignored)

For each (method, material) it searches OpenAlex and keeps the best article whose TITLE contains the exact
formula and whose title or abstract names the method. growth_type / phase are keyword SUGGESTIONS only:
every row is reviewed by hand before it is copied into data/known_methods.csv.
"""
import csv
import re
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from explorer.formula import normalized_formula  # noqa: E402

API = "https://api.openalex.org/works"
MAILTO = "you@example.com"
METHODS = {
    "chemical_vapor_deposition": ("chemical vapor deposition", r"chemical vapou?r deposition|\bCVD\b"),
    "ALD": ("atomic layer deposition", r"atomic layer deposition|\bALD\b"),
    "molecular_beam_epitaxy": ("molecular beam epitaxy", r"molecular[- ]beam epitaxy|\bMBE\b"),
    "magnetron_sputtering": ("magnetron sputtering", r"magnetron[- ]sputter"),
    "pulsed_laser_deposition": ("pulsed laser deposition", r"pulsed[- ]laser deposition|\bPLD\b"),
}
MATERIALS = ["MoS2", "WS2", "MoSe2", "WSe2", "MoTe2", "WTe2", "ReS2", "ReSe2", "PtSe2", "PtS2", "SnS2", "SnSe2",
             "SnS", "SnSe", "GeSe", "GeS", "GaSe", "GaS", "GaTe", "InSe", "HfS2", "ZrS2", "HfSe2", "ZrSe2", "BN",
             "Bi2Se3", "Bi2Te3", "Bi2S3", "Sb2Se3", "Ga2S3", "TiS3", "FePS3", "NiPS3", "As2S3", "SiP"]
GROWTH = [  # first match wins
    ("conversion", r"sulfuri[sz]|seleni[sz]|telluri[sz]|chalcogeni[sz]|conversion of|converted"),
    ("hybrid", r"hybrid"),
    ("bulk", r"bulk crystal|single crystals? (grown|growth)"),
    ("direct_2d", r"monolayer|bilayer|few[- ]layer|atomically thin|two[- ]dimensional|\b2D\b|ultrathin"),
]
PHASE = r"\b(1T'|1T′|1Td|1T|2H|3R|[αβγ]-|hexagonal|orthorhombic|rhombohedral|cubic)"


def abstract_text(inv):
    if not inv:
        return ""
    words = sorted((pos, w) for w, positions in inv.items() for pos in positions)
    return " ".join(w for _, w in words)


def best_hit(material, phrase, method_re):
    formula_re = re.compile(r"(?<![A-Za-z])" + re.escape(material) + r"(?![a-z0-9])")
    if material == "BN":
        formula_re = re.compile(r"(?<![A-Za-z])h?-?BN(?![a-z0-9])|boron nitride", re.I)
    params = {"filter": f"title.search:{material if material != 'BN' else 'boron nitride'},"
                        f"title_and_abstract.search:{phrase},type:article",
              "per-page": 50, "mailto": MAILTO, "select": "doi,title,publication_year,abstract_inverted_index,cited_by_count"}
    resp = requests.get(API, params=params, timeout=60)
    resp.raise_for_status()  # a rate-limit error must not be mistaken for "no literature"
    hits = []
    for w in resp.json()["results"]:
        title = re.sub(r"<[^>]+>", "", w.get("title") or "")
        abstract = abstract_text(w.get("abstract_inverted_index"))
        if not w.get("doi") or not formula_re.search(title):
            continue
        in_title = bool(re.search(method_re, title, re.I))
        sentences = [s for s in re.split(r"(?<=[.!?])\s+", abstract) if re.search(method_re, s, re.I)]
        if in_title or sentences:
            hits.append((in_title, w.get("cited_by_count", 0), w, title, abstract, sentences))
    if not hits:
        return None
    in_title, _, w, title, abstract, sentences = max(hits, key=lambda h: (h[0], h[1]))
    text = f"{title} {abstract}"
    growth = next((g for g, pat in GROWTH if re.search(pat, text, re.I)), "thin_film")
    phase = sorted(set(m.strip("-") for m in re.findall(PHASE, text)))
    return {"doi": w["doi"].replace("https://doi.org/", ""), "year": w.get("publication_year"), "title": title,
            "evidence": title if in_title else sentences[0][:300], "evidence_in": "title" if in_title else "abstract",
            "growth_suggested": growth, "phase_suggested": ";".join(phase)}


def best_hit_crossref(material, phrase, method_re):
    """Fallback without an API key: Crossref titles only, so formula AND method must both be in the title."""
    name = material if material != "BN" else "boron nitride"
    formula_re = re.compile(r"(?<![A-Za-z])" + re.escape(material) + r"(?![a-z0-9])") if material != "BN" \
        else re.compile(r"(?<![A-Za-z])h?-?BN(?![a-z0-9])|boron nitride", re.I)
    resp = requests.get("https://api.crossref.org/works", timeout=60,
                        headers={"User-Agent": f"mat2vec-2d-explorer (mailto:{MAILTO})"},
                        params={"query.bibliographic": f"{phrase} {name}", "rows": 40, "filter": "type:journal-article",
                                "select": "DOI,title,issued,is-referenced-by-count"})
    resp.raise_for_status()
    hits = []
    for it in resp.json()["message"]["items"]:
        title = re.sub(r"<[^>]+>", "", " ".join(it.get("title") or []))
        if formula_re.search(title) and re.search(method_re, title, re.I):
            hits.append((it.get("is-referenced-by-count", 0), it, title))
    if not hits:
        return None
    _, it, title = max(hits, key=lambda h: h[0])
    growth = next((g for g, pat in GROWTH if re.search(pat, title, re.I)), "thin_film")
    return {"doi": it["DOI"], "year": (it.get("issued", {}).get("date-parts") or [[None]])[0][0], "title": title,
            "evidence": title, "evidence_in": "title", "growth_suggested": growth,
            "phase_suggested": ";".join(sorted(set(m.strip("-") for m in re.findall(PHASE, title))))}


def main():
    source = sys.argv[1] if len(sys.argv) > 1 else "openalex"
    only = sys.argv[2:] or list(METHODS)
    finder = best_hit if source == "openalex" else best_hit_crossref
    out = Path(f"data/literature_candidates_{source}.csv")
    rows = []
    for method, (phrase, method_re) in METHODS.items():
        if method not in only:
            continue
        for material in MATERIALS:
            hit = finder(material, phrase, method_re)
            print(method, material, hit["doi"] if hit else "-", flush=True)
            if hit:
                rows.append({"method": method, "formula": material, "token": normalized_formula(material), **hit})
            time.sleep(0.15)
    Path("data").mkdir(exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(len(rows), f"candidates -> {out}")


if __name__ == "__main__":
    main()
