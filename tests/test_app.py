"""Streamlit AppTest smoke tests. All but the first need local artifacts (python -m explorer.build_data)."""
import pandas as pd
from streamlit.testing.v1 import AppTest

APP = "../explorer/app.py"   # AppTest resolves relative to this test file
CAVEAT = "Literature association before 2018, not feasibility."


def text(at):
    """All visible text of the rendered app, for substring checks."""
    parts = [e.value for kind in ("markdown", "caption", "info", "warning", "error") for e in getattr(at, kind)]
    return "\n".join(str(p) for p in parts)


def run():
    return AppTest.from_file(APP, default_timeout=60).run()


def test_missing_artifacts_shows_build_command(monkeypatch, tmp_path):
    monkeypatch.setenv("EXPLORER_ARTIFACTS", str(tmp_path))
    at = run()
    assert any("python -m explorer.build_data" in e.value for e in at.error)


def test_explore_renders_with_default_target():
    at = run()
    assert not at.exception and len(at.dataframe) >= 2
    assert any(CAVEAT in c.value for c in at.caption)


def test_min_count_default_matches_paper_and_filters_both_tables():
    n_paper = int((pd.read_parquet("artifacts/candidates.parquet")["count"] > 3).sum())
    at = run()
    assert at.slider(key="min_count").value == 4
    assert f"{n_paper} of 591 candidates shown" in text(at)
    assert len(at.dataframe[0].value) == len(at.dataframe[1].value)   # same filter for both modes
    at.slider(key="min_count").set_value(1).run()
    assert "591 of 591 candidates shown" in text(at)


def test_explains_token_count_bias_coverage_and_pca_limits():
    at = run()
    body = text(at)
    assert "training corpus" in body and "not the number of publications" in body
    assert "favours frequently discussed materials" in body and "rare tokens" in body
    assert "591 of 1,362" in body
    assert "200-dimensional" in body and "19" in body


def test_3d_toggle_renders():
    at = run()
    at.radio(key="dims").set_value("3D").run()
    assert not at.exception


def test_unknown_free_text_target_is_friendly():
    at = run()
    at.text_input(key="free").input("unobtainium").run()
    assert not at.exception and any("not in vocabulary" in w.value for w in at.warning)


def eval_page():
    at = run()
    at.radio(key="page").set_value("Evaluation").run()
    return at


def test_evaluation_page_reports_bias_without_precision_metrics():
    at = eval_page()
    assert not at.exception
    assert any("Spearman" in m.label for m in at.metric)
    assert not any("AUPRC" in m.label or "precision" in m.label.lower() for m in at.metric)
    body = text(at)
    assert "unlabeled, not confirmed negatives" in body and "n = " in body
    assert "in-sample" in body   # most verified papers predate the 2018 training cut-off


def test_evaluation_page_lists_references_and_method_selector():
    at = eval_page()
    at.selectbox(key="eval_method").set_value("Molecular beam epitaxy (MBE)").run()
    assert not at.exception
    refs = at.dataframe[-1].value
    assert "DOI" in refs.columns and len(refs) > 0


def test_evaluation_page_marks_insufficient_data():
    at = eval_page()
    at.toggle(key="direct2d").set_value(True).run()
    assert not at.exception and "insufficient data" in text(at).lower() + " ".join(
        str(v) for df in at.dataframe for v in df.value.astype(str).values.ravel()).lower()
