import subprocess
import sys


def test_no_personal_email_in_tracked_files():
    hits = subprocess.run(["git", "grep", "-n", "-I", "-E", r"[A-Za-z0-9._%+-]+@gmail\.com"],
                          capture_output=True, text=True).stdout
    assert hits == "", hits


def test_mc2d_attribution_discloses_changes_and_links_licence():
    notices = open("THIRD_PARTY_NOTICES.md").read()
    assert "https://creativecommons.org/licenses/by/4.0/" in notices
    for change in ("band gap > 0", "normalized", "merged", "Mat2vec vocabulary"):
        assert change in notices, change


def test_readme_test_count_matches_suite():
    import re
    collected = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q"], capture_output=True, text=True).stdout
    n = int(re.search(r"(\d+) tests? collected", collected).group(1))
    readme = open("README.md").read()
    assert f"{n} tests" in readme and not any(f"{m} tests" in readme for m in re.findall(r"(\d+) tests", readme) if int(m) != n)


ARTIFACT_FILES = {"artifacts/vectors.npz", "artifacts/candidates.parquet", "artifacts/coords.parquet", "artifacts/meta.json"}


def test_raw_model_files_stay_ignored_and_only_the_four_artifacts_are_shipped():
    """Policy: the four derived artifacts/ files are distributed (MIT, confirmed); the ~880 MB raw model never is."""
    for path in ("pretrained_embeddings", "pretrained_embeddings.wv.vectors.npy",
                 "models/pretrained_embeddings.trainables.syn1neg.npy", "anything/vectors.npz", "artifacts/other.npz"):
        assert subprocess.run(["git", "check-ignore", "-q", path]).returncode == 0, path
    assert subprocess.run(["git", "check-ignore", "-q", "artifacts/vectors.npz"]).returncode == 1
    tracked = subprocess.run(["git", "ls-files"], capture_output=True, text=True).stdout.split()
    assert not [f for f in tracked if f.endswith(".npy") or "pretrained_embeddings" in f]
    assert {f for f in tracked if f.endswith((".npz", ".parquet"))} == ARTIFACT_FILES - {"artifacts/meta.json"}
    assert {f for f in tracked if f.startswith("artifacts/")} == ARTIFACT_FILES
    from pathlib import Path
    assert max(Path(f).stat().st_size for f in tracked) < 50_000_000   # GitHub warns at 50 MB


def test_readme_states_confirmed_mit_licence_for_the_weights():
    readme = open("README.md").read()
    for phrase in ("MIT License", "confirmed by first author Vahe Tshitoyan via email in October 2026",
                   "Copyright (c) 2019 Authors of Tshitoyan et al. Nature (2019)", "10.1038/s41586-019-1335-8"):
        assert phrase in readme, phrase


def test_no_outdated_licensing_claims():
    hits = subprocess.run(["git", "grep", "-n", "-I", "-i", "-E",
                           r"licen[cs]ing (status )?(of [^.]* )?(is |remains )?unresolved|redistribution terms (are|have not been) "
                           r"(unresolved|unverified|verified)|does not cover the (pretrained|weights)|not cover the weights|"
                           r"no licen[cs]e for them has been confirmed|weights are \*\*not\*\* redistributed|embeddings are \*\*not\*\* redistributed",
                           "--", ".", ":!tests", ":!docs/superpowers"], capture_output=True, text=True).stdout
    assert hits == "", hits


def test_no_email_addresses_in_tracked_files():
    hits = subprocess.run(["git", "grep", "-n", "-I", "-E", r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+\.[A-Za-z]{2,}",
                           "--", ".", ":!tests/test_repo_hygiene.py"], capture_output=True, text=True).stdout
    assert hits == "", hits


def test_license_holder_and_no_local_paths():
    assert "Copyright (c) 2026 Alex Xu" in open("LICENSE").read()
    hits = subprocess.run(["git", "grep", "-n", "-I", "-E", r"/Users/|/opt/anaconda", "--", ".", ":!tests/test_repo_hygiene.py"], capture_output=True, text=True).stdout
    assert hits == "", hits


def test_no_categorical_reconstruction_claims():
    hits = subprocess.run(["git", "grep", "-n", "-I", "-i", "-E",
                           r"cannot reconstruct|can.t reconstruct|impossible to reconstruct|cannot be reconstructed",
                           "--", ".", ":!tests/test_repo_hygiene.py"], capture_output=True, text=True).stdout
    assert hits == "", hits


def test_streamlit_cloud_uses_requirements_txt_with_runtime_deps_only():
    """Community Cloud takes the first of uv.lock, Pipfile, environment.yml, requirements.txt, pyproject.toml,
    looking in the entrypoint folder (explorer/) and then the repo root."""
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    for folder in (root / "explorer", root):
        for name in ("uv.lock", "Pipfile", "environment.yml", "pyproject.toml"):
            assert not (folder / name).exists(), folder / name
    assert not (root / "explorer" / "requirements.txt").exists()
    deps = {line.split("==")[0] for line in (root / "requirements.txt").read_text().split() if line}
    assert deps == {"streamlit", "pandas", "numpy", "scipy", "plotly", "pyarrow"}   # scipy: pandas Spearman
    readme = (root / "README.md").read_text()
    assert "explorer/app.py" in readme and "Python 3.11" in readme and "Advanced settings" in readme
