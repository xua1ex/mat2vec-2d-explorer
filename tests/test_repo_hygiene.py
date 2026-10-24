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
