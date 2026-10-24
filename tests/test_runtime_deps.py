"""Regression: the hosted app must run with only what requirements.txt installs (as on Streamlit Community Cloud).

The local environment also has build-only packages (gensim, pymatgen, and what they pull in, e.g. scipy), which can
hide a missing runtime dependency. This test runs every hosted page in a fresh process that blocks any installed
package outside the dependency closure of requirements.txt.
"""
import subprocess
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SCRIPT = textwrap.dedent('''
    import importlib.abc, os, sys
    from importlib.metadata import PackageNotFoundError, packages_distributions, requires
    from packaging.requirements import Requirement

    def closure(names):
        seen, todo = set(), list(names)
        while todo:
            name = todo.pop().lower().replace("_", "-")
            if name in seen:
                continue
            seen.add(name)
            try:
                reqs = requires(name) or []
            except PackageNotFoundError:
                continue
            todo += [r.name for r in map(Requirement, reqs) if r.marker is None or r.marker.evaluate({"extra": ""})]
        return seen

    allowed = closure(line.split("==")[0] for line in open("requirements.txt").read().split())
    owners = packages_distributions()

    class OnlyRuntimeDeps(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path, target=None):
            top = name.split(".")[0]
            dists = {d.lower().replace("_", "-") for d in owners.get(top, [])}
            if dists and not dists & allowed:
                raise ModuleNotFoundError(f"No module named {top!r} (not installed by requirements.txt)", name=top)
            return None

    sys.meta_path.insert(0, OnlyRuntimeDeps())
    os.environ["EXPLORER_ARTIFACTS"] = os.path.join(os.getcwd(), "no-artifacts-here")   # hosted: bundle only
    from streamlit.testing.v1 import AppTest

    failures = []
    for query in ({}, {"view": "3D"}, {"page": "Evaluation"}, {"page": "Literature"}, {"page": "About"}):
        at = AppTest.from_file("explorer/app.py", default_timeout=60)
        for k, v in query.items():
            at.query_params[k] = v
        at.run()
        failures += [f"{query}: {e.value}" for e in at.exception]
    print("\\n".join(failures) or "ok")
    sys.exit(1 if failures else 0)
''')


def test_hosted_pages_run_with_only_requirements_txt():
    result = subprocess.run([sys.executable, "-c", SCRIPT], cwd=ROOT, capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-2000:]
