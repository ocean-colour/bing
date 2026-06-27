# HOWTO: Release BING to PyPI

This guide walks through publishing the **BING** package
(distribution name `bing-ocean`, import name `bing`) to the Python
Package Index. It reflects the current state of the repo after the
packaging work in `prompts/pip.md` (Tasks 1–2).

---

## 0. Current state (what is already done)

These are in place and need no further work:

- ✅ `pyproject.toml` — PEP 621 metadata, trimmed dependencies, dynamic
  version from `bing/__init__.py:__version__`, `bing_fit_Rrs` console
  entry point, and `package-data` globs that bundle `bing/data/**`.
- ✅ `setup.py` — thin shim (metadata lives in `pyproject.toml`).
- ✅ `MANIFEST.in` — controls the source distribution (sdist) contents.
- ✅ `.github/workflows/tests.yml` — CI runs the test suite on 3.11/3.12.
- ✅ `README.md` — long description rendered on the PyPI project page.
- ✅ `LICENSE` — BSD 3-Clause.
- ✅ Build verified: `python -m build` produces a valid sdist + wheel with
  the data files and entry point included.

---

## 1. Blockers to resolve BEFORE the first upload

These must be decided/fixed or the release will be broken or impossible.

### 1a. The `ocpy` dependency (hard blocker)

`bing` imports `ocpy` in ~26 places, but:

- `ocpy` (this project) lives only at
  `https://github.com/ocean-colour/ocpy` and is **not** on PyPI.
- A **different, unrelated** package named `ocpy` (v0.5.x) already
  occupies that name on PyPI.

So `pip install bing-ocean` cannot pull the correct `ocpy` automatically,
and we must NOT list the bare name `ocpy` as a dependency. Choose one:

- **(Recommended)** Publish *this* `ocpy` to PyPI under a unique name
  (e.g. `ocpy-ocean`), then add it to `[project].dependencies` in
  `pyproject.toml`. This makes `pip install bing-ocean` fully self-contained.
- **(Interim)** Keep `ocpy` out of the dependency list and document the
  manual install (already done in `README.md`):
  `pip install git+https://github.com/ocean-colour/ocpy.git`.
  Note: PyPI **rejects** direct-URL/VCS dependencies in uploaded metadata,
  so a `git+` dependency cannot live in `pyproject.toml` for a PyPI release.

The optional `correct_atmosphere` import (in `bing/fitting/l23.py`) has the
same issue; consider making it a lazy/optional import so importing the
subpackage does not hard-fail without it.

### 1b. Confirm the distribution name

`bing-ocean` is set in `pyproject.toml`. Confirm it is free / owned:

```bash
pip index versions bing-ocean   # should 404 / show nothing if available
```

If taken, pick another name and update `[project].name`.

### 1c. Pick the release version

Currently `bing/__init__.py:__version__ = "0.1.0"`. The README previously
claimed "2.0.0". Decide the real first-release version and set it in
`bing/__init__.py` (the single source of truth). PyPI versions are
**immutable** — you cannot re-upload the same version.

---

## 2. One-time account setup

1. Create accounts on **TestPyPI** (https://test.pypi.org) and
   **PyPI** (https://pypi.org). Enable 2FA on both.
2. Create an **API token** for each (Account Settings → API tokens).
   Scope it to "Entire account" for the first upload, then re-scope to the
   project afterwards.
3. Store credentials in `~/.pypirc` (or use the token at the `twine`
   prompt, or environment variables `TWINE_USERNAME=__token__` /
   `TWINE_PASSWORD=<token>`):

   ```ini
   [distutils]
   index-servers =
       pypi
       testpypi

   [pypi]
   username = __token__
   password = pypi-AgEI...        # your PyPI token

   [testpypi]
   repository = https://test.pypi.org/legacy/
   username = __token__
   password = pypi-AgEI...        # your TestPyPI token
   ```

   `chmod 600 ~/.pypirc` to protect it.

---

## 3. Install the build tooling

`twine` is **not** currently installed in the `ocean14` env.

```bash
conda activate ocean14
pip install --upgrade build twine
```

---

## 4. Build the distributions

From the repo root:

```bash
# Clean any stale artifacts first
rm -rf dist build *.egg-info

python -m build        # writes dist/bing_ocean-<version>.tar.gz and .whl
```

---

## 5. Validate the artifacts

```bash
twine check dist/*     # verifies metadata + README renders on PyPI
```

Optionally inspect that the data files and entry point are bundled:

```bash
unzip -l dist/bing_ocean-*.whl | grep -E 'data/|entry_points'
```

---

## 6. Test on TestPyPI first (strongly recommended)

```bash
twine upload --repository testpypi dist/*
```

Then install from TestPyPI into a fresh environment and smoke-test. Use
the main PyPI as the fallback index so real dependencies resolve, and
install `ocpy` from git (until blocker 1a is resolved):

```bash
conda create -n bing-test python=3.11 -y
conda activate bing-test
pip install git+https://github.com/ocean-colour/ocpy.git
pip install \
  --index-url https://test.pypi.org/simple/ \
  --extra-index-url https://pypi.org/simple/ \
  bing-ocean

python -c "import bing; print(bing.__version__)"
bing_fit_Rrs --help
```

---

## 7. Upload to the real PyPI

Once TestPyPI looks good:

```bash
twine upload dist/*
```

Verify the project page at `https://pypi.org/project/bing-ocean/` and a
clean install:

```bash
pip install bing-ocean
```

---

## 8. Tag the release in git

Keep the git tag in sync with the published version:

```bash
git tag -a v0.1.0 -m "BING 0.1.0 — first PyPI release"
git push origin v0.1.0
```

Optionally create a GitHub Release from the tag (changelog + attach the
sdist/wheel).

---

## 9. (Optional) Automate releases with GitHub Actions

Use PyPI **Trusted Publishing** (OIDC, no long-lived tokens):

1. On PyPI: project → Settings → Publishing → add a trusted publisher
   pointing at `ocean-colour/bing`, workflow `publish.yml`,
   environment `pypi`.
2. Add `.github/workflows/publish.yml` that triggers `on: release:
   types: [published]`, runs `python -m build`, and uses
   `pypa/gh-action-pypi-publish` (which needs `permissions:
   id-token: write`). No API token is stored in the repo.

---

## 10. Post-release housekeeping

- Bump `__version__` to the next dev version (e.g. `0.1.1.dev0`) so future
  builds are not mistaken for the release.
- Update `CHANGES`/release notes.
- Confirm the ReadTheDocs build (`oc-bing`) picks up the tag.

---

## Quick reference (the happy path, after blockers resolved)

```bash
conda activate ocean14
pip install --upgrade build twine
rm -rf dist build *.egg-info
python -m build
twine check dist/*
twine upload --repository testpypi dist/*    # test first
twine upload dist/*                          # then real PyPI
git tag -a v<version> -m "BING <version>" && git push origin v<version>
```
