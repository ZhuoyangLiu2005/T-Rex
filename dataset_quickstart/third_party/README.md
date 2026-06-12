# third_party

Vendored robot descriptions, added as **git subtrees** (both Apache-2.0). Subtrees keep the files
in-repo — a normal `git clone` gets them (no `--recursive`) — with upstream history squashed.

- **`dexmate-urdf`** — the Dexmate Vega arm URDF, imported in `robot.py` as `from dexmate_urdf import robots`.
- **`sharpa-urdf-usd-xml`** — the Sharpa hand, **`wave`** variant. Its joint *ordering* is semantically
  the same as the dataset's hand DOFs (verified manually against the data), so `observation.state` maps
  onto the model joints correctly — no remap needed. Loaded by path (no install).

## Install the Dexmate arm package (needed for replay)

`robot.py` imports the `dexmate_urdf` Python package. Install the **vendored** copy from source
(per its own `third_party/dexmate-urdf/README.md`):

```bash
cd third_party/dexmate-urdf
cp -r robots/* src/dexmate_urdf/robots/
python scripts/workflows/generate_content.py
uv pip install -e .
```

Or, as an alternative, the public package: `uv pip install dexmate_urdf`.

The Sharpa hand needs no install — `robot.py` reads the MJCF directly via
`SHARPA_LEFT_HAND_MJCF_PATH` / `SHARPA_RIGHT_HAND_MJCF_PATH`.

## Add a subtree (one-time, from the repo root, clean working tree)

```bash
git subtree add --prefix third_party/dexmate-urdf        <DEXMATE_REPO_URL> <ref> --squash
git subtree add --prefix third_party/sharpa-urdf-usd-xml <SHARPA_REPO_URL>  <ref> --squash
```

- `<ref>` = a branch, tag, or commit.
- Keep each upstream `LICENSE`/`NOTICE`.

## Update later

```bash
git subtree pull --prefix third_party/dexmate-urdf        <DEXMATE_REPO_URL> <ref> --squash
git subtree pull --prefix third_party/sharpa-urdf-usd-xml <SHARPA_REPO_URL>  <ref> --squash
```

## Licensing & attribution

Both vendored packages are third-party software under the **Apache License 2.0**; they remain under
their original license, and this project's MIT license does not apply to them. They are vendored
**unmodified**, and each subtree retains its upstream `LICENSE`/`NOTICE`:

- **`dexmate-urdf`** — © Dexmate Inc., Apache-2.0 — see [`dexmate-urdf/LICENSE`](dexmate-urdf/LICENSE).
- **`sharpa-urdf-usd-xml`** — © Sharpa Group, Apache-2.0 — see
  [`sharpa-urdf-usd-xml/LICENSE.txt`](sharpa-urdf-usd-xml/LICENSE.txt). Per its
  [`NOTICE.txt`](sharpa-urdf-usd-xml/NOTICE.txt), it also incorporates MuJoCo
  (© DeepMind, Apache-2.0).
