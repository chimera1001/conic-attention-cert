"""Shared helpers of the review computations (``experiments/review_*.py``): the computations added to Section 7 for the
journal version (the curvature certificate, CROWN with the certificate's score intervals, the terms of bound (39) on
gpt-oss-20b, the GPT-2 logit-margin exceptions).

They read the stored records of the other experiments (``--refs``, default ``results_dir()``; with ``--quick`` the
``*_quick`` directories of those experiments) and never write there. Every unit writes one JSON under
``<out>/<task>/`` (default ``results_dir()/review``, with ``--quick`` ``results_dir()/review_quick``; atomic, so a rerun
skips finished units), and the pools are the spawn pools of the other scripts (``jmlr_extensions.run_part``: workers x
threads within ``AIMP_MAX_THREADS``, log ``<out>/run.log``).
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "experiments"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # CPU only

from aimp.io import env_info, git_commit, jsonable, results_dir, write_json_atomic  # noqa: E402

VALID_TOL = 1e-7  # a bound is valid if it is >= reference_hi - 1e-7 max(1, |reference_hi|)
POS = 4
NAME_RE = re.compile(r"L(\d+)H(\d+)_P(\d+)_i(\d+)_r([0-9.e-]+?)(_fq)?\.json$")
_PROV: dict = {}


def dirs(out: Path | None, refs: Path | None, quick: bool) -> tuple[Path, Path, str]:
    """``(out, refs, suffix)``: the output directory, the root of the stored records, and ``"_quick"`` with ``--quick``."""
    sfx = "_quick" if quick else ""
    return (out or results_dir() / f"review{sfx}"), (refs or results_dir()), sfx


def parse(f: Path) -> tuple[int, int, int, int, float]:
    """``(layer, head, prompt, query position, rho_rel)`` of a unit file ``L<ll>H<hh>_P<p>_i<ii>_r<rho>[_fq].json``."""
    m = NAME_RE.search(Path(f).name)
    if not m:
        raise ValueError(f"not a unit file: {f}")
    return int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4)), float(m.group(5))


def read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def provenance() -> dict:
    """Machine, versions, commit and thread budget (once per process), for every record."""
    if not _PROV:
        from aimp import compute

        _PROV.update({"git": git_commit(), "env": env_info(), "threads": compute.thread_budget()})
    return dict(_PROV)


def write_unit(path: Path, rec: dict) -> None:
    write_json_atomic(path, jsonable({**rec, **provenance()}))


def relpath(f: Path, refs: Path) -> str:
    """``f`` relative to the root of the stored records (for the ``record`` columns)."""
    try:
        return str(Path(f).resolve().relative_to(Path(refs).resolve()))
    except ValueError:
        return Path(f).name


def load_units(out: Path, task: str) -> list[dict]:
    return [read_json(f) for f in sorted((out / task).rglob("*.json"))]
