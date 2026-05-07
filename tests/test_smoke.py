"""Lightweight smoke tests: package imports cleanly and configs load."""
from __future__ import annotations

import importlib
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_package_importable():
    """envisage package must import without optional heavy deps."""
    mod = importlib.import_module("envisage")
    assert mod is not None
    assert hasattr(mod, "__doc__")


@pytest.mark.parametrize("submodule", [
    "envisage.depth",
    "envisage.evaluation",
    "envisage.fairness",
    "envisage.hybrid",
    "envisage.landmarks",
    "envisage.masks",
    "envisage.pipeline",
    "envisage.postprocess",
    "envisage.tps_augment",
])
def test_submodule_importable(submodule):
    """Each submodule should at least parse + import."""
    try:
        importlib.import_module(submodule)
    except ImportError as e:
        # Heavy deps (torch/diffusers/mediapipe) are optional in CI;
        # only fail on syntax/structural errors.
        msg = str(e).lower()
        if any(x in msg for x in ["torch", "diffusers", "mediapipe",
                                   "insightface", "cv2", "lpips",
                                   "transformers"]):
            pytest.skip(f"{submodule}: optional dep missing ({e})")
        raise


def test_rhinoplasty_config_loads():
    cfg_path = REPO_ROOT / "configs" / "rhinoplasty.yaml"
    assert cfg_path.exists(), f"missing config: {cfg_path}"
    cfg = yaml.safe_load(cfg_path.read_text())
    assert isinstance(cfg, dict)


def test_paper_figures_present():
    """Figures referenced from paper/main.tex must be committed."""
    fig_dir = REPO_ROOT / "paper" / "figures"
    required = [
        "figM1_pipeline.png",
        "figM2_qualitative.png",
        "figM3_decomposed_arcface.png",
        "figS1_conditioning.png",
    ]
    for name in required:
        assert (fig_dir / name).exists(), f"missing figure: {name}"


def test_readme_links_resolve():
    """README must point at real on-disk files (no broken local refs)."""
    readme = (REPO_ROOT / "README.md").read_text()
    # Every local path that looks like a relative file ref should exist
    import re
    for m in re.finditer(r"\]\(([^)]+\.(?:png|pdf|yaml|py|md|sh))\)", readme):
        path = m.group(1).split("#")[0]
        if path.startswith(("http", "https", "mailto")):
            continue
        candidate = REPO_ROOT / path
        assert candidate.exists(), f"README links to missing file: {path}"
