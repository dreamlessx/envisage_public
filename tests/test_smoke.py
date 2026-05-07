"""Lightweight smoke tests: package imports cleanly, configs load, paper artifacts present."""
from __future__ import annotations

import importlib
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent

# All envisage submodules currently in the package
SUBMODULES = [
    "envisage.depth",
    "envisage.evaluation",
    "envisage.fairness",
    "envisage.hybrid",
    "envisage.landmarks",
    "envisage.masks",
    "envisage.measurements",
    "envisage.pipeline",
    "envisage.pipeline_v2",
    "envisage.postprocess",
    "envisage.scorer",
    "envisage.candidates",
    "envisage.gt_analysis",
    "envisage.tps_augment",
    "envisage.rhino_config",
    "envisage.bleph_config",
    "envisage.rhytid_config",
    "envisage.composite_selector",
    "envisage.statistics",
]

# Heavy / optional deps that may be missing in CI; skip rather than fail.
OPTIONAL_DEPS = (
    "torch", "diffusers", "mediapipe", "insightface",
    "cv2", "opencv", "lpips", "transformers", "controlnet",
    "skimage", "matplotlib", "scipy", "sklearn",
    "huggingface_hub", "PIL.Image", "torchvision",
)


def test_package_importable():
    mod = importlib.import_module("envisage")
    assert mod is not None


@pytest.mark.parametrize("submodule", SUBMODULES)
def test_submodule_importable(submodule):
    try:
        importlib.import_module(submodule)
    except ImportError as e:
        msg = str(e).lower()
        if any(x.lower() in msg for x in OPTIONAL_DEPS):
            pytest.skip(f"{submodule}: optional dep missing ({e})")
        raise


def test_rhinoplasty_config_loads():
    cfg_path = REPO_ROOT / "configs" / "rhinoplasty.yaml"
    assert cfg_path.exists(), f"missing config: {cfg_path}"
    cfg = yaml.safe_load(cfg_path.read_text())
    assert isinstance(cfg, dict)


def test_paper_figures_present():
    """Figures referenced from paper/main_neurips_v1.tex must be committed."""
    fig_dir = REPO_ROOT / "paper" / "figures"
    required = [
        "fig1_pipeline.pdf",
        "fig3_decomposed_arcface.png",
        "fig_qualitative_v26.png",
        "fig_other_procedures.png",
    ]
    for name in required:
        assert (fig_dir / name).exists(), f"missing figure: {name}"


def test_paper_source_present():
    paper_dir = REPO_ROOT / "paper"
    required = [
        "main_neurips_v1.tex",
        "checklist_neurips_v1.tex",
        "refs.bib",
        "main_neurips.pdf",
    ]
    for name in required:
        assert (paper_dir / name).exists(), f"missing paper artifact: {name}"


def test_evaluation_jsons_present():
    """Load-bearing evaluation JSONs cited in paper tables must be committed."""
    import json
    eval_dir = REPO_ROOT / "evaluation"
    required = [
        "strict_n211_all_methods.json",
        "strict_n211_baselines/empirical_lipschitz.json",
        "strict_n211_baselines/sensitivity_t1_dirichlet_strict.json",
        "strict_n211_baselines/paired_stats_v1.json",
        "strict_n211_baselines/per_method_ci.json",
    ]
    for rel in required:
        path = eval_dir / rel
        assert path.exists(), f"missing eval JSON: {rel}"
        with path.open() as fh:
            json.load(fh)  # parses cleanly


def test_preset_configs_define_taxonomies():
    """24-preset taxonomy: 8 rhino + 8 bleph + 8 rhytid presets."""
    rhino = importlib.import_module("envisage.rhino_config")
    bleph = importlib.import_module("envisage.bleph_config")
    rhytid = importlib.import_module("envisage.rhytid_config")
    rhino_count = len(getattr(rhino, "PRIORITY_ORDER", []))
    bleph_count = len(getattr(bleph, "PRIORITY_ORDER", []))
    rhytid_count = len(getattr(rhytid, "PRIORITY_ORDER", []))
    assert rhino_count == 8, f"rhino presets: {rhino_count}"
    assert bleph_count == 8, f"bleph presets: {bleph_count}"
    assert rhytid_count == 8, f"rhytid presets: {rhytid_count}"
