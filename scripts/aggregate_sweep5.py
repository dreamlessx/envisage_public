"""Aggregate 5-seed Envisage sweep into Table 2 mean+/-std.

Expects outputs at evaluation/burn/sweep5/<procedure>/seed_<N>/<case>/output.png
plus score.json (from eval_poc_surgical equivalent) OR raw PNGs that the script
can score via the existing envisage.evaluation helpers.

Usage:
    python scripts/aggregate_sweep5.py --sweep-dir evaluation/burn/sweep5 \
        --out paper/arxiv_v1/sweep5_aggregate.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean, pstdev

PROCEDURES = ("rhinoplasty", "blepharoplasty", "rhytidectomy")
SEEDS = (42, 123, 456, 789, 1024)
EYEBROW_28_CASE = "Eyebrow_28"  # for bleph N=26 sensitivity row


def _load_scores(proc_dir: Path) -> dict | None:
    """Try eval_poc_surgical summary first, fall back to per-case aggregate."""
    summary = proc_dir / "surgical_score.json"
    if summary.exists():
        return json.loads(summary.read_text(encoding="utf-8"))
    return None


def aggregate_procedure(sweep_dir: Path, procedure: str, exclude_cases: set[str] | None = None) -> dict:
    """Aggregate per-seed mean, then compute mean+/-std across 5 seeds."""
    exclude = exclude_cases or set()
    per_seed: dict[int, dict[str, float]] = {}

    for seed in SEEDS:
        seed_dir = sweep_dir / procedure / f"seed_{seed}"
        if not seed_dir.exists():
            continue
        scores = _load_scores(seed_dir)
        if scores is None or "cases" not in scores:
            continue

        kept = [c for c in scores["cases"] if c.get("name") not in exclude]
        if not kept:
            continue
        per_seed[seed] = {
            "n_cases": len(kept),
            "outside_ssim": mean(c["outside_ssim"] for c in kept if "outside_ssim" in c),
            "full_arcface": mean(c["full_arcface"] for c in kept if "full_arcface" in c),
            "gt_arcface": mean(c["gt_arcface"] for c in kept if "gt_arcface" in c),
            "baseline_arcface": mean(c["baseline_arcface"] for c in kept if "baseline_arcface" in c),
            "inside_lpips": mean(c["inside_lpips"] for c in kept if "inside_lpips" in c),
        }

    seeds_list = sorted(per_seed)
    if not seeds_list:
        return {"procedure": procedure, "n_seeds": 0, "error": "no seeds scored"}

    metrics = ("outside_ssim", "full_arcface", "gt_arcface", "baseline_arcface", "inside_lpips")
    out: dict = {
        "procedure": procedure,
        "n_seeds": len(seeds_list),
        "seeds": seeds_list,
        "excluded_cases": sorted(exclude),
    }
    for m in metrics:
        vals = [per_seed[s][m] for s in seeds_list if m in per_seed[s]]
        if not vals:
            continue
        out[m] = {"mean": mean(vals), "std": pstdev(vals) if len(vals) > 1 else 0.0,
                  "values_per_seed": dict(zip(seeds_list, vals))}
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--sweep-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path("paper/arxiv_v1/sweep5_aggregate.json"))
    args = p.parse_args(argv)

    result = {
        "envisage_poc_params": {
            "strength": 0.75, "guidance": 3.5, "cn_scale": 0.5,
            "intensity_pct": 100, "seeds": SEEDS,
        },
        "per_procedure": {},
        "bleph_sensitivity_exclude_eyebrow_28": {},
    }

    for proc in PROCEDURES:
        result["per_procedure"][proc] = aggregate_procedure(args.sweep_dir, proc)

    # Sensitivity: bleph with Eyebrow_28 excluded
    result["bleph_sensitivity_exclude_eyebrow_28"] = aggregate_procedure(
        args.sweep_dir, "blepharoplasty", exclude_cases={EYEBROW_28_CASE}
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"wrote {args.out}")
    for proc, data in result["per_procedure"].items():
        if "gt_arcface" in data:
            m, s = data["gt_arcface"]["mean"], data["gt_arcface"]["std"]
            print(f"  {proc}: GT Arc = {m:.3f} +/- {s:.3f}  (n_seeds={data['n_seeds']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
