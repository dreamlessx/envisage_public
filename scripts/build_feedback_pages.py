r"""Build one Obsidian page per case from the burn-minimal outputs.

Reads evaluation/burn/<procedure>/<case_id>/ which contains input.png,
target.png, output.png, params.json -- writes ONE markdown page per case
into the Obsidian vault's feedback_images/ folder so Mudit can leave
feedback on each case individually.

Each page embeds the three images side-by-side and has an explicit
## Feedback section at the bottom with free-form prompts Mudit fills in.

Usage (on laptop after scp'ing outputs down):
  python -m scripts.build_feedback_pages \
      --outputs-dir ~/envisage-workspace/envisage/evaluation/burn \
      --vault ~/Library/Mobile\ Documents/iCloud~md~obsidian/Documents/Dreamless_Machine/03-Research/Envisage/feedback_images
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path


PAGE_TEMPLATE = """---
tags: [envisage, feedback, poc, {procedure}]
case: {case_id}
procedure: {procedure}
date: 2026-04-18
status: awaiting-feedback
---

# {procedure} — {case_id}

## Images

| Input | Target (GT) | Envisage PoC output |
|---|---|---|
| ![[{img_input}]] | ![[{img_target}]] | ![[{img_output}]] |

## Mask (region model was allowed to edit)

![[{img_mask}]]

## Config used for this case

- active presets detected: **{active_presets}**
- prompt: `{prompt}`
- strength: {strength} · steps: {steps} · guidance: {guidance} · seed: {seed}
- resolution: {resolution} · runtime: {runtime_s}s

## Feedback

_Mudit fills this in_

- **Visible change vs input?** yes / no / partial:
- **Right surgical direction?** (do the detected presets match what the surgery should do?):
- **Mask correct region?** (e.g. for an eyebag case, is lower-lid included?):
- **Artifacts?** (hue shift, mask bleed, melted features, doubled anatomy):
- **Surgeon-style verdict**: ship / reject / needs specific change
- **Free notes**:

---
[[{prev_nav}|← prev]] · [[INDEX|index]] · [[{next_nav}|next →]]
"""


INDEX_TEMPLATE = """---
tags: [envisage, feedback, poc, index]
date: 2026-04-18
---

# Envisage PoC — feedback index

One page per case. Click into each to leave feedback.

{procedure_sections}

## Envisage PoC config

- Base: FLUX.1-dev + jasperai Flux.1-dev-Controlnet-Depth
- Depth conditioning: Depth Anything V2 + per-procedure Gaussian modification
- Mask: MediaPipe 478-landmark convex hull, feathered; preset-aware for bleph
- Composite: hard-mask composite at original resolution (outside-mask pixels byte-identical to input)
- strength: 0.75 · steps: 30 · guidance: 3.5 · cn_scale: 0.5 · seed: 42
- Resolution: 1024x1024
- No LoRA, no scorer, no ensemble, single seed

## What to look for

1. **Visible surgical change vs input?** (prior v2 ensemble was often passthrough — we need the change to be visible AND correct.)
2. **Right surgical direction?** Bleph = upper-lid skin tightening + crease. Rhino = narrower bridge / refined tip / alar. Rhytid = straighter jawline + smoother neck.
3. **Hallucinations?** Dark nostril holes (Rhino known failure), color drift, melted features, weird asymmetry.
4. **Mask boundary?** Does the edit bleed beyond where it should (e.g., rhytid mask leaking into image watermark), or is it contained?
"""


def build_pages(outputs_dir: Path, vault_dir: Path) -> int:
    """Create per-case pages + index in vault_dir. Returns page count."""
    vault_dir.mkdir(parents=True, exist_ok=True)

    # Collect cases, grouped by procedure
    procedures: dict[str, list[tuple[str, Path]]] = {}
    for proc_dir in sorted(outputs_dir.iterdir()):
        if not proc_dir.is_dir():
            continue
        procedure = proc_dir.name
        for case_dir in sorted(proc_dir.iterdir()):
            if not case_dir.is_dir():
                continue
            if not (case_dir / "output.png").exists():
                continue
            procedures.setdefault(procedure, []).append((case_dir.name, case_dir))

    if not procedures:
        print(f"No case directories found under {outputs_dir}", file=sys.stderr)
        return 0

    pages_written = 0
    procedure_sections: list[str] = []

    for procedure, cases in procedures.items():
        section_lines = [f"\n## {procedure} ({len(cases)} cases)\n"]

        for i, (case_id, case_dir) in enumerate(cases):
            # Copy images into vault with unique filenames
            prefix = f"{procedure}_{case_id}"
            img_input = f"{prefix}_input.png"
            img_target = f"{prefix}_target.png"
            img_output = f"{prefix}_output.png"
            img_mask = f"{prefix}_mask.png"

            shutil.copy2(case_dir / "input.png", vault_dir / img_input)
            if (case_dir / "target.png").exists():
                shutil.copy2(case_dir / "target.png", vault_dir / img_target)
            shutil.copy2(case_dir / "output.png", vault_dir / img_output)
            if (case_dir / "mask.png").exists():
                shutil.copy2(case_dir / "mask.png", vault_dir / img_mask)

            # Params
            params_path = case_dir / "params.json"
            if params_path.exists():
                params = json.loads(params_path.read_text(encoding="utf-8"))
            else:
                params = {}

            # Prev/next navigation
            prev_nav = f"{procedure}_{cases[i - 1][0]}" if i > 0 else "INDEX"
            next_nav = f"{procedure}_{cases[i + 1][0]}" if i + 1 < len(cases) else "INDEX"

            page_name = f"{procedure}_{case_id}.md"
            active = params.get("active_presets") or []
            active_str = ", ".join(active) if active else "(none detected)"
            page_body = PAGE_TEMPLATE.format(
                procedure=procedure,
                case_id=case_id,
                img_input=img_input,
                img_target=img_target,
                img_output=img_output,
                img_mask=img_mask,
                active_presets=active_str,
                prompt=params.get("prompt", "(not recorded)"),
                strength=params.get("strength", "?"),
                resolution=params.get("resolution", "?"),
                steps=params.get("steps", "?"),
                guidance=params.get("guidance", "?"),
                seed=params.get("seed", "?"),
                runtime_s=params.get("runtime_s", "?"),
                prev_nav=prev_nav,
                next_nav=next_nav,
            )
            (vault_dir / page_name).write_text(page_body, encoding="utf-8")
            pages_written += 1

            status = "✅" if (case_dir / "output.png").exists() else "❌"
            section_lines.append(
                f"- {status} [[{procedure}_{case_id}|{case_id}]]"
            )

        procedure_sections.append("\n".join(section_lines))

    index_path = vault_dir / "INDEX.md"
    index_path.write_text(INDEX_TEMPLATE.format(
        procedure_sections="\n".join(procedure_sections),
    ), encoding="utf-8")
    print(f"Wrote {pages_written} case pages + INDEX.md to {vault_dir}")
    return pages_written


def _cli() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--outputs-dir", required=True,
                   help="Local dir containing <procedure>/<case_id>/ output.png etc")
    p.add_argument("--vault",
                   default=str(Path.home() / "Library/Mobile Documents/iCloud~md~obsidian/Documents/Dreamless_Machine/03-Research/Envisage/feedback_images"),
                   help="Obsidian vault feedback_images/ path")
    args = p.parse_args()

    outputs_dir = Path(args.outputs_dir).expanduser()
    vault_dir = Path(args.vault).expanduser()

    if not outputs_dir.exists():
        print(f"ERROR: outputs dir does not exist: {outputs_dir}", file=sys.stderr)
        return 1
    return 0 if build_pages(outputs_dir, vault_dir) else 2


if __name__ == "__main__":
    raise SystemExit(_cli())
