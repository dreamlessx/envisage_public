"""Build a manifest of (label, case, procedure, input, target, output) for the
full SurgicalScore v5 sweep across all available data."""
from __future__ import annotations

import json
import sys
from pathlib import Path

VAULT = Path("/Users/dreamless/Library/Mobile Documents/iCloud~md~obsidian/Documents/Dreamless_Machine/03-Research/Envisage/feedback_images/2026-04-29_smoke_AB")
SWEEP = Path("/tmp/sweep_outputs")

# Each rhino case: which versions have output
RHINO_CASES = ["Nose_102", "Nose_113", "Nose_120", "Nose_122", "Nose_129", "Nose_142"]
RHINO_FULL = [f"rhinoplasty_{c}" for c in RHINO_CASES]

BLEPH_CASES = ["Eyebrow_125", "Eyebrow_53"]
BLEPH_FULL = [f"blepharoplasty_{c}" for c in BLEPH_CASES]

RHYT_CASES = ["Facelift_33", "Facelift_56"]
RHYT_FULL = [f"rhytidectomy_{c}" for c in RHYT_CASES]

# Old versions had ACCRE-saved per-procedure outputs in /tmp/sweep_outputs/<v>/
# Pattern in /tmp/sweep_outputs: rhinoplasty_rhinoplasty_Nose_<id>.png
# New versions in vault: <stem_short>_v<v>_output.png

manifest = []

# Rhino across versions (v8-v33)
for v in ["v8", "v9", "v15", "v22", "v22b", "v23", "v26", "v29b", "v31", "v32", "v33"]:
    for stem_short, stem_full in zip(RHINO_CASES, RHINO_FULL):
        candidates = [
            SWEEP / v / f"rhinoplasty_{stem_full}.png",  # ACCRE pull format
            VAULT / f"{stem_short}_{v}_output.png",       # vault format
        ]
        out = next((c for c in candidates if c.exists()), None)
        if out:
            manifest.append({
                "label": v, "case": stem_full, "procedure": "rhinoplasty",
                "input": str(VAULT / f"{stem_full}_input.png"),
                "target": str(VAULT / f"{stem_full}_target.png"),
                "output": str(out),
            })

# Bleph across versions (v22b had outputs, v22/v23/v24b too, v26-v33 in vault)
for v in ["v22", "v22b", "v23", "v24b", "v26", "v27", "v29b", "v31", "v32", "v33"]:
    for stem_short, stem_full in zip(BLEPH_CASES, BLEPH_FULL):
        candidates = [
            SWEEP / v / "bleph_E125.png" if stem_short == "Eyebrow_125" else SWEEP / v / "bleph_E53.png",
            SWEEP / v / f"blepharoplasty_{stem_full}.png",
            VAULT / f"{stem_short}_{v}_output.png",
        ]
        out = next((c for c in candidates if c and c.exists()), None)
        if out:
            manifest.append({
                "label": v, "case": stem_full, "procedure": "blepharoplasty",
                "input": str(VAULT / f"{stem_full}_input.png"),
                "target": str(VAULT / f"{stem_full}_target.png"),
                "output": str(out),
            })

# Rhyt across versions
for v in ["v22", "v22b", "v23", "v24b", "v26", "v27", "v29b", "v31", "v32", "v33"]:
    for stem_short, stem_full in zip(RHYT_CASES, RHYT_FULL):
        candidates = [
            SWEEP / v / "rhyt_F56.png" if stem_short == "Facelift_56" else SWEEP / v / "rhyt_F33.png",
            SWEEP / v / f"rhytidectomy_{stem_full}.png",
            VAULT / f"{stem_short}_{v}_output.png",
        ]
        out = next((c for c in candidates if c and c.exists()), None)
        if out:
            manifest.append({
                "label": v, "case": stem_full, "procedure": "rhytidectomy",
                "input": str(VAULT / f"{stem_full}_input.png"),
                "target": str(VAULT / f"{stem_full}_target.png"),
                "output": str(out),
            })

# Passthrough (output = input) for each case in matched-N as the floor
for stem_full in RHINO_FULL + BLEPH_FULL + RHYT_FULL:
    proc = stem_full.split("_")[0]
    inp = VAULT / f"{stem_full}_input.png"
    if inp.exists():
        manifest.append({
            "label": "PASSTHROUGH", "case": stem_full, "procedure": proc,
            "input": str(inp),
            "target": str(VAULT / f"{stem_full}_target.png"),
            "output": str(inp),  # passthrough = output is input
        })

print(f"Manifest: {len(manifest)} entries", file=sys.stderr)
print(json.dumps(manifest, indent=2))
