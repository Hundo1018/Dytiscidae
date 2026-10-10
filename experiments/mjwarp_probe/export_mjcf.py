#!/usr/bin/env python3
"""N9 step 0 (project venv only): write the MJCF of arch49's median-dof elite.

    PYTHONPATH=. MUJOCO_GL=disable .venv/bin/python experiments/mjwarp_probe/export_mjcf.py [outdir [rotor]]

The scratch venv (mjwarp-venv) never imports the project, so the XML is the interface.
"""
import json, sys
from pathlib import Path
import numpy as np
from dytiscidae.ops.run import load_run_archive
from dytiscidae.core.phenotype import build
from dytiscidae.core.mjcf import build_model_xml, scene_xml

out = Path(sys.argv[1] if len(sys.argv) > 1 else "experiments/mjwarp_probe")
seen, rows = set(), []
for isl in ["air", "water", "land", "amphibian", "aerial_diver", "land_air", "triphibian", "generalist"]:
    arc, _ = load_run_archive("runs/arch49", isl)
    for e in arc.cells.values():
        gid = getattr(e.genome, "genome_id", "") or f"{isl}:{e.cell}"
        if gid in seen:
            continue
        seen.add(gid); rows.append((isl, e))
tag = "elite_median_dof"
if len(sys.argv) > 2 and sys.argv[2] == "rotor":      # median dof among elites with rotors (velocity servos)
    rows = [(i, e) for i, e in rows if (e.meta.get("n_rotors") or 0) > 0]; tag = "elite_rotor_median_dof"
dofs = np.array([e.meta.get("dof", -1) for _, e in rows])
print("n unique elites", len(rows), "dof keys present:", sum(1 for _, e in rows if "dof" in e.meta))
order = np.argsort(dofs, kind="stable"); mid = order[len(order) // 2]
isl, e = rows[mid]
print("median dof", np.median(dofs), "chosen", isl, getattr(e.genome, "genome_id", None), "dof", dofs[mid], "n_rotors", e.meta.get("n_rotors"))
p = build(e.genome)
# the env compiles with scene_xml(timestep=0.004) (triphibian.py:643), not the 0.002 default
xml, names = build_model_xml(p, scene=scene_xml(timestep=0.004))
(out / f"{tag}.xml").write_text(xml)
(out / f"{tag}.json").write_text(json.dumps({"island": isl, "genome_id": str(getattr(e.genome, "genome_id", None)), "dof": int(dofs[mid]), "n_rotors": e.meta.get("n_rotors"), "actuators": names, "n_unique": len(rows)}, indent=1))
print("wrote", len(xml), "bytes;", len(names), "actuators")
