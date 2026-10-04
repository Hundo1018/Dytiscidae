import numpy as np
from dytiscidae.core.bodyplans import BODY_PLANS
from dytiscidae.core.phenotype import build
from dytiscidae.control.cpg import CPGParams
from dytiscidae.envs.evaluate import Controller
from dytiscidae.envs.transitions import run_transition
from dytiscidae.envs.triphibian import TriphibianEnv
for kind in ("air_to_water","water_to_land"):
    for back in (0.0, 1.0, 2.0, 4.0):
        r={"gait":[], "still":[]}
        for plan in BODY_PLANS:
            p=build(BODY_PLANS[plan]())
            for seed in (0,1):
                e=TriphibianEnv(p,seed=seed); b=e.cpg.base
                still=CPGParams(amplitude=np.zeros(e.cpg.n),phase=np.asarray(b.phase,float),offset=np.asarray(b.offset,float),frequency=float(b.frequency))
                for name,prm in (("gait",b),("still",still)):
                    r[name].append(float(run_transition(e,kind,Controller(params=prm),back=back).crossed))
        print(f"{kind:14s} back={back:3.1f}m  crossed: gait {np.mean(r['gait']):.2f}  still {np.mean(r['still']):.2f}", flush=True)
