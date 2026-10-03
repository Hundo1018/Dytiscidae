import numpy as np, collections
from dytiscidae.core.bodyplans import BODY_PLANS
from dytiscidae.core.phenotype import build
from dytiscidae.control.cpg import CPGParams
from dytiscidae.envs.evaluate import Controller
from dytiscidae.envs.transitions import run_transition
from dytiscidae.envs.triphibian import TriphibianEnv
C=("shock","control","settle","economy","exit_state")
acc=collections.defaultdict(list)
for plan in BODY_PLANS:
    p=build(BODY_PLANS[plan]())
    for seed in (0,1):
        e=TriphibianEnv(p,seed=seed); b=e.cpg.base
        still=CPGParams(amplitude=np.zeros(e.cpg.n),phase=np.asarray(b.phase,float),offset=np.asarray(b.offset,float),frequency=float(b.frequency))
        for kind in ("air_to_water","water_to_land"):
            for name,prm in (("gait",b),("still",still)):
                t=run_transition(e,kind,Controller(params=prm))
                acc[(kind,name)].append([t.components.get(c,0.0) for c in C]+[float(t.crossed)])
                if not acc.get(('keys',)): acc[('keys',)]=sorted(t.components)
print('all component keys:',acc[('keys',)])
for k,v in acc.items():
    if k==('keys',): continue
    v=np.array(v); print(f"{k[0]:14s} {k[1]:5s}", "  ".join(f"{c}={m:.3f}" for c,m in zip(C+("crossed",),v.mean(0))))
