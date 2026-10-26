# D-A4 (IMPL_1010_SPEC §4): device memory of put_data at a given nworld, njmax 512; run per XML; see the spec D-A4 note for the numbers.
import sys, mujoco, mujoco_warp as mjw, warp as wp, numpy as np
path = sys.argv[1]; nworld = int(sys.argv[2])
mjm = mujoco.MjModel.from_xml_path(path); mjm.geom_margin[:] = 0.0
mjd = mujoco.MjData(mjm)
free0 = wp.get_device().total_memory if hasattr(wp.get_device(), "total_memory") else None
m = mjw.put_model(mjm)
d = mjw.put_data(mjm, mjd, nworld=nworld, njmax=512)
mjw.forward(m, d); wp.synchronize()
import subprocess
used = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout.strip()
print(f"{path.split('/')[-1]} nworld={nworld} njmax=512 nv={mjm.nv} nbody={mjm.nbody} device_used_MiB={used}", flush=True)
