"""S2 helpers: structured S8R shell mesh of the cambered wing, B32R tube along a
chord line, CalculiX deck writer, ccx runner, .frd reader, twist fit.
Units SI (m, N, Pa, kg). Span along x, chord along y (LE at -c/2), camber up z."""
import os, subprocess, time, json, math
import numpy as np

WORK = '/home/hundo/Projects/Dytiscidae/.cfd/cases/S2'
PIXI = '/home/hundo/Projects/Dytiscidae/.cfd'
CCX = PIXI + '/.pixi/envs/default/bin/ccx'
L = 0.70; C = 0.258; CAMBER = 0.058
RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results.jsonl')

# materials (spec section 1)
CARBON_SHEET = dict(E=60e9, G=4.5e9, nu12=0.05, rho=1550.)
TUBE = dict(E=120e9, G=4.5e9, rho=1600.)
DEPRON = dict(E=15e6, G=6e6, rho=40.)

def arc_z(y, c=C, camber=CAMBER):
    h = camber * c
    if h == 0: return np.zeros_like(y)
    R = (c * c / 4 + h * h) / (2 * h)
    return np.sqrt(R * R - y * y) - (R - h)

class ShellMesh:
    """nx x ny elements (S8R), nodes on a (2nx+1) x (2ny+1) grid; node id = 1 + i*(2ny+1)+j."""
    def __init__(self, nx, ny, camber=CAMBER, c=C, L_=L):
        self.nx, self.ny, self.camber, self.c, self.L = nx, ny, camber, c, L_
        ni, nj = 2 * nx + 1, 2 * ny + 1
        xs = np.linspace(0, L_, ni); ys = np.linspace(-c / 2, c / 2, nj)
        self.xyz = np.zeros((ni * nj, 3))
        for i in range(ni):
            for j in range(nj):
                self.xyz[i * nj + j] = (xs[i], ys[j], arc_z(np.array(ys[j]), c, camber))
        self.nj, self.ni = nj, ni
        els = []
        for i in range(nx):
            for j in range(ny):
                g = lambda a, b: 1 + (2 * i + a) * nj + (2 * j + b)
                els.append([g(0, 0), g(2, 0), g(2, 2), g(0, 2), g(1, 0), g(2, 1), g(1, 2), g(0, 1)])
        self.els = els
    def nid(self, i, j): return 1 + i * self.nj + j
    def line_ids(self, j):  # span line at chord index j (half-element units)
        return [self.nid(i, j) for i in range(self.ni)]
    def tip_ids(self): return [self.nid(self.ni - 1, j) for j in range(self.nj)]
    def root_ids(self): return [self.nid(0, j) for j in range(self.nj)]
    def area(self):
        # area of each quad by 3x3 Gauss on the 8-node shape
        tot = 0.0
        gp = [-math.sqrt(.6), 0, math.sqrt(.6)]; gw = [5 / 9, 8 / 9, 5 / 9]
        for e in self.els:
            X = self.xyz[np.array(e) - 1]
            for a, wa in zip(gp, gw):
                for b, wb in zip(gp, gw):
                    dN = dshape8(a, b)
                    t1 = dN[:, 0] @ X; t2 = dN[:, 1] @ X
                    tot += wa * wb * np.linalg.norm(np.cross(t1, t2))
        return tot

def dshape8(x, y):
    # derivatives of 8-node serendipity (corner order 1-4, mids 5:(1-2) 6:(2-3) 7:(3-4) 8:(4-1))
    cx = [-1, 1, 1, -1]; cy = [-1, -1, 1, 1]
    d = np.zeros((8, 2))
    for k in range(4):
        xi, et = cx[k], cy[k]
        d[k, 0] = .25 * xi * (1 + et * y) * (2 * xi * x + et * y)
        d[k, 1] = .25 * et * (1 + xi * x) * (xi * x + 2 * et * y)
    d[4, 0] = -x * (1 - y); d[4, 1] = -.5 * (1 - x * x)
    d[5, 0] = .5 * (1 - y * y); d[5, 1] = -y * (1 + x)
    d[6, 0] = -x * (1 + y); d[6, 1] = .5 * (1 - x * x)
    d[7, 0] = -.5 * (1 - y * y); d[7, 1] = -y * (1 - x)
    return d

def fmt(a): return ', '.join(f'{v:.10g}' if isinstance(v, float) else str(v) for v in a)

def wrap_ids(ids, per=8):
    return '\n'.join(', '.join(str(i) for i in ids[k:k + per]) for k in range(0, len(ids), per))

class TubeShell:
    """Equivalent thin-wall tube as S8R shells: mid radius rm=sqrt((ro^2+ri^2)/2), wall
    t=(ro^2-ri^2)/(2 rm) so that area, J and I equal the real ro/ri tube's exactly.
    Axis at (y0,z0); node 0 of the ring points toward angle th0 (radians in the y-z plane).
    If plate is given, the ring node at th0 on every span line is the plate's own node (shared)."""
    BASE = 500000
    def __init__(self, nx, nth, ro, ri, y0, z0, th0, share_ids=None, L_=L):
        self.nx, self.nth = nx, nth
        self.rm = math.sqrt((ro ** 2 + ri ** 2) / 2); self.t = (ro ** 2 - ri ** 2) / (2 * self.rm)
        self.ro, self.ri, self.y0, self.z0 = ro, ri, y0, z0
        ni, nk = 2 * nx + 1, 2 * nth
        xs = np.linspace(0, L_, ni)
        self.ids = {}; self.xyz = {}
        for i in range(ni):
            for k in range(nk):
                if k == 0 and share_ids is not None: nid = share_ids[i]
                else: nid = self.BASE + i * nk + k
                self.ids[(i, k)] = nid
                ang = th0 + 2 * math.pi * k / nk
                self.xyz[nid] = (xs[i], y0 + self.rm * math.cos(ang), z0 + self.rm * math.sin(ang))
        self.els = []
        for i in range(nx):
            for k in range(nth):
                g = lambda aa, bb: self.ids[(2 * i + aa, (2 * k + bb) % nk)]
                self.els.append([g(0, 0), g(2, 0), g(2, 2), g(0, 2), g(1, 0), g(2, 1), g(1, 2), g(0, 1)])
        self.ni, self.nk = ni, nk
    def ring(self, i): return [self.ids[(i, k)] for k in range(self.nk)]
    def new_nodes(self, shared):
        return [(n, p) for n, p in self.xyz.items() if n not in shared]
    def area(self):
        return 2 * math.pi * self.rm * L * 1.0

def couple(ids, P, T, closed=False):
    """nodal forces for a pure couple T about x, nodes ids (ordered along an edge chain of
    quadratic elements: weights 1/6,4/6,1/6 of the segment) positions P (n,3); tangential force
    proportional to weight*radius from the weighted centroid."""
    n = len(ids); w = np.zeros(n)
    rng = range(0, n if closed else n - 2, 2)
    for k in rng:
        k2 = (k + 2) % n; km = k + 1
        s = np.linalg.norm(P[k2] - P[k])  # chord of the element edge (enough at these resolutions)
        w[k] += s / 6; w[km] += 4 * s / 6; w[k2] += s / 6
    cy = (w * P[:, 1]).sum() / w.sum(); cz = (w * P[:, 2]).sum() / w.sum()
    yy = P[:, 1] - cy; zz = P[:, 2] - cz
    k_ = T / (w * (yy ** 2 + zz ** 2)).sum()
    return [(b, -k_ * wi * z_, k_ * wi * y_) for b, wi, y_, z_ in zip(ids, w, yy, zz)]

def ortho_block(name, m, elset, thick):
    return (f'*MATERIAL, NAME={name}\n*ELASTIC, TYPE=ENGINEERING CONSTANTS\n'
            + fmt([m['E'], m['Eh'] if 'Eh' in m else m['E'], m['E'], m['nu12'], m['nu12'], m['nu12'], m['G'], m['G']])
            + '\n' + fmt([m['G'], 0.0]) + f'\n*DENSITY\n{m["rho"]}\n'
            + f'*SHELL SECTION, ELSET={elset}, MATERIAL={name}, ORIENTATION=OR\n{thick:.10g}\n')

def write_deck(name, plate=None, thick=None, plate_mat=None, tshell=None, tshell_mat=None, beam=None,
               bc='clamp', load=None):
    """plate: ShellMesh; plate_mat ('ortho', dict) or ('iso', E, nu, rho); tshell: TubeShell;
    beam: dict(ro, ri, nelem, mat=('iso',E,nu,rho)) tube alone as B32R along the x axis (y=z=0).
    load: dict(kind='torque'|'pressure', val, on='plate'|'tube'|'beam')."""
    out = []; A = out.append
    A('*HEADING\n' + name)
    shared = set()
    A('*NODE')
    if plate is not None:
        for k, p in enumerate(plate.xyz): A(f'{k+1}, {p[0]:.10g}, {p[1]:.10g}, {p[2]:.10g}')
        shared = set(range(1, len(plate.xyz) + 1))
    if tshell is not None:
        for n, p in tshell.new_nodes(shared): A(f'{n}, {p[0]:.10g}, {p[1]:.10g}, {p[2]:.10g}')
    tn = None
    if beam is not None:
        nn = 2 * beam['nelem'] + 1; xs = np.linspace(0, L, nn); tn = [100000 + k for k in range(nn)]
        for k in range(nn): A(f'{tn[k]}, {xs[k]:.10g}, 0., 0.')
    if plate is not None:
        A('*ELEMENT, TYPE=S8R, ELSET=PLATE')
        for k, e in enumerate(plate.els): A(f'{k+1}, ' + ', '.join(map(str, e)))
    if tshell is not None:
        A('*ELEMENT, TYPE=S8R, ELSET=TSHELL')
        for k, e in enumerate(tshell.els): A(f'{300000+k}, ' + ', '.join(map(str, e)))
    if beam is not None:
        A('*ELEMENT, TYPE=B32R, ELSET=BEAM')
        for k in range(beam['nelem']): A(f'{200000+k}, {tn[2*k]}, {tn[2*k+1]}, {tn[2*k+2]}')
    A('*ORIENTATION, NAME=OR\n1.,0.,0.,0.,1.,0.')
    if plate is not None:
        if plate_mat[0] == 'ortho': A(ortho_block('MP', plate_mat[1], 'PLATE', thick).rstrip('\n'))
        else:
            _, E, nu, rho = plate_mat
            A(f'*MATERIAL, NAME=MP\n*ELASTIC\n{E}, {nu}\n*DENSITY\n{rho}\n*SHELL SECTION, ELSET=PLATE, MATERIAL=MP\n{thick:.10g}')
    if tshell is not None: A(ortho_block('MTS', tshell_mat, 'TSHELL', tshell.t).rstrip('\n'))
    if beam is not None:
        _, E, nu, rho = beam['mat']
        A(f'*MATERIAL, NAME=MB\n*ELASTIC\n{E}, {nu}\n*DENSITY\n{rho}')
        A('*BEAM SECTION, ELSET=BEAM, MATERIAL=MB, SECTION=PIPE\n' + f'{beam["ro"]:.10g}, {beam["ro"]-beam["ri"]:.10g}\n0., 0., -1.')
    # boundary conditions
    if plate is not None and bc != 'none':
        ids = plate.root_ids()
        if bc == 'clamp': A('*BOUNDARY\n' + '\n'.join(f'{i}, 1, 6' for i in ids))
        else:  # root free to warp: in-plane displacement restrained, ux free except two corners
            A('*BOUNDARY\n' + '\n'.join(f'{i}, 2, 3' for i in ids) + f'\n{ids[0]}, 1, 1\n{ids[-1]}, 1, 1')
    if tshell is not None:
        A('*BOUNDARY\n' + '\n'.join(f'{i}, 1, 6' for i in tshell.ring(0)))
    if beam is not None: A(f'*BOUNDARY\n{tn[0]}, 1, 6')
    A('*STEP\n*STATIC')
    on = load['on']
    if load['kind'] == 'torque':
        T = load['val']
        if on == 'plate':
            tip = plate.tip_ids(); f = couple(tip, plate.xyz[np.array(tip) - 1], T)
        elif on == 'tube':
            ring = tshell.ring(tshell.ni - 1); P = np.array([tshell.xyz[i] for i in ring]); f = couple(ring, P, T, closed=True)
        if on in ('plate', 'tube'):
            A('*CLOAD'); [A(f'{b}, 2, {fy:.10g}\n{b}, 3, {fz:.10g}') for b, fy, fz in f]
        else: A(f'*CLOAD\n{tn[-1]}, 4, {T}')
    else:
        p = load['val']
        if on == 'plate':
            A('*DLOAD\n' + '\n'.join(f'{k+1}, P, {p}' for k in range(len(plate.els))))
        else:
            q = p * C; A('*CLOAD')
            for k in range(beam['nelem']):
                s = L / beam['nelem']
                for m, fr in ((0, 1 / 6), (1, 4 / 6), (2, 1 / 6)): A(f'{tn[2*k+m]}, 3, {-q*s*fr:.10g}')
    A('*NODE FILE\nU\n*EL FILE\nS\n*END STEP')
    return '\n'.join(out) + '\n'

def merge_cload(txt):
    """ccx keeps only the last value for repeated (node,dof): sum them."""
    lines = txt.split('\n'); res = []; acc = None; inc = False
    for ln in lines:
        if ln.startswith('*CLOAD'):
            inc = True; acc = {}; res.append(ln); mark = len(res); continue
        if inc and not ln.startswith('*'):
            if ln.strip():
                a, b, v = ln.split(','); key = (a.strip(), b.strip()); acc[key] = acc.get(key, 0.0) + float(v)
            continue
        if inc:
            for (a, b), v in acc.items(): res.append(f'{a}, {b}, {v:.10g}')
            inc = False
        res.append(ln)
    return '\n'.join(res)

def run_ccx(name, txt, threads=4):
    d = os.path.join(WORK, name); os.makedirs(d, exist_ok=True)
    open(f'{d}/{name}.inp', 'w').write(merge_cload(txt))
    env = dict(os.environ, OMP_NUM_THREADS=str(threads), CCX_NPROC_STIFFNESS=str(threads), CCX_NPROC_EQUATION_SOLVER=str(threads))
    t = time.time()
    r = subprocess.run([CCX, '-i', name], cwd=d, env=env, capture_output=True, text=True)
    wall = time.time() - t
    open(f'{d}/{name}.log', 'w').write(r.stdout + r.stderr)
    ok = os.path.exists(f'{d}/{name}.frd') and 'Job finished' in r.stdout
    return ok, wall, r.stdout + r.stderr

def read_frd(path):
    nodes = {}; disp = {}; stress = {}
    mode = None; blk = None
    for ln in open(path):
        if ln.startswith('    2C'): mode = 'xyz'; continue
        if ln.startswith('    3C'): mode = 'el'; continue
        if ln.startswith(' -4'):
            if 'DISP' in ln: blk = 'U'
            elif 'STRESS' in ln: blk = 'S'
            else: blk = None
            mode = 'res' if blk else None; continue
        if ln.startswith(' -3'): mode = None; continue
        if ln.startswith(' -1') and mode in ('xyz', 'res'):
            nid = int(ln[3:13]); vals = [float(ln[13 + 12 * k:25 + 12 * k]) for k in range((len(ln.rstrip('\n')) - 13) // 12)]
            if mode == 'xyz': nodes[nid] = vals
            elif mode == 'res':
                (disp if blk == 'U' else stress)[nid] = vals
    return nodes, disp, stress

def fit_twist(P, U):
    """rotation about x of a section: P (n,3) coordinates, U (n,3) displacements. Least squares
    on in-plane displacement after removing the mean translation:
    theta = sum(y' (uz-mean uz) - z' (uy-mean uy)) / sum(y'^2+z'^2), y',z' about the section centroid."""
    y = P[:, 1] - P[:, 1].mean(); z = P[:, 2] - P[:, 2].mean()
    uy = U[:, 1] - U[:, 1].mean(); uz = U[:, 2] - U[:, 2].mean()
    return float((y * uz - z * uy).sum() / (y * y + z * z).sum())

def post(name, axis=None, rm=None):
    """axis=(y0,z0), rm: tube ring (shell tube) -> twist_tube from nodes with 0.5rm<dist<1.5rm at the tip,
    twist_plate_tip from the other tip nodes more than 0.02 m from the axis."""
    d = os.path.join(WORK, name)
    nodes, disp, stress = read_frd(f'{d}/{name}.frd')
    ids = np.array(sorted(nodes)); P = np.array([nodes[i] for i in ids]); U = np.array([disp[i][:3] for i in ids])
    tip = np.abs(P[:, 0] - L) < 1e-9
    res = {}
    res['twist_all_tip'] = fit_twist(P[tip], U[tip])
    if axis is not None:
        dist = np.hypot(P[:, 1] - axis[0], P[:, 2] - axis[1])
        ring = tip & (dist > 0.5 * rm) & (dist < 1.5 * rm)
        res['twist_tube'] = fit_twist(P[ring], U[ring]); res['n_ring'] = int(ring.sum())
        pl = tip & (dist > 0.02)
        if pl.sum() > 3: res['twist_plate_tip'] = fit_twist(P[pl], U[pl])
    res['tip_uz_max'] = float(U[tip][:, 2][np.argmax(np.abs(U[tip][:, 2]))])
    res['tip_uz_mean'] = float(U[tip][:, 2].mean())
    sid = [k for k, i in enumerate(ids) if i in stress]
    S = np.array([stress[ids[k]] for k in sid]) if stress else None
    xs_ = P[sid, 0] if S is not None else None
    if S is not None and len(S):
        sxx, syy, szz, sxy, syz, szx = S.T[:6]
        vm = np.sqrt(.5 * ((sxx - syy) ** 2 + (syy - szz) ** 2 + (szz - sxx) ** 2) + 3 * (sxy ** 2 + syz ** 2 + szx ** 2))
        res['vm_max'] = float(vm.max())
        # max shear = half the largest principal difference
        tens = np.zeros((len(S), 3, 3))
        tens[:, 0, 0], tens[:, 1, 1], tens[:, 2, 2] = sxx, syy, szz
        tens[:, 0, 1] = tens[:, 1, 0] = sxy; tens[:, 1, 2] = tens[:, 2, 1] = syz; tens[:, 0, 2] = tens[:, 2, 0] = szx
        ev = np.linalg.eigvalsh(tens); res['tau_max'] = float(((ev[:, 2] - ev[:, 0]) / 2).max())
        mid = (xs_ > 0.15 * L) & (xs_ < 0.85 * L)
        res['tau_mid'] = float(((ev[mid, 2] - ev[mid, 0]) / 2).max()); res['vm_mid'] = float(vm[mid].max())
    res['n_nodes_frd'] = len(ids)
    # twist at stations x = 0.5L and 0.9L (fit over every node of the section) for the local rate
    def tw(xx):
        m = np.abs(P[:, 0] - xx) < 1e-7
        if axis is not None:  # joined models: the tube ring is the twist reference
            dd = np.hypot(P[:, 1] - axis[0], P[:, 2] - axis[1]); m = m & (dd > 0.5 * rm) & (dd < 1.5 * rm)
        return fit_twist(P[m], U[m]) if m.sum() > 3 else float('nan')
    res['twist_x50'] = tw(0.5 * L); res['twist_x90'] = tw(0.9 * L)
    res['tip_uz_chord_mean'] = float(U[tip][:, 2].mean())
    if axis is not None:
        res['tip_uz_axis'] = float(U[ring][:, 2].mean())
    return res

def row(**kw):
    with open(RESULTS, 'a') as f: f.write(json.dumps(kw) + '\n')
