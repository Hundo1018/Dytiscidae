"""S2 driver. `python run_s2.py checks` then `python run_s2.py cases` (from the pixi env:
cd /home/hundo/Projects/Dytiscidae/.cfd && pixi run python <this file> checks|cases|all).
Rewrites results.jsonl on `all`."""
import sys, os, math, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from s2lib import *

G_CARBON = 4.5e9
PRESS_SPEC = 8.6 * 3        # N/m^2 as written in the spec
PRESS_W = 0.60 * 9.81 / 0.34 * 3   # N/m^2: 0.60 kg on 0.34 m^2 is 17.3, times 3
NU_G = 0.2222222222         # E=11.0 GPa, nu=0.2222 -> G=4.5 GPa (beam torsion legs)
TUBE_SHELL = dict(E=120e9, nu12=0.05, G=4.5e9, rho=1600.)
H = CAMBER * C
RC = (C * C / 4 + H * H) / (2 * H)

def tube_mass(ro, ri): return TUBE['rho'] * math.pi * (ro ** 2 - ri ** 2) * L

def tube_geom_on_plate(sm, ro, ri, nx, nth, frac=0.25):
    yt = -C / 2 + frac * C; j = int(round(frac * 2 * sm.ny))
    zt = float(arc_z(np.array(yt)))
    slope = -yt / math.sqrt(RC * RC - yt * yt)
    n = np.array([-slope, 1.0]); n /= np.linalg.norm(n)
    rm = math.sqrt((ro ** 2 + ri ** 2) / 2)
    y0, z0 = yt + rm * n[0], zt + rm * n[1]
    th0 = math.atan2(-n[1], -n[0])
    return TubeShell(nx, nth, ro, ri, y0, z0, th0, share_ids=sm.line_ids(j)), (y0, z0)

def finish(case, kind, val, name, deck_info, r, wall, Gref, mass, extra=None, T=None):
    d = dict(case=case, **deck_info, load=f'{kind} {val}', wall_s=round(wall, 2), mass_per_wing_kg=mass)
    if kind == 'torque':
        tw = r.get('twist_tube', r['twist_all_tip']) if deck_info.get('twist_from') == 'tube' else r['twist_all_tip']
        d['twist_rad'] = tw; d['twist_deg'] = math.degrees(tw)
        d['GJ_Nm2'] = val * L / tw; d['J_geom_m4'] = d['GJ_Nm2'] / Gref
        if 'twist_plate_tip' in r: d['twist_plate_tip_rad'] = r['twist_plate_tip']
        dx = 0.4 * L; tl = (r['twist_x90'] - r['twist_x50'])
        d['GJ_local_x50_90_Nm2'] = val * dx / tl if tl else None
        d['tau_max_mid_Pa'] = r.get('tau_mid'); d['tau_max_all_Pa'] = r.get('tau_max'); d['vm_max_all_Pa'] = r.get('vm_max')
    else:
        d['tip_defl_chordmean_m'] = abs(r['tip_uz_chord_mean']); d['tip_defl_max_m'] = abs(r['tip_uz_max'])
        if 'tip_uz_axis' in r: d['tip_defl_tube_axis_m'] = abs(r['tip_uz_axis'])
        d['peak_vm_Pa'] = r.get('vm_max')
    if extra: d.update(extra)
    d['n_nodes_frd_expanded'] = r['n_nodes_frd']
    row(**d); return d

def plate_case(case, t, nx=40, ny=16, camber=CAMBER, bc='clamp', kind='torque', val=1.0, mat=CARBON_SHEET):
    sm = ShellMesh(nx, ny, camber=camber)
    name = f'{case}_{kind}'
    pm = ('ortho', mat) if 'nu12' in mat else ('iso', mat['E'], mat['nu'], mat['rho'])
    txt = write_deck(name, plate=sm, thick=t, plate_mat=pm, bc=bc,
                     load=dict(kind=kind, val=val, on='plate'))
    ok, wall, out = run_ccx(name, txt)
    if not ok: print('FAILED', name, out[-300:]); return None
    r = post(name)
    mass = mat['rho'] * sm.area() * t
    info = dict(element='S8R shell (orthotropic eng. const.)', nodes=len(sm.xyz), elements=len(sm.els),
                thickness_m=t, camber=camber, root=bc, mesh=f'{nx}x{ny}')
    return finish(case, kind, val, name, info, r, wall, mat['G'], mass)

def tube_beam_case(case, ro, ri, nelem=35, kind='torque', val=1.0):
    E, nu = (2 * G_CARBON * (1 + NU_G), NU_G) if kind == 'torque' else (120e9, 0.3)
    beam = dict(ro=ro, ri=ri, nelem=nelem, mat=('iso', E, nu, TUBE['rho']))
    name = f'{case}_{kind}'
    txt = write_deck(name, beam=beam, load=dict(kind=kind, val=val, on='beam'))
    ok, wall, out = run_ccx(name, txt)
    if not ok: print('FAILED', name, out[-300:]); return None
    r = post(name)
    info = dict(element='B32R pipe (expanded to C3D20R by ccx)', nodes=2 * nelem + 1, elements=nelem, ro_m=ro, ri_m=ri,
                root='clamp', mesh=f'{nelem} beam el',
                material=f'E={E:.4g} nu={nu} ' + ('(G=4.5 GPa leg)' if kind == 'torque' else '(E=120 GPa leg)'))
    d = finish(case, kind, val, name, info, r, wall, G_CARBON, tube_mass(ro, ri))
    if kind == 'torque':
        d['GJ_closed_Nm2'] = G_CARBON * math.pi * (ro ** 4 - ri ** 4) / 2
        d['ratio_to_closed'] = d['GJ_Nm2'] / d['GJ_closed_Nm2']
    else:
        EI = 120e9 * math.pi * (ro ** 4 - ri ** 4) / 4; q = val * C
        d['tip_defl_closed_m'] = q * L ** 4 / (8 * EI); d['ratio_to_closed'] = d['tip_defl_chordmean_m'] / d['tip_defl_closed_m']
    return d

def tube_shell_case(case, ro, ri, nx=40, nth=12, kind='torque', val=1.0):
    ts = TubeShell(nx, nth, ro, ri, 0., 0., 0.)
    name = f'{case}_{kind}'
    txt = write_deck(name, tshell=ts, tshell_mat=TUBE_SHELL, load=dict(kind=kind, val=val, on='tube'))
    ok, wall, out = run_ccx(name, txt)
    if not ok: print('FAILED', name, out[-300:]); return None
    r = post(name, axis=(0., 0.), rm=ts.rm)
    info = dict(element='S8R equivalent thin-wall shell tube', nodes=len(ts.xyz), elements=len(ts.els), ro_m=ro, ri_m=ri,
                rm_m=ts.rm, wall_m=ts.t, root='clamp', mesh=f'{nx}x{nth}', twist_from='tube')
    d = finish(case, kind, val, name, info, r, wall, G_CARBON, tube_mass(ro, ri))
    d['GJ_closed_Nm2'] = G_CARBON * math.pi * (ro ** 4 - ri ** 4) / 2; d['ratio_to_closed'] = d['GJ_Nm2'] / d['GJ_closed_Nm2']
    return d

def joined_case(case, t, plate_mat, ro=4e-3, ri=3e-3, nx=40, ny=16, nth=12, kind='torque', val=1.0, plate_root='clamp', foam=False):
    sm = ShellMesh(nx, ny)
    ts, axis = tube_geom_on_plate(sm, ro, ri, nx, nth)
    name = f'{case}_{kind}'
    txt = write_deck(name, plate=sm, thick=t, plate_mat=plate_mat, tshell=ts, tshell_mat=TUBE_SHELL,
                     bc=('clamp' if plate_root == 'clamp' else 'none'),
                     load=dict(kind=kind, val=val, on='tube' if kind == 'torque' else 'plate'))
    ok, wall, out = run_ccx(name, txt)
    if not ok: print('FAILED', name, out[-300:]); return None
    r = post(name, axis=axis, rm=ts.rm)
    rho_p = plate_mat[1]['rho'] if plate_mat[0] == 'ortho' else plate_mat[3]
    mass = rho_p * sm.area() * t + tube_mass(ro, ri)
    info = dict(element='S8R plate + S8R equivalent tube, shared nodes along one generator', nodes=len(sm.xyz) + len(ts.new_nodes(set(range(1, len(sm.xyz) + 1)))),
                elements=len(sm.els) + len(ts.els), thickness_m=t, ro_m=ro, ri_m=ri, root=f'tube clamped, plate root {plate_root}',
                mesh=f'{nx}x{ny} plate, {nx}x{nth} tube', twist_from='tube')
    return finish(case, kind, val, name, info, r, wall, G_CARBON, mass)

def checks():
    out = {}
    # 1. tube GJ vs closed form (beam), three tubes + halved
    for lab, ro, ri in (('6x4', 3e-3, 2e-3), ('8x6', 4e-3, 3e-3), ('10x8', 5e-3, 4e-3)):
        d = tube_beam_case(f'CHK1_tube{lab}', ro, ri); print('CHK1', lab, d['GJ_Nm2'], d['ratio_to_closed'])
    d = tube_beam_case('CHK3_tube8x6_halved_70el', 4e-3, 3e-3, nelem=70); print('CHK3 tube 70 el', d['GJ_Nm2'], d['ratio_to_closed'])
    # shell-tube (equivalent) too, since TP/F use it
    for lab, ro, ri in (('6x4', 3e-3, 2e-3), ('8x6', 4e-3, 3e-3), ('10x8', 5e-3, 4e-3)):
        d = tube_shell_case(f'CHK1s_tubeshell{lab}', ro, ri); print('CHK1s', lab, d['GJ_Nm2'], d['ratio_to_closed'])
    d = tube_shell_case('CHK3s_tubeshell8x6_halved', 4e-3, 3e-3, nx=80, nth=24); print('CHK3s', d['GJ_Nm2'], d['ratio_to_closed'])
    # 2. flat strip, root free to warp
    for t in (0.5e-3, 1.0e-3):
        d = plate_case(f'CHK2_strip_t{int(t*1e6)}um', t, camber=0.0, bc='warp')
        d['ratio_to_Gct3_3'] = d['GJ_Nm2'] / (G_CARBON * C * t ** 3 / 3); print('CHK2', t, d['GJ_Nm2'], d['ratio_to_Gct3_3'])
        row(case=f'CHK2_strip_t{int(t*1e6)}um_ratio', ratio_to_Gct3_3=d['ratio_to_Gct3_3'], GJ_open_section_Nm2=G_CARBON * C * t ** 3 / 3)
    d = plate_case('CHK3_strip_t500um_halved', 0.5e-3, nx=80, ny=32, camber=0.0, bc='warp'); print('CHK3 strip halved', d['GJ_Nm2'])
    d = plate_case('CHK3_P500um_clamp_cambered_halved', 0.5e-3, nx=80, ny=32); print('CHK3 cambered clamp halved', d['GJ_Nm2'])

def cases():
    for t in (0.3e-3, 0.5e-3, 1.0e-3):
        lab = f'P{int(t*1e6)}um'
        d = plate_case(lab, t); print(lab, 'torque GJ', d['GJ_Nm2'], 'local', d['GJ_local_x50_90_Nm2'], 'mass', d['mass_per_wing_kg'])
        d = plate_case(f'{lab}_rootfree', t, bc='warp'); print(lab, 'root free to warp GJ', d['GJ_Nm2'], 'ratio to G c t^3/3', d['GJ_Nm2'] / (G_CARBON * C * t ** 3 / 3))
        for pr, pl in ((PRESS_SPEC, 'spec25.8'), (PRESS_W, 'weight51.9')):
            d = plate_case(f'{lab}_{pl}', t, kind='pressure', val=pr); print(lab, pl, d['tip_defl_chordmean_m'])
    d = plate_case('Pfoam6mm_alone', 6e-3, mat=dict(E=15e6, nu=0.25, G=6e6, rho=40.)); print('foam alone', d['GJ_Nm2'], 'open-section', 6e6 * C * 6e-3 ** 3 / 3)
    for lab, ro, ri in (('6x4', 3e-3, 2e-3), ('8x6', 4e-3, 3e-3), ('10x8', 5e-3, 4e-3)):
        d = tube_beam_case(f'T{lab}', ro, ri); print('T', lab, d['GJ_Nm2'], d['ratio_to_closed'], d['tau_max_mid_Pa'])
        for pr, pl in ((PRESS_SPEC, 'spec25.8'), (PRESS_W, 'weight51.9')):
            d = tube_beam_case(f'T{lab}_{pl}', ro, ri, kind='pressure', val=pr); print('T', lab, pl, d['tip_defl_chordmean_m'], d['ratio_to_closed'])
    d = joined_case('TP8x6_P300um', 0.3e-3, ('ortho', CARBON_SHEET)); print('TP', d['GJ_Nm2'], d['GJ_local_x50_90_Nm2'], d['mass_per_wing_kg'])
    d = joined_case('TP8x6_P300um_plateRootFree', 0.3e-3, ('ortho', CARBON_SHEET), plate_root='free'); print('TP free', d['GJ_Nm2'])
    d = joined_case('TP8x6_P300um_halved', 0.3e-3, ('ortho', CARBON_SHEET), nx=80, ny=32, nth=24); print('TP halved', d['GJ_Nm2'])
    foam = ('iso', DEPRON['E'], 0.25, DEPRON['rho'])
    d = joined_case('F8x6_depron6mm', 6e-3, foam); print('F', d['GJ_Nm2'], d['mass_per_wing_kg'])
    for pr, pl in ((PRESS_SPEC, 'spec25.8'), (PRESS_W, 'weight51.9')):
        d = joined_case(f'TP8x6_P300um_{pl}', 0.3e-3, ('ortho', CARBON_SHEET), kind='pressure', val=pr); print('TP', pl, d['tip_defl_axis_m'] if 'tip_defl_axis_m' in d else d['tip_defl_tube_axis_m'], d['tip_defl_chordmean_m'])
        d = joined_case(f'F8x6_depron6mm_{pl}', 6e-3, foam, kind='pressure', val=pr); print('F', pl, d['tip_defl_tube_axis_m'], d['tip_defl_chordmean_m'])
    # reference: shell tube alone (like-for-like for S3, S4)
    d = tube_shell_case('Tshell8x6', 4e-3, 3e-3); print('Tshell 8x6', d['GJ_Nm2'])

if __name__ == '__main__':
    mode = sys.argv[1]
    if mode == 'all' and os.path.exists(RESULTS): os.remove(RESULTS)
    if mode in ('checks', 'all'): checks()
    if mode in ('cases', 'all'): cases()
