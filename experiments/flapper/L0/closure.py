"""L0: paper closure of the built machine, one line per candidate.

Every constant names its source (reports A11, A13, A14 under
experiments/triphibian_diagnosis/reports) or is marked TYPED: a number nobody
measured, swept here so that its weight in the answer shows.

    python experiments/flapper/L0/closure.py
"""
from __future__ import annotations

import itertools
import math

G = 9.81
RHO = 1.2

# --- sourced ---------------------------------------------------------------
PACK_WH_KG = {"lipo": 125.0,      # A14 A.1: 106-142 Wh/kg at the pack, mid
              "cyl": 182.0}       # A13 limit 2: welded P45B at 4C
CELL_21700_G = 70.0               # A13 A.1
CELL_21700_WH = 13.65             # A13 limit 6, 15-minute discharge
CYL_MAX_W_KG = 2200.0             # A13 limit 3, datasheet continuous maximum
CRUISE_W_KG = (122.0, 175.0)      # A13 limit 4: Phoenix, E-Flap (verified)
DRIVE_FRACTION_EFLAP = 0.14       # A13 limit 9: 71 g on 0.51 kg
EFLAP_PEAK_W_KG = 392.0           # A14 B.5: 200 W electrical on 0.51 kg
WING_KG_M2 = 0.187                # A13 limit 12
WING_LOADING = (1.78, 2.5, 3.64)  # kg/m2: Kestrel (A13), between, RoboFalcon (A14)
TORQUE_REF = (4.43, 1.17)         # N m at span m (A13 limit 10, +-30%)
WATER_TORQUE_RATIO = 833.0        # rho_water / rho_air at equal stroke
DRIVE_HZ_CEILING = 5.5            # A13 limit 9 (E-Flap); RoboFalcon ran 7 (A14)
STROKE = math.radians(85.0)       # RoboFalcon 2.0 full amplitude (A14 B.5)
R2 = 0.30                         # A11 D: second moment of area, 0.30-0.39
HUMP = (0.16, 0.37)               # A14 C.3: hump drag over weight, full-size hulls
ROBO_RAVEN_TW = (0.46, 0.78)      # A14 C.3, thrust in flight tests over weight

# --- typed -----------------------------------------------------------------
FIXED_KG = 0.080                  # TYPED: controller 10, ESC 8, depth 14 (A13 D),
#                                   receiver, two tail servos, wiring, sealing
STRUCTURE = (0.20, 0.30)          # TYPED: hull, tail, thumbs, seals over all-up mass
ASPECT = 6.0                      # TYPED
TORQUE_HEADROOM = 2.0             # TYPED: drive torque over cruise shoulder torque


def hover_hz(m, S, R, cl):
    """A11 D, blade elements: frequency at which the wings alone carry m."""
    return math.sqrt(4 * m * G / (RHO * cl * S * R2)) / (math.pi * STROKE * R)


def ideal_hover_w(m, R):
    disc = STROKE * R * R                     # swept by two wings
    return m * G * math.sqrt(m * G / disc / (2 * RHO))


# RoboFalcon 2.0 (A14 B.5): 0.8 kg, 1.2 m, 3.64 kg/m2, 7 Hz and 400 W at lift-off.
_S, _R = 0.8 / 3.64, 0.6
CL_REF = (hover_hz(0.8, _S, _R, 1.0) / 7.0) ** 2      # lift coefficient it implies
MERIT = ideal_hover_w(0.8, _R) / 400.0                 # ideal over electrical peak


def close(w, wl, cell, s, three_21700=False):
    """All-up mass at which the fractions close, or None."""
    wing = WING_KG_M2 / wl
    out = None
    for _ in range(60):                       # drive fraction depends on the answer
        m = out or 0.5
        S = m / wl
        R = math.sqrt(ASPECT * S) / 2
        burst = ideal_hover_w(m, R) / MERIT   # electrical W for a standing take-off
        drive = DRIVE_FRACTION_EFLAP * max(1.0, burst / m / EFLAP_PEAK_W_KG)
        if three_21700:
            fixed = FIXED_KG + 3 * CELL_21700_G * 1.04 / 1000
            rest = 1 - drive - wing - s
        else:
            fixed = FIXED_KG
            rest = 1 - 0.25 * w / PACK_WH_KG[cell] - drive - wing - s
        if rest <= 0.02:
            return None
        new = fixed / rest
        if out and abs(new - out) < 1e-6:
            break
        out = new
    m = out
    S = m / wl
    b = math.sqrt(ASPECT * S)
    R = b / 2
    burst = ideal_hover_w(m, R) / MERIT
    batt = 3 * CELL_21700_G / 1000 if three_21700 else 0.25 * w * m / PACK_WH_KG[cell]
    wh = 3 * CELL_21700_WH if three_21700 else 0.25 * w * m
    torque = TORQUE_REF[0] * (b / TORQUE_REF[1]) ** 3
    f_to = hover_hz(m, S, R, CL_REF)
    return dict(
        m=m, S=S, b=b, batt=batt, burst=burst, burst_kg=burst / m,
        energy_margin=wh / (0.25 * w * m),
        f_takeoff=f_to, f_margin=DRIVE_HZ_CEILING / f_to, f_margin7=7.0 / f_to,
        cyl_power_margin=CYL_MAX_W_KG * batt / burst,
        torque=torque,
        f_water=f_to * math.sqrt(TORQUE_HEADROOM / WATER_TORQUE_RATIO),
        displaced_cm3=m * 1000,
    )


def main():
    print(f"calibration on RoboFalcon 2.0: CL = {CL_REF:.2f}, "
          f"ideal/electrical at lift-off = {MERIT:.3f}")
    head = ("cell      w    WL    s  |    m     S    span  batt | take-off: Hz  /5.5  /7.0"
            "   W   W/kg | N m  water Hz")
    print(head)
    rows = []
    for cell, w, wl, s in itertools.product(("lipo", "cyl", "3x21700"), CRUISE_W_KG,
                                            WING_LOADING, STRUCTURE):
        r = close(w, wl, "cyl" if cell == "3x21700" else cell, s, cell == "3x21700")
        tag = f"{cell:8s} {w:4.0f} {wl:5.2f} {s:4.2f} | "
        if r is None:
            print(tag + "does not close")
            continue
        if cell == "3x21700" and r["energy_margin"] < 1:
            print(tag + f"m {r['m']:.2f} kg but the cells hold {r['energy_margin']:.2f} of 15 min")
            continue
        rows.append((cell, w, wl, s, r))
        print(tag + f"{r['m']:5.2f} {r['S']:5.3f} {r['b']:5.2f} {r['batt'] * 1000:5.0f} | "
              f"         {r['f_takeoff']:5.2f} {r['f_margin']:5.2f} {r['f_margin7']:5.2f} "
              f"{r['burst']:4.0f} {r['burst_kg']:5.0f} | {r['torque']:4.1f}  {r['f_water']:5.2f}")
    ms = [r["m"] for *_, r in rows]
    print(f"\n{len(rows)} of 36 close; mass {min(ms):.2f}-{max(ms):.2f} kg")
    lo, hi = ROBO_RAVEN_TW[0] / HUMP[1], ROBO_RAVEN_TW[1] / HUMP[0]
    print(f"planing run, thrust over hump drag: {lo:.2f}-{hi:.2f} "
          "(full-size hull data, a flapper's in-flight thrust)")
    for mu in (0.2, 0.4):
        print(f"ground run on skids, friction {mu}: thrust over friction "
              f"{ROBO_RAVEN_TW[0] / mu:.2f}-{ROBO_RAVEN_TW[1] / mu:.2f}")


if __name__ == "__main__":
    main()
