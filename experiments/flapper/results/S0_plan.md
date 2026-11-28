# S0 plan: how the six scenarios are assembled (written before any scored row)

2026-11-19. Spec: `PREREG_S0_continuation.md`, frozen at dd89844, not edited.
Code: `s0_machine.py` (the machine of each rung), `s0_run.py` (rows),
`s0_read.py` (the frozen gates, imported, not copied), `s0_draw.py`.
Rows `results/S0_rows.jsonl`, tuning `results/S0_tuning.jsonl`, logs
`results/s0_logs/`.

## The machine at each rung (`s0_machine.build`)

- **A0.** C1 = `c1_pilot.body()` on `mission.Machine`, 0.6 kg, the hull's
  transverse inertia set to 0.012 kg m2 as `m4_common.build` does ("as C1").
  M1b, M2b and M3b ran without that override (default 0.0076); this is the
  one machine for every scenario, so the override is applied everywhere
  (disclosed). Every hinge on `tau_s = 1e4`, `omega_0 = 1e7` (`motors=`
  argument; `J_r` and gears as frozen, so the copper term `omega_0/tau_s`
  is 1e3 as in the M series and `E_elec` stays comparable). The strip
  twist is enabled exactly where the frozen law uses it (M4's legs).
- **A1.** `m3b_sweep.SweepMachine` (0.66 kg), the sweep hinge on the ideal
  motor too. Taken at once for land, land<->air, three-media.
- **A2.** Hull ballast at the hull's centre so that the solver's fully
  submerged buoyancy (`solver.diag.buoyancy`, read on the built machine at
  5 m depth, held) over the weight is b; b = 1.00, then 0.97, then 1.03.
- **A3** (only if no b passes both dive and exit): hull `_dry_mass` and
  `body_mass` changed by a ballast law, ±8% of displacement, ≤2%/s, then
  `mj_setConst` on a scratch MjData.
- **A5p.2.** The A2 ballast mass (the mass added to reach b) moved from the
  hull's centre to a point 0.4 x hull half-height (0.016 m) below it, as a
  contact-free point mass in the hull body at compile time.
- **A6.** `body.built` with (1) tip chord x1.5, root chord and span
  unchanged (`wing_shape` taper 1.5 with `wing_chord` x1.25, since `built`
  keeps the mean chord), (2) `wing_length` x1.25, or (3) `hind_u` set for a
  second pair 0.3 m behind the first, its strokes at the fore pair's phase
  plus 180 deg.

**Cumulative reading (reported to the lead, not resolved here).** The
machine at a rung carries every rung taken so far; only scenarios that have
not passed are re-run (§2: "every scenario that has not passed yet is run").
A scenario that passes is established on the machine of the rung where it
passed; the union is tested only by the 15-minute run, which exists only at
S0*. A1 is in every machine from A2 on, held at 0 by laws that do not use it.

## The scenarios, each from its frozen gates

Pass rule with n = 2 scored starts per gate: the frozen count is a fraction
(M4 H/T 5/6, entries 3/4, M3b walk 5/6, turn 2/3, M2b GX30 4/6, GW and GS
5/6); with 2 starts each fraction needs 2/2. Every gate also needs its
do-nothing arm to fail on every start; a null that passes makes the gate
"cannot tell", and no verdict rests on it. Water gates: both entrainment
modes ("wetting" and the default `True`), each 2/2; a split is "not
established".

| scenario | legs (gate, code) | do-nothing |
|---|---|---|
| air | M4 cruise at P35 from M4's assisted launch (decision 2): H and the ±90 pair T (`m4_lib.run_half`, `m4_read.cruise_half`, pair P_t >= 60 and hashes) | N |
| land (A0+A1) | KS stand (`m3_read.stand_pass` as m3b_read), K walk pairs at 12 s with U per pair (`m3_read.walk_pairs`), K turn pairs at 14 s (`m3b_read.turn_pairs`); `m3b_land.run`, window 14 s | N0 |
| water | dive D: float start (M2 GS start), WT told depth 3.0 m, M2's held clause (mean depth 35-40 s within 0.5 m of 3.0, depth >= 0.3 m over 15-40 s); swim GW (M2b, depth start, halves A/B, `m2_read.gate_gw`); surface U: depth start, WT told depth 0, M2/GS's surface clause (depth <= 0.5 m at every sample of 35-40 s); surface turn GS (M2b ST, `m2_read.gate_gs`) | still per leg |
| land<->air (A0+A1) | take-off: V+ (`m1_takeoff.run`, G1-G4); cruise: M4 H from the assisted launch; landing: M4 land entry pair near/far (`m4_read.entry_half` E1-E5) with law E (A0) or A4; take off again: V+ from the landed rest state, same rollout, G1-G4 | N0 / N / NL |
| water<->air | exit + 30 s: M2b XS (`m2b_read.gate_gx30`); water entry: M4 water pair with E (A0) or A4, both modes; exit again: XS from the entry's rest state, same rollout, GX30 | X0f / NW |
| three media | only at S0*: stop and report the cost first | - |

Legs that need a transition no frozen law provides (hover after take-off
into cruise; XS hover into the M4 entry approach) start from the leg's own
frozen start (M4's assisted launch). That is assembly, not chaining, and is
disclosed; the 15-minute run is the chained test.

A4 (law): M4's cruise with L until the CoM passes x_e, then V+ (`m1_takeoff.Fly`,
V+ gains) with H read above the surface under it and H_t falling at 0.5 m/s
from the switch, until first contact, then M4's frozen contact hold. No free
parameter is named in the prereg; it is run on the tuning starts and, if it
fails there, the rung fails (nothing to tune). Expected risk, stated now:
V+ hovers nose-up ~87 deg and E2 needs -5 <= pitch_rel <= 30 at contact.

## Starts (disjoint from every M-series tuning start)

Avoided: M1 TUNING (17,0) (18,180) (27,0) (27,180) (22,90) and STRESS
(25,90) (25,270); M3 tuning yaw 7, -25, 97 / x 21 / phase 0.5; M4 U1, U2;
M2/M2b seeds 100-103.

| gate | tuning | scored |
|---|---|---|
| cruise (roll, yaw, V) | QT1 (2, 20, 8.0), QT2 (-2, -110, 8.1) | Q1 (3, 30, 8.2), Q2 (-4, -150, 7.8) |
| take-off (x, yaw) | (23.5, 330), (20.5, 120) | (19.5, 30), (26.5, 210) |
| land walk (yaw, phase, x, heading) | TW1 (-8, 0.1, 23, 0) | SW1 (5, 0.25, 20, 0) up/down; SW2 (82, 0.75, 26, 90) across |
| land turn (yaw) | TR1 (60) | SR1 (40), SR2 (130) |
| stand (yaw, x) | - | SS1 (20, 19), SS2 (200, 27) |
| entries (roll, yaw, V, dz) | ET1 (5, -2, 8.0, 0.25), ET2 (-5, 4, 8.2, -0.1) | ES1 (3, 2, 8.1, 0.1), ES2 (-3, -4, 7.9, -0.2) |
| water seeds (`m2_common.draw`) | 200, 201 | 20, 21 |

A0 runs the frozen laws on the scored starts directly (nothing to tune).
Every later rung: tuning starts first, under §5's 24-rollout budget per rung
per scenario, every rollout logged to `S0_tuning.jsonl`; the first set that
passes every tuning gate is frozen by sha256 (`results/S0_frozen.json`),
then the scored rows. A rung with no named free parameter runs once on the
tuning starts.

## Free parameters (only what §2 names)

- A2: b in {1.00, 0.97, 1.03}, in that order.
- A3: the ballast law's depth gain and its sign (a P law on depth error,
  rate-limited), nothing else.
- A5: XS's frozen tuning knobs of M2b (strike depth, kdx, kpP/kdP, k_v and
  lean, kpz/kdz).
- A5w: WT's feather offset (sgn, a_set as the flow angle) and amplitude
  terms, stroke amplitude A and frequency f in 2.5-6 Hz; ST's amplitude
  split (s1/s2 share) plus the same; nothing else. This is a different
  regime from anything M2b tuned (WT 0.25 Hz, ST 0.6 Hz).
- A5p.1: sweep attitude gains kp, kd (pitch common, roll differential).
- A6: none (the three candidates re-run the frozen laws; 24 rollouts shared
  over the three, tuning starts first).

## §4 guards, read per row

- Stroke frequency: each row carries its law's stroke frequency and a flag
  when outside 2.5-6 Hz. A flapping gate passed outside the band does not
  count toward S0* (the guard says the frequency stays inside it); the K
  walking gait (contact, not fluid) is flagged and reported, not excluded.
- X2: from a per-step instrument on `Machine.step` (every 10th step: CoM
  speed relative to the still medium, the root feather angles): an airborne
  gate whose window has speed < 6 m/s with |feather| > 30 deg on more than
  half its samples is "passes in a regime the model is known wrong in".
- Contact: peak vertical contact over the weight, every step; > 10 weights
  flags the row and the gate leaning on it is re-run at solref time
  constant x2 (`geom_solref[:, 0] *= 2`), which must also pass.
- dt/2: one row per scenario at dt 5e-5.
- Recorded per row (never gated): peak torque, speed and shaft power per
  hinge, energy per leg, peak contact.

## Compute

Probe: 12 s wall per simulated second, one process. A0 is about 4,100
simulated s (about 4-5 h wall on 4 workers). At most 4 processes.
