# Pass S1 pre-registration: the entrainment reaction under a driven wing in water (2026-11-15)

Frozen before the sweep. Code at this commit: `FluidSolver.entrainment =
"wetting"` (opt-in, numpy path only, default unchanged), `amref.py`,
`rig.run(solver_flags=, added_mass_ref=, legacy_added_weight=)`,
`s1_entrain.py`, `s1_read.py`.

## Found before freezing (probes `probe_entrain.py`, `probe_entrain2.py`; wing 0.4, 0.3 m/s unless said, stroke 85 deg, 0.5 Hz)

1. **The instability is the projection, not the wetting.** A wing strip's
   added mass in the mass matrix is its tensor projected on the direction of
   the relative flow, so it is a function of velocity. `entrainment_reaction`
   differences it in time and applies `-dm/dt v`: an acceleration fed back
   explicitly, one-sided. Reacting only the growth a strip's density brings
   (`"wetting"`) is stable on the torque-driven rig at feather 30 and 60 deg,
   and submerged it is the same as the reaction off (identical to the digits
   printed).
2. **The held rig's torque and power series carried the entrained weight.**
   The solver cancels the weight MuJoCo gives the added mass with an upward
   force at each body's centre of mass. `rig.run` took it out of the net
   force only. In water the stroke torque read 11.0 N m where the fluid's is
   6.1 and the torque-driven rig reads 5.5; in still water the instantaneous
   power swung -34 to +50 W where the fluid's is +1 to +10 W. Mean force is
   unaffected; mean power is unaffected at these two points. Every water pass
   (1b-1e, 1j, 1l) and, by about 2% of the torque, every air pass read
   torque and crank metric through it. Fixed in the rig
   (`legacy_added_weight=True` reproduces the old reading).
3. With that fixed the two rigs agree at feather 30: Fx +7.28 held (reaction
   off) against +7.39 torque-driven, power 6.76 against 6.76 W.
4. The solver's default reaction costs the held rig thrust: +5.92 against
   +7.28 N at feather 30, +11.01 against +12.98 N at 60.
5. `amref.py`, the plate's normal added-mass force from the prescribed
   motion: in still water it changes mean power 5.48 to 5.49 W (it should do
   no net work: control passes). It raises mean thrust from +8.55 to +13.31 N
   in still water and +7.28 to +12.04 N at 0.3 m/s, and at 0.3 m/s mean power
   6.76 to 8.19 W. A first form that kept the chordwise term gave mean force
   unchanged and mean power up 67% at feather 60; dropped, because a separated
   plate has no leading-edge suction and that form does net work.
6. On peak instantaneous power at that point the torque-driven rig reads
   12.4 W, the held rig 13.2 and the held rig with the reference 14.9.

## 0 Question and decision

Can the torque-driven rig be used in water, what do pass 1l's gates read with
the held rig's torque corrected and the water's inertia in it, and is the
flap's negative power (pass 1j, R5) the same accounting fault? Decides whether
water measurements of the free body can start, and whether pass 1l stands.

## 1 Rivals

(a) "wetting" is stable everywhere on 1l's grid and the rigs agree: the
projection's reaction was the only fault. (b) Stable but the rigs disagree:
the scalar mass matrix itself misstates the wing's inertia. (c) Unstable
somewhere else on the grid (high frequency, the 0.6 wing). (d) Nothing moves
pass 1l: the correction is small where the gate points are.

## 3 Predictions (grid: pass 1l's 192 moving points and 2 at rest)

- **P1** Torque-driven, "wetting": 192 of 192 finite with tracking error
  under 0.5 deg. Halving the step on 8 points moves Fx and mean power by
  under 3% and the crank metric by under 5%. Falsified by any divergence.
- **P2** Torque-driven against held (reaction off): mean Fx within the larger
  of 5% and 0.3 N, and mean power within 5%, at 173 or more of 192.
- **P3** The corrected held crank metric is below pass 1l's as run at 173 or
  more of 192, median ratio 0.4-0.8. The 0.6 wing gains at least one gate
  point under a centre-of-pressure law (1l: none).
- **P4** Held, the solver's default reaction against off: Fx is lower with it
  on at 154 or more of 192; median loss 10-30% of |Fx|.
- **P5** Crank metric, torque-driven against held with the reference: within
  20% at 134 or more of 192. One probe point (item 6) says the solver's
  inertia is not nearer the reference than no inertia; no count predicted.
- **P6** The reference raises held mean Fx at 154 or more of 192, median by
  30-80%. This is two added-mass models disagreeing, not a measurement of
  either; it is reported as the band on water thrust.
- **P7** The 10 flap points of `L1j_flap_r` with negative mean power: as run
  they reproduce to 1e-6 W (control); on the corrected held rig at most 2 are
  negative; on the torque-driven rig none is.

Gate reading for pass 1l: a point meets it only if it does on the corrected
held rig both without and with the reference.

## 4-5 Measurement

`s1_entrain.py`: arms `ctrl` (held as run, 8 points, must equal
`L1l_camwater.jsonl`), `held` (corrected, reaction off), `heldent`
(corrected, default reaction), `heldref` (corrected, reference), `free`
(torque-driven, "wetting", 1 beat held then 2 measured), `freedt` (the same
8 points at half the step), and the three flap arms. One pass, no tuning
after it. Not covered: the search's default (unchanged), the Mojo kernel, a
released body.
