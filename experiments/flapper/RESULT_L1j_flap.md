# L1 pass 1j result: the one-way flap as a built panel (2026-11-15)

Pre-registration: `PREREG_L1j_flap.md` (commit 832d541; a direction bug and
two probes disclosed there). Data: `results/L1j_flap.jsonl`, 306 rows, all
finite, one pass. Reader: `l1j_read.py`. Held body, water. The rear 60% of
each wing is a hinged body with a stop; "welded" is the same split wing with
no hinge.

**Erratum 2 (found 2026-11-15 while building `freerig.py`): the stroke's
inertial push on the free joint never acted in this pass.** The rig wrote it
to `qfrc_applied` before `env.step`; the fluid solver rewrites all of
`qfrc_applied` inside the step and wiped it. `probe_push.py`: on the still-water rowing point the flap-up arm changes from +1.37 N at 17.9 N m to +2.33 N at 6.1 N m, and on pass 1e's up-down point from -1.79 N to +0.24 N. In water the push includes the flap's added mass, so every flap number below is void until re-run.
The rig is fixed (`legacy_coupling=True` reproduces this pass as run). Re-run with the push acting as `results/L1j_flap_r.jsonl` (`PREREG_L1r_reruns.md`).

## Block A: pass 1e's up-down stroke, 0.3 m/s

Best net forward force under both torque caps:

| wing (length) | welded | flap up | flap down | pass 1e, chord law |
|---|---|---|---|---|
| 0.4 | **+6.71 N** (0.5 Hz, feather 30, 10.3 N m, 7.6 W) | -0.07 N | -0.24 N | 18.19 N |
| 0.6 | **+3.30 N** (0.25 Hz, feather 30, 10.8 N m, 3.1 W) | +0.18 N | -0.29 N | 11.77 N |
| points meeting 1e's gate | 7 / 2 | 0 / 0 | 0 / 0 | 9 / 5 |

- **P1 confirmed.** The better flap arm gives -0.01 and 0.06 of the welded
  wing's best, not more. No flap point meets the gate. Pass 1e's 3.1-3.6
  times is not reproduced by a flap that opens by itself.
- **P2 confirmed.** The welded wing reads 6.71 and 3.30 N against pass 1e's
  fixed wing at 5.92 and 3.28 N (13% and 1%): the split did not move the
  baseline.
- **P3 failed.** Mean power is negative at 8 of 128 flap points (6.3%; the
  prediction was under 2%).
  - All 8 are flap-up points at stroke 40 deg with feather 20 or 30 deg. No
    welded point and no rowing point (0 of 64) is negative.
  - The largest, -4.7 W, comes with -7.4 N at 0.3 m/s, so the tow supplies
    only 2.2 W: the stream does not account for it.
  - Suspected cause, not tested: the rig charges the feather axis with the
    flap's whole fluid moment about the span, but a free hinge parallel to
    that axis does not pass that moment on. If so it is the rig's power
    accounting for a prescribed feather with a free flap, and forces and
    stroke torque do not depend on it.
  - Until tested, power figures of flap arms with a feather are not to be
    quoted. P1 rests on force and stroke torque.

## Block B: a rowing stroke (hull vertical, no feather)

Still water, stroke 85 deg:

| wing | frequency | welded | flap up | flap down |
|---|---|---|---|---|
| 0.4 | 0.25 Hz | +0.01 N | +0.40 N, 4.1 N m | -0.38 N |
| 0.4 | 0.5 Hz | +0.03 N, 5.8 N m | **+1.37 N**, 17.9 N m | **-1.48 N**, 12.2 N m |
| 0.4 | 0.75 Hz | +0.05 N | +2.45 N, 12.3 N m | -3.05 N |
| 0.6 | 0.5 Hz | +0.11 N | +5.40 N, 28.0 N m | -4.86 N |
| 0.6 | 0.75 Hz | +0.20 N | +10.83 N, 62.7 N m | -9.77 N |

- **P4, made blind: direction and symmetry confirmed, size refuted.**
  - Welded: 0.03 N, inside the predicted 0.5 N.
  - Flap up +1.37 N and flap down -1.48 N: opposite and within 8% of each
    other, as predicted. But I predicted 2 to 10 N; both are below it.
- The valve works as a valve: a welded paddle gives nothing in still water
  and the flap gives thrust in the direction it is built for. It costs
  torque: most points above 0.5 Hz are over the caps.
- At 0.3 m/s every arm at every point reads negative (-5.8 to -68.9 N): the
  broadside posture's own drag (-16.1 and -27.9 N at rest, welded) is never
  overcome. **P5's first half confirmed.**
- **P5's second half refuted at 1 of 16 points**: flap up beats welded
  everywhere (by 3.9-18.3 N) except wing 0.6, stroke 85, 1.0 Hz (-57.6
  against -43.5 N), a point far over the caps.
- Much of flap up's lead is not rowing: at rest the stream already holds the
  flap open and the lead is 4.4 and 6.0 N.

## Block C: the flap itself (still water, wing 0.4, stroke 85 deg, 0.5 Hz)

| cut at | spring 0.005 N m/rad | spring 0.05 N m/rad |
|---|---|---|
| 0.2 chord (flap 80%) | +0.56 N, 29.3 N m | **+2.34 N, 6.8 N m** |
| 0.4 | +1.37 N, 17.9 N m | +1.41 N, 8.6 N m |
| 0.7 (flap 30%) | +0.52 N, 8.8 N m | +0.34 N, 10.0 N m |

- A stiffer return spring cuts the peak torque by two to four times at the
  same or more thrust. The soft spring's torque peaks look like the flap
  hitting its stop.
- The best point within the caps anywhere in still water is 2.34 N.

## Reading

- Rival (a) holds for the stroke of passes 1b-1e. It is lift-based; both
  halves make thrust through a controlled feather, and a flap that lets go
  throws its half away.
- Rival (c) holds too. The flap is a one-way valve and gives thrust only in
  a drag-based stroke. There it gives 1.4-2.3 N within the caps in still
  water, against 6.7 N from the welded wing in the lift-based stroke at
  0.3 m/s.
- Rival (b), pass 1e's reading, is not supported by a passive flap.
  - This does not show that 1e's gain was created energy. 1e narrowed the
    chord while the feather still set the incidence; a free flap gives up
    the incidence as well.
  - A chord that narrows under control on one half (a fold driven per beat)
    is still untested, and 1e's own artefact check still stands failed.
- Rival (d) is open: strips do not see each other, so nothing here includes
  the open flap sitting in the front part's wake or flow through the gap.
- **Decision: the water wing does not carry a passive per-beat flap.** The
  shape change for water stays the fold set once (passes 1c-1d: 5.9-15.5 N).
  The user's point, least drag on the recovery, is met in the lift-based
  stroke by the feather, which is why the cam of pass 1i matters in water
  too.

## Not covered

Wake and gap flow between the parts; a flexible flap; impact at the stop; a
rowing stroke in a streamlined posture; the suspected power-accounting fault
for flap arms with a feather; a driven per-beat chord change; free body; air.
