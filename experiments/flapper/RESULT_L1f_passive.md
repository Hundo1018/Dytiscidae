# L1 pass 1f result: the air passes with a passive feather (2026-11-15)

Pre-registration: `PREREG_L1f_passive.md` (commit 2ef7cf1; rig fix and probes
disclosed there). Data: `results/L1f_passive.jsonl`, 250 rows, all finite,
one pass. Stroke prescribed, feather a spring hinge at the leading edge,
damping 0.3 of critical, mass at the quarter chord, body held.

## (a) Lift from rest

| frequency | best vertical force / weight | aero power | peak drive torque | feather taken |
|---|---|---|---|---|
| 3.5 Hz | 1.15 | 82 W | 13.2 N m | +-36 deg |
| 4.0 | 1.58 | 122 | 17.8 | +-43 |
| 4.5 | 2.08 | 168 | 22.0 | +-44 |
| 5.5 | 3.28 | 301 | 34.1 | +-54 |

Best with the body pitched -90 deg (trailing edges up) and the softest
springs (0.3-1.2 N m/rad). Gate (1.6 weights, <= 12 N m, <= 150 W): **0
points.** The cheapest 1.6-weight point needs 20.7 N m and 164 W.
Prescribed feather (1a): 1.6 weights at 4.0 Hz for 64 W and 10.7 N m.

- Frozen prediction: not met; 14 N m or more needed. **Held** (20.7).
- So a passive leading-edge hinge doubles the torque and power of a take-off.

## (b) Level cruise

149 of 150 points trimmed to the weight; **0 sustain**: every one has net
drag. The best is -0.11 N at 8 m/s, 4.5 Hz, stroke 85 deg, k 5-10, at
180-220 W of aerodynamic power. Stiffer springs come closer (best net force
-0.42 N at k 0.6, -0.11 N at k 5-10), so the trend runs toward a feather
held still, which is not what cruise needs either.

- Frozen prediction: gate met, 20-35 W. **Falsified.**
- With the feather prescribed (1a), the same body sustained at 19.5-38 W.

## Reading

In this model a wing that pitches freely about its leading edge weathervanes
into its own stroke: it carries lift (or, from rest, a lot of it at a high
incidence and a high cost), but in forward flight it gives up the thrust.
Rival (b) stands. Rival (c) stands beside it: the hinge's behaviour is set
entirely by the model's strip loads at the quarter chord and its pitch
damping, neither checked against a hinged flapping wing; real ornithopters
(E-Flap, Robo Raven) fly on passive twist of a membrane behind a stiff
leading spar, which is a distributed, spanwise-varying compliance, not one
rigid plate on one hinge.

**Consequence for K1.** "One drive, passive feather" is not shown to work.
The two passes that met the air gates (1a) did so with a feather this drive
does not have. Either the feather is given another way, or the passive wing
is modelled as the real ones are built. In order of cost:

1. The hinge axis behind the leading edge (at 0.1-0.25 of the chord the
   aerodynamic moment shrinks and changes sign; the spring then sets the
   incidence instead of the weathervane). One parameter in the rig.
2. A spanwise-varying passive twist (stiff root, compliant tip), which is
   the real wings' layout; needs a wing in several hinged panels.
3. A feather driven from the stroke by a cam or linkage (no second motor),
   or a second actuator.

Not covered: free body, wing bending, a hinge anywhere but the leading edge.
