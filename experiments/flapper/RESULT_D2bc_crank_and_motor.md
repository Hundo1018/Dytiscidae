# Passes D2b and D2c, result: the feather crank away from its toggle; the rotor's inertia

2026-11-17. Frozen text: `PREREG_D2bc_crank_and_motor.md` (`36b3daf`).
Code: `drive/drive.py` (`arm_scale`, `feather["motor"]`, the design's
range and `transmission_min`), `drive/d2_run.py` (stages `d2b`, `d2c`),
`drive/d2bc_read.py`, `drive/draw_d2.py` (takes a geometry). Rows:
`results/D2b_geometry.jsonl` (30), `results/D2c_rotor.jsonl` (26), none
raised. Nothing is chosen here.

## Outcome in one paragraph

**The geared feather crank (FX2) is readable once its shaft sits deep
enough under the horn: 10 of 15 geometries close their energy in a vacuum
to 0.7% or better and follow their design within 3 deg; the five at a
transmission margin of 0.14 or less (D2's among them) lock or run away.**
The readable ones make the take-off feather from the one stroke motor for
197-228 W of machine electrical power against the ideal servo's 178 W, with
lift from -0.6% to +8.7% of the servo's. **The price of a reversing drive
is mostly the rotor: with the witness's rotor at a twentieth, a motor on
the feather costs 187 W for the machine (296 W before) and direct drive of
the stroke 161 W at take-off (295 W) and 8.0 W at cruise (14.6 W).** The
same lighter rotor takes the four-bar's flywheel away: at take-off its beat
falls to 0.907 and its lift by 17%.

## D2b: FX2's geometry (take-off, stroke by FB, lever 60 mm, gear 24)

Readable = B1's rule (vacuum input within 1% of peak shaft power, gap under
0.5 mm, feather range within 3 deg of design at both ends).

| horn | shaft depth | arm scale | transmission_min | vacuum input / peak | readable | air: feather amp | lead | beat | motor on limit | gap mm | lift vs F0 | machine W |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 30 | 80 | 1.00 | 0.14 | +3.22% | **no** | 61.7 | -72 | 0.899 | 0.23 | 2.15 | -14.9% | 302.4 |
| 30 | 80 | 0.85 | 0.40 | +0.06% | yes | 44.2 | -83 | 0.996 | 0.05 | 0.44 | +1.0% | 216.7 |
| 30 | 120 | 1.00 | 0.33 | +0.15% | yes | 54.5 | -83 | 0.999 | 0.00 | 0.42 | +8.0% | 200.2 |
| 30 | 120 | 0.85 | 0.51 | +0.01% | yes | 42.3 | -84 | 0.994 | 0.06 | 0.38 | -0.2% | 212.8 |
| 30 | 160 | 1.00 | 0.42 | +0.01% | yes | 52.6 | -84 | 0.999 | 0.00 | 0.34 | +7.2% | 196.6 |
| 30 | 160 | 0.85 | 0.56 | +0.00% | yes | 42.1 | -85 | 0.993 | 0.06 | 0.39 | -0.6% | 212.4 |
| 45 | 80 | 0.85 | 0.13 | -7.00% | **no** (locks) | 0.1 | | 0.000 | 1.00 | 0.55 | | |
| 45 | 120 | 1.00 | 0.14 | +21.3% | **no** | 61.0 | -74 | 0.829 | 0.27 | 1.97 | -19.8% | 350.6 |
| 45 | 120 | 0.85 | 0.40 | +0.01% | yes | 43.3 | -83 | 0.995 | 0.05 | 0.39 | +1.1% | 215.5 |
| 45 | 160 | 1.00 | 0.28 | +0.16% | yes | 54.7 | -83 | 0.999 | 0.00 | 0.38 | +8.7% | 202.4 |
| 45 | 160 | 0.85 | 0.49 | +0.00% | yes | 42.4 | -84 | 0.994 | 0.06 | 0.39 | +0.0% | 213.5 |
| 60 | 80 | 0.85 | 0.12 | -6.50% | **no** (locks) | 0.1 | | 0.000 | 1.00 | 0.40 | | |
| 60 | 120 | 0.85 | 0.24 | +0.71% | yes | 46.1 | -82 | 0.996 | 0.05 | **0.58** | +3.4% | 227.6 |
| 60 | 160 | 1.00 | 0.14 | +45.7% | **no** (locks) | 5.6 | | 0.024 | 1.00 | 0.53 | | |
| 60 | 160 | 0.85 | 0.40 | +0.01% | yes | 43.0 | -83 | 0.995 | 0.05 | 0.38 | +1.0% | 215.3 |

F0 (servo): 1.927 weights, 178.2 W. FM best with the witness rotor: 295.5 W.

| | frozen | read | |
|---|---|---|---|
| B1 | readable: the seven at 0.40 or more; not: the five at 0.14 or less; three between unpredicted | as predicted; the three between (0.24, 0.28, 0.33) are readable too | **holds** |
| B2 | readable rows in air: amplitude within 3 deg of design, lead within 20 deg of a quarter beat, beat 0.98+, on limit under 10%, gap under 0.5 mm | amplitude 0.0-1.7 deg from design; lead 5-8 deg short of a quarter beat; beat 0.993-0.999; limit 0-6%; gap 0.34-0.44 mm, **0.58 at 60/120/0.85** | holds for nine; the tenth's gap is over |
| B3 | 30/160/1.00: lift within 10% of F0; 170-215 W, under FM's best by 80 W | +7.2%; 196.6 W, 99 W under | **holds** |
| B4 | arm 0.85: lift 75-100% of F0; 160-215 W | lift 99.4-103.4%; 212.4-227.6 W, four of seven over 215 | **wrong on both, narrowly**: 42-46 deg of feather lifts as much as the servo's 50, and costs more than the fuller arm, with the motor on its limit 5-6% of the beat |
| B5 | peak loop force under 400 N (horn 30), 250 N (horn 60) | **not measured**: the recorded peak (323-693 N) is the largest over every loop, the stroke's coupler included (296 N without any feather crank); the rod's own force was not recorded | not read |

What the sweep says about the geometry: the margin is set by the shaft's
depth against the horn's length and by the arm's radius; a deeper shaft or
a smaller arm moves the loop off its toggle. A deeper shaft is a bigger
mechanism: at 160 mm the feather crank's lowest point is 183 mm below the
shoulder line (the drawing), against a hull of 40 mm radius.

The drawing (`figures/D2b_fx2_30_160.png`, 30/160/1.00, from the built
model, vacuum, held body): arm 23.0 mm, rod 161.2 mm, horn 30.0 mm aft of
the hinge at all four angles, shaft 160 mm under the horn; feather -45.1,
+12.7, +51.4, -15.1 deg at crank 0, 90, 180, 270; largest gap over the
whole beat 0.044 mm (D2's geometry: 0.98 mm). Matches the table's row.

## D2c: the rotor's inertia (J_r 2e-5, 5e-6, 1e-6 kg m^2)

Best gear by D1's rule with the beat held; electrical power.

| | 2e-5 (witness) | 5e-6 | 1e-6 |
|---|---|---|---|
| FM, machine W (feather motor W), gear | 295.5 (148.2), 16 | 205.5 (57.3), 32 | 186.8 (39.2), 32 |
| DD cruise W, gear | 14.59, 48 | 9.05, 90 | 7.97, 90 |
| DD take-off W, gear; on limit | 294.9, 32; 41% | 179.0, 32; 0 | 161.2, 32; 0 |
| FB cruise W (gear 48); beat | 7.32; 1.000 | 7.32; 1.000 | 7.28; 0.996 |
| FB take-off W (gear 24, lever 60); beat; lift | 149.4; 1.000; 1.927 | 138.1; 0.981; 1.918 | **101.4; 0.907; 1.592** |

| | frozen | read | |
|---|---|---|---|
| C1 | FM falls with J_r at every gear; at 1e-6 best 210 W or less at gear 32; at 5e-6, 215-260 W | falls at every gear; 186.8 W at gear 32; **205.5 W** at 5e-6 | first two hold; the 5e-6 range is wrong (it is cheaper than I predicted) |
| C2 | DD cruise at 1e-6 within 15% of FB's 7.32 W; take-off under 175 W, on limit under 10% | 7.97 W (+8.9%); 161.2 W, 0 | **holds** |
| C3 | FB changes under 2% across rotors at both points | cruise 0.5%; **take-off -32%, with the beat falling to 0.907** | **wrong at take-off** |
| C4 | lift within 3% of the same candidate's at 2e-5 where amplitude and beat hold | DD within 3.1%; FM at 1e-6 down 2.5-4.2% (gears 32-16) with the amplitude and beat held; FB at 1e-6 take-off down 17% (beat lost) | wrong for FM at gears 16 and 24, narrowly; the cause was not looked for |

C3 was the control for "the saving is the reversal's". It failed in a
direction that does not undo C1 and C2 but adds a fact: **the rotor is the
crank's flywheel.** A crank's motor sees a torque that swings through the
beat; the reflected rotor (J_r N^2) carries it through. Take the rotor
away and the four-bar's speed controller cannot hold the beat at take-off.
So the two kinds of drive want opposite rotors, and the comparison "DD 161
W against FB 149 W at take-off" is between a DD with a light rotor and an
FB with a heavy one. As the pre-registration says, the savings in C1 and
C2 are reported as measured and not as proof of the mechanism.

## What this changes in D1 and D2, as ranges

- Feather at take-off, machine electrical power: servo 178 W; crank geared
  to the stroke 197-228 W (ten readable geometries); a motor on the
  feather 187-296 W over a twentyfold range of rotor inertia.
- Stroke at take-off: four-bar 149 W (witness rotor); direct drive 161-295
  W over the same range. At cruise: four-bar 7.3 W; direct drive 8.0-14.6 W.
- Direct drive remains the only stroke drive here that can change its
  amplitude; D4 is building crank candidates that can.

## What these numbers rest on

As D1 and D2: witness motor, no gearbox or bearing loss, ideal
regeneration, ideal 1:1 gear between the two cranks, links without fluid
load, body held. The rotor sweep changes one number of a motor model;
whether a motor with that torque, speed and rotor exists is not asked. Not
checked by the model: the mechanisms pass through the hull.
