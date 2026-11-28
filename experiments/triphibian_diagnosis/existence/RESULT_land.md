# E0b result — land with the weight on the ground

Spec: `PREREG_land.md` (frozen 2026-11-11, not edited). Run 2026-11-11, main @
a7770c2, MuJoCo 3.15, numpy path, no training, no optimiser. `triquad.py`
sha256 before and after: `1cc5f4bd199505f412146f468e3a14f71b20ec55f7409491d50103ba99ecee67`.
Code: `landgait.py` (imports `triquad` and `nulls`, edits neither). Rows:
`results_land.jsonl` (280), tuning rollouts `trials_land.jsonl` (31, seeds
100-103 only). No row in `results_land.jsonl` failed: 0 `bad_qacc`, 0
divergences, 0 flat packs, 0 exceptions.

**Arm L passes every row.** B1 with four jointed limbs walks the commanded way
with its whole weight on the ground and its rotors commanded to zero (land
0.829-0.948 on 8 of 8 seeds, support share 0.998-1.001), and the same body
still flies, swims, makes the four crossings and both mission transitions from
each start under E0's law. The three nulls read 1e-16 or less. Arm W (the same
limbs made of paddles) does the same.

## 1 The support share `s` and its four probes

`s` = mean vertical ground-contact force on the machine over the scored window
of the walk phase (the late half of the land segment's cruise phase), divided by
dry mass x g; read each step from `mj_contactForce` over machine-world contacts
(`landgait.ground_fz`, `Recorder`). Reported per half; a row's `s` is the lower
of its two halves. Probes, tuning seed 100, `logs/land_probes_final.log`:

| probe | bar | reading |
|---|---|---|
| L1 resting still on the beach (still arm) | 0.95-1.05 | 0.9999991 in both halves (stop window 0.9999990) |
| E0's hover taxi (triquad's `Hand` on B1), land score 0.742 | <= 0.05 | 0.0 in both halves |
| skid, rotors at 94% of the weight (`c_ref` -0.012 on B1), land score 0.631 | <= 0.15 | 0.086 and 0.081 |
| airborne (L1 air segment) | exactly 0, and `s` missing | Fz mean over the rollout 0.0; `s` None (no land steps) |

All four pass. One instrument error was found by the resting read and fixed
before any arm L number was taken: `model.body_mass` still carries the fluid
solver's added mass after a water rollout on the same environment, and a
resting body read 0.541. The normaliser is the solver's dry mass.

## 2 Arm L

Body L1 and its gait are in sections 4 and 5. Segments 8 s, wired as E0's
table (one environment per seed; air, water, land in that order).

| seed | L1 land score | L1 `s` (two halves) | L1 rotor speed in the walk window, max rad/s (commanded 0; top speed 1181) | L1 bad_qacc | B1 as built (hover taxi) land score | B1 `s` |
|---|---|---|---|---|---|---|
| 0 | 0.948 | 1.001, 1.000 | <= 0.40 | 0 | 0.713 | 0.000 |
| 1 | 0.914 | 1.000, 1.000 | <= 0.40 | 0 | 0.741 | 0.000 |
| 2 | 0.845 | 1.000, 1.000 | <= 0.40 | 0 | 0.747 | 0.000 |
| 3 | 0.829 | 0.999, 1.000 | <= 0.40 | 0 | 0.761 | 0.000 |
| 4 | 0.857 | 1.000, 1.000 | <= 0.40 | 0 | 0.751 | 0.000 |
| 5 | 0.941 | 0.998, 1.000 | <= 0.40 | 0 | 0.711 | 0.000 |
| 6 | 0.914 | 1.000, 1.000 | <= 0.40 | 0 | 0.753 | 0.000 |
| 7 | 0.918 | 1.001, 1.000 | <= 0.40 | 0 | 0.737 | 0.000 |

(0.40 rad/s is the largest over the eight seeds; the per-seed values are in the
rows. B1's rotors turn at up to 649 rad/s in the same window. B1's eight land
scores re-run here equal E0's stored ones to three decimals.) L1 land detail:
walk progress 0.86-0.99, across-track 0.01-0.21 of the commanded speed, stop
0.97-1.00, contact-while-upright 0.97-0.99, upright 0.93-0.94.

E0's rows on the legged body, E0's law unchanged (`LegHand`: `Hand` with the
land segment's rotors off), beside B1 from `results_final.jsonl`:

| item | L1 | B1 as built |
|---|---|---|
| air, seeds 0-7 | 0.908 0.898 0.910 0.892 0.921 0.920 0.899 0.908 (8/8 >= 0.5) | 0.891-0.927 |
| water, seeds 0-7 | 0.209 0.747 0.440 0.436 0.394 0.196 0.337 0.189 (8/8 >= 0.012, 8/8 >= 0.15) | 0.211-0.717 |
| air_to_water | crossed, hold 0.797 | crossed, hold 0.871 |
| water_to_air | crossed, hold 1.00 | crossed, hold 1.00 |
| water_to_land | crossed, hold 1.00 | crossed, hold 1.00 |
| land_to_air | crossed, hold 1.00 | crossed, hold 1.00 |
| mission from air | 2/2, `continuous_score` 0.687; land leg entered 5.37 s after the command; landed 2.6 s, `s` 1.03, walked 0.10 m | 2/2, 0.704 |
| mission from water | 2/2, 0.771; entered at 5.42 s; landed 2.5 s, `s` 1.03, walked 0.04 m | 2/2, 0.775 |
| mission from land | 2/2, 0.786; on the ground 7.96 s, `s` 1.005, walked 0.28 m | 2/2, 0.792 |

The crossing and mission rows are identical on all eight seeds (they do not
read the seed): n = 1 each. In the missions the machine reaches the beach by
flying, as in E0, and walks from touch-down until the next command (heading
+x); rotor speed after touch-down settles to under 0.6 rad/s.

Nulls on L1, land segment, seeds 0-7:

| arm | land score, 8 seeds | `s` |
|---|---|---|
| still (`held_still_params`) | max 1.0e-19 | 1.000 |
| gait with the heading frozen at the first one seen | max 1.0e-16 | 0.998-1.001 |
| gait fed seed (s+1)%8's task | max 4.9e-17 | 0.999-1.000 |

## 3 The five predictions

1. *"Arm L, unmodified land score through `evaluate.run_segment` (antipodal
   pair): >= 0.15 on >= 7 of 8 seeds, with `s` >= 0.9 and every rotor stopped
   on each of them. I hold this at about 0.5. Falsified by fewer than 7 seeds,
   or by any passing seed with `s` < 0.9."* **Held.** 8 of 8 (0.829-0.948), `s`
   0.998-1.001, rotors commanded 0 and measured at 0.40 rad/s or less.
2. *"Arm L's body through E0's law, rows and bars unchanged (`existence/PREREG.md`
   §3): air >= 0.5 on >= 7 of 8, water >= 0.012 on >= 7 of 8, all four
   crossings, both mission transitions from each start with the land leg
   walked, not taxied. I expect water or `water_to_land` to be what breaks if
   anything does (E0m: `water_to_land` was lost by 16 of 29 mutants)."*
   **Held.** Air 8/8, water 8/8, four of four crossings, 2/2 from each start
   with the landed part walked (`s` 1.00-1.03). Nothing broke; `water_to_land`
   crossed with hold 1.00.
3. *"Nulls on arm L's body, land segment, 8 seeds each: still (every actuator,
   `held_still_params`) < 0.012 on 8 of 8; the gait with the command frozen
   (heading fixed at the first one seen) < 0.012 on 8 of 8, because the
   antipodal pair cancels motion the command did not choose; the gait fed
   another seed's task < 0.15 on 8 of 8."* **Held.** Maxima 1.0e-19, 1.0e-16,
   4.9e-17.
4. *"The land score, as it stands, pays the walker less than the taxi (the
   taxi's 0.71-0.76 was earned at a speed legs will not reach). I expect arm L
   between 0.15 and 0.5."* **Refuted.** The walker is paid more on every seed:
   0.829-0.948 against 0.711-0.761. It reaches the commanded 0.04 m/s (walk
   progress 0.86-0.99) and keeps contact, which the taxi gives up (contact
   0.97-0.99 against 0.52).
5. *"Arm W: `s` >= 0.9 and land score >= 0.012 on at least 4 of 8 seeds. I hold
   this at about 0.3. Nothing in the plan depends on it."* **Held.** 8 of 8:
   land 0.784 0.894 0.866 0.795 0.685 0.796 0.827 0.847, `s` 1.000 on each,
   rotors at 0.51 rad/s or less. Section 6.

## 4 Body edits and Tier-0

L1 is B1 (`triquad.b1_genome`) with these edits and no others
(`landgait.L1_EDITS`, `l1_genome`):

| gene | B1 | L1 |
|---|---|---|
| ground gear | four fixed `FOOT` legs (part 2, two mirrored edges) | removed |
| limbs | none | four: a `STRUT` (length 0.09 m, radius 6 mm, CFRP) on a `universal` joint, actuated, out to the side from the hull at 42% and 58% of its length (mirrored, roll pi/2), stroke axis local z with range +-0.7 rad (raises and lowers the foot), feather hinge +-1.0 rad as `mjcf` builds it (swings the foot fore and aft); each carries one fixed `FOOT` (0.075 m, 6 mm) straight down from its tip |
| limb motors | none | 8 (`bldc`, 0.03 kg, gear 6, not sealed): stall torque 1.115 N m, servo kp 2.23 N m/rad; `stroke_amplitude` 0 so the body's own base gait holds them still |
| hull radius | 0.059 m | 0.064 m (one re-set, for the density ratio: 1.21 at 0.059, 1.10 at 0.064) |
| `rotor_throttle` | 0.555 | B1's rule re-applied to the new mass (1.05 x hover) |

L1 as built: 2.437 kg (B1 2.045; +0.24 kg of limb motors, the rest hull),
density ratio 1.102, 12 actuators (4 rotors, 8 limb), level-rig margin 0.933,
no air gates. **Tier-0 as the search runs it: passes** (`gate_margin` +0.151,
report `PASS`, worst check `can_submerge` +0.15; `logs/land_facts_v1.log` is the
first build at radius 0.059, also a pass at +0.19). Feet sit 7.5 cm under the
hull axis, 1.3 cm below the hull's belly.

W1 (`w1_genome`) is L1 with the jointed part a `PADDLE` (span 0.09 m, chord
0.05 m) and the fixed part a `PADDLE` (span 0.075 m, chord 0.04 m): 2.443 kg,
density ratio 1.103, Tier-0 passes (+0.152).

## 5 The gait

`landgait.LegLaw`, `GAIT`. All four limbs in step at 1.5 Hz: stance (stroke
+0.25 rad, feet pressed down, body carried) while the feet sweep back, swing
(stroke -0.35 rad, feet lifted, body on its belly) while they sweep forward;
sweep amplitude 0.6 rad. Steering is by heading error only: left limbs sweep
with gain v - w, right with v + w, w = clip(2 e), v fading to 0 at 0.6 rad of
error; an error over 90 degrees is walked backward. Told to stop, the limbs
lift and the body sits. Off land they hold one pose. It reads the commanded
heading and the body's yaw (`env._phase_now`, the root's rotation matrix) and
the clock; nothing else.

Tuning: two settings were run on seeds 100-103 (`trials_land.jsonl`): the first
(2 Hz, sweep 0.8, down 0.35, up -0.5) scored 0.70-0.91 and bounded; the second,
kept, 0.88-0.95. The first rollout of all had every limb on the same side
(a sign read from the wrong field) and went downhill whatever the command;
fixed before those two.

## 6 Arm W (exploratory)

Same gait, not re-tuned (one check on seeds 100-103: 0.66-0.94). Seeds 0-7:
land 0.685-0.894 (8/8), `s` 1.000; still 0 on 8/8; frozen heading 0 on 8/8; air
0.886-0.911; water 0.204-0.748; four crossings crossed (hold 0.796, 1, 1, 1);
missions 2/2 from each start (0.709, 0.777, 0.787). The surfaces are limbs
here, driven as limbs; they are not flapped and do not fly the body.

## 7 Reproduce (one line per row kind)

```bash
cd /home/hundo/Projects/Dytiscidae/dytiscidae; export MUJOCO_GL=disable PYTHONPATH=.; E=experiments/triphibian_diagnosis/existence
.venv/bin/python $E/landgait.py facts --body L1                                                      # body, Tier-0, limb kinematics
.venv/bin/python $E/landgait.py run --body L1 --arm still --items land --seeds 100 --out /dev/stdout # probe 1 (2: --body B1 --arm hand; 3: add --gains '{"c_ref": -0.012}'; 4: --body L1 --arm walk --items air)
.venv/bin/python $E/landgait.py run --body L1 --arm walk --items air,water,land --seeds 5 --out /dev/stdout       # a media row (land needs air and water first)
.venv/bin/python $E/landgait.py run --body L1 --arm walk --items water_to_land,mission_water --seeds 0 --out /dev/stdout   # a crossing and a mission
.venv/bin/python $E/landgait.py run --body L1 --arm frozen --items air,water,land --seeds 5 --out /dev/stdout     # a null (still, frozen, shuffled)
.venv/bin/python $E/landgait.py run --body B1 --arm hand --items air,water,land --seeds 5 --out /dev/stdout       # B1 as built beside it
.venv/bin/python $E/landgait.py run --body W1 --arm walk --items air,water,land --seeds 5 --out /dev/stdout       # arm W
python3 $E/logs/land_summary.py                                                                      # the tables above from results_land.jsonl
```

## 8 What I could not check

* **`s` is a time mean.** It reads 1 for any machine that neither leaves the
  ground for good nor is held up by something else, including one that bounds.
  The kept gait keeps contact 97-99% of the land segment; the short flights that
  remain are in that 1-3%. `s` also does not say which part carries the weight:
  in swing the body is on its belly, in stance on its feet.
* **Tuning seeds do not exist for crossings and missions.** They ignore the
  seed, so the one rollout per kind is both the first try and the scored row.
  None was tuned: each passed on its first run (`trials_land.jsonl`, "L1
  through E0's law, first pass").
* **Entry-state robustness of the legged body's crossings** (E0's X1) and the
  blind and base nulls on its crossings (E0n) were not run.
* **Limb and propeller clearance.** The limbs' hips are placed clear of the four
  discs in plan view, but the simulator does not collide a rotor with anything,
  so a strike would not have shown.
* **The walk inside a mission is short**: 0.04-0.28 m in the 2.5-8 s on the
  ground, toward a fixed heading. No row asks more of it.
* **Rotor "stopped"** is a command of 0 and a measured 0.4-0.6 rad/s (the
  rotors turn freely on their hinges as the body rocks).
* **Whether a learned policy can drive this gait**, and wheels: not measured,
  as the spec says.
* Energy was booked per rollout (land segment 95 W mean on L1 against 214 W
  for B1's taxi; air 533 W against 433 W, the price of 0.39 kg) and not
  extrapolated.
