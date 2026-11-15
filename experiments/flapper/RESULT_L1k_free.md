# L1 pass 1k result: the cam drive with the body let go (2026-11-15)

Pre-registration: `PREREG_L1k_free.md` (commit 7f40010; one take-off and one
cruise probe point disclosed there). Data: `results/L1k_free.jsonl`, 168
rows, all finite; worst tracking error 0.71 deg. Reader: `l1k_read.py`.
Points: pass 1i's 16 take-off and 26 cruise gate points. No tail, no
controller. Joint damping is zero unless stated.

## The new rig against pass 1i, body held (P1)

| | lift | power | crank metric |
|---|---|---|---|
| take-off, 16 points | -4.0% to -0.6% | -3.0% to -0.1% | -14.5% to +14.7% |
| cruise, 26 points | -2.3% to +0.4% | -0.1% to +1.6% | -7.3% to -0.6% |

- **P1 confirmed for lift and power** (both within 5%).
- **P1 refuted for the crank metric**: I predicted it would only fall, by up
  to 25%. At take-off it moves both ways, by up to 15%.

## Take-off

| arm | result |
|---|---|
| held, new rig | **13 of 16** meet the gate; lift 1.58-1.83 weights, 90-128 W, crank 9.4-11.1 N m |
| x and z free | climbs 0.89-1.83 m in 4 beats (14 of 16 by 1 m or more); 73-100 W; crank 6.4-10.2 N m; bob 21-54 mm |
| x, z and pitch free | climbs 0.96-2.16 m; pitch swings 25-118 deg; crank 8.5-20.9 N m (10 of 16 within 12) |
| held, built joint damping | **2 of 16** meet the gate; 126-166 W, crank 11.0-12.9 N m |

- **P2 confirmed**: 13 of 16 (predicted 10-14).
- **P3 mostly refuted.**
  - Every point climbs, but 2 of 16 by less than 1 m (both with the hinge at
    0.2 chord: 0.89 m).
  - Lift over the first beat after release is 0.72-0.84 of the held lift
    (1.17-1.40 weights), not the 0.85-1.0 I predicted.
  - Read against the frozen gate, **0 of 16** reach 1.6 weights over that
    first beat.
  - That gate was badly chosen, and is recorded as a fault, not repaired: a
    released body is climbing at over 1 m/s by the end of its first beat, and
    the lift falls because of the climb. Static thrust is what the held arm
    measures.
  - The crank metric is lower free than held at all 16 points, as predicted.
- **P6 refuted for take-off**: only 6 of 16 stay within 45 deg of pitch and
  climb 1 m (predicted at least 12). All 16 still climb, but 6 exceed the
  torque cap while pitching.

## Cruise

| arm | result |
|---|---|
| held, new rig | **19 of 26** meet the gate |
| x and z free, cam offset re-trimmed | **13 of 26** meet the gate; cheapest 28.3 W at crank 5.2 N m (8 m/s, 3.5 Hz, stroke 60); 24 of 26 trimmed; bob 52-95 mm |
| x, z and pitch free, no tail | **all 26 pitch beyond 45 deg within 4 beats** (91-360 deg); 15-74 deg in the first beat alone |
| held, built joint damping | 10 of 26 meet the gate; cheapest 33.8 W |

- **P4: two parts confirmed, one refuted.** 13 of 26 still meet the gate
  (predicted 8-18) and the cheapest costs 28.3 W (predicted 27-33). But the
  change in net force is not a uniform loss of 0.05-0.4 N: it runs from
  -1.46 to +1.21 N, because the re-trim moves each point.
- The lowest-power points lose their margin: the six cheapest free points
  read -0.07 to -0.25 N net.
- **P6 confirmed for cruise.**

## Pitch (P5) and joint damping (P7)

- The holder takes a pitching moment that peaks at 5.3-8.2 N m at take-off
  and 2.1-4.2 N m in cruise, with means of -0.11 to +0.73 N m. **P5 refuted
  narrowly**: I predicted peaks of 2-8 and means under 0.5.
- This is the cam's reaction: the feather axes lie along the span, so the
  torque that turns the wings turns the hull the other way in pitch. The
  hull's pitch inertia is about 0.005 kg m2.
- **P7 confirmed**: the built joint damping raises power by 31% at take-off
  and 27% in cruise (medians), and the crank metric by 17% and 12%.

## Reading

- Rival (a) holds for take-off in x and z: letting the body move costs no
  torque and it climbs.
- Rival (b) holds for cruise: the bob takes the margin from the cheapest
  points, and half of pass 1i's cruise points survive.
- **Rival (c) is the result.** With pitch free and no tail, no cruise point
  flies and most take-off points swing beyond 45 deg. The cam drive cannot be
  called a flying machine until something takes 2-8 N m of oscillating pitch
  moment.
- **The margin does not survive joint friction at the model's generic
  value**: 2 of 16 take-off points. That figure is TYPED, not measured; it
  is now the most important unmeasured number in the drive.
- Decision: K1 with a cam stays the reference, **conditionally**. Before any
  further gate is claimed it needs:
  - a tail and pitch inertia sized against the measured moment, and a
    stabiliser, then the pitch-free arm again;
  - a measured or sourced joint and cam friction;
  - a feather drive whose reaction does not go into the hull's pitch (for
    example a hinge placed where the feather's torque is least), as an
    alternative to a larger tail.

## Not covered

Tail, stabiliser, roll and yaw; ground contact; a real cam profile; the
motor's torque-speed line; water.
