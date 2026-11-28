# What "a triphibian flapper is found" means here

2026-11-18. The user set the goal "keep pushing until a triphibian flapping
machine is found". This file fixes, before any pass toward it, what will
count as found, so that the condition can be checked and cannot drift
with the results. It is built only from the user's own decisions
(`PLAN.md`, "The user's decisions (2026-11-13)" and later) and the
project's rules; where it reads one of them, the reading is stated.

## Found, in simulation

One simulated machine, at the L0 mass (0.6 kg all-up unless an L0 pass
moves it), with flapping wings as its only propulsor in air and water
(no rotor, no jet), that passes, each in its own free-body run:

1. the six scenarios of decision 1, each on its own charge: water; land;
   air; land<->air; water<->air; land<->air<->water, a mixed scenario 15
   minutes in total (the energy column read from the run, not assumed);
2. with every crossing between media under its own power, except the
   first take-off of a scenario, which may be assisted (decision 2);
3. with actuators limited by a motor's torque-speed curve (the witness
   motor of D1 or a named better one), never ideal servos;
4. with the control laws reading only modelled sensors (attitude, rates,
   depth or pressure, contact), not the true state;
5. with a do-nothing arm for every gate (actuators held still, and the
   machine dropped, floated or set down the same way), which must fail
   each gate the machine passes (`CLAUDE.md`, "Designing a measurement");
6. on more than one seed or entry state per gate, pre-registered, with a
   figure from the built model opened and checked at each stage.

The mechanisms of the D series (stroke linkages, feather crank, tip
tube) and torque-limited hinges are interchangeable candidates inside
this machine. **Which mechanism, which material and which construction
are not part of "found"** (the user, 2026-11-17: nothing of that kind is
fixed until several candidates are simulated); a machine found with one
candidate is reported beside the others, with no pick.

## What "found" does not claim

- Hardware. Nothing is bought or built: that needs the user, and the
  bench (PLAN L2) is the only thing that can correct the model.
- Accuracy where the model is known blind. The strip solver has no
  dynamic stall (X2: forces off by 1.5-3.5 times at large feather and low
  speed, which is where take-off and water exit happen); no wake between
  wings; the water's added mass and entrainment follow pass S1. Every
  verdict that rests on such a regime says so in its result, in the
  outcome paragraph, not a footnote.
- That the search found it. An evolutionary run (the arch series) is not
  the route here: PLAN puts the search inside L1 only after the bench has
  corrected the model once.

## Not found, also a result

If a requirement is shown not to close at this mass or with flapping
alone (an L0 row, or a pass whose do-something arm cannot pass a gate at
any point of a pre-registered sweep), that is reported to the user as the
outcome it is, with the measurement, and work continues on the rows that
do not depend on it. Loosening a gate after seeing data is not a way to
"find" the machine.

## Order of passes (by what would end the concept first)

- M0: one harness for a free six-axis body with ground, water surface and
  air, carrying the 0.6 kg reference machine (BODY_S, the W6 wing, the
  1m tail), with torque-limited hinges. Probed, not assumed.
- M1: standing take-off from land, free body (L0 row 4).
- M2: free swim with the same wings, and water to air (L0 rows 2, 5, 9).
- M3: land: standing still, walking, steering (L0 row 3).
- M4: entries: air to water, air to land (L0 row 6).
- M5: the six scenarios end to end, modelled sensors, do-nothing arms.
- Then the D-series mechanisms in place of the torque-limited hinges, and
  the packaging (every mechanism so far sits outside a 40 mm hull).

M1-M3 do not depend on each other once M0 exists and may run in
parallel. Each is pre-registered and frozen before it runs.

## Known before starting

- The panel's mass box sits about a quarter chord ahead of the fluid
  surface's mid-chord (D3). It changes a free body's pitch inertia and
  trim; it is fixed or stated before M1.
- Water code after the S1 erratum is the one to use; the torque figures
  of L1b, L1c, L1d and L1l are overstated by 0.36-0.96.
