# S0b, design note: the first machine starts with every freedom, not the fewest

2026-11-19, the lead. Not a pre-registration. It states the machine and the
probes that come before one. The user, 2026-11-19: "設計可以大膽一點
盡力找到第一個機構設計".

## Why S0's ladder was the wrong side to start from

`PREREG_S0_continuation.md` (dd89844) starts from C1 and adds one freedom per
measured failure, with 24 tuning rollouts each. Its own prediction S7 (0.55)
is that it stops at a water gate. The user's method is the other direction:
a machine that surely does all three media first, the limits afterwards.
S0's ladder is held after its A0 read; its A0 numbers are kept as the
reference for what C1 does on ideal actuators.

## The idea: one thrust-vector mode in every medium

The only machine in this repository that has done every part of the mission
is the hand-flown quad B1 (`experiments/triphibian_diagnosis/existence`). It
does one thing everywhere: a thrust vector it can point, plus attitude
control on three axes. It hovers, hover-taxis on land, swims by pointing the
thrust, enters by descending, and leaves the water level and staged.

The flapping machine that has come closest is the same thing in one medium:
V+ is an axial thrust with attitude control, nose up (M1b: 13/14, 12/12).
Every measured failure is a place where the flapper is not that machine:

| failure | what the thrust-vector machine does instead |
|---|---|
| swim: no commanded progress (M2b WT) | points the body, thrust along the axis, at the stroke frequency water asks for |
| surface turn: nose-stood, inverted (M2b ST) | attitude is controlled on three axes under water, as in hover; ballast low |
| dive: 1.85x buoyant (M2) | b = 1.00, and swims down |
| exit: the vault never clears under the default mode (M2b) | swims up to speed and breaches, then the air stroke |
| after the exit: falls in 3-8 s on saturated motors (M2b XS) | ideal actuators, and attitude authority on more than one pair |
| entries: 9-14 m long, hard on land (M4) | stops in hover over the place and descends |

## The machine, T4

Flapping only. No typed force. No material, construction or drive is chosen.

- **Four wings in two pairs** (fore and hind, `body.flapper(hind_u=...)`),
  the W6 wing on each. Two pairs give pitch authority by fore-aft thrust
  difference and roll by left-right, in every medium, without leaning on
  the tail.
- **Three axes per shoulder**: stroke, fore-aft sweep (M3b's hinge), feather,
  plus the per-wing strip-twist stand-in. With sweep as a stroke the wing
  can beat in any plane, so the thrust vector can be turned without turning
  the hull.
- **Ideal actuators** on every hinge (`tau_s` 1e4, `omega_0` 1e7), every
  torque, speed and power recorded.
- **Hull**: ballast to b = 1.00 placed low (keel), so the hull rights itself
  in water. Variable ballast (±8% of displacement, 2%/s, by `body_mass` and
  `mj_setConst`) is allowed from the start.
- **Tail**: kept as C1's, so the frozen cruise law still applies.
- **Mass is whatever this comes to.** It is recorded; 0.6 kg is a limit for
  later.

## Laws, and a bolder budget

One law family, "axial thrust plus three-axis attitude", written once and
used in hover, under water (frequency scaled by the medium), and for the
breach. The frozen laws (V+, M4 L, M3b K) are tried first where they still
apply to T4.

Free parameters are tuned by an optimiser (CMA-ES or a plain random search
with restarts), not by hand: **up to 400 rollouts per manoeuvre**, on tuning
starts only. The guards are unchanged from S0 §4, and the verdicts are
unchanged: frozen gates, scored starts disjoint from tuning, do-nothing
arms, water in both entrainment modes with a split counted "not
established", the X2 flag, contact above 10 weights re-run at time constant
x2, one dt/2 re-run per scenario. An optimiser finds a model's blind
regions faster than a hand does, so the X2 flag and the both-modes rule
carry more weight here, not less.

## Probes before the pre-registration (to be disclosed in it)

Each is a few rollouts. They decide what the pre-registration freezes.

- **B0, build.** T4 builds on `mission.Machine` with twelve shoulder hinges
  on ideal motors; a figure from the built model; mass, inertia, buoyancy
  ratio before and after ballast.
- **B1, hover.** V+ on the fore pair with the hind pair in phase opposition,
  untuned: does it leave the ground, and what are the attitude moments
  available per axis (a held-body force map, as `hover_map.py`)?
- **B2, water thrust.** Held at 3 m depth, both modes: axial force against
  stroke frequency 0.5-6 Hz for the hover kinematics. Is there a frequency
  with thrust above the drag at 0.3 m/s and moments on three axes?
- **B3, breach.** Free, b = 1.00, from 3 m, nose up, B2's best frequency:
  the speed at the surface and the height the centre of mass reaches with
  the wings stopped at the surface, both modes. Does it clear by more than
  one wing length?
- **B4, still machine.** B2 and B3 with the actuators held: thrust 0, no
  breach.

If B2 or B3 fails in the default mode with every freedom above, that is the
first finding about the mechanism, and it goes to the user before anything
else is built.
