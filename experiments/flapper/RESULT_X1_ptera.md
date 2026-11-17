# Cross-check X1: the cruise against an outside solver

2026-11-17. PteraSoftware 5.1.0 (unsteady vortex-lattice, MIT), installed
with the user's approval in its own environment outside the repository
(`../.venv-ptera`, 1.3 GB). Set up and run by a subagent to a written brief;
its script, 159 rows and notes are in `ptera/`. Not pre-registered as a pass:
the brief fixed the cases and forbade tuning, and gave the strip solver's
numbers as context only. I read `ptera/read.txt` and the trims below; I did
not re-derive its power formula.

What it is and is not: inviscid, attached flow, a thin surface on Goe 417a's
mean line (5.83% camber), prescribed wake (a free wake moved lift 0.3% and
power 0.5%), no hull, no profile drag, no stall. So its thrust is an upper
bound and it cannot say a wing stalls.

## Controls

- Static wing, roots sealed: lift slope 0.956 of Helmbold's, induced drag
  0.96-1.03 of elliptic. With the 8 cm gap between the roots left open (no
  hull): slope 0.784 of Helmbold's, induced drag 1.5-1.7 of elliptic. **The
  gap is the largest effect in the comparison.** The strip solver treats the
  pair as one wing of the full span, i.e. as sealed.
- Step halved, panels doubled, five cycles, free wake: lift and power within
  2%, thrust within 0.011 N.
- Twist sign read off the built mesh: tip leading-edge down at mid-downstroke.
  The opposite sign costs 16.4 W for the same lift.

## Trimmed to lift = weight

| V | f, stroke, feather | strip solver: net thrust, power | outside: inviscid thrust, power (gap open) | (gap sealed) |
|---|---|---|---|---|
| 8 | 2.5, 40, tip 15 | +0.045 N, 7.24 W | +0.177 N, 4.37 W | +0.416 N, 5.80 W |
| 8 | 2.0, 50, tip 15 | +0.060, 7.34 | +0.177, 4.36 | |
| 8 | 3.0, 40, tip 20 | +0.084, 7.50 | +0.280, 5.55 | |
| 8 | 3.5, 50, rigid 15 | +0.127, 21.5 | +0.925, 14.46 | |
| 6 | 2.0, 60, tip 15 | -0.698, 7.46 | +0.363, 7.04 | |
| 6 | 2.5, 50, tip 15 | -0.683, 8.60 | +0.443, 7.78 | +0.816, 9.99 |
| 6 | 3.0, 40, tip 15 | -0.655, 7.67 | +0.352, 7.01 | |
| 6 | 2.5, 60, tip 20 | -0.525, 12.39 | +0.786, 10.74 | |
| 6 | 3.5, 60, tip 20 | -0.174, 34.32 | +2.226, 26.06 | |

Root incidence at trim: 2.4 deg nose-up at 8 m/s and 7.9-8.4 at 6 m/s
(outside, gap open; 0.9 and 5.0 sealed); the strip solver's cam offset is
-3.0 and -10.9 to -11.8, so its negative offset is nose-up and the two agree
within 2-4 deg.

## Reading

- **Power agrees.** At 6 m/s the strip solver is 6-32% above the inviscid
  figure at every point; at 8 m/s it is 1.4-3 W above, about what the profile
  drag of the section costs. The ranking is the same: the tip feathering is
  cheaper than the rigid feather in both, by a similar factor.
- **The 8 m/s margin is inside the root gap's uncertainty.** The strip
  solver's +0.045 N net is the sealed case less about 0.37 N of profile and
  hull drag. With the gap open the inviscid thrust is 0.24 N less, which
  would leave nothing. The real machine's hull fills the gap in part. So "8
  m/s sustains for 7.2 W" holds only as far as the hull seals the roots, and
  neither solver models that.
- **"Nothing sustains at 6 m/s" is not established.** The two solvers
  differ by 1.0-1.3 N of thrust there at the same lift and nearly the same
  power. The outside solver has no stall; the strip solver has the section's
  measured drag only to 11.65 deg (Cd 0.044 at CL 1.37) and beyond that
  charges a fully separated strip's drag (lift times tan alpha, six times
  as much at 12 deg), although the section's own lift run does not stall
  until 13.5-14.5 deg. At 6 m/s the root sits at 8-11 deg and the tip
  feathers 15 either way, so much of the beat is in that band. Profile drag
  at the measured Cd would be about 0.3 N against 0.35-0.44 N of inviscid
  thrust: marginal, not a 0.7 N shortfall. The pass-W5 extension of the
  section is the weak link and is pessimistic there.
- Also seen: the section's lift runs carry its measured pitching moment
  (-0.05 to -0.085 at 8-13 deg), about half the arc's theoretical -0.18 for
  5.8% that pass W7's twist torque used a stand-in for (W7 used -0.126 for
  4%). The hinge torque of W7 is high by that much.

## Not verified

The subagent's power formula against an energy balance; unsteady loads
against Theodorsen; the twist is piecewise linear over six segments.
