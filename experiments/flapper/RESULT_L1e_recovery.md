# L1 pass 1e result: a recovery half-stroke in a low-drag shape (2026-11-15)

Pre-registration: `PREREG_L1e_recovery.md` (commit 6ac3f35; six probe points
seen before freezing, disclosed there). Data: `results/L1e_recovery.jsonl`,
258 points, all finite, one pass. Held body, water, 0.3 m/s; chord and area
of the fluid strips scaled to 0.4 on one half of the beat, blended over the
reversal.

## Best net forward force under the 15.8 N m shoulder cap

| wing (length) | fixed | narrow, positive half | narrow, negative half | narrow throughout |
|---|---|---|---|---|
| 0.4 | 5.92 N (0.52 N per N m) | 3.86 N | **18.19 N** at 1.0 Hz, 11.9 N m, 24 W (1.53 N per N m) | 9.51 N |
| 0.6 | 3.28 N | 2.00 N | **11.77 N** at 0.5 Hz, 12.0 N m, 8.4 W | 8.52 N |
| gate points (twice the drag, both caps) | 4 / 1 | 3 / 0 | 9 / 5 | 9 / 7 |

## Against the frozen text

- P1 (the better per-half arm beats fixed by more than 20%): 3.07 and 3.59
  times. **Held.**
- P2 (beats narrow-throughout by more than 20%): 1.91 and 1.38 times.
  **Held.**
- P3 (negative mean power at fewer than 10% of per-half points, else every
  per-half gain is unproven): **13 of 128, 10.2%. Failed, narrowly.** By the
  frozen rule the gains above are **reported as unproven**: the model may be
  creating energy when it changes the strips' added mass in time. All 13 are
  in the negative-half arm (7 and 6 of 32); the best points quoted above have
  positive power (24.4 and 8.4 W), which does not clear them, since the same
  mechanism can add energy at a point whose net power stays positive.
- Rival (a) (idling a half only costs thrust) is ruled out for the negative
  half and stands for the positive half: which half is the recovery depends
  on the feather's phase, not on up or down.

## Reading

If real, narrowing the wing on its recovery half triples the force a fixed
wing gives under the same torque and doubles a wing kept narrow throughout:
the user's point, in the model's numbers. It is not yet established, because
the one check written for this artefact failed. What would establish it: the
same comparison with the area change built into the wing's geometry (two
wing panels, one that feathers edge-on on the recovery like a one-way flap),
so no strip changes size in time; or a bench paddle in a tub with a flap and
without.

## Next

A one-way flap as a built panel in the model: a rear panel hinged to the
front one, free to swing edge-on in one direction and stopped in the other
(the diving beetle's fringe, as a part). It needs no actuator and changes no
strip in time, so it answers the artefact question and is a design candidate
at once.
