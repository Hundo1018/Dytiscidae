# L0: the drive mechanism, opened (2026-11-15)

The user, on reading L0 pass 1: the mechanism need not be a crank; try more
effective schemes or design one. This is the paper pass over that. Arithmetic:
`mechanism.py`. Precedents are from A11, A13 and A14 as labelled there; what
is called a design below is the lead's proposal and has been built by no one.

## What the mechanism has to do (from L0 pass 1)

| | air | water | land |
|---|---|---|---|
| stroke | about 85 deg | small, or slow | 30-60 deg, planted on the downstroke |
| frequency | 4-5.5 Hz | 0.2-1 Hz | 1-1.5 Hz |
| shoulder torque | 9-12 N m at take-off | limited by structure, not by need | under 1 N m to lift the body |
| power | 120-150 W mechanical for seconds, a third of it in cruise | under 10 W | small |
| also | stroke biased upward for a water take-off | wet for 15 minutes | stand still; left and right different |

A fixed crank does the first column and nothing else well: one stroke, no
difference between sides, no standing still.

## Candidates

| | scheme | precedent | what it gains | what it costs | verdict |
|---|---|---|---|---|---|
| K0 | one motor, fixed crank | E-Flap, Phoenix, 0.5-1.2 kg | lightest known (14% of mass), efficient: the motor never reverses | everything outside the air column | the baseline to beat |
| K1 | **one motor, crank, and a rocker whose throw is set per side by a self-locking screw** | a cable-driven variable-amplitude ornithopter flew at 6 Hz (Cal Poly thesis, A14 D; mass not read) | the stroke becomes a gear: see below. Per-side throw steers in all three media; zero throw parks the wing | two screw drives and sliders, about 40 g (TYPED); a slow change of throw (seconds), so it is trim and mode, not beat-by-beat control | **carried to L1** |
| K2 | one position-controlled motor per wing, oscillating, through a cable capstan, with a spring across the shoulder | Robo Raven is this with servos (0.3-0.4 kg, 11-83 times less power per kilogram); a resonant drive is not found above 12 g, and the spring's benefit is disputed in the literature (A14 D) | any stroke, any bias, any frequency, each side its own, beat by beat | the motor reverses twice a beat: peak about twice the mean power, so two motors each sized like K0's one; about 200 g in all (TYPED, from 50 g motor + 25 g capstan + 15 g driver + 15 g spring a side), a third of the machine against K0's seventh; the spring fights the motor in water and on land | carried to L1 **only as the comparison**: it is the scheme with the most authority, so it bounds what authority is worth |
| K3 | one motor and fixed crank per side | none found | sides differ by frequency and phase | two gearboxes; still one stroke; flight needs the two sides phase-locked by control | dropped: K1 gives the same with one motor |
| K4 | fixed crank for flight, clutch, and servos for water and land | none found (A14 D) | each regime gets its own actuator | servos that are waterproof give 0.6-1.9 W (A13); a clutch at 10 N m | dropped |
| K5 | fixed crank, wing compliance cuts the stroke in water | FAAV, under 300 g, 5 Hz in both media | no mechanism at all | the compliance that sheds a factor of hundreds in water is also the wing's stiffness in air; nothing for land | kept as a property of the wing in K1, not as the drive |

## K1 in numbers (`mechanism.py`)

The coupler meets the rocker at a distance L from the shoulder; the stroke is
2 asin(r / L) and the crank sees the shoulder torque times r / L. Moving the
joint outward shortens the stroke and lengthens the lever together.

In water, with the crank torque held at its take-off value and the shoulder
allowed 1.5 times its take-off torque (TYPED):

| stroke | water frequency | crank torque | swim speed (St 0.3, TYPED) | limited by |
|---|---|---|---|---|
| 85 deg | 0.15 Hz | 1.00 of take-off | 0.32 m/s | the motor |
| 40 deg | 0.38 Hz | 0.76 | 0.42 m/s | the structure |
| 15 deg | 1.01 Hz | 0.29 | 0.43 m/s | the structure |

- **It does not buy thrust**: the shoulder's strength caps the force at any
  stroke, and the estimated swim speed is about 0.4 m/s whichever stroke is
  used. Saying otherwise would be wrong.
- **It buys a place for the motor to work**: at 15 deg the motor turns seven
  times faster than at full stroke for a third of the torque, instead of
  crawling at 0.15 Hz at full torque. And the same impulse arrives in seven
  small strokes instead of one large one, which is what a free body thrown
  about by its own wings needs (F3: 2.8-19 body weights on a held body).
- Swimming costs under 10 W mechanical, as A13 expected.

On land, as a crutch (mudskipper, hatchling turtle; A11 C): both wings plant
on a thumb and push down and back along a stroke plane tilted 45 deg (TYPED),
the body lifts and is set down, the wings recover in the air. Lifting the
body needs 0.88 N m, 8% of the take-off torque. Stride 6-11 cm, so 0.06-0.17
m/s at 1-1.5 Hz if the thumb does not slip. It is slow, the hull touches down
every stride, and a difference in throw between the sides turns it. A11
records that fins alone slid back on a 20 deg slope without a tail's push.

**The tilted stroke plane is one choice that serves three rows**: it is the
attitude RoboFalcon lifted off from, it gives the crutch its backward push,
and it lifts the wing tips off the water for the take-off run.

## What this does not settle

- Stroke bias (upward, for the water take-off) is a second adjustment: a
  coupler of variable length, or a fixed bias chosen in L1. Not designed.
- With one flapping degree of freedom per wing, thrust in water comes from
  passive twist, as in air. A13 notes the present model's hinge is too
  heavily damped to show this; the compliance model of the plan comes first.
- Every mass above is typed. A screw that moves a joint carrying 100-400 N,
  printed, wet, 3,600 cycles a flight: bench.
- The 1.5 shoulder cap and the Strouhal number are typed.
- Whether a beat-by-beat authority (K2) is needed in flight at all is the
  passive-stability gate of the plan; K1 assumes a tail does it.

## For L1

Fixture 1 is K1 on the L0 reference machine: 0.60 kg, one pair, 1.4 m, tail,
stroke plane tilted, throw 15-85 deg per side, crank limited to its take-off
torque with a torque-speed line. Fixture 2 is K2 at its typed mass. K0 is the
control: K1 with the throw locked at 85 deg. A scheme stays only if it beats
K0 on a gate K0 fails.

## Correction, same day (L1 pass 1a, `../RESULT_L1a.md`)

The water table above is wrong. On the held rig the three points of
`mechanism.py` ask 25-230 N m at the shoulder against the 15.8 N m cap, and
the small fast strokes are the worst, not the best: the rule used here left
out added mass and the wing's own turning in water. The claim "it buys a
place for the motor to work" is withdrawn until re-derived with added mass.
What stands: the per-side throw for steering and parking, the crutch
arithmetic, and the tilted stroke plane.
