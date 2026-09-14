# A pulsed jet produces thrust and nothing pays for the water

```
PYTHONPATH=. python experiments/jet_energy/run.py
```

## The question

`JetSet.apply` produces thrust `rho Q^2 / A` and injects it into
`data.xfrc_applied` as a pure force at the body's centre of mass.
`JetSet.actuator_work` returns

```python
return 0.0  # accounted through the driving actuator's torque
```

and `TriphibianEnv.step` charges the battery from
`|data.actuator_force| * |data.actuator_velocity|`. So the claim is that
expelling the water shows up as extra torque on the joint driving the bell.
`JetSet.apply` writes no joint torque, no `qfrc_applied` and no body torque.

## What the work should be

The jet leaves at `v_e = Q/A`. The cavity pressure needed to drive it is the
stagnation pressure `p = (1/2) rho v_e^2`, so the mechanical power the muscle
supplies is

    P_jet = p * Q = (1/2) rho Q^3 / A^2 = (1/2) * thrust * v_e

— the ideal minimum, with no nozzle loss, no leakage and no refill cost. The
`JetSet` docstring states this same relation in its `actuator_work` docstring
and then returns zero.

## What was measured

One free body carrying one 4 L bell with a 12 cm^2 orifice, submerged, driven
by a position servo through a 1.5 Hz sinusoid over the joint's full range, for
4 s.

| | |
|---|---|
| peak thrust | 1 649 N |
| peak jet velocity | 36.6 m/s |
| thrust impulse | 19.11 N.s |
| **ideal pumping work** | **141.61 J** |
| actuator mechanical work charged, jet ON | 114.1222 J |
| actuator mechanical work charged, jet OFF | 114.1111 J |
| **difference attributable to the jet** | **+0.0111 J**, i.e. **+0.008%** of 141.61 J |

**100.0% of the pumping work is uncharged.** The 114 J the actuator does pay is
the cost of swinging the bell's own inertia and is identical whether the jet is
firing or not. The comment's claim is false, and it is false in the direction
that produces free thrust in water. Recorded as **J-01**.

H1 — "MuJoCo's constraint solver reflects the thrust back into the joint,
because the force acts at the body CoM which is off the joint axis" — is
rejected by the jet-on/jet-off comparison. Whatever moment that lever arm
produces is a different quantity from the pressure the pump works against, of a
different size, and of either sign.

## A second finding from the same run

Peak thrust is 1 649 N on an 8.6 kg machine: **20 g of acceleration** from one
bell. `thrust = rho Q^2 / A` takes `Q` straight from the joint rate:

```python
dv_dt = -self.volume * self.stroke_fraction * (omega / span)
thrust_mag = coeff * rho * q * np.abs(q) / np.maximum(self.orifice_area, 1e-6)
```

so thrust goes as the **square of the joint rate**, with no upper bound and no
limiter. `FluidSolver` clamps its own forces at 60x vehicle weight and raises
`diag.clamped` so the scorer knows the run left the model's valid domain;
`JetSet` has no equivalent, so a fast joint produces an arbitrarily large force
with nothing flagged. Recorded as **J-02**.

## What would fix each

**J-01.** Apply the reaction as a torque on the driving joint:
`tau_reaction = p * dV/dtheta = (1/2) rho (Q/A)^2 * V0 * stroke_fraction / span`,
opposing the contraction. That makes `data.actuator_force` carry it, so the
existing battery accounting picks it up with no change, and it makes the
orifice-area trade real — the `jet.py` module docstring already argues that a
small orifice "costs more pressure and therefore more actuator work", which is
the term this would add.

**J-02.** Either cap the jet velocity at something a muscle can produce, or
route the jet force through the same limiter and `clamped` flag `FluidSolver`
uses, so an out-of-domain result is marked rather than scored.

Both change what a medusa scores, so they are measurements to run rather than
patches to apply.
