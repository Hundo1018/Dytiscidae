"""A ladder of analytic benchmarks, from a bare rigid body up to the controller.

The question this package answers is not "do I trust the simulator". It is
**at which layer of modelling does the simulator start departing from an answer
that can be written down, and by how much.**

Each layer adds exactly one piece of physics to the one below it and carries a
reference that does not come from the simulator:

    1  rigid body           a = F/m, alpha = I^-1 tau, ballistic flight
    2  added mass           a = F/(m + m_a) with m_a a tensor
    3  drag                 terminal velocity, and the exact tanh approach
    4  buoyancy             rho g V, the equilibrium draft, the heave period
    5  jet propulsion       momentum flux in, and the work that must pay for it
    6  articulated body     internal torques conserve total momentum
    7  fluid surrogate      the assembled force against its own coefficients
    8  controller           the commanded twist against the delivered one

Where a closed form exists it is used. Where one does not, the reference is a
tight Runge-Kutta integration of the *same force law*, which separates an error
in the force law from an error in the integration -- two failures that look
identical in a trajectory and need completely different fixes.

A layer that departs is not a bug report by itself. It is a statement about
where the model's fidelity ends, which is the thing a reader of a result needs
and which no amount of passing unit tests provides.

Run:  PYTHONPATH=. python benchmarks/run.py
"""
