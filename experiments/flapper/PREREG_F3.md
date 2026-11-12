# F3 — the same wing under water: which strokes swim, and what they ask

Frozen 2026-11-12 before any rollout, commit 65b0f0a. No learning. Run by
the lead while F2 (land) is measured by an agent; `body.py` and `rig.py` are
not edited here.

## 0 Question and decision

The one-panel 0.25 m wing of the air measurements, body level and submerged,
carried at 0.25, 0.5 and 1.0 m/s: which strokes make at least as much thrust
as the body's drag, at what hydrodynamic power and peak shoulder torque, and
how hard do they shake the body? The decision is what the water asks of the
design: a frequency and amplitude band for the swimming stroke, whether the
wing has to shed area or load there (a fold, a wrist, a softer drive), and
whether the body needs something against the stroke's reaction (a tail, a
second pair in antiphase).

## 1 Rivals

- **Easy and violent**: water is 800 times denser, so a small slow stroke is
  thrust enough and anything like the air stroke is tens of newtons on a
  0.75 N body.
- **No thrust without feathering**: a plate that only heaves pushes equally
  both ways.
- **Instrument**: the rig's force is the solver's fluid wrench summed over
  bodies, including buoyancy; read against the still wing at each speed.

## 2 Method

A copy of `rig.run`'s loop in `water_map.py` that also keeps the per-step
force and the shoulder torques. Body level at 3 m depth, speed U along its
axis, stroke `(Phi/2) sin(wt)`, feather `-Psi cos(wt)`, no offset. Frequency
0.5, 1, 2, 3 Hz; stroke 30, 60, 100 deg; feather amplitude 0, 20, 40, 55
deg; U 0.25, 0.5, 1.0 m/s; and the wing held still at each U (drag and
buoyancy). Five beats, the last three read; time step 1 ms.

Read: mean force along the path (thrust minus drag); the vertical force's
swing over a beat against the weight; mean hydrodynamic power; the largest
fluid torque about a shoulder's stroke axis and about its feather axis.
**Swims**: mean force along the path >= 0.

## 3 Predictions (written before running)

1. A stroke that swims exists at each speed at 2 Hz or less.
2. At 0.5 m/s the cheapest stroke that swims takes under 2 W and asks a
   shoulder for under 0.42 N m (what the hover stroke asks by inertia).
3. With no feathering (amplitude 0) nothing swims at 1.0 m/s.
4. At the cheapest stroke that swims at 1.0 m/s the vertical force swings by
   more than three body weights each way.
5. The cheapest strokes have a Strouhal number (frequency times tip travel
   over speed) between 0.2 and 0.6.
6. The air stroke carried into water unchanged (8 Hz is off the grid; read
   at 3 Hz, 100 deg, feather 20) asks a shoulder for more than ten times the
   0.42 N m of hover.
