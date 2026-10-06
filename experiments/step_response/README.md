# step_response: does ReCo's closed-loop response model add anything to `_turn_authority`?

papers-2610, `docs/papers/2610.01612.md` §3.2. ReCo fits `v' = (k*c - v)/tau` per command
channel. Decision it feeds: publish a response-model measurement or descriptor from Tier-1, or drop it.

**Verdict: drop it.** On all 200 arch48 elites the fitted dR2 (what the command explains beyond
momentum) is ~0.002, indistinguishable in scale from the shuffled-command control, uncorrelated with
competence in every medium, and adds nothing to `_turn_authority` (which is itself uncorrelated with
competence). The gain `k` is unbounded and `tau` is mostly the control interval's own scale.

## Method

- Model, per angular channel of one segment: `r[t+1] = a*r[t] + b*c[t] + e` (least squares, no
  intercept); `tau = -dt/ln(a)` and `k = b/(1-a)` for 0<a<1, else NaN. `R2` centred on `mean(r[1:])`;
  `dR2 = R2_full - R2_AR` where AR is `r[t+1] = a*r[t]`.
- `c` = commanded body angular rate (`basis.twist_of(coeffs)[3:]`), `r` = `env.body_twist()[3:]`, one
  pair per control decision, dt = 10 * 0.004 = **0.04 s** (read from `rollout`: `control_hz=25`, `timestep=0.004`).
- Segment summary: **median across the 3 channels** of dR2 (primary); `dr2_max` also stored. `k`, `tau`,
  `a`, `b` are those of the best-explained channel.
- **Path: the offline path** (`TriphibianEnv.rollout`, with `evaluate_tier1`'s per-medium reset /
  scatter / task / `air_launch_height`, `identify_axes=False`, stored bases, own policy + the network that
  scored the elite's generation). **It is not the search's path** (`batchroll.rollout_batch`). Capture
  wraps `env._score_segment` on the instance; no library code was changed. The offline competence
  reproduces the recorded one within 0.005 for air 200/200, water 186/200, land 169/200 (largest gap
  0.28 on land: the known offline/batched difference). Correlations are also reported on the reproduced
  subset (below, `analysis_arch48.txt`).
- Segments with `bad_qacc > 0` or not survived are excluded from fits (1 air, 2 water, 5 land), since a MuJoCo reset corrupts the response record.
- Still-machine rule: no commands means NaN, not 0 (7-12 segments per medium have no fit). Shuffled-command
  control: 50 permutations in time of the 3-channel command sequence per segment, refit, dR2 recorded.
- Sample: **all 200 elites** of `runs/arch48` (not a stratified n=40-60: it cost 6 s/elite, 1165 s total,
  and the full archive has no sampling bias). 3 media each.

## Run

```bash
systemd-run --user --unit sr-arch48 -p MemoryMax=3000M --same-dir bash -c \
  'MUJOCO_GL=disable PYTHONPATH=. .venv/bin/python -u experiments/step_response/run.py capture \
     --run runs/arch48 --n 0 --out experiments/_cache/step_response/captures_arch48.npz > ~/.cache/dyt/sr_arch48.log 2>&1'
PYTHONPATH=. .venv/bin/python experiments/step_response/run.py analyze \
  --captures experiments/_cache/step_response/captures_arch48.npz --out experiments/step_response/results_arch48.json
```

Capture wall 1165 s (n=200 elites, 0 errors); analysis seconds. Outputs: `captures_arch48.npz` (raw
command/response sequences + elite metadata), `results_arch48.json` (every row), `analysis_arch48.txt`.

## Results (Spearman rho, p in brackets; n = segments with a fit)

| medium | n fit/200 | dR2 vs competence | dR2-minus-shuffled vs competence | turn_authority vs competence | dR2 vs turn_authority | partial: dR2 given TA vs competence |
|---|---|---|---|---|---|---|
| air   | 193 | -0.09 (0.21) | -0.05 (0.46) | +0.05 (0.51) | -0.04 (0.62) | -0.09 (0.22) |
| water | 190 | -0.02 (0.79) | -0.04 (0.57) | -0.07 (0.30) | -0.12 (0.11) | -0.03 (0.70) |
| land  | 188 | -0.03 (0.72) | +0.08 (0.28) | +0.03 (0.65) | +0.10 (0.16) | -0.03 (0.69) |

Air competence is 0 for 199 of 200 arch48 elites (one is 0.06): the air column has nothing to
predict. Water has 94 segments >= 0.05, land 60. Restricted to competence >= 0.05: water dR2 -0.14 (0.20),
TA +0.04, partial -0.14; land dR2 +0.03 (0.84), TA -0.07, partial +0.06. Restricted to reproduced
segments: water dR2 -0.01, TA -0.05; land dR2 -0.04, TA +0.01. `dr2_max`: -0.12 / +0.07 / -0.09.
`k` vs competence: +0.01 / -0.01 / -0.10; `tau`: +0.17 (p=0.02, air, nothing to predict) / -0.02 / +0.11.
Leave-one-out R2 of rank-OLS on competence: TA -0.02, dR2 -0.01..-0.02, TA+dR2 -0.02..-0.03 (all
negative, i.e. worse than the mean) in every medium.

### Real dR2 vs shuffled-command control

| medium | median real dR2 | median shuffled dR2 | real 95th pct | segments beating shuffle at p<0.05 (5% expected) |
|---|---|---|---|---|
| air   | 0.0022 | 0.0017 | 0.020 | 44/193 (23%) |
| water | 0.0016 | 0.0011 | 0.055 | 61/190 (32%) |
| land  | 0.0018 | 0.0020 | 0.021 | 43/188 (23%) |

The command carries real signal for a minority of segments (more than the 5% chance rate), but the
typical dR2 is 0.2%, equal to the shuffled one, and where it exists it is unrelated to competence.

### Fitted values

`a` in (0,1) for 89-90% of fits. `tau` median 0.10 s (air), 0.28 s (water), 0.065 s (land), range 0.007-67 s
(i.e. 0.2 to 1700 control intervals). `k` spans -12944..19943 (air), -5075..43418 (water), -4454..11091
(land): with dR2 ~ 0.002 `b` is poorly determined, so `k = b/(1-a)` explodes when `a` is near 1.

### Tier-2

Only 2 of 200 final-archive elites join to a `promote` event (same island and cell, event gen >= born_at;
the archive holds the final occupant, and promoted elites were mostly displaced). Not testable; rows are in
`results_arch48.json` under `tier2_joined`.

## Caveats

- The command is the policy's output, which sees the angular rate as an observation, so `b` mixes the
  plant's gain with the controller's feedback. This is closed-loop identification without an excitation
  signal; ReCo drives the plant with designed excitation.
- The realised rate also carries the flapping oscillation, which a first-order lag in the commanded rate
  cannot represent. One decision of lag only; no longer-lag variants were tried.
- Competence ties at 0 are heavy in air and land; Spearman handles them but power is low there.
