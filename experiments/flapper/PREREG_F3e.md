# F3e — does the two-pair machine go to a depth when told, and stay there?

Frozen 2026-11-12 before any rollout, commit 9682696. No learning, no
optimiser. Written by the lead; the agent that runs it may not edit this
file.

## 0 Where this stands

F3d: on a neutral hull (`FLAPPER_BODY_B='{"hull_radius": 0.0209}'`) and with
the water hold paired (`DYTISCIDAE_DEPTH_PAIR=1`), the diagonal stroke swims
on 16 of 16 fresh seeds and both null arms read under 1e-16. It answers the
heading and not the depth: its pair hold is 0.00-0.01. The user asked for
depth control next.

What the body allows, from F3 and from the geometry: with four wings spread
the body cannot be moved up or down broadside at any useful speed (the
wings' added mass normal to themselves is about 3 kg on a 0.105 kg body), so
depth has to be changed by pointing the body and swimming along it, wings
edge-on to the travel, and then levelling and stopping. Variable buoyancy
(the genome's BALLAST part) is the other mechanism and is not used here.

## 1 Question and decision

With the shoulders the machine has, the diagonal stroke, and a hand law that
reads what F3b's law read: does it make at least half of a commanded
difference in depth inside the water segment, and does it reach and keep a
commanded depth when given time?

Decision. Both bars below met: the water is done for stage 1 (it swims
where it is told, at the depth it is told). The step bar met and the segment
bar not: the machine controls depth and the 8 s segment is too short for
it; reported as that, with the speed that would be needed. Neither: depth by
swimming is not enough on this body, and a ballast is measured next under a
new frozen file.

## 2 Method

Body B with the neutral hull, ideal servos, 1 ms step, `DYTISCIDAE_DEPTH_PAIR=1`
throughout. Law and stroke tuned by hand on seeds 100-103 only (frequency up
to 2 Hz), every tuning rollout appended to `trials_F3e.jsonl`. `swim.py` is
extended to keep `pair_depth_response`, `pair_second`, `pair_hold_cmd` and
`pair_hold_depth` in its rows; nothing under `dytiscidae/` changes.

- **Segment bar**, seeds 40-55 (16, never used), unmodified water score
  through `evaluate.run_segment`: score 0.15 or more on at least 14 (it still
  swims); `pair_depth_response` 0.5 or more on at least 12; `bad_qacc` 0;
  every actuator held, and the command frozen, under 0.012 on 16 of 16.
- **Step bar**, outside the score: released level and still at 4 m, told a
  depth of 2.5 m, and separately 5.5 m, for 12 s, three seeds each: at the
  end within 0.3 m of the target, and over the last 2 s a net vertical rate
  under 0.05 m/s and a horizontal drift under 0.3 m. Six of six.
- Read beside both: the time to come within 0.3 m of the target; the pitch
  it used; the largest shoulder torque; the pair hold score.

## 3 Predictions (written before running)

1. The step bar is met, six of six. I hold this at 0.6.
2. The segment bar's response clause is met (0.5 or more on 12 of 16). I
   hold this at 0.5.
3. It still swims: 0.15 or more on at least 14 of 16.
4. The pair hold score stays under 0.3 on most seeds even where the response
   is met, because the hold wants the depth reached within about two seconds
   and then stillness.
5. Going up and going down take the same time within 30% (the body is
   neutral).
6. Null arms under 0.012 on 16 of 16 each.

## 4 Limits fixed now

Quasi-steady model, no wake between the pairs, rigid wings, one buoyancy.
Stopping rule: rows filled, or a row shown blocked with its trace, or about
three hours. A clean negative is a result; no ballast, jets, rotors or extra
fins here, and no bar is loosened.
