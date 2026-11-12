"""L0: can a feed-forward network learn the teacher?  (PREREG.md, frozen first.)

A probe, not one of the three runs: about 3e5 decisions.  Round 0 is plain
imitation of the teacher's own missions; each later round the student flies,
the teacher labels every state it visits, and the network is refitted from
scratch on everything so far (DAgger).

    PYTHONPATH=. MUJOCO_GL=disable .venv/bin/python \
        experiments/triphibian_diagnosis/learnability/probe.py --rounds 3 --jobs 8
"""
import argparse
import json
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
STARTS = ("air", "water", "land")
CONT = [0, 1, 2, 3, 6, 7]          # acceleration (3), tilt limit, limb drive, limb turn
FLAG = [4, 5, 8, 9]                # thrust below the horizon, rotors on, limb pose, memory
HIDDEN, EPOCHS, BATCH, LR, FIT_SEED = 128, 60, 512, 1e-3, 0


def student_law(w):
    """``law(o) -> u`` for fitted weights ``w`` (plain numpy, as a worker runs it)."""
    def law(o):
        x = (np.asarray(o, float) - w["x_mean"]) / w["x_std"]
        if w.get("blind"):
            x = np.zeros_like(x)
        h = np.tanh(w["W0"] @ x + w["b0"])
        h = np.tanh(w["W1"] @ h + w["b1"])
        y = w["W2"] @ h + w["b2"]
        u = np.zeros(len(CONT) + len(FLAG))
        u[CONT] = y[:len(CONT)] * w["y_std"] + w["y_mean"]
        u[FLAG] = (y[len(CONT):] > 0.0).astype(float)
        return u
    return law


def fly(args):
    """One mission.  Returns the verdict row and every decision's (o, teacher's u)."""
    who, w, start, seed = args
    from dytiscidae.control.innerloop import InnerLoop, body_model
    from dytiscidae.control.outer import ShoreMap, SplitController, Teacher
    from dytiscidae.core.bodyplans import ANCHOR_PLANS
    from dytiscidae.core.phenotype import build
    from dytiscidae.envs import mission as M
    from dytiscidae.envs.triphibian import Domain, TriphibianEnv

    t0 = time.time()
    env = TriphibianEnv(build(ANCHOR_PLANS["legged_quad"]()), seed=seed)
    model = body_model(env)
    teach = Teacher(model, ShoreMap()).teach
    law = teach if who == "teacher" else student_law(w)
    obs, lab, act = [], [], []

    def logged(o):
        u = law(o)
        obs.append(np.asarray(o, np.float32))
        lab.append(np.asarray(teach(o), np.float32))
        act.append(np.asarray(u, np.float32))
        return u

    row = {"who": who, "start": start, "seed": seed}
    try:
        with SplitController(logged, InnerLoop(env, model).act).installed(env) as sc:
            read = M.evaluate_mission(env, sc.controller, Domain(start), seed)
        row.update({"held": read.held, "completed": read.completed, "failure": read.failure,
                    "transitions": [read.transitions_completed, read.transitions_commanded],
                    "shares": [round(x, 3) for x in read.shares],
                    "legs": [[d.value, round(s, 2)] for d, s in read.schedule]})
    except Exception as exc:  # a broken rollout is a row, not a crash and not a zero
        row.update({"completed": False, "failure": f"exception: {exc!r}", "held": [],
                    "transitions": [0, 2]})
    row["decisions"] = len(obs)
    if obs:
        a, l = np.array(act), np.array(lab)
        row["flag_agree"] = [float((a[:, j] == l[:, j]).mean()) for j in FLAG]
    row["wall_s"] = round(time.time() - t0, 1)
    return row, (np.array(obs), np.array(lab))


def fit(X, Y, hold):
    """Fit on rows ``~hold``; report error on rows ``hold``."""
    import torch

    torch.manual_seed(FIT_SEED)
    torch.set_num_threads(8)
    Xt, Yt = X[~hold].astype(np.float64), Y[~hold].astype(np.float64)
    x_mean, x_std = Xt.mean(0), Xt.std(0)
    x_std = np.where(x_std < 1e-9, 1.0, x_std)
    y_mean, y_std = Yt[:, CONT].mean(0), Yt[:, CONT].std(0)
    y_std = np.where(y_std < 1e-9, 1.0, y_std)

    def prep(Xa, Ya):
        x = torch.as_tensor((Xa - x_mean) / x_std)
        yc = torch.as_tensor((Ya[:, CONT] - y_mean) / y_std)
        yf = torch.as_tensor(Ya[:, FLAG])
        return x, yc, yf

    x, yc, yf = prep(Xt, Yt)
    net = torch.nn.Sequential(torch.nn.Linear(X.shape[1], HIDDEN), torch.nn.Tanh(),
                              torch.nn.Linear(HIDDEN, HIDDEN), torch.nn.Tanh(),
                              torch.nn.Linear(HIDDEN, len(CONT) + len(FLAG))).double()
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    bce = torch.nn.BCEWithLogitsLoss()
    g = torch.Generator().manual_seed(FIT_SEED)
    n = len(x)
    for _ in range(EPOCHS):
        perm = torch.randperm(n, generator=g)
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            out = net(x[idx])
            loss = ((out[:, :len(CONT)] - yc[idx]) ** 2).mean() + bce(out[:, len(CONT):], yf[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
    stats = {"train_rows": int(n), "held_rows": int(hold.sum())}
    with torch.no_grad():
        for name, (xa, ya) in (("train", (Xt, Yt)), ("held", (X[hold].astype(np.float64),
                                                              Y[hold].astype(np.float64)))):
            if not len(xa):
                continue
            xx, cc, ff = prep(xa, ya)
            out = net(xx)
            stats[name + "_cont_rmse_std"] = [float(v) for v in
                                              ((out[:, :len(CONT)] - cc) ** 2).mean(0).sqrt()]
            stats[name + "_flag_acc"] = [float(v) for v in
                                         ((out[:, len(CONT):] > 0).double() == ff).double().mean(0)]
    sd = net.state_dict()
    w = {"x_mean": x_mean, "x_std": x_std, "y_mean": y_mean, "y_std": y_std,
         "W0": sd["0.weight"].numpy(), "b0": sd["0.bias"].numpy(),
         "W1": sd["2.weight"].numpy(), "b1": sd["2.bias"].numpy(),
         "W2": sd["4.weight"].numpy(), "b2": sd["4.bias"].numpy()}
    return w, stats


def tally(rows):
    out = {}
    for s in STARTS:
        v = [r for r in rows if r["start"] == s]
        out[s] = {"completed": sum(bool(r["completed"]) for r in v), "of": len(v),
                  "transitions": sum(r["transitions"][0] for r in v),
                  "failed": sum(bool(r["failure"]) for r in v)}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--n", type=int, default=32)
    a = ap.parse_args()
    DATA.mkdir(exist_ok=True)
    out = HERE / "results.jsonl"
    ctx = mp.get_context("spawn")
    summary = {"prereg_sha256": __import__("hashlib").sha256((HERE / "PREREG.md").read_bytes()).hexdigest()}

    def run(pool, who, w, seeds, tag):
        jobs = [(who, w, s, seed) for s in STARTS for seed in seeds]
        res = pool.map(fly, jobs, chunksize=1)
        rows = [dict(r, tag=tag) for r, _ in res]
        with open(out, "a") as f:
            for r in rows:
                f.write(json.dumps(r, default=float) + "\n")
        X = np.concatenate([d[0] for _, d in res if len(d[0])])
        Y = np.concatenate([d[1] for _, d in res if len(d[0])])
        ids = np.concatenate([np.full(len(d[0]), i) for i, (_, d) in enumerate(res) if len(d[0])])
        return rows, X, Y, ids

    with ctx.Pool(a.jobs) as pool:
        val = list(range(200, 200 + a.n))
        t0 = time.time()
        rows, _, _, _ = run(pool, "teacher", None, val, "teacher-val")
        summary["teacher_val"] = tally(rows)
        print("teacher on validation seeds", json.dumps(summary["teacher_val"]), flush=True)

        rows, X, Y, ids = run(pool, "teacher", None, list(range(1000, 1000 + a.n)), "collect-0")
        flags = Y[:, FLAG]
        summary["flags_binary"] = bool(np.isin(flags, (0.0, 1.0)).all())
        summary["flag_means_round0"] = [float(v) for v in flags.mean(0)]
        # one mission in ten held out of the round-0 fit, by mission
        hold = np.isin(ids, np.arange(0, ids.max() + 1, 10))
        Xs, Ys, Hs = [X], [Y], [hold]
        for r in range(a.rounds + 1):
            Xa, Ya, Ha = np.concatenate(Xs), np.concatenate(Ys), np.concatenate(Hs)
            np.savez_compressed(DATA / f"round{r}.npz", X=Xa, Y=Ya, hold=Ha)
            w, stats = fit(Xa, Ya, Ha)
            np.savez(DATA / f"weights{r}.npz", **w)
            rows, _, _, _ = run(pool, "student", w, val, f"student-{r}-val")
            summary[f"round{r}"] = {"fit": stats, "val": tally(rows), "rows": int(len(Xa)),
                                    "elapsed_s": round(time.time() - t0)}
            print(f"round {r}: rows {len(Xa)}", json.dumps(summary[f"round{r}"]["val"]),
                  "held flag acc", [round(v, 4) for v in stats.get("held_flag_acc", [])], flush=True)
            if r == 0:
                rows, _, _, _ = run(pool, "student", dict(w, blind=True), val, "blind-val")
                summary["blind"] = tally(rows)
                print("arm Z (input ignored)", json.dumps(summary["blind"]), flush=True)
            if r < a.rounds:
                seeds = list(range(1000 * (r + 2), 1000 * (r + 2) + a.n))
                rows, X, Y, ids = run(pool, "student", w, seeds, f"collect-{r + 1}")
                summary[f"collect{r + 1}"] = tally(rows)
                Xs.append(X); Ys.append(Y); Hs.append(np.zeros(len(X), bool))
            with open(HERE / "summary.json", "w") as f:
                json.dump(summary, f, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
