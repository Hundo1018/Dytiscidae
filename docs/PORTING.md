# Moving this off the throwaway container

Everything here was developed in an ephemeral cloud container. The code is in
git and safe; **the runs are not** — `runs/` is gitignored, and a run's archive
is the one artefact that cannot be recomputed. Every cell in it cost seconds of
simulation.

## What has to move

| | where it lives | in git? | size |
|---|---|---|---|
| code | this repo, `main` | yes | — |
| search state | `runs/<name>/` | **no** | ~3.5 MB per run |
| session transcript | `~/.claude/projects/<project>/<id>.jsonl` | no | ~21 MB (8 MB gzipped) |
| rendered video | `runs/*/mission.mp4` | no (`*.mp4` ignored) | ~1 MB each |

## Setting up locally

```bash
git clone https://github.com/hundo1018/dytiscidae && cd dytiscidae
python -m venv .venv && source .venv/bin/activate     # 3.11 or newer
pip install -r requirements.txt
python -m dytiscidae.ops.run verify                   # 115 physics checks
python tests/test_search.py                           # 194 search checks
```

CPU only — no GPU anywhere, and `torch`/`ray` are commented out in
`requirements.txt` because nothing on the main path needs them. A run is
single-process and roughly 130 s per generation on four cores.

Rendering is the only part with a system dependency. On a desktop with a
display it works as-is; headless needs an offscreen GL backend:

```bash
export MUJOCO_GL=osmesa      # or egl, if the machine has a GPU
```

## Continuing a run rather than restarting it

Unpack the run beside the repo and resume it:

```bash
tar xzf dytiscidae-runs.tar.gz          # restores runs/arch08/ etc.
python -m dytiscidae.ops.run search --run runs/arch08 --resume \
    --generations 600 --batch 6 --segment-seconds 8 --checkpoint-every 10
```

It prints what it recovered before doing anything:

```
resumed runs/arch08 at generation 61 (0 evaluations, 213 elites)
```

`--resume` is safe to leave on permanently. On a directory with no checkpoint it
just starts a normal run.

Two levels of recovery, because the second one is what you actually get when a
machine disappears:

- **With `search_state.pkl`** — exact. Judge bars, curriculum stages, critic,
  scout, operator bandit and generation counter all come back.
- **Archives only** — the population and the generation come back; everything
  learned is relearned from the designs that are still there. This is the case
  for any run started before resume existed, and for a checkpoint that was being
  written when the power went out.

Both are tested in `tests/test_search.py::test_a_run_can_be_picked_up_where_it_stopped`.

## Reading a run you have moved

```bash
python -m dytiscidae.ops.run dashboard --run runs/arch08     # regenerate the HTML
python -m dytiscidae.ops.run cohort    --run runs/arch08 -n 6
python -m dytiscidae.ops.run render    --run runs/arch08 --top 3
python -m dytiscidae.ops.run showcase  --run runs/film --design runs/arch08 --train
```

`runs/<name>/events.jsonl` and `generations.jsonl` are append-only JSONL and are
the honest record of what happened — regime changes, judge tightenings, audits,
vetoes, scout fits. The dashboard is a view of them, not a separate source.

## Session transcript

`session-transcript.jsonl.gz` is the conversation this was built in: 5,494
records, of which 1,420 are prompts and 2,405 are responses. It is a plain JSONL
of `{"type": "user"|"assistant"|"system"|...}` records, so it reads with
anything:

```bash
zcat session-transcript.jsonl.gz | jq -r 'select(.type=="user") | .message.content'
```

Dropping it at `~/.claude/projects/<slugified-project-path>/<uuid>.jsonl` on a
machine running Claude Code puts it back in that project's history. The
directory name is the project path with `/` replaced by `-`; on the container it
was `-home-user-Dytiscidae`, so a local clone at `/home/you/dytiscidae` wants
`-home-you-dytiscidae`.

The transcript is not needed to continue the work. The reasoning that mattered is
in the commit messages and in the comments next to the code it explains — that
was deliberate, and it is why the commit messages are long.
