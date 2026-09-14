"""Reproducible experiments that check the project's own mathematics.

Every number quoted in a source comment, a docstring or `docs/ROADMAP.md` is
supposed to be reproducible by running something in here.  A number that only
exists in a comment is a story about one run that happened once; a number that
comes out of `python experiments/<name>/run.py` is a measurement.

Each experiment directory holds:

    config.json   the inputs: seeds, sample sizes, sweep points
    run.py        the experiment, runnable as `python experiments/<name>/run.py`
    results/      what the last run wrote, committed so a claim has a receipt
    README.md     the question, the design, and the answer as measured

`run.py` never invents a number.  Anything it prints it computed in the same
process.
"""
