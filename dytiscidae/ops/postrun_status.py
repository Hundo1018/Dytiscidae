"""What a finished run's post-run left behind, readable without numpy.

``ops.run.launch_postrun`` writes ``postrun_status.json``; ``job status``
reads it through ``postrun_verdict`` (arch48 item 2: two runs' reports were
silently missing because a ``post-run exited 1`` line went unread).
"""
from __future__ import annotations

import json
from pathlib import Path

#: What ``launch_postrun`` leaves in the run directory: its exit, whether the
#: report and the film manifest exist, and the tail of ``postrun.log``.
POSTRUN_STATUS = "postrun_status.json"


def postrun_verdict(run_dir) -> str | None:
    """One line for ``job status`` when post-run failed or never wrote a
    status; None when it succeeded."""
    path = Path(run_dir) / POSTRUN_STATUS
    if not path.exists():
        return None if (Path(run_dir) / "report.html").exists() else (
            "post-run has not recorded a status and there is no report.html")
    try:
        st = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        return f"{POSTRUN_STATUS} unreadable: {exc}"
    if st.get("ok"):
        return None
    last = (st.get("log_tail") or [""])[-1]
    return (f"POST-RUN FAILED at {st.get('at')} (exit {st.get('exit')}, "
            f"report {'present' if st.get('report') else 'missing'}): {last}")
