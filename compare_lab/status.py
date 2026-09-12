"""Summarize all long tasks from their status.json files (plan section 10).

    python -m compare_lab.status

Reports each task's state, progress, tokens/s, recent losses and ETA, and flags
DEAD tasks: state "running" but either not updated for >5 min or whose PID is
gone (crash / power loss / closed window).
"""

import json
import time
from pathlib import Path

from compare_lab.train.tasks import RUNS_DIR

STALE_SECONDS = 5 * 60


def _pid_alive(pid) -> bool | None:
    if not pid:
        return None
    try:
        import psutil
        return psutil.pid_exists(int(pid))
    except Exception:
        return None  # unknown without psutil


def _age_seconds(updated_at: str | None) -> float | None:
    if not updated_at:
        return None
    try:
        t = time.mktime(time.strptime(updated_at, "%Y-%m-%d %H:%M:%S"))
        return time.time() - t
    except Exception:
        return None


def collect() -> list[dict]:
    rows = []
    if not RUNS_DIR.exists():
        return rows
    for d in sorted(RUNS_DIR.iterdir()):
        f = d / "status.json"
        if not f.exists():
            continue
        try:
            s = json.loads(f.read_text())
        except Exception:
            continue
        age = _age_seconds(s.get("updated_at"))
        alive = _pid_alive(s.get("pid"))
        # The PID is authoritative: if the process is alive, the task is alive,
        # however old the last update is. Data prep legitimately goes quiet for
        # minutes at a time (training the BPE, filling the HF shuffle buffer over
        # the network), and calling that DEAD would be a false alarm on exactly
        # the workflow this is meant to watch. Only report DEAD when the process
        # is really gone, or when it is stale AND we cannot check the PID.
        dead = s.get("state") == "running" and (
            alive is False or (alive is None and age is not None and age > STALE_SECONDS)
        )
        s["_age"] = age
        s["_alive"] = alive
        s["_dead"] = dead
        s["_stale"] = bool(s.get("state") == "running" and age is not None
                           and age > STALE_SECONDS and alive)
        rows.append(s)
    return rows


def _fmt_eta(sec) -> str:
    if not sec:
        return "-"
    sec = int(sec)
    h, rem = divmod(sec, 3600)
    m, _ = divmod(rem, 60)
    return f"{h}h{m:02d}m" if h else f"{m}m"


def main() -> None:
    rows = collect()
    if not rows:
        print("No tasks found in", RUNS_DIR)
        return
    print(f"{'task':22} {'model':14} {'state':9} {'step':>13} {'pct':>6} "
          f"{'tok/s':>8} {'val':>7} {'eta':>7}  flags")
    print("-" * 110)
    for s in rows:
        step = f"{s.get('step','?')}/{s.get('total_steps','?')}"
        flags = []
        if s.get("_dead"):
            flags.append("DEAD(process gone)")
        elif s.get("_stale"):
            flags.append(f"quiet {int(s['_age'] // 60)}m (process alive)")
        if s.get("error"):
            flags.append(f"error: {s['error'][:60]}")
        print(f"{s.get('task',''):22} {str(s.get('model','')):14} {s.get('state',''):9} "
              f"{step:>13} {str(s.get('percent','')):>6} {str(s.get('tokens_per_sec') or '-'):>8} "
              f"{str(s.get('val_loss') or '-'):>7} {_fmt_eta(s.get('eta_seconds')):>7}  {' '.join(flags)}")


if __name__ == "__main__":
    main()
