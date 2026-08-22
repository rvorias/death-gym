#!/usr/bin/env python3
"""Live multi-run training dashboard. Stdlib only — run it in a tmux pane:

    just tui                 # refreshes every 2s
    python tui.py --once     # single render (debug)
    python tui.py --focus 2  # start focused on run #2

Shows every run at once — both queue lanes, anything launched by hand, and the
pending queue behind them — then a detail panel for whichever run has focus.

Keys (when stdin is a tty):  1-9 focus a run   j/k move focus   q quit

Runs are discovered from /proc rather than from a fixed log path: any live
python under this directory is matched to the log on its stdout, so a run shows
up here whether it came from a lane runner or was started by hand. Finished runs
are recovered from the lane done-files, which carry exit code and wall time.

Reads (all optional — a missing file just leaves that panel empty):
  /proc                  live processes, their logs and GPU use
  local/logs/run.log           default train.py log (what `just train` writes)
  local/logs/*.log             any other run log becomes its own row
  local/logs/reference.log     a past run, drawn as the comparison bar
  local/logs/queue{,2}.txt     pending work per lane
  local/logs/queue{,2}_done.txt  completed work: exit code + wall time
  local/logs/results.tsv       experiment verdicts
  local/logs/state.json        behavioural eval (side-car)

Override the directory with DM_LOGS=/path/to/logs.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import select
import shutil
import subprocess
import sys
import termios
import time
import tty
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent

# Everything the dashboard reads lives under local/logs/. Each path is optional —
# a missing file just means that panel stays empty.
LOGS = Path(os.environ.get("DM_LOGS", HERE / "local" / "logs"))
RUN_LOG = LOGS / "run.log"          # `just train` tees here
REF_LOG = LOGS / "reference.log"    # a past run to draw the comparison bar from
RESULTS = LOGS / "results.tsv"      # experiment verdicts, one row per run
STATE = LOGS / "state.json"         # behavioural eval side-car
EXP_LOGS = LOGS                     # any other *.log in here becomes a row

# Queue lanes: a runner script that pulls commands off a queue file and appends
# to a done file. Left in place for anyone who wires up their own runner.
LANES = [
    ("L1", LOGS / "queue.txt", LOGS / "queue_done.txt", "exp_runner.py"),
    ("L2", LOGS / "queue2.txt", LOGS / "queue2_done.txt", "exp_runner2.py"),
]

# The score a finished run is graded against — green if it beats this, red if
# not. Set it to whatever your current best run scores.
BAR_EVAL_AVG = float(os.environ.get("DM_BAR_EVAL_AVG", 281.6))

# Action mix of human players, for comparison against the policy's mix (%).
HUMAN_MIX = {"explore": 10.8, "fight": 27.9, "flee": 4.5,
             "equip": 30.0, "shop": 16.9, "levelup": 9.9}

# A run whose log has not advanced in this long is stalled, not running.
STALL_SECONDS = 180
# Finished runs older than this are not worth a row.
RECENT_DONE_SECONDS = 6 * 3600

IT_RE = re.compile(
    r"^it (\d+)/(\d+) \| (\d+)s \| fps: (\d+) \| pg: [-\d.]+ \| v: ([\d.]+) "
    r"\| ent: ([\d.]+) \| death_xp: (\d+)")
HDR_RE = re.compile(r"\| (\d+) envs \| (\d+) steps/rollout \| (\d+) iters \| ([\d,]+) env steps")
RUN_NAME_RE = re.compile(r"--run-name[= ]+(\S+)")

DIM, BOLD, RESET = "\033[2m", "\033[1m", "\033[0m"
GREEN, RED, YELLOW, CYAN, BLUE = (
    "\033[32m", "\033[31m", "\033[33m", "\033[36m", "\033[34m")
SPARK = "▁▂▃▄▅▆▇█"

# Run lifecycle. Mirrors how work actually moves through a lane: it waits in
# the queue, runs, and ends up judged. STALLED is broken out from RUNNING
# because a wedged run looks identical to a healthy one in a process list.
STATE_STYLE = {
    "RUNNING": (GREEN, "●"),
    "STALLED": (YELLOW, "◐"),
    "DONE":    (DIM, "○"),
    "ENDED":   (DIM, "◌"),
    "FAILED":  (RED, "✗"),
    "QUEUED":  (DIM, "·"),
}


# ─── Log parsing ────────────────────────────────────────────────────────

def parse_log(path: Path, tail_bytes: int = 400_000):
    """Header info, iteration rows and final summary from a training log.

    Only the tail is read: these logs reach tens of MB and the TUI re-reads
    every run every couple of seconds."""
    out = {"rows": [], "envs": None, "iters": None, "total_steps": None,
           "final": {}, "device": None, "run_name": None}
    try:
        size = path.stat().st_size
        with open(path, "rb") as f:
            if size > tail_bytes:
                f.seek(size - tail_bytes)
                f.readline()  # drop the partial line
            text = f.read().decode("utf-8", errors="replace")
    except OSError:
        return out
    for line in text.splitlines():
        m = IT_RE.match(line)
        if m:
            it, tot, sec, fps, v, ent, dxp = m.groups()
            out["rows"].append((int(it), int(tot), int(sec), int(fps),
                                float(v), float(ent), int(dxp)))
            continue
        m = HDR_RE.search(line)
        if m:
            out["envs"] = int(m.group(1))
            out["iters"] = int(m.group(3))
            out["total_steps"] = int(m.group(4).replace(",", ""))
            continue
        if "  >> saved " in line and "/checkpoints/" in line:
            out["run_name"] = line.split("/checkpoints/", 1)[1].split("/")[0]
            continue
        if line.startswith("Device:"):
            out["device"] = line.split(":", 1)[1].strip()
            continue
        for key in ("avg_xp", "max_xp", "median_xp", "eval_avg_xp",
                    "eval_max_xp", "eval_median_xp", "fps", "training_seconds",
                    "stitched_eval_avg_xp", "stitched_eval_max_xp"):
            if line.startswith(f"{key}:"):
                try:
                    out["final"][key] = float(line.split(":")[1])
                except ValueError:
                    pass
    return out


# ─── Process / GPU discovery ────────────────────────────────────────────

def _read(path: Path, binary: bool = False):
    try:
        return path.read_bytes() if binary else path.read_text(errors="replace")
    except OSError:
        return b"" if binary else ""


def live_processes() -> list[dict]:
    """Python processes running out of this directory, with the log they write.

    /proc/<pid>/fd/1 is the child's stdout, which the lane runners point at the
    experiment log — that gives an exact process→log mapping without having to
    guess from filenames."""
    procs = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        raw = _read(entry / "cmdline", binary=True)
        if not raw:
            continue
        # Parse the real argv. Substring-matching the joined command line
        # matches any SHELL whose command merely mentions train.py — including
        # the `just` recipe that launched it, and any tooling that echoes the
        # path — which shows up as a phantom run.
        argv = [a.decode("utf-8", errors="replace")
                for a in raw.split(b"\x00") if a]
        if not argv:
            continue
        cmd = " ".join(argv)
        if not Path(argv[0]).name.startswith("python"):
            continue
        script = next((a for a in argv[1:] if a.endswith(".py")), None)
        if script is None:
            continue
        parts = Path(script).parts
        if not (Path(script).name in ("train.py", "grpo_train.py")
                or "analysis" in parts or "dashboard" in parts or "llm" in parts):
            continue
        if Path(script).name in ("tui.py", "arena.py", "watch.py") \
                or "exp_runner" in script:
            continue
        try:
            log = os.readlink(f"/proc/{pid}/fd/1")
        except OSError:
            log = ""
        procs.append({
            "pid": pid,
            "cmd": cmd,
            "log": Path(log) if log.startswith("/") and not log.startswith("/dev") else None,
            "ppid": _ppid(pid),
            "start": _proc_start_time(pid),
        })
    return procs


def _ppid(pid: int) -> int | None:
    stat = _read(Path(f"/proc/{pid}/stat"))
    if not stat:
        return None
    # comm may contain spaces/parens — everything after the final ')' is safe.
    try:
        return int(stat[stat.rindex(")") + 1:].split()[1])
    except (ValueError, IndexError):
        return None


def _proc_start_time(pid: int) -> float | None:
    try:
        return Path(f"/proc/{pid}").stat().st_mtime
    except OSError:
        return None


def lane_of(proc: dict, runner_pids: dict[int, str]) -> str:
    """Which queue lane launched this, via its parent process."""
    ppid = proc.get("ppid")
    if ppid in runner_pids:
        return runner_pids[ppid]
    return "—"


def runner_pids() -> dict[int, str]:
    """pid → lane label, for the lane runners themselves."""
    out = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        raw = _read(entry / "cmdline", binary=True)
        if not raw:
            continue
        cmd = raw.replace(b"\x00", b" ").decode("utf-8", errors="replace")
        for label, _q, _d, script in LANES:
            if script in cmd:
                out[int(entry.name)] = label
    return out


def gpus() -> list[dict]:
    try:
        q = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=index,uuid,utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5).stdout
    except Exception:
        return []
    out = []
    for line in q.splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) < 5:
            continue
        out.append({"index": int(parts[0]), "uuid": parts[1],
                    "util": int(parts[2]), "mem_used": int(parts[3]),
                    "mem_total": int(parts[4])})
    return out


def compute_apps(gpu_list: list[dict]) -> dict[int, tuple[str, int]]:
    """pid → ("gpu<id>", MB). Reports the GPU a run is ACTUALLY on, which
    is what you want here: CUDA_VISIBLE_DEVICES ordering does not match
    nvidia-smi ordering on this box."""
    by_uuid = {g["uuid"]: g for g in gpu_list}
    try:
        q = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory,gpu_uuid",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5).stdout
    except Exception:
        return {}
    out = {}
    for line in q.splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) < 3:
            continue
        try:
            pid, mem = int(parts[0]), int(parts[1])
        except ValueError:
            continue
        g = by_uuid.get(parts[2])
        out[pid] = (f"gpu{g['index']}" if g else "?", mem)
    return out


# ─── Queue state ────────────────────────────────────────────────────────

def _finished_at(stamp: str, now: float) -> float:
    """exp_runner writes 'MM-DD HH:MM' with no year. Assume the most recent
    such date that is not in the future."""
    try:
        parsed = time.strptime(stamp.strip(), "%m-%d %H:%M")
    except ValueError:
        return 0.0
    year = time.localtime(now).tm_year
    for candidate in (year, year - 1):
        t = time.mktime((candidate, parsed.tm_mon, parsed.tm_mday,
                         parsed.tm_hour, parsed.tm_min, 0, 0, 0, -1))
        if t <= now + 86400:
            return t
    return 0.0


def done_entries(done_file: Path, now: float) -> dict[str, tuple[int, str, float]]:
    """cmd → (exit_code, wall, finished_at). Mirrors exp_runner's own parsing.

    The per-line timestamp is what dates an entry — using the file mtime would
    make every historical row look freshly finished the moment a lane appends
    its next result."""
    out = {}
    for line in _read(done_file).splitlines():
        if "\t" not in line:
            continue
        stamp, rest = line.split("\t", 1)
        cmd = rest.rsplit("\t", 2)[0]
        tail = rest.rsplit("\t", 2)[1:]
        code, wall = 0, ""
        if len(tail) == 2:
            try:
                code = int(tail[0])
            except ValueError:
                code = 0
            wall = tail[1]
        out[cmd] = (code, wall, _finished_at(stamp, now))
    return out


def queue_state(now: float) -> list[dict]:
    """Per-lane pending commands and completion history."""
    lanes = []
    for label, queue_file, done_file, script in LANES:
        done = done_entries(done_file, now)
        pending = []
        for line in _read(queue_file).splitlines():
            cmd = line.strip()
            if not cmd or cmd.startswith("#"):
                continue
            if cmd not in done:
                pending.append(cmd)
        lanes.append({"label": label, "pending": pending, "done": done,
                      "script": script})
    return lanes


# ─── Run model ──────────────────────────────────────────────────────────

@dataclass
class Run:
    name: str
    state: str
    cmd: str = ""
    lane: str = "—"
    gpu: str = ""
    gpu_mem: int = 0
    pid: int | None = None
    log: Path | None = None
    exit_code: int | None = None
    wall: str = ""
    updated: float = 0.0
    parsed: dict = field(default_factory=dict)

    @property
    def last_row(self):
        rows = self.parsed.get("rows") or []
        return rows[-1] if rows else None

    @property
    def sort_key(self):
        order = {"RUNNING": 0, "STALLED": 1, "FAILED": 2, "DONE": 3,
                 "ENDED": 4, "QUEUED": 5}
        return (order.get(self.state, 9), -self.updated)


def run_name_for(cmd: str, log: Path | None) -> str:
    m = RUN_NAME_RE.search(cmd)
    if m:
        return m.group(1)
    for token in cmd.split():
        if token.endswith(".py"):
            stem = Path(token).stem
            if stem != "train":
                return stem
    if log is not None:
        # exp_runner names logs "<epoch>_<slugified command>.log"
        stem = log.stem
        return stem.split("_", 1)[1][:28] if "_" in stem else stem[:28]
    return "train"


def discover_runs(now: float) -> list[Run]:
    lanes = queue_state(now)
    gpu_list = gpus()
    apps = compute_apps(gpu_list)
    rpids = runner_pids()
    runs: list[Run] = []
    claimed_logs: set[Path] = set()

    # 1. Live processes — the runs that matter most.
    for proc in live_processes():
        log = proc["log"]
        parsed = parse_log(log) if log else {}
        updated = log.stat().st_mtime if log and log.exists() else (
            proc.get("start") or now)
        stalled = parsed.get("rows") and (now - updated) > STALL_SECONDS
        gpu, mem = apps.get(proc["pid"], ("", 0))
        runs.append(Run(
            name=run_name_for(proc["cmd"], log),
            state="STALLED" if stalled else "RUNNING",
            cmd=proc["cmd"], lane=lane_of(proc, rpids), gpu=gpu, gpu_mem=mem,
            pid=proc["pid"], log=log, updated=updated, parsed=parsed))
        if log:
            claimed_logs.add(log)

    # 2. Recently finished work, from the lane done-files.
    for lane in lanes:
        for cmd, (code, wall, finished) in lane["done"].items():
            if now - finished > RECENT_DONE_SECONDS:
                continue
            runs.append(Run(
                name=run_name_for(cmd, None),
                state="DONE" if code == 0 else "FAILED",
                cmd=cmd, lane=lane["label"], exit_code=code, wall=wall,
                updated=finished, parsed={}))

    # 3. Recently-finished experiment logs with no live process. Without this a
    # hand-started run DISAPPEARS the moment it finishes: DONE rows otherwise
    # come only from the lane done-files, which a manual launch never touches.
    if EXP_LOGS.is_dir():
        for log in EXP_LOGS.glob("*.log"):
            if log in claimed_logs:
                continue
            try:
                updated = log.stat().st_mtime
            except OSError:
                continue
            if now - updated > RECENT_DONE_SECONDS:
                continue
            parsed = parse_log(log)
            if not (parsed.get("rows") or parsed.get("final")):
                continue
            stem = log.stem
            name = parsed.get("run_name") or (
                stem.split("_", 1)[1] if "_" in stem else stem)
            runs.append(Run(
                name=name,
                state="DONE" if parsed["final"].get("eval_avg_xp") else "ENDED",
                cmd=f"(finished) {log.name}", log=log, updated=updated,
                parsed=parsed))
            claimed_logs.add(log)

    # 4. A hand-started run writing the default log, if no process claimed it.
    if RUN_LOG.exists() and RUN_LOG not in claimed_logs:
        parsed = parse_log(RUN_LOG)
        updated = RUN_LOG.stat().st_mtime
        age = now - updated
        # Only STALLED if it looks like something that just stopped advancing.
        # An old log is not a wedged run — showing a weeks-old file as STALLED
        # reads as "needs attention" for something nobody is waiting on.
        if parsed.get("rows") and age <= RECENT_DONE_SECONDS:
            if parsed["final"].get("eval_avg_xp"):
                state = "DONE"
            elif age > STALL_SECONDS:
                state = "ENDED"
            else:
                state = "RUNNING"
            runs.append(Run(name="run.log", state=state, cmd="(manual)",
                            log=RUN_LOG, updated=updated, parsed=parsed))

    # 5. What is waiting behind them.
    for lane in lanes:
        for cmd in lane["pending"][:3]:
            runs.append(Run(name=run_name_for(cmd, None), state="QUEUED",
                            cmd=cmd, lane=lane["label"], updated=0.0))

    runs.sort(key=lambda r: r.sort_key)
    return runs


# ─── Rendering helpers ──────────────────────────────────────────────────

def sparkline(values, width):
    if not values or width <= 0:
        return ""
    step = max(1, len(values) // width)
    vs = [values[i] for i in range(0, len(values), step)][:width]
    lo, hi = min(vs), max(vs)
    rng = (hi - lo) or 1
    return "".join(SPARK[int((v - lo) / rng * (len(SPARK) - 1))] for v in vs)


def chart(series, ref_series, width, height):
    """Multi-row area chart: current run solid, reference dim dots overlay.
    Both series are (frac, value) lists; frac in [0,1] maps to x."""
    if not series or width < 12 or height < 3:
        return []
    plot_w = width - 8
    grid = [[" "] * plot_w for _ in range(height)]
    vals = [v for _, v in series] + [v for _, v in (ref_series or [])]
    lo, hi = min(vals), max(vals)
    rng = (hi - lo) or 1.0

    def put(frac, val, solid):
        x = min(plot_w - 1, int(frac * plot_w))
        y = min(height - 1, int((val - lo) / rng * (height - 1)))
        row = height - 1 - y
        if solid:
            for r in range(row, height):
                grid[r][x] = "█"
        elif grid[row][x] == " ":
            grid[row][x] = "·"

    for frac, val in (ref_series or []):
        put(frac, val, solid=False)
    for frac, val in series:
        put(frac, val, solid=True)

    lines = []
    for r, row in enumerate(grid):
        label = f"{hi:5.0f} ┤" if r == 0 else (f"{lo:5.0f} ┤" if r == height - 1 else "      │")
        lines.append(f"{DIM}{label}{RESET}{''.join(row)}")
    return lines


def ref_at_frac(ref, frac):
    """death_xp of the reference run at the same env-step fraction."""
    if not ref["rows"] or not ref["iters"]:
        return None
    target = frac * ref["iters"]
    return min(ref["rows"], key=lambda r: abs(r[0] - target))[6]


def fmt_eta(seconds: float) -> str:
    if seconds <= 0:
        return "—"
    return f"{int(seconds // 3600)}:{int(seconds % 3600 // 60):02d}h"


def results_tail(n=5):
    rows = []
    for line in _read(RESULTS).splitlines()[1:]:
        parts = line.split("\t")
        if len(parts) >= 5:
            rows.append(parts)
    return rows[-n:]


def behavior_section(width):
    lines = []
    try:
        s = json.loads(_read(STATE))
    except Exception:
        return lines
    ev = s.get("policy_eval") or {}
    groups = {g["group"]: g["pct"] for g in ev.get("strategy_groups", [])}
    if not groups:
        return lines
    ck = (ev.get("checkpoint") or "?").split("/")[-2:]
    lines.append(f"{BOLD}BEHAVIOR{RESET} {DIM}(ckpt {'/'.join(ck)}, argmax avg_xp "
                 f"{ev.get('xp_avg', 0):.0f}){RESET}")
    name_map = {"equip/manage": "equip", "level-up": "levelup"}
    barw = max(10, min(34, width - 44))
    for label in ("explore", "fight", "flee", "equip", "shop", "levelup"):
        pol = next((v for k, v in groups.items()
                    if name_map.get(k, k) == label or k == label), None)
        if pol is None:
            continue
        hum = HUMAN_MIX[label]
        nbar = int(pol / 40.0 * barw)
        hpos = min(barw - 1, int(hum / 40.0 * barw))
        bar = ["█" if i < nbar else "─" for i in range(barw)]
        bar_s = "".join(bar[:hpos]) + f"{YELLOW}┃{RESET}" + "".join(bar[hpos + 1:])
        lines.append(f"  {label:8s} {pol:5.1f}%  {bar_s}  {DIM}human {hum:.1f}{RESET}")
    return lines


# ─── Panels ─────────────────────────────────────────────────────────────

def gpu_panel(gpu_list, width):
    if not gpu_list:
        return [f"  {DIM}GPU n/a{RESET}"]
    cells = []
    for g in gpu_list:
        col = GREEN if g["util"] > 50 else (YELLOW if g["util"] > 5 else DIM)
        cells.append(f"gpu{g['index']:<3d} {col}{g['util']:3d}%{RESET} "
                     f"{DIM}{g['mem_used'] / 1024:4.1f}/{g['mem_total'] / 1024:.0f}G{RESET}")
    return ["  " + "   ".join(cells)]


def lane_panel(lanes, rpids_by_lane, width):
    parts = []
    for lane in lanes:
        up = lane["label"] in rpids_by_lane.values()
        col = GREEN if up else RED
        state = "up" if up else "down"
        parts.append(f"{lane['label']} {col}{state}{RESET} "
                     f"{DIM}{len(lane['pending'])} queued{RESET}")
    return ["  lanes: " + "   ".join(parts)]


def run_rows(runs, focus_idx, width):
    """One or two lines per run: identity + live stats."""
    lines = []
    for i, r in enumerate(runs):
        col, glyph = STATE_STYLE.get(r.state, (DIM, "·"))
        marker = f"{CYAN}▸{RESET}" if i == focus_idx else " "
        idx = f"{DIM}{i + 1}{RESET}" if i < 9 else " "
        where = f"{r.lane}/{r.gpu}" if r.gpu else r.lane
        head = (f"{marker}{idx} {col}{glyph} {r.state:<8}{RESET}"
                f" {DIM}{where:<12}{RESET} {BOLD}{r.name[:26]:<26}{RESET}")

        row = r.last_row
        if row and r.state in ("RUNNING", "STALLED"):
            it, tot, sec, fps, v, ent, dxp = row
            frac = it / tot if tot else 0.0
            barw = max(8, min(20, width - 92))
            filled = int(frac * barw)
            eta = (tot - it) * (sec / it) if it else 0
            lines.append(f"{head} [{('█' * filled).ljust(barw, '░')}] "
                         f"{frac * 100:5.1f}%  {DIM}ETA{RESET} {fmt_eta(eta)}")
            spark = sparkline([x[6] for x in r.parsed["rows"]], 24)
            lines.append(f"     {DIM}it {it}/{tot}  fps {fps:,}  ent {ent:.2f}  "
                         f"v {v:.1f}{RESET}  death_xp {BOLD}{dxp}{RESET} {DIM}{spark}{RESET}")
        elif r.state in ("DONE", "FAILED", "ENDED"):
            bits = []
            if r.exit_code is not None:
                bits.append(f"exit {r.exit_code}")
            if r.wall:
                bits.append(r.wall)
            fin = r.parsed.get("final", {}) if r.parsed else {}
            for key, label in (("eval_avg_xp", "eval_avg"),
                               ("stitched_eval_avg_xp", "stitched")):
                if key in fin:
                    bits.append(f"{label} {fin[key]:.1f}")
            lines.append(f"{head} {DIM}{'  '.join(bits)}{RESET}")
        else:
            lines.append(f"{head} {DIM}{r.cmd[:max(0, width - 60)]}{RESET}")
    return lines


def focus_panel(run: Run, ref, width, height):
    """The chart + bar comparison for the focused run."""
    out = []
    if run is None:
        return out
    rows = run.parsed.get("rows") if run.parsed else None
    if not rows:
        out.append(f"{BOLD}FOCUS{RESET} {run.name} {DIM}— no iteration data{RESET}")
        if run.cmd:
            out.append(f"  {DIM}{run.cmd[:width - 4]}{RESET}")
        return out

    it, tot, sec, fps, v, ent, dxp = rows[-1]
    frac = it / tot if tot else 0.0
    out.append(f"{BOLD}FOCUS{RESET} {BOLD}{run.name}{RESET} "
               f"{DIM}death_xp — current (█) vs bar run (·){RESET}")
    cur_series = [(r[0] / tot, r[6]) for r in rows]
    ref_series = ([(r[0] / ref["iters"], r[6]) for r in ref["rows"]]
                  if ref["rows"] and ref["iters"] else None)
    out.extend(chart(cur_series, ref_series, width, max(5, height)))
    out.append(f"  {DIM}entropy  {sparkline([r[5] for r in rows], width - 14)}{RESET}")

    rd = ref_at_frac(ref, frac)
    tail = ""
    if rd is not None:
        delta = dxp - rd
        col = GREEN if delta >= 0 else RED
        tail = f"   vs bar-run @ same step-frac: {col}{delta:+d}{RESET} {DIM}(ref {rd}){RESET}"
    fin = run.parsed.get("final", {})
    if fin.get("eval_avg_xp") is not None:
        ea = fin["eval_avg_xp"]
        col = GREEN if ea > BAR_EVAL_AVG else RED
        out.append(f"  {BOLD}FINAL{RESET} eval_avg {col}{ea:.1f}{RESET} / "
                   f"max {fin.get('eval_max_xp', 0):.0f}   bar {BAR_EVAL_AVG}{tail}")
    elif tail:
        out.append(" " + tail)
    return out


# ─── Frame ──────────────────────────────────────────────────────────────

def render(focus_idx: int, interactive: bool) -> int:
    now = time.time()
    runs = discover_runs(now)
    ref = parse_log(REF_LOG)
    gpu_list = gpus()
    lanes = queue_state(now)
    rpids = runner_pids()

    width, height = shutil.get_terminal_size((120, 40))
    focus_idx = max(0, min(focus_idx, len(runs) - 1)) if runs else 0

    out = []
    title = " DEATH MOUNTAIN — LIVE RUNS "
    stamp = time.strftime("%H:%M:%S")
    out.append(f"{BOLD}{CYAN}{title}{RESET}"
               f"{DIM}{'─' * max(0, width - len(title) - 10)} {stamp}{RESET}")
    out.extend(gpu_panel(gpu_list, width))
    out.extend(lane_panel(lanes, rpids, width))
    out.append("")

    active = [r for r in runs if r.state in ("RUNNING", "STALLED")]
    out.append(f"{BOLD}RUNS{RESET} {DIM}({len(active)} active, {len(runs)} shown){RESET}")
    if runs:
        out.extend(run_rows(runs, focus_idx, width))
    else:
        out.append(f"  {DIM}nothing running and nothing queued{RESET}")
    out.append("")

    used = len(out)
    n_results = 4
    behavior = behavior_section(width)
    budget = height - used - n_results - len(behavior) - 6
    out.extend(focus_panel(runs[focus_idx] if runs else None, ref, width,
                           max(5, min(14, budget))))
    out.append("")

    out.append(f"{BOLD}RESULTS{RESET} {DIM}(last {n_results}){RESET}")
    for parts in results_tail(n_results):
        avg, status, desc = parts[1], parts[3], parts[4]
        scol = {"keep": GREEN, "discard": RED, "crash": RED}.get(status, YELLOW)
        out.append(f"  {scol}{status:8s}{RESET} {avg:>7s}  "
                   f"{DIM}{desc[:max(0, width - 22)]}{RESET}")

    if behavior:
        out.append("")
        out.extend(behavior)

    if interactive:
        out.append(f"{DIM}  1-9 focus   j/k move   q quit{RESET}")

    print("\033[2J\033[H" + "\n".join(out[:height]), flush=True)
    return focus_idx


# ─── Input ──────────────────────────────────────────────────────────────

def read_key(timeout: float) -> str | None:
    """Non-blocking single keypress; None if nothing arrived in `timeout`."""
    if select.select([sys.stdin], [], [], timeout)[0]:
        return sys.stdin.read(1)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true", help="render once and exit")
    ap.add_argument("--interval", type=float, default=2.0)
    ap.add_argument("--focus", type=int, default=1, help="1-based run to focus")
    args = ap.parse_args()

    focus = max(0, args.focus - 1)
    if args.once:
        render(focus, interactive=False)
        return

    interactive = sys.stdin.isatty()
    old = None
    try:
        print("\033[?25l", end="")  # hide cursor
        if interactive:
            old = termios.tcgetattr(sys.stdin)
            tty.setcbreak(sys.stdin.fileno())
        while True:
            focus = render(focus, interactive)
            if not interactive:
                time.sleep(args.interval)
                continue
            key = read_key(args.interval)
            if key is None:
                continue
            if key in ("q", "\x03"):
                break
            if key.isdigit() and key != "0":
                focus = int(key) - 1
            elif key == "j":
                focus += 1
            elif key == "k":
                focus = max(0, focus - 1)
    except KeyboardInterrupt:
        pass
    finally:
        if old is not None:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old)
        print("\033[?25h", end="")  # restore cursor


if __name__ == "__main__":
    main()
