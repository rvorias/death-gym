#!/usr/bin/env python3
"""Automatic evaluator: score every Taskmarket submission as it lands.

    python tools/watch_submissions.py <taskId>              # public board
    python tools/watch_submissions.py <taskId> --private --once
    python tools/watch_submissions.py <taskId> --private --once --reveal \
        --gist <id>                                        # publish, once

Polls `taskmarket task submissions`, downloads each new entry, scores it with
tools/evaluate_submission.py, and keeps a leaderboard JSON keyed by submission
id. Already-scored ids are skipped, so the loop is safe to restart and safe to
run past the deadline.

--private rescores every entry on the private group and publishes nothing.
Add --reveal to write FINAL.md into the gist with the seeds and salt exposed,
which is what lets anyone recompute the commitment and reproduce a row. Do it
once, after the deadline: revealing early hands out the test set.

By default it scores the PUBLIC seed group and writes a leaderboard entrants
can check and reproduce themselves -- the prize is decided later on the private
group, with --private. Publishing a live board on the private seeds would leak
the test set one score at a time.

The scoring itself is deterministic, so the leaderboard is reproducible by
anyone holding the seeds. This does NOT pay anyone: it produces the
ranking a human then acts on with `taskmarket task accept-submissions`.

Requires the native engine (`just build`) and the `taskmarket` CLI on PATH.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import tools.evaluate_submission as scorer  # noqa: E402
import tools.validate_submission as validator  # noqa: E402

SAFE_ID = re.compile(r"\A[A-Za-z0-9_-]{1,128}\Z")


def taskmarket(*args: str) -> dict:
    """Run the CLI and unwrap its {ok, data} envelope. Never shell=True."""
    proc = subprocess.run(["taskmarket", *args], capture_output=True, text=True)
    try:
        envelope = json.loads(proc.stdout or proc.stderr)
    except json.JSONDecodeError:
        raise RuntimeError(f"taskmarket {' '.join(args)}: {proc.stderr.strip()}")
    if not envelope.get("ok"):
        raise RuntimeError(f"taskmarket {' '.join(args)}: {envelope.get('error')}")
    return envelope.get("data")


def rows_of(data) -> list[dict]:
    """The submissions list, whether the CLI hands back a list or wraps it."""
    if isinstance(data, list):
        return data
    for key in ("submissions", "items", "results"):
        if isinstance(data, dict) and isinstance(data.get(key), list):
            return data[key]
    raise RuntimeError(f"unrecognised submissions payload: {type(data).__name__}")


def new_rows(rows: list[dict], scored: dict) -> list[dict]:
    """Active, not-yet-scored submissions with a usable id.

    Rejected rows are skipped: the requester already ruled them out, and
    scoring one would put it back on the leaderboard. Ids are checked because
    they are used as filenames below.
    """
    out = []
    for row in rows:
        sid = str(row.get("id", ""))
        if row.get("rejectedAt") or sid in scored:
            continue
        if not SAFE_ID.match(sid):
            print(f"skip: unusable submission id {sid!r}", file=sys.stderr)
            continue
        out.append(row)
    return out


def score_one(task_id: str, row: dict, entries: Path, worlds: int | None,
              seeds=None) -> dict:
    """Download and score one submission. A rejection is a result, not a crash."""
    sid = row["id"]
    zip_path = entries / f"{sid}.zip"
    if not zip_path.exists():
        taskmarket("task", "download", task_id, "--submission", sid,
                   "--output", str(zip_path))
    entry = {"worker": row.get("workerAddress"),
             "submittedAt": row.get("submittedAt"),
             # Hashed here, from what is on disk at the moment it is scored, so
             # a rescored entry attests to the same bytes rather than a refetch.
             "sha256": hashlib.sha256(zip_path.read_bytes()).hexdigest()}
    try:
        scored = scorer.score_zip(zip_path, worlds=worlds, seeds=seeds,
                                  verbose=False)
    except (validator.Invalid, KeyError) as exc:
        return {**entry, "rejected": str(exc)}
    entry.update(
        architecture=scored["architecture"],
        mean_xp=round(float(scored["mean_xp"]), 4),
        # Indexed, not unpacked: train.competition_score has already grown a
        # fourth per-bank element once, and the leaderboard only wants two.
        per_seed={str(bank[0]): round(float(bank[1].mean()), 4)
                  for bank in scored["per_seed"]},
        truncated=int(scored["truncated"]),
        worlds=scored["worlds"],
        # Strings: a seed can exceed 2**53 and JSON readers with float
        # numbers (every JS one) would silently round it.
        seeds=[str(x) for x in scored["seeds"]],
    )
    return entry


def write_board(path: Path, board: dict) -> None:
    """Atomic, so a kill mid-write cannot lose the whole leaderboard."""
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(board, indent=2, sort_keys=True))
    tmp.replace(path)


def git_sha() -> str:
    """The evaluator commit, so a published board says what produced it.

    DM_EVALUATOR_SHA wins: a container runs from a copy of the tree with no
    .git and no git binary, so the sha has to be baked in at deploy time.
    """
    stamped = os.environ.get("DM_EVALUATOR_SHA")
    if stamped:
        return stamped
    try:
        proc = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True)
    except OSError:
        return "unknown"
    return proc.stdout.strip() or "unknown"


def write_markdown(path: Path, board: dict, seeds) -> None:
    """The board entrants read. It names the seeds and the evaluator commit, so
    anyone can re-run `just score-submission` and reproduce their own row.

    Only entries that passed completely appear. A rejection reason is an
    evaluator internal and is never published: it stays in leaderboard.json.
    A nonzero `truncated` is a failed run by the rules, so it is not a row
    either. An entrant checks their own entry with `just check-submission`.
    """
    rows = ["# Leaderboard", "",
            f"Public seeds `{','.join(map(str, seeds))}` | evaluator `{git_sha()}`",
            "",
            "Scored on the **public** seed group, which is published so you can",
            "reproduce your own row. The prize is decided on a **private** group,",
            "released with the final results. A high public score is not a win.",
            "",
            # No arch column: it is the entrant's business, and publishing
            # what the field leaders picked turns the board into a copy sheet.
            # Full sha, not a prefix: an entrant checks it against their own
            # `sha256sum submission.zip`, and a truncated one proves less.
            "| # | worker | mean XP | submitted | zip sha256 |",
            "|--:|---|--:|---|---|"]
    passed = [(sid, e) for sid, e in ranking(board)
              if "rejected" not in e and e.get("truncated") == 0]
    for rank, (sid, e) in enumerate(passed, 1):
        worker = e.get("worker") or "?"
        # Taskmarket sends an ISO-8601 UTC timestamp; minutes are enough here.
        when = (e.get("submittedAt") or "")[:16].replace("T", " ")
        rows.append(f"| {rank} | `{worker}` | {e['mean_xp']:.1f} | {when} "
                    f"| `{e.get('sha256', '?')}` |")
    path.write_text("\n".join(rows) + "\n")


def gist_request(gist_id: str, token: str, body: bytes | None = None):
    """One authenticated gist call. stdlib only -- no dependency for two."""
    import urllib.error
    import urllib.request

    req = urllib.request.Request(
        f"https://api.github.com/gists/{gist_id}", data=body,
        method="PATCH" if body else "GET",
        headers={"Authorization": f"Bearer {token}",
                 "Accept": "application/vnd.github+json",
                 "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read())
    except urllib.error.URLError as exc:
        raise RuntimeError(f"gist call failed: {exc}") from None


def fetch_gist_board(gist_id: str, token: str) -> dict:
    """The board as the gist holds it. The gist IS the state: no volume, and a
    fresh runner picks up where the last one stopped."""
    files = gist_request(gist_id, token).get("files", {})
    entry = files.get("leaderboard.json")
    return json.loads(entry["content"]) if entry and entry.get("content") else {}


def publish_gist(gist_id: str, text: str, token: str, board: dict | None = None) -> None:
    """PATCH the board into a gist. Raises RuntimeError; the caller keeps
    watching either way."""
    files = {"LEADERBOARD.md": {"content": text}}
    if board is not None:
        files["leaderboard.json"] = {"content": json.dumps(board, indent=2,
                                                           sort_keys=True)}
    body = json.dumps({"files": files}).encode()
    gist_request(gist_id, token, body)


def write_final(path: Path, board: dict, seeds, salt: str,
                public: dict | None = None) -> None:
    """The final board, published once with the seeds revealed.

    The reveal is the point: with the seeds and the salt anyone can recompute
    the commitment from the task brief, and re-run `just score-submission` on a
    published zip. Without them the ranking is only a claim.
    """
    seed_str = ",".join(map(str, seeds))
    digest = hashlib.sha256(f"{seed_str}|{salt}".encode()).hexdigest()
    rows = ["# Final results", "",
            f"Private seeds `{seed_str}` | salt `{salt}` | evaluator `{git_sha()}`",
            "",
            f"Commitment `{digest}` — recompute as `sha256(\"<seeds>|<salt>\")` "
            "and compare with the task brief. Set `DM_COMPETITION_SEEDS` to "
            "those seeds and `just score-submission` reproduces any row, within "
            "the cross-GPU margin.",
            "",
            # The public column is the story: a large drop is an entry fitted to
            # the published seeds rather than to the game.
            "| # | worker | mean XP | public XP | submitted | zip sha256 |",
            "|--:|---|--:|--:|---|---|"]
    passed = [(sid, e) for sid, e in ranking(board)
              if "rejected" not in e and e.get("truncated") == 0]
    for rank, (sid, e) in enumerate(passed, 1):
        pub = (public or {}).get(sid, {}).get("mean_xp")
        rows.append(
            f"| {rank} | `{e.get('worker') or '?'}` | {e['mean_xp']:.1f} "
            f"| {'—' if pub is None else format(pub, '.1f')} "
            f"| {(e.get('submittedAt') or '')[:16].replace('T', ' ')} "
            f"| `{e.get('sha256', '?')}` |")
    path.write_text("\n".join(rows) + "\n")


def publish_final(gist_id: str, board: dict, seeds, token: str,
                  public_board: Path, out: Path) -> None:
    """Write FINAL.md next to the public board in the same gist."""
    salt = os.environ.get("DM_SEED_SALT")
    if not salt:
        raise RuntimeError("--reveal needs DM_SEED_SALT: without the salt the "
                           "commitment cannot be checked")
    public = json.loads(public_board.read_text()) if public_board.exists() else {}
    write_final(out, board, seeds, salt, public)
    gist_request(gist_id, token, json.dumps(
        {"files": {"FINAL.md": {"content": out.read_text()}}}).encode())


def ranking(board: dict) -> list[tuple[str, dict]]:
    """Best first. Rejected and truncated entries sort last: a nonzero
    `truncated` means episodes hit the step cap instead of ending in death,
    which the rules require to be 0."""
    def key(item):
        entry = item[1]
        ok = "rejected" not in entry and entry.get("truncated") == 0
        return (not ok, -entry.get("mean_xp", 0.0))
    return sorted(board.items(), key=key)


def print_board(board: dict) -> None:
    print(f"\n{'#':>3}  {'worker':<24} {'mean XP':>9}  {'arch':<18} worker")
    for rank, (sid, entry) in enumerate(ranking(board), 1):
        if "rejected" in entry:
            print(f"{'--':>3}  {'REJECTED':<24} {'':>9}  {entry['rejected'][:60]}")
            continue
        flag = "" if entry["truncated"] == 0 else f"  !! truncated={entry['truncated']}"
        print(f"{rank:>3}  {(entry.get('worker') or '?')[:24]:<24} {entry['mean_xp']:>9.1f}  "
              f"{entry['architecture']:<18} {entry['worker']}{flag}")
    print()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("task_id")
    ap.add_argument("--interval", type=int, default=300,
                    help="seconds between polls (default 300)")
    ap.add_argument("--once", action="store_true", help="one pass, then exit")
    ap.add_argument("--worlds", type=int, default=None,
                    help="override the bank size; leave unset to score for real")
    ap.add_argument("--private", action="store_true",
                    help="final scoring on the private group from "
                         "DM_COMPETITION_SEEDS; default is the public group")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--entries", type=Path, default=ROOT / "local" / "entries")
    ap.add_argument("--markdown", type=Path, default=ROOT / "LEADERBOARD.md",
                    help="the board entrants read; not written with --private")
    ap.add_argument("--gist", default=os.environ.get("DM_LEADERBOARD_GIST"),
                    help="gist id to publish the board to; needs GITHUB_TOKEN")
    ap.add_argument("--reveal", action="store_true",
                    help="with --private: publish FINAL.md with the seeds and "
                         "salt revealed. Once, after the deadline.")
    args = ap.parse_args()

    import train
    if args.private:
        if "DM_COMPETITION_SEEDS" not in os.environ:
            print("--private needs DM_COMPETITION_SEEDS set to the private seeds.",
                  file=sys.stderr)
            return 2
        seeds = tuple(train.COMPETITION_SEEDS)
        if not args.reveal:
            args.gist = None      # nothing private is published without --reveal
    else:
        # Always the public group, whatever the environment says. A live board
        # on the private seeds would leak the test set one score at a time.
        seeds = tuple(train.PUBLIC_SEEDS)
        if tuple(train.COMPETITION_SEEDS) != seeds:
            print("note: DM_COMPETITION_SEEDS is set and ignored; the public "
                  "board always scores the public group.", file=sys.stderr)
    args.out = args.out or ROOT / "local" / (
        "leaderboard-private.json" if args.private else "leaderboard.json")

    args.entries.mkdir(parents=True, exist_ok=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    board = json.loads(args.out.read_text()) if args.out.exists() else {}

    while True:
        try:
            pending = new_rows(rows_of(taskmarket("task", "submissions", args.task_id)),
                               board)
            for row in pending:
                print(f"scoring {row['id']} from {row.get('workerAddress')}", flush=True)
                try:
                    board[row["id"]] = score_one(args.task_id, row, args.entries,
                                                 args.worlds, seeds)
                except Exception as exc:
                    # Unexpected (OOM, disk, a bug): leave the row unscored so
                    # the next pass retries it. Recording it as rejected would
                    # drop an entrant for our fault, and raising would stop the
                    # sentinel and quietly score nothing from here on.
                    print(f"score failed for {row['id']}, will retry: {exc}",
                          file=sys.stderr)
                    continue
                write_board(args.out, board)
                if args.private and args.reveal and args.gist:
                    try:
                        publish_final(args.gist, board, seeds,
                                      os.environ["GITHUB_TOKEN"],
                                      ROOT / "local" / "leaderboard.json",
                                      ROOT / "local" / "FINAL.md")
                    except (RuntimeError, KeyError) as exc:
                        print(f"reveal failed: {exc}", file=sys.stderr)
                if not args.private:
                    write_markdown(args.markdown, board, seeds)
                    if args.gist:
                        try:
                            publish_gist(args.gist, args.markdown.read_text(),
                                         os.environ["GITHUB_TOKEN"])
                        except (RuntimeError, KeyError) as exc:
                            # A publish failure must not lose a scored entry.
                            print(f"publish failed: {exc}", file=sys.stderr)
        except RuntimeError as exc:            # CLI/network hiccup: keep watching
            pending = []
            print(f"poll failed: {exc}", file=sys.stderr)
        if pending or args.once:
            print_board(board)
        if args.once:
            return 0
        time.sleep(args.interval)


def self_check() -> None:
    rows = [
        {"id": "a1", "workerAddress": "0x1", "rejectedAt": None},
        {"id": "b2", "workerAddress": "0x2", "rejectedAt": "2026-01-01"},
        {"id": "c3", "workerAddress": "0x3"},
        {"id": "../../etc/passwd", "workerAddress": "0x4"},
        {"id": "d4", "workerAddress": "0x5"},
    ]
    assert [r["id"] for r in new_rows(rows, {"d4": {}})] == ["a1", "c3"]
    assert rows_of([{"id": "x"}]) == [{"id": "x"}]
    assert rows_of({"submissions": [{"id": "x"}]}) == [{"id": "x"}]
    board = {
        "a": {"worker": "0x1", "mean_xp": 200.0, "truncated": 0,
              "architecture": "dm_lstm_v1", "sha256": "a" * 64},
        "b": {"worker": "0x2", "mean_xp": 300.0, "truncated": 4,   # truncated: last
              "architecture": "dm_mlp_v1", "sha256": "b" * 64},
        "c": {"worker": "0x3", "mean_xp": 250.0, "truncated": 0,
              "submittedAt": "2026-01-02T03:04:05.000Z",
              "architecture": "dm_lstm_v1", "sha256": "c" * 64},
        "d": {"rejected": "bad zip"},
    }
    assert [sid for sid, _ in ranking(board)] == ["c", "a", "b", "d"], ranking(board)

    import tempfile
    md = Path(tempfile.mkdtemp()) / "LEADERBOARD.md"
    write_markdown(md, board, (3930, 7717, 20477))
    text = md.read_text()
    assert "| 1 | `0x3` | 250.0" in text, text          # best first
    assert "2026-01-02 03:04" in text, text              # submitted, minutes
    assert "rejected" not in text, text                  # never published
    assert "truncated" not in text and "0x2" not in text, text   # excluded
    assert "3930,7717,20477" in text, text               # reproducible
    assert "c" * 64 in text, text                        # full zip sha

    # The final board: reveal present, public column carried across, and the
    # commitment recomputable from what it prints.
    final = md.parent / "FINAL.md"
    write_final(final, board, (11, 22, 33), "s" * 64,
                public={"c": {"mean_xp": 240.0}})
    ftext = final.read_text()
    assert "11,22,33" in ftext and "s" * 64 in ftext, ftext
    want = hashlib.sha256(f"11,22,33|{'s' * 64}".encode()).hexdigest()
    assert want in ftext, ftext
    assert "| 1 | `0x3` | 250.0 | 240.0 " in ftext, ftext   # private then public
    assert "| — |" in ftext, ftext                          # no public row
    assert "rejected" not in ftext, ftext
    print("self-check OK")


if __name__ == "__main__":
    if "--self-check" in sys.argv:
        self_check()
    else:
        raise SystemExit(main())
