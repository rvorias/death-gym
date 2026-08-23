"""Scoring and the public leaderboard, on Modal. Nothing else.

The wallet, acceptance and the private seeds stay on the operator's machine.
This app sees submission zips, the public seed group, and a read-only Taskmarket
credential, so a compromise here cannot move money or leak the test set.

Intake runs here too, but without the key. `taskmarket:read:<address>` is a
static message with no nonce -- the API says so explicitly -- so the operator
signs it once locally and stores only the signature. It authorises gated reads
and nothing else: it cannot accept, reject, pay, or submit.

    modal deploy tools/modal_app.py
    modal run tools/modal_app.py --zip submission.zip --sid s1 --gist <id>
    modal run tools/modal_app.py::poll        # one intake pass, on demand

The gist is the state: the board is read from it and written back, so there is
no volume and a cold runner picks up exactly where the last one stopped.

One GPU type is pinned for the whole competition. That is not tidiness: the
same submission scores 167.6 on an A5000 and 167.9 on an RTX 4000 Ada, because
cuBLAS picks reduction orders per device. Entries scored on different cards are
not comparable, so GPU is part of the protocol.
"""
from __future__ import annotations

import modal

GPU = "A10G"                      # pinned; changing it invalidates the board
REPO = "/repo"

def _local_sha() -> str:
    """Stamped into the image so the published board names its evaluator."""
    import subprocess
    proc = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                          capture_output=True, text=True)
    return proc.stdout.strip() or "unknown"


image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("build-essential", "make", "git")
    .pip_install("numpy>=1.26", "safetensors>=0.4")
    .pip_install("torch==2.8.0", index_url="https://download.pytorch.org/whl/cu128")
    .add_local_dir(
        ".", REPO, copy=True,
        ignore=["local", ".venv", ".git", "**/__pycache__", "engine/build"],
    )
    .run_commands(f"make -C {REPO}/engine")
    .env({"PYTHONPATH": REPO, "DM_EVALUATOR_SHA": _local_sha()})
)

app = modal.App("death-mountain-scoring", image=image)
github = modal.Secret.from_name("dm-github-token")   # GITHUB_TOKEN, gist scope
# TM_CALLER_ADDRESS + TM_CALLER_SIGNATURE: read-only, see the module docstring.
taskmarket = modal.Secret.from_name("dm-taskmarket-read")

API = "https://api.taskmarket.dev"
TASK_ID = "0xace815c521a866aee6b474ed379160e73a933552b01c990b36b8937b88f3295a"
GIST_ID = "545d0b413e31b315a017157339adca9e"

MAX_TRIES = 3          # attempts before an entry is quarantined

# Every submission id ever handled, passed or rejected. True means finished
# (scored or rejected); an int is how many times it has errored unexpectedly. Rejections are NOT in
# the public board by design, so without this the poller would re-download and
# re-score every failing entry on every pass, forever.
seen = modal.Dict.from_name("dm-scored-ids", create_if_missing=True)


@app.function(gpu=GPU, secrets=[github], timeout=900)
def score(zip_bytes: bytes, sid: str, gist_id: str, worker: str = "?",
          submitted_at: str = "") -> dict:
    """Score one submission on the public seeds and republish the board.

    Returns the leaderboard entry. A rejection is a result, not a crash: it is
    recorded for the operator and never published.
    """
    import hashlib
    import os
    import sys
    import tempfile
    from pathlib import Path

    sys.path.insert(0, REPO)
    import tools.evaluate_submission as scorer
    import tools.validate_submission as validator
    import tools.watch_submissions as board_io
    import train

    token = os.environ["GITHUB_TOKEN"]
    work = Path(tempfile.mkdtemp())
    zip_path = work / f"{sid}.zip"
    zip_path.write_bytes(zip_bytes)

    entry = {"worker": worker, "submittedAt": submitted_at,
             "sha256": hashlib.sha256(zip_bytes).hexdigest()}
    try:
        scored = scorer.score_zip(zip_path, seeds=train.PUBLIC_SEEDS, verbose=False)
    except (validator.Invalid, KeyError) as exc:
        entry["rejected"] = str(exc)
    else:
        entry.update(
            architecture=scored["architecture"],
            mean_xp=round(float(scored["mean_xp"]), 4),
            per_seed={str(b[0]): round(float(b[1].mean()), 4)
                      for b in scored["per_seed"]},
            truncated=int(scored["truncated"]),
            worlds=scored["worlds"],
            seeds=[str(x) for x in scored["seeds"]],
        )

    board = board_io.fetch_gist_board(gist_id, token)
    # Only passing entries are kept: the gist is world-readable by URL, and a
    # rejection reason is an evaluator internal. The local sentinel keeps its
    # own record of what was rejected and why.
    if "rejected" not in entry:
        board[sid] = entry
        md = work / "LEADERBOARD.md"
        board_io.write_markdown(md, board, train.PUBLIC_SEEDS)
        board_io.publish_gist(gist_id, md.read_text(), token, board=board)
    return entry


def _api(path: str, method: str = "GET") -> object:
    """One Taskmarket read, carrying the operator's read-only proof.

    The headers are optional to the API: omit them and the request silently
    falls back to the anonymous view, which under `winner_only` is an empty
    list. A missing signature therefore looks exactly like "no entries yet",
    which is why the caller checks the credential rather than trusting a 200.
    """
    import json
    import os
    import urllib.request

    req = urllib.request.Request(
        f"{API}{path}", method=method,
        headers={"X-Taskmarket-Caller-Address": os.environ["TM_CALLER_ADDRESS"],
                 "X-Taskmarket-Caller-Signature": os.environ["TM_CALLER_SIGNATURE"],
                 "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


# max_containers=1: poll() must never overlap itself. An id is marked seen only
# after score() returns, so two concurrent passes would both pick up the same
# unscored entries and pay for them twice.
# ponytail: scoring is sequential, so a big backlog drains at ~3 min/entry.
# If that is ever too slow, shard by submission id across parallel pollers and
# give each its own gist file, or move the board off the gist into a Dict.
@app.function(schedule=modal.Cron("*/30 * * * *"), max_containers=1,
              secrets=[github, taskmarket], timeout=6 * 3600)
def poll() -> dict:
    """Score every submission that has not been scored yet, then republish.

    No GPU on this function: it downloads and dispatches, and the GPU is taken
    by score() one entry at a time. Sequential on purpose -- score() does a
    read-modify-write of the gist, so scoring in parallel would let two
    containers publish boards that each omit the other's row.
    """
    import urllib.error
    import urllib.request

    rows = _api(f"/api/tasks/{TASK_ID}/submissions")
    if not isinstance(rows, list):
        return {"error": f"unexpected submissions payload: {type(rows).__name__}"}

    # A wrong or expired credential is not an error to this API: it answers 200
    # with [], exactly like a task nobody has entered. submissionCount is public
    # and needs no credential, so it is the one honest cross-check available --
    # without it a dead poller reports "0 new" forever and looks healthy.
    public = _api(f"/api/tasks/{TASK_ID}")
    declared = (public.get("data") or public).get("submissionCount") or 0
    if declared and not rows:
        raise RuntimeError(
            f"read credential is not working: the task declares {declared} "
            "submissions but the gated read returned none. Re-sign "
            "taskmarket:read:<lowercase requester address> and update the "
            "dm-taskmarket-read secret.")

    pending = [r for r in rows
               if r.get("id") and seen.get(r["id"]) is not True
               and not r.get("rejectedAt")]
    print(f"{len(rows)} submissions, {len(pending)} new", flush=True)

    scored, failed, errored, no_zip = 0, 0, 0, []
    for row in pending:
        sid = row["id"]
        # The artifact route, not POST submissions/{id}/preview: that one wants
        # a deviceId and apiToken the CLI holds, which a read signature is not.
        # This one is gated purely on submissionVisibility, so it answers us.
        art = next((a for a in (row.get("artifacts") or [])
                    if a.get("mimeType") == "application/zip"
                    or str(a.get("fileName", "")).endswith(".zip")), None)
        if art is None:
            # Benchmark mode accepts a proof with no artifact. That entry is a
            # real delivery by the platform's rules and unscoreable by ours, so
            # it is reported rather than silently dropped. Not marked seen: an
            # artifact appearing later should still get picked up.
            no_zip.append(sid)
            continue
        try:
            url = _api(f"/api/tasks/{TASK_ID}/artifacts/{art['id']}/preview")["previewUrl"]
            with urllib.request.urlopen(url, timeout=300) as resp:
                zip_bytes = resp.read()
        except (urllib.error.URLError, KeyError, TypeError) as exc:
            # A presign expires in an hour. Leave it unseen so the next pass
            # retries rather than dropping a paying entrant on a transient fail.
            print(f"fetch failed for {sid}: {exc}", flush=True)
            failed += 1
            continue

        try:
            entry = score.remote(zip_bytes, sid, GIST_ID,
                                 row.get("workerAddress") or "?",
                                 row.get("submittedAt") or "")
        except Exception as exc:
            # score() turns every malformed submission into a recorded
            # rejection, so reaching here means infrastructure (OOM, a CUDA
            # fault, a dead container) or an input nobody anticipated. Bounded
            # retries, then quarantine: unbounded, one poison entry aborts
            # every pass at the same row and nothing behind it is ever scored.
            tries = (seen.get(sid) or 0) + 1
            seen[sid] = True if tries >= MAX_TRIES else tries
            print(f"score errored for {sid} (attempt {tries}/{MAX_TRIES}): {exc}",
                  flush=True)
            if tries >= MAX_TRIES:
                print(f"QUARANTINED, needs a human: {sid}", flush=True)
            errored += 1
            continue
        seen[sid] = True          # only after score() returned: a crash retries
        scored += 1
        print(f"{sid} {entry.get('rejected') or format(entry['mean_xp'], '.1f') + ' XP'}",
              flush=True)

    if no_zip:
        print(f"UNSCOREABLE, no zip artifact: {len(no_zip)} -> {no_zip}", flush=True)
    return {"submissions": len(rows), "new": len(pending), "scored": scored,
            "fetch_failed": failed, "errored": errored,
            "no_zip_artifact": no_zip}


@app.local_entrypoint()
def main(zip: str, sid: str, gist: str, worker: str = "?", submitted_at: str = ""):
    from pathlib import Path

    entry = score.remote(Path(zip).read_bytes(), sid, gist, worker, submitted_at)
    print(entry.get("rejected") or f"{entry['mean_xp']:.1f} XP  {entry['sha256'][:12]}")
