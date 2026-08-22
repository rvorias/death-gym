"""Scoring and the public leaderboard, on Modal. Nothing else.

The wallet, Taskmarket intake, acceptance and the private seeds stay on the
operator's machine. This app only ever sees a submission zip and the public
seed group, so a compromise here cannot move money or leak the test set.

    modal deploy tools/modal_app.py
    modal run tools/modal_app.py --zip submission.zip --sid s1 --gist <id>

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


@app.local_entrypoint()
def main(zip: str, sid: str, gist: str, worker: str = "?", submitted_at: str = ""):
    from pathlib import Path

    entry = score.remote(Path(zip).read_bytes(), sid, gist, worker, submitted_at)
    print(entry.get("rejected") or f"{entry['mean_xp']:.1f} XP  {entry['sha256'][:12]}")
