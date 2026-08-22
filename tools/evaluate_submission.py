#!/usr/bin/env python3
"""Score a submission zip on the competition bank.

    python tools/evaluate_submission.py submission.zip

This is the evaluator's own path, end to end: validate the archive, build the
whitelisted architecture, load the weights, and score. It exists so the number
you see is the number the evaluator gets — scoring the checkpoint a submission
came from proves nothing about the submission.

Requires the native engine (`just build`).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import tools.validate_submission as validator  # noqa: E402


def score_zip(path: Path, worlds: int | None = None, progress: bool = False,
              verbose: bool = True, seeds=None) -> dict:
    """Validate, load, and score one submission zip. Raises validator.Invalid.

    The single scoring path: the CLI below and tools/watch_submissions.py both
    go through here, so a watched score and a hand-run score cannot drift.
    """
    result = validator.validate(path, verbose=verbose)
    model = validator.load_into_model(result)

    import train
    device = train.select_device()
    model = model.to(device)
    worlds = worlds or train.COMPETITION_WORLDS
    seeds = tuple(seeds) if seeds else train.COMPETITION_SEEDS
    if verbose:
        print(f"device            {device}")
        print(f"scoring           {len(seeds)} banks of {worlds} "
              f"worlds, seeds {','.join(map(str, seeds))}, "
              f"batch {train.COMPETITION_BATCH}")
    per_seed, mean_xp, truncated = train.competition_score(
        model, device, seeds=seeds, worlds=worlds, progress=progress)
    return {
            "architecture": result["config"]["architecture"],
            "per_seed": per_seed, "mean_xp": mean_xp, "truncated": truncated,
            "worlds": worlds, "seeds": list(seeds)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("submission", type=Path)
    ap.add_argument("--worlds", type=int, default=None,
                    help="override the bank size (default: the competition bank)")
    ap.add_argument("--quiet", action="store_true", help="suppress per-chunk progress")
    args = ap.parse_args()

    try:
        scored = score_zip(args.submission, worlds=args.worlds,
                           progress=not args.quiet)
    except validator.Invalid as exc:
        print(f"REJECTED  {args.submission}: {exc}", file=sys.stderr)
        return 1

    import train
    print("---")
    train.print_competition_result(scored["per_seed"], scored["mean_xp"],
                                   scored["truncated"], train.COMPETITION_BATCH)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
