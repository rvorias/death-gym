#           Death Mountain RL — task runner
#
#   just              list every recipe
#   just build        compile the native engine
#   just test         run the engine test suite
#   just train        start a training run
#   just tui          live training dashboard
#
# Every recipe uses one interpreter. It uses the project .venv if that
# directory exists. If not, it uses the `python3` on your PATH.

root := justfile_directory()
py   := if path_exists(root / ".venv/bin/python") == "true" {
            root / ".venv/bin/python"
        } else { "python3" }

export PYTHONPATH := root

_default:
    @just --list --unsorted

# ─── Engine ─────────────────────────────────────────────────────────────

# Compile the native engine to engine/build/libdmfast.so
build:
    make -C {{root}}/engine

# The strict check: newest gcc here, warnings fatal. The default build stays
# portable; this is what stops us shipping code another compiler rejects.
# Restores the normal build afterwards.
build-strict:
    #!/usr/bin/env bash
    set -euo pipefail
    cc="$(ls /usr/bin/gcc-1? 2>/dev/null | sort -V | tail -1 || echo gcc)"
    echo "building with $cc, warnings fatal"
    make -C {{root}}/engine clean
    make -C {{root}}/engine CC="$cc" \
        CFLAGS="-O3 -fPIC -std=c11 -Wall -Wextra -Werror -march=native"
    make -C {{root}}/engine clean
    make -C {{root}}/engine

# Remove all engine build artifacts
clean:
    make -C {{root}}/engine clean

# Engine, reward, checkpoint, and environment tests
test: build
    {{py}} -m pytest {{root}}/tests -q

# Measure engine throughput in env-steps/s (4096 parallel envs)
bench: build
    #!/usr/bin/env bash
    set -euo pipefail
    script="$(mktemp --suffix=.py)"
    trap 'rm -f "$script"' EXIT
    cat > "$script" <<'PY'
    import time, numpy as np
    from dmfast.env import BatchEnv
    N, STEPS = 4096, 300
    env = BatchEnv(num_envs=N, seed=1, max_steps=512)
    env.reset(seed=1)
    rng = np.random.default_rng(0)
    for _ in range(50):                       # warm up
        env.step(env.sample_masked_actions(rng))
    best = 0.0
    for _ in range(5):
        t0 = time.perf_counter()
        for _ in range(STEPS):
            env.step(env.sample_masked_actions(rng))
        best = max(best, N * STEPS / (time.perf_counter() - t0))
    print(f"{best/1e6:.2f}M env-steps/s")
    PY
    {{py}} "$script"

# ─── Training ───────────────────────────────────────────────────────────

#   just train                        # defaults
#   just train --total-steps 50000000 # anything train.py accepts
# Train a policy. Write the log to local/logs/run.log for `just tui`
train *ARGS: build
    #!/usr/bin/env bash
    set -euo pipefail
    mkdir -p {{root}}/local/logs
    {{py}} {{root}}/train.py {{ARGS}} 2>&1 | tee {{root}}/local/logs/${DM_LOG:-run}.log

#   just eval local/checkpoints/my-run/final.safetensors
#   DM_EVAL_SEEDS=1,2,3 just eval ...   # score on other worlds
# The full eval, not the cheap one the trainer runs per checkpoint.
# Score a checkpoint on the standard banks: 3 x 16384 worlds
eval CKPT *ARGS: build
    {{py}} {{root}}/train.py --full-eval --resume {{CKPT}} {{ARGS}}

# ─── Dashboard ──────────────────────────────────────────────────────────

# Show the live training dashboard (1-9 focus, j/k move, q quit)
tui *ARGS:
    {{py}} {{root}}/tui.py {{ARGS}}
