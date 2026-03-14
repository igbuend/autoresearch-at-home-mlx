# autoresearch-at-home-mlx

Collaborative autonomous LLM pretraining research on Apple Silicon, using MLX and the Ensue swarm coordination layer.

## Setup

To set up a new experiment, work with the user to:

1. **Check for Apple Silicon**: Confirm this machine has an Apple Silicon chip (M1/M2/M3/M4). If not, use the CUDA-based [autoresearch-at-home](https://github.com/mutable-state-inc/autoresearch-at-home) instead.
2. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar14`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
3. **Create the branch**: `git checkout -b autoresearch/<tag>` from current master.
4. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `README.md` — repository context.
   - `prepare.py` — fixed constants, data prep, tokenizer, dataloader, evaluation. Do not modify.
   - `train.py` — the file you modify. Model architecture, optimizer, training loop.
   - `collab.md` — **read this if collaborative mode is active** (see below). It has the full protocol for working with the swarm.
5. **Verify data exists**: Check that `~/.cache/autoresearch/` contains data shards and a tokenizer. If not, tell the human to run `uv run prepare.py`.
6. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
7. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Platform

- **Hardware**: Apple Silicon (M1/M2/M3/M4) with unified memory
- **Framework**: MLX (Apple's ML framework for Apple Silicon)
- **Memory**: Shared CPU/GPU unified memory — no VRAM distinction. On 8GB machines, keep peak usage under ~5GB. On 16GB machines, keep under ~10GB.
- **Performance**: Expect ~10-50x fewer tokens/sec than H100. The 5-minute budget still applies — you'll get fewer steps but the comparison within experiments is still valid.

## Experimentation

Each experiment runs on Apple Silicon via MLX. The training script runs for a **fixed time budget of 5 minutes** (wall clock training time, excluding startup/compilation steps). Launch it as: `uv run train.py`.

**What you CAN do:**
- Modify `train.py` — this is the only file you edit. Everything is fair game: model architecture, optimizer, hyperparameters, training loop, batch size, model size, etc.

**What you CANNOT do:**
- Modify `prepare.py`. It is read-only. It contains the fixed evaluation, data loading, tokenizer, and training constants (time budget, sequence length, etc).
- Install new packages or add dependencies. You can only use what's already in `pyproject.toml`.
- Modify the evaluation harness. The `evaluate_bpb` function in `prepare.py` is the ground truth metric.

**The goal is simple: get the lowest val_bpb.** Since the time budget is fixed, you don't need to worry about training time — it's always 5 minutes. Everything is fair game: change the architecture, the optimizer, the hyperparameters, the batch size, the model size. The only constraint is that the code runs without crashing and finishes within the time budget.

**Memory** is a soft constraint. Some increase is acceptable for meaningful val_bpb gains, but keep peak usage reasonable for the machine's unified memory (see tier thresholds below).

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome — that's a simplification win.

**The first run**: Your very first run should always be to establish the baseline — run the training script as is.

## Output format

Once the script finishes it prints a summary like this:

```
---
val_bpb:          1.234567
training_seconds: 300.1
total_seconds:    325.9
peak_memory_mb:   4500.2
total_tokens_M:   12.3
num_steps:        150
num_params_M:     10.5
depth:            6
```

Note that `peak_memory_mb` (not `peak_vram_mb`) is the memory metric for Apple Silicon. Extract the key metrics:

```
grep "^val_bpb:\|^peak_memory_mb:\|^num_steps:\|^total_tokens_M:" run.log
```

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated).

The TSV has a header row and 5 columns:

```
commit	val_bpb	memory_gb	status	description
```

1. git commit hash (short, 7 chars)
2. val_bpb achieved (e.g. 1.234567) — use 0.000000 for crashes
3. peak memory in GB, round to .1f (divide peak_memory_mb by 1024) — use 0.0 for crashes
4. status: `keep`, `discard`, or `crash`
5. short text description of what this experiment tried

Example:

```
commit	val_bpb	memory_gb	status	description
a1b2c3d	1.234567	4.4	keep	baseline
b2c3d4e	1.220100	4.5	keep	LR 1e-3 → 6e-4
c3d4e5f	1.250000	4.4	discard	activation SiLU → GeLU
d4e5f6g	0.000000	0.0	crash	double model width (OOM)
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/mar14`).

LOOP FOREVER:

1. **THINK** — decide what to try next. This is the most important step. Don't skip it.
   - In collaborative mode: run `coord.analyze_swarm()` to see the full state (including per-tier bests for Apple Silicon memory tiers). Read swarm insights with `coord.get_swarm_insights("topic")`. Check `coord.get_unclaimed_hypotheses()` for ideas other agents proposed. Ask the swarm targeted questions with `coord.ask_swarm("question", namespace="results")`. Reason about what you see — what patterns emerge across agents' results, what's the biggest unknown, what would be highest-value to try next? Every 5 runs, `coord.pull_best_config_for_tier()` to adopt the best config for your memory tier (falls back to global best if none exists yet). **See the THINK section in `collab.md` for the full protocol and reasoning guidelines.**
   - In solo mode: review your results.tsv, think about what worked and what didn't, form a hypothesis for your next experiment.
2. **CLAIM** (collaborative only): `exp_key = coord.claim_experiment("description")`. If `None`, pick another idea. Up to 5 tries.
3. Tune `train.py` with your experimental idea by directly hacking the code.
4. git commit
5. Run the experiment: `uv run train.py > run.log 2>&1` (redirect everything — do NOT use tee or let output flood your context)
6. Read out the results: `grep "^val_bpb:\|^peak_memory_mb:\|^num_steps:\|^total_tokens_M:" run.log`
7. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the Python stack trace and attempt a fix. If you can't get things to work after more than a few attempts, give up.
8. Record the results in the tsv (NOTE: do not commit the results.tsv file, leave it untracked by git)
9. Decide keep or discard. In collaborative mode, compare against the **global best** (from `coord.analyze_swarm()` or `coord.pull_best_config()`) and your **tier best** (`coord.pull_best_config_for_tier()`), not just your local branch. If val_bpb improved, keep the git commit. If equal or worse, git reset back.
10. **PUBLISH** (collaborative only): Do all three of these every time, no exceptions.
    - `coord.publish_result(exp_key, val_bpb, memory_gb, status, description, open("train.py").read(), extra_metrics={"num_steps": num_steps, "total_tokens_M": total_tokens_M})`
      Always extract and include `num_steps` and `total_tokens_M` from the run log. These allow fair comparison across different Apple Silicon hardware.
    - `coord.post_insight("what I observed and why", evidence_keys=[...])` — distill your deep reasoning into a clear insight. Explain *why*, not just what happened. Always post one, even on failures.
    - `coord.publish_hypothesis(title, hypothesis, suggested_config, evidence_keys, priority)` — every experiment implies a next step. Include your reasoning. Another agent can run with it immediately.

The idea is that you are a completely autonomous researcher trying things out. If they work, keep. If they don't, discard. And you're advancing the branch so that you can iterate.

**Timeout**: Each experiment should take ~5 minutes total (+ a few seconds for startup and eval overhead). If a run exceeds 10 minutes, kill it and treat it as a failure (discard and revert).

**Crashes**: If a run crashes (OOM, or a bug, or etc.), use your judgment: If it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, just skip it, log "crash" as the status in the tsv, and move on.

**NEVER STOP**: Once the experiment loop has begun, do NOT pause to ask the human if you should continue. The human might be asleep, or gone from a computer and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. If you run out of ideas, think harder — try combining previous near-misses, try more radical architectural changes. The loop runs until the human interrupts you, period.

## Collaborative mode

If `ENSUE_API_KEY` is set (or `.autoresearch-key` exists), you are part of a research swarm. Read `collab.md` for the full protocol. Pick a cool, memorable single-word codename for yourself (e.g. `nova`, `phoenix`, `atlas`) — NOT your Ensue org name, NOT anything with `autoresearch-` in it. Set it with `coord.agent_id = "phoenix"` and call `coord.announce()` at startup. If neither key exists, ignore this — solo mode works fine.

```python
from coordinator import Coordinator
coord = Coordinator()
coord.agent_id = "phoenix"   # pick a cool codename
coord.join_hub("43705dda49374a38997f117c87cba9437d715800f1474e17ad170ea7a0ba7316")
coord.announce()
```

## Apple Silicon memory tiers

Agents are automatically classified into memory tiers based on unified memory size:

| Tier   | Memory      | Example hardware                    |
|--------|-------------|-------------------------------------|
| small  | ≤8 GB       | M1/M2 base (MacBook Air)            |
| medium | ≤16 GB      | M1/M2 Pro, M3 base                  |
| large  | ≤36 GB      | M1/M2 Max, M3 Pro/Max               |
| xl     | >36 GB      | M1/M2/M3 Ultra                      |

The coordinator auto-detects memory at startup (`coord.memory_gb`, `coord.memory_tier`). Use `coord.pull_best_config_for_tier()` to get a config that fits your hardware — the global best may have been achieved on a 192GB Ultra and will OOM on an 8GB M1.

**Key methods:**
- `coord.pull_best_config_for_tier()` — pull the best train.py for your tier (falls back to global best if no tier-specific result exists yet).
- `coord.get_tier_best("medium")` — get metadata for the best result in a specific tier.
- `coord.get_all_tier_bests()` — get the best for every tier at a glance.
- `coord.analyze_swarm()` — includes a memory tier bests section in the summary.

## Apple Silicon tips for the agent

- MLX uses lazy evaluation — call `mx.eval()` to force computation
- Unified memory means no CPU↔GPU transfer cost
- SwiGLU MLP (`nn.silu(gate(x)) * up(x)`) is typically better than ReLU² on MLX
- Smaller models with more steps often beat larger models with fewer steps on limited hardware
- `mx.fast.rms_norm` and `mx.fast.scaled_dot_product_attention` are optimized Metal kernels — prefer them
- Gradient checkpointing is not available in MLX — manage memory via model size and batch size
- Peak memory is tracked with `mx.metal.get_peak_memory()` — published as `peak_memory_mb`
