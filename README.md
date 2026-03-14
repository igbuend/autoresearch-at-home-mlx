# autoresearch-at-home-mlx

Collaborative autonomous LLM pretraining research for Apple Silicon, using [MLX](https://github.com/ml-explore/mlx) and the [Ensue](https://ensue-network.ai) swarm coordination layer.

## What is this?

This project combines two things:

- **[autoresearch-at-home](https://github.com/mutable-state-inc/autoresearch-at-home)** — a SETI@home-style collaborative fork of autoresearch where multiple agents running on different machines share results, avoid redundant work, and collectively drive down val_bpb through a shared Ensue workspace.
- **[autoresearch-mlx](https://github.com/ElixirLabsUK/autoresearch-mlx)** — an Apple Silicon port of autoresearch using Apple's MLX framework instead of PyTorch/CUDA.

This repo is the intersection: **collaborative swarm research that runs on your Mac**.

The agent modifies `train.py`, claims experiments through Ensue to avoid duplicates, trains for 5 minutes via MLX, publishes results and insights to the swarm, and loops forever.

## Setup

### 1. Install dependencies

```bash
uv sync
```

### 2. Prepare data and tokenizer

```bash
uv run prepare.py
```

This downloads training data from Hugging Face and trains a BPE tokenizer. Data is cached in `~/.cache/autoresearch/`.

### 3. Run a training experiment

```bash
uv run train.py
```

### 4. Enable collaborative mode (optional)

```bash
# Register your agent with Ensue
curl -sf -X POST https://api.ensue-network.ai/auth/agent-register \
  -H "Content-Type: application/json" \
  -d '{"name": "autoresearch-<your-name>"}'

# Save the api_key from the response
echo "lmn_..." > .autoresearch-key
```

Then point an AI agent at `program.md` to run the experiment loop.

## Project structure

```
train.py        — MLX model, optimizer, training loop (agent modifies this)
prepare.py      — data download, tokenizer, dataloader, evaluation (read-only)
coordinator.py  — Ensue integration for the research swarm (Apple Silicon version)
program.md      — agent instructions for the experiment loop
collab.md       — collaborative mode protocol
setup_hub.py    — one-time hub org setup script
pyproject.toml  — dependencies
```

## Upstream repos

This project tracks two upstream repositories with orthogonal concerns:

| Remote | URL | Owns |
|--------|-----|------|
| `upstream-ath` | `https://github.com/mutable-state-inc/autoresearch-at-home.git` | `coordinator.py`, `collab.md`, `program.md`, `setup_hub.py` |
| `upstream-mlx` | `https://github.com/ElixirLabsUK/autoresearch-mlx.git` | `train.py`, `prepare.py`, `pyproject.toml` |

### Merging from upstream-ath (coordination layer changes)

```bash
git fetch upstream-ath
git checkout -p upstream-ath/master -- coordinator.py collab.md setup_hub.py
# Then manually review program.md diffs and adapt
# (upstream-ath program.md references CUDA/VRAM — translate to MLX/memory terms)
```

### Merging from upstream-mlx (MLX training changes)

```bash
git fetch upstream-mlx
git checkout -p upstream-mlx/autoresearch/mar12 -- train.py prepare.py
# Then manually review pyproject.toml diffs and adapt
# (add requests back if upstream-mlx drops it — coordinator.py requires it)
```

The two upstreams are orthogonal: upstream-ath never touches train.py/prepare.py, and upstream-mlx never touches coordinator.py/collab.md. Conflicts are unlikely. When they do occur, they will be in program.md or pyproject.toml, and are straightforward to resolve manually.

## Apple Silicon memory tiers

The coordinator auto-detects unified memory and classifies agents into tiers:

| Tier   | Memory  | Example hardware          |
|--------|---------|---------------------------|
| small  | ≤8 GB   | M1/M2 base                |
| medium | ≤16 GB  | M1/M2 Pro, M3 base        |
| large  | ≤36 GB  | M1/M2 Max, M3 Pro/Max     |
| xl     | >36 GB  | M1/M2/M3 Ultra            |

Use `coord.pull_best_config_for_tier()` to pull a config that fits your hardware.

## License

MIT
