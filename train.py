"""
Autoresearch pretraining script for Apple Silicon. Single-file, MLX.
Usage: uv run train.py
"""

import math
import time
from dataclasses import dataclass, asdict

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_map

from prepare import MAX_SEQ_LEN, TIME_BUDGET, Tokenizer, make_dataloader, evaluate_bpb

# ---------------------------------------------------------------------------
# GPT Model
# ---------------------------------------------------------------------------

@dataclass
class GPTConfig:
    sequence_len: int = 2048
    vocab_size: int = 32768
    n_layer: int = 8
    n_head: int = 6
    n_kv_head: int = 6
    n_embd: int = 384


class RMSNorm(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.weight = mx.ones((dims,))

    def __call__(self, x):
        return mx.fast.rms_norm(x, self.weight, eps=1e-6)


class RotaryEmbedding:
    def __init__(self, head_dim, max_seq_len=8192, base=10000.0):
        inv_freq = 1.0 / (base ** (mx.arange(0, head_dim, 2).astype(mx.float32) / head_dim))
        t = mx.arange(max_seq_len).astype(mx.float32)
        freqs = mx.outer(t, inv_freq)
        self.cos = mx.cos(freqs)
        self.sin = mx.sin(freqs)
        mx.eval(self.cos, self.sin)

    def __call__(self, x, offset=0):
        T = x.shape[1]
        # cos/sin: (T, D/2) -> (1, T, 1, D/2) for broadcasting with (B, T, H, D/2)
        cos = self.cos[offset:offset + T][None, :, None, :]
        sin = self.sin[offset:offset + T][None, :, None, :]
        # x: (B, T, H, D)
        d = x.shape[-1] // 2
        x1, x2 = x[..., :d], x[..., d:]
        y1 = x1 * cos + x2 * sin
        y2 = x1 * (-sin) + x2 * cos
        return mx.concatenate([y1, y2], axis=-1)


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0

        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)

        self.scale = self.head_dim ** -0.5

    def __call__(self, x, rotary):
        B, T, C = x.shape
        q = self.c_q(x).reshape(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).reshape(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).reshape(B, T, self.n_kv_head, self.head_dim)

        # Apply rotary embeddings
        q = rotary(q)
        k = rotary(k)

        # Transpose for attention: (B, N, T, D) — GQA handled natively
        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        # Fast scaled dot-product attention (optimized Metal kernel, handles GQA + causal)
        y = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask="causal")

        y = y.transpose(0, 2, 1, 3).reshape(B, T, -1)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden = 4 * config.n_embd
        self.gate = nn.Linear(config.n_embd, hidden, bias=False)
        self.up = nn.Linear(config.n_embd, hidden, bias=False)
        self.down = nn.Linear(hidden, config.n_embd, bias=False)

    def __call__(self, x):
        return self.down(nn.silu(self.gate(x)) * self.up(x))


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = RMSNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln2 = RMSNorm(config.n_embd)
        self.mlp = MLP(config)

    def __call__(self, x, rotary):
        x = x + self.attn(self.ln1(x), rotary)
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.blocks = [Block(config) for _ in range(config.n_layer)]
        self.ln_f = RMSNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.rotary = RotaryEmbedding(config.n_embd // config.n_head, config.sequence_len * 2)

    def __call__(self, idx, targets=None):
        x = self.wte(idx)
        for block in self.blocks:
            x = block(x, self.rotary)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits

    def num_params(self):
        from mlx.utils import tree_flatten
        leaves = tree_flatten(self.parameters())
        return sum(p.size for _, p in leaves)

# ---------------------------------------------------------------------------
# Hyperparameters (edit these directly, no CLI flags needed)
# ---------------------------------------------------------------------------

# Model architecture
DEPTH = 2               # number of transformer layers
N_HEAD = 4              # attention heads
N_KV_HEAD = 4           # key/value heads (set < N_HEAD for GQA)
N_EMBD = 128            # embedding dimension

# Optimization
BATCH_SIZE = 2           # smaller batch for more gradient steps
MATRIX_LR = 0.032        # Muon learning rate for weight matrices
EMBED_LR = 0.3           # Adam learning rate for embeddings
SCALAR_LR = 0.3          # Adam learning rate for scalars (norms)
WEIGHT_DECAY = 0.1       # AdamW weight decay for embeddings
MUON_MOMENTUM = 0.95     # Muon momentum
WARMUP_RATIO = 0.0       # no warmup (Muon converges from step 0)
WARMDOWN_RATIO = 1.0     # linear cooldown throughout training
FINAL_LR_FRAC = 0.01     # final LR as fraction of peak
GRAD_ACCUM_STEPS = 1

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

mx.random.seed(42)

tokenizer = Tokenizer.from_directory()
vocab_size = tokenizer.get_vocab_size()
print(f"Vocab size: {vocab_size:,}")

config = GPTConfig(
    sequence_len=MAX_SEQ_LEN,
    vocab_size=vocab_size,
    n_layer=DEPTH,
    n_head=N_HEAD,
    n_kv_head=N_KV_HEAD,
    n_embd=N_EMBD,
)
print(f"Model config: {asdict(config)}")

model = GPT(config)
num_params = model.num_params()
print(f"Parameters: {num_params:,} ({num_params / 1e6:.1f}M)")

total_batch_tokens = BATCH_SIZE * MAX_SEQ_LEN * GRAD_ACCUM_STEPS
print(f"Tokens per optimizer step: {total_batch_tokens:,}")

# ---------------------------------------------------------------------------
# Muon optimizer
# ---------------------------------------------------------------------------

def zeropower_via_newtonschulz5(G, steps=5):
    """Newton-Schulz iteration to produce a near-orthogonal matrix from G."""
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.astype(mx.float32)
    X = X / (mx.linalg.norm(X) + 1e-7)
    transposed = X.shape[0] > X.shape[1]
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        X = a * X + (b * A + c * (A @ A)) @ X
    if transposed:
        X = X.T
    return X.astype(G.dtype)

# Adam optimizer for embeddings and scalars
adam_optimizer = optim.AdamW(
    learning_rate=EMBED_LR,
    betas=(0.9, 0.95),
    eps=1e-8,
    weight_decay=WEIGHT_DECAY,
)

# Muon momentum state: {param_path: momentum_array}
muon_state = {}

def apply_updates(model, grads, matrix_lr, embed_lr, scalar_lr, lr_multiplier):
    """Apply Muon to 2D weight matrices, Adam to everything else."""
    from mlx.utils import tree_flatten, tree_unflatten

    flat_params = dict(tree_flatten(model.trainable_parameters()))
    flat_grads = dict(tree_flatten(grads))

    new_params = {}
    adam_params = {}
    adam_grads = {}

    for k, p in flat_params.items():
        g = flat_grads.get(k)
        if g is None:
            new_params[k] = p
            continue

        if p.ndim == 2 and 'wte' not in k and 'lm_head' not in k:
            # Muon update for weight matrices
            g_orth = zeropower_via_newtonschulz5(g)
            prev_m = muon_state.get(k, mx.zeros_like(g_orth))
            m = MUON_MOMENTUM * prev_m + g_orth
            muon_state[k] = m
            # Nesterov look-ahead
            eff_g = MUON_MOMENTUM * m + g_orth
            scale = math.sqrt(max(p.shape))
            new_params[k] = p - (matrix_lr * lr_multiplier) * scale * eff_g
        else:
            # Adam for embeddings and scalars
            adam_params[k] = p
            adam_grads[k] = g

    # Apply Adam to non-matrix params
    if adam_grads:
        # Scale embed_lr / scalar_lr (use embed_lr for simplicity)
        adam_optimizer.learning_rate = embed_lr * lr_multiplier
        # Temporarily update only these params via tree operations
        # Build a sub-model update: update new_params with Adam results
        # We'll do manual Adam here to avoid needing a sub-model
        for k, p in adam_params.items():
            g = adam_grads[k]
            # Get or init Adam state
            m_key = f"adam_m_{k}"
            v_key = f"adam_v_{k}"
            step_key = f"adam_step_{k}"
            m = muon_state.get(m_key, mx.zeros_like(p))
            v = muon_state.get(v_key, mx.zeros_like(p))
            t = muon_state.get(step_key, 0) + 1
            muon_state[m_key] = 0.9 * m + 0.1 * g
            muon_state[v_key] = 0.95 * v + 0.05 * (g * g)
            muon_state[step_key] = t
            m_hat = muon_state[m_key] / (1 - 0.9 ** t)
            v_hat = muon_state[v_key] / (1 - 0.95 ** t)
            lr = (embed_lr if p.ndim >= 2 else scalar_lr) * lr_multiplier
            update = lr * m_hat / (mx.sqrt(v_hat) + 1e-8)
            if p.ndim >= 1:
                update = update + lr * WEIGHT_DECAY * p
            new_params[k] = p - update

    model.update(tree_unflatten(list(new_params.items())))
    mx.eval(model.parameters())
    # Force eval of muon state to avoid memory buildup
    mx.eval(*[v for v in muon_state.values() if hasattr(v, 'shape')])

# Loss function
def loss_fn(model, x, y):
    logits = model(x)
    logits_flat = logits.reshape(-1, logits.shape[-1])
    targets_flat = y.reshape(-1)
    return mx.mean(nn.losses.cross_entropy(logits_flat, targets_flat))

loss_and_grad = nn.value_and_grad(model, loss_fn)

# LR schedule
def get_lr_multiplier(progress):
    if progress < WARMUP_RATIO:
        return progress / WARMUP_RATIO if WARMUP_RATIO > 0 else 1.0
    elif progress < 1.0 - WARMDOWN_RATIO:
        return 1.0
    else:
        cooldown = (1.0 - progress) / WARMDOWN_RATIO
        return cooldown * 1.0 + (1 - cooldown) * FINAL_LR_FRAC

train_loader = make_dataloader(tokenizer, BATCH_SIZE, MAX_SEQ_LEN, "train")

print(f"Time budget: {TIME_BUDGET}s")
print(f"Gradient accumulation steps: {GRAD_ACCUM_STEPS}")

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

t_start = time.time()
smooth_train_loss = 0
total_training_time = 0
step = 0
warmup_steps = 2  # steps excluded from timing (MLX compiles on first call)

while True:
    t0 = time.time()

    # Gradient accumulation
    total_loss = 0.0
    accumulated_grads = None

    for micro_step in range(GRAD_ACCUM_STEPS):
        x, y, epoch = next(train_loader)
        loss, grads = loss_and_grad(model, x, y)
        mx.eval(loss, grads)
        total_loss += loss.item()

        if accumulated_grads is None:
            accumulated_grads = grads
        else:
            accumulated_grads = tree_map(
                lambda a, b: a + b, accumulated_grads, grads
            )

    # Average gradients and eval once
    accumulated_grads = tree_map(
        lambda g: g / GRAD_ACCUM_STEPS, accumulated_grads
    )
    mx.eval(accumulated_grads)
    avg_loss = total_loss / GRAD_ACCUM_STEPS

    # Update LR schedule
    progress = min(total_training_time / TIME_BUDGET, 1.0)
    lr_mult = get_lr_multiplier(progress)

    # Apply Muon + Adam updates
    apply_updates(model, accumulated_grads, MATRIX_LR, EMBED_LR, SCALAR_LR, lr_mult)

    # Fast fail
    if math.isnan(avg_loss) or avg_loss > 100:
        print("FAIL")
        exit(1)

    t1 = time.time()
    dt = t1 - t0

    if step > warmup_steps:
        total_training_time += dt

    # Logging
    ema_beta = 0.9
    smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * avg_loss
    debiased_smooth_loss = smooth_train_loss / (1 - ema_beta ** (step + 1))
    pct_done = 100 * progress
    tok_per_sec = int(total_batch_tokens / dt)
    remaining = max(0, TIME_BUDGET - total_training_time)

    print(f"\rstep {step:05d} ({pct_done:.1f}%) | loss: {debiased_smooth_loss:.6f} | lr: {lr_mult * MATRIX_LR:.2e} | dt: {dt*1000:.0f}ms | tok/sec: {tok_per_sec:,} | epoch: {epoch} | remaining: {remaining:.0f}s    ", end="", flush=True)

    step += 1

    if step > warmup_steps and total_training_time >= TIME_BUDGET:
        break

print()

total_tokens = step * total_batch_tokens

# Final eval
print("Evaluating...")
val_bpb = evaluate_bpb(model, tokenizer, BATCH_SIZE)

# Final summary
t_end = time.time()
peak_mem_bytes = mx.metal.get_peak_memory()
peak_mem_mb = peak_mem_bytes / 1024 / 1024

print("---")
print(f"val_bpb:          {val_bpb:.6f}")
print(f"training_seconds: {total_training_time:.1f}")
print(f"total_seconds:    {t_end - t_start:.1f}")
print(f"peak_memory_mb:   {peak_mem_mb:.1f}")
print(f"total_tokens_M:   {total_tokens / 1e6:.1f}")
print(f"num_steps:        {step}")
print(f"num_params_M:     {num_params / 1e6:.1f}")
print(f"depth:            {DEPTH}")
