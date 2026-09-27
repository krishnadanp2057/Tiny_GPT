# Tiny GPT — A Transformer Built From Scratch

A minimal, fully from-scratch implementation of a GPT-style Transformer in PyTorch — built while studying **["Attention Is All You Need" (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762)** as a way to translate the paper's ideas into working code rather than just reading about them.

## Motivation

Reading the Transformer paper is one thing; implementing masked self-attention, positional encodings, and a full decoder stack by hand is what actually makes the architecture click. This project is that exercise — every component (attention heads, multi-head attention, feed-forward layers, residual + layer norm blocks) is implemented manually with no high-level Transformer library, so the mechanics stay visible end to end.

## Architecture

The model follows the **decoder-only** half of the original Transformer (the GPT-style variant):

```
Input tokens
   │
   ▼
Token Embedding  +  Positional Embedding
   │
   ▼
┌─────────────────────────────┐
│  Transformer Block  × N     │
│  ┌────────────────────────┐ │
│  │ LayerNorm               │ │
│  │ Masked Multi-Head       │ │
│  │ Self-Attention           │ │
│  │ + residual connection    │ │
│  ├────────────────────────┤ │
│  │ LayerNorm               │ │
│  │ Feed-Forward Network     │ │
│  │ + residual connection    │ │
│  └────────────────────────┘ │
└─────────────────────────────┘
   │
   ▼
Final LayerNorm
   │
   ▼
Linear Head → vocabulary logits
   │
   ▼
Softmax → next-token prediction
```

Key ideas from the paper implemented here:
- **Scaled dot-product attention** — `softmax(QKᵀ / √d_k) V`
- **Multi-head attention** — multiple attention heads run in parallel and their outputs are concatenated and projected
- **Causal masking** — a lower-triangular mask ensures each token can only attend to itself and previous tokens (autoregressive generation)
- **Positional embeddings** — since attention has no inherent notion of order, position information is injected explicitly
- **Residual connections + LayerNorm (pre-norm)** — stabilizes training in deeper stacks
- **Position-wise Feed-Forward Networks** — a small MLP applied independently to each position

## Files

| File | Description |
|---|---|
| `transformer_block.py` | Core Transformer components: `SelfAttentionHead`, `MultiHeadAttention`, `FeedForward`, and the `Block` (attention + FFN with residuals) |
| `demo.py` | Builds a toy word-level vocabulary, defines `TinyGPT` (embeddings + stacked blocks + output head), trains it on a small corpus, and generates text |

## How It Works

1. **Tokenization** — a tiny custom corpus is split into words and mapped to integer IDs (word-level, not BPE — kept simple on purpose).
2. **Embeddings** — each token ID is converted into a learned vector, combined with a learned positional vector.
3. **Transformer blocks** — the sequence passes through `n_layers` blocks, each combining masked multi-head self-attention with a feed-forward network via residual connections.
4. **Prediction** — a final linear layer projects each position's representation into vocabulary-sized logits, trained with cross-entropy against the next-token target.
5. **Generation** — starting from a seed token, the model autoregressively samples one new token at a time using `torch.multinomial` over the softmax output.

## Running It

```bash
pip install torch
python demo.py
```

This will print training loss every 300 steps and generate a short sample sequence from a seed word at the end.

## What I Learned

- Why attention needs to be **scaled** by `√d_k` (and what happens if you get the scaling factor wrong)
- How **causal masking** enforces autoregressive behavior during training and generation
- Why **positional embeddings** are necessary — attention alone is permutation-invariant
- How **pre-norm residual blocks** (`x + sublayer(LayerNorm(x))`) differ from the original paper's post-norm design, and why pre-norm is more common in modern implementations
- The practical difference between `vocab_size` (unique tokens) and total corpus length

## Next Steps

- [ ] Fix attention scaling to divide by `head_size` instead of full embedding dimension
- [ ] Add dropout for regularization
- [ ] Move from word-level to character-level or BPE tokenization
- [ ] Add a train/validation split and loss curves
- [ ] Scale up to a larger corpus and compare generation quality

## Reference

Vaswani, A. et al. (2017). *Attention Is All You Need*. [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)
