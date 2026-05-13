# NLP Transformer — Chatbot from Scratch

A conversational chatbot built entirely from scratch using a Transformer encoder-decoder architecture in PyTorch, trained on ~1 million English movie conversations.

## Overview

This project implements the full Transformer architecture (as in "Attention Is All You Need") without relying on pre-trained models or high-level NLP libraries. It includes multi-head attention with **Rotary Positional Embeddings (RoPE)**, a custom vocabulary class, greedy decoding for inference, and a training loop with checkpoint saving.

## Architecture

| Component | Detail |
|---|---|
| Model | Encoder-Decoder Transformer |
| `d_model` | 512 |
| Attention heads | 8 |
| Encoder layers | 4 |
| Decoder layers | 4 |
| Feed-forward dim | 2048 |
| Positional encoding | Rotary (RoPE) |
| Dropout | 0.1 |
| Max sequence length | 350 tokens |

## Project Structure

```
nlp_transformer/
├── classes/
│   ├── transformer.py                    # Full encoder-decoder model
│   ├── encoder.py                        # Encoder layer
│   ├── decoder.py                        # Decoder layer
│   ├── multi_head_attention.py           # Multi-head attention with RoPE
│   ├── rotary_positional_embedding.py    # RoPE implementation
│   ├── position_wise_feed_forward.py     # FFN sublayer
│   ├── positional_encoding.py            # Sinusoidal PE (alternate)
│   ├── greedy_search_transformer_decoder.py
│   └── vocabulary.py                     # Custom vocabulary (Voc)
├── train/
│   └── transformer_train.py              # Training loop + evaluation
├── util/
│   └── transformer_util.py               # Data loading and batching
├── datasets/
│   └── movie-corpus/                     # Conversation data (not included)
├── transformer_main_train.py             # Entry point for training
└── transformer_evaluate.py              # Entry point for inference (chat)
```

## Training

```bash
python transformer_main_train.py
```

**Hyperparameters used:**

- Optimizer: Adam (`lr=1e-4`, `betas=(0.9, 0.98)`)
- Loss: CrossEntropyLoss (ignores padding)
- Gradient clipping: max norm 1.0
- Iterations: 10,000
- Batch size: 8

The trained checkpoint is saved to `datasets/save/transformer_model2/10000_checkpoint.tar`.

## Inference

After training, run the interactive chatbot:

```bash
python transformer_evaluate.py
```

Type a sentence and the model responds. Enter `exit`, `quit`, or `q` to stop.

```
> hello how are you
Bot: i m fine thank you
```

## Dataset

The model is trained on `ingles_conversation_1M.txt` — a file of ~1 million English conversation pairs in the format:

```
input_sentence%response_sentence
```

Rare words (frequency < 3) are trimmed from the vocabulary before training.

## Requirements

- Python 3.10+
- PyTorch 2.x (with `torch.accelerator` support)
- CUDA / MPS / CPU (auto-detected)

```bash
pip install torch
```

## Key Design Choices

- **RoPE instead of sinusoidal PE**: Rotary embeddings are applied directly inside the attention computation (on Q and K), providing better relative position awareness.
- **No external NLP libraries**: Vocabulary, tokenization, and data loading are all custom-built.
- **Causal masking**: A no-peak (upper-triangular) mask is applied on the decoder to prevent attending to future tokens.
