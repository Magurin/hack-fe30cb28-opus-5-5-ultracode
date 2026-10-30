# Submission: TODO название команды

## Links
- Repository, final commit: https://github.com/BAITC-Hacks/hack-fe30cb28-opus-5-5-ultracode/commit/b6dd912d78d8c4ec61d6e354662498c8aa68957b
- Public Kaggle notebook, the saved version (`COMMIT` in it = the sha above; the same link is in
  the README): https://www.kaggle.com/code/magurin/soup-kernel-challenge-opus-ultracode
- Demo video (up to 3 minutes): TODO
- Presentation: TODO

## Team (up to 4 people)
| Name | Role | GitHub | Kaggle | Contact (captain) |
|---|---|---|---|---|
| TODO | | | | |

## Our kernels
| Kernel (`@triton.jit` name) | File | What it replaces | Share of the training time (check 4; 15.2% together) |
|---|---|---|---|
| `nf4_dequant_kernel` | `solution/kernels.py` | bitsandbytes NF4 dequantization (absmax dequant + 4-bit dequant, 2 kernels) | 7.2% |
| `rms_fwd_kernel` | `solution/kernels.py` | fp32 RMSNorm (5-6 PyTorch ops), residual add, fp16 casts, LoRA dropout | 2.3% |
| `rms_bwd_kernel` | `solution/kernels.py` | RMSNorm backward, LoRA dropout backward, gradient sums | 1.2% |
| `qkv_rope_fwd_kernel` | `solution/kernels.py` | q/k/v split, fp32 casts, LoRA adds, `apply_rotary_pos_emb` | 0.6% |
| `qkv_rope_bwd_kernel` | `solution/kernels.py` | RoPE backward, `repeat_kv` backward, gradient concat | 0.3% |
| `glu_fwd_kernel`, `glu_bwd_kernel` | `solution/kernels.py` | fp32 silu * up and its backward | 1.1 + 1.8% |
| `lm_head_ce_kernel` | `solution/kernels.py` | fp32 logits, cross-entropy, TRL entropy and accuracy, rewind row loss, logits gradient | 0.7% |

Ideas taken from other kernels, with links (we wrote the code ourselves): Liger-Kernel fused
RMSNorm and fused linear cross-entropy (https://github.com/linkedin/Liger-Kernel), Unsloth
(https://github.com/unslothai/unsloth), Cut Cross-Entropy (https://arxiv.org/abs/2411.09009),
online softmax (https://arxiv.org/abs/1805.02867), QLoRA NF4 / double quantization
(https://arxiv.org/abs/2305.14314), activation recomputation (https://arxiv.org/abs/1604.06174),
ZeRO-Offload / ZeRO-Infinity weight offloading (https://arxiv.org/abs/2101.06840).

## Results on Kaggle T4 (from `runs/result.json` and `runs/checks.json` of the notebook above)
| | Base Soup | Ours | Ratio |
|---|---|---|---|
| Time of `soup train`, s | 90.2 | 55.8 | 1.618x speedup (1.42x in our `--repeats 2` run) |
| Peak GPU memory, MiB | 14,345 | 1,275 | 11.25x base/ours |
| Eval loss (public rows) | 2.04966 | 2.04955 | -0.005% vs base |

- [x] All 10 checks of `harness/check_rules.py` passed (if not: which failed and why)
- [x] README.md rewritten for our solution: what we did, how to run it, results, limits

## Presentation covers
Which operations we sped up and why (padding, fp32 activation copies, full-vocabulary logits,
resident weights), how the kernels work, results against the base (speed, memory, quality),
limits and next steps.

## Limitations
See README.md, "Ограничения и что не получилось".
