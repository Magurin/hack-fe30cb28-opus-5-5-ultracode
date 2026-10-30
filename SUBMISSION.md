# Submission: TODO (заполняет команда) название команды

Слитый шаг обучения Qwen2 + NF4 + LoRA на восьми своих Triton-кернелах: без паддинга, с пересчётом
слоя в backward, с NF4-весами слоёв в pinned-памяти CPU и головой кусками по 256 строк. Подробно -
`README.md`.

## Links
- Repository, final commit: https://github.com/BAITC-Hacks/hack-fe30cb28-opus-5-5-ultracode/commit/[финал]
- Public Kaggle notebook, the saved version (`COMMIT` in it = the sha above; the same link is in
  the README): https://www.kaggle.com/code/magurin/soup-kernel-challenge-opus-ultracode
  (TODO (заполняет команда): ссылка на сохранённую версию `?scriptVersionId=...` после финального
  Save & Run All; ноутбук и репозиторий открыты для жюри)
- Demo video (up to 3 minutes): TODO (заполняет команда)
- Presentation: TODO (заполняет команда)

## Team (up to 4 people)
| Name | Role | GitHub | Kaggle | Contact (captain) |
|---|---|---|---|---|
| TODO (заполняет команда) | | | | |

## Our kernels
Доля во времени обучения - как в проверке 4 (CUDA events harness, шаги 3-64); прогон коммита
`fbe7502` на T4, публичные строки. Финальные доли - в `runs/checks.json` ноутбука.

| Kernel (`@triton.jit` name) | File | What it replaces | Share of the training time (check 4) |
|---|---|---|---|
| `nf4_dequant_kernel` | `solution/kernels.py` | деквантизация NF4 в bitsandbytes (absmax + 4 бита, 2 кернела); у нас один проход, один запуск на [q\|k\|v] и один на [gate\|up], побитно как bnb | 8,4% |
| `rms_fwd_kernel` | `solution/kernels.py` | fp32 RMSNorm (5-6 операций PyTorch), residual add, касты в fp16, dropout входов LoRA | 2,6% |
| `rms_bwd_kernel` | `solution/kernels.py` | backward RMSNorm, backward dropout LoRA, суммы градиентов | 1,3% |
| `qkv_rope_fwd_kernel` | `solution/kernels.py` | разрезка q/k/v, fp32-касты, прибавка LoRA, `apply_rotary_pos_emb`, `repeat_kv` | 1,1% |
| `qkv_rope_bwd_kernel` | `solution/kernels.py` | backward RoPE, backward `repeat_kv`, сборка градиентов под один GEMM | 0,4% |
| `glu_fwd_kernel`, `glu_bwd_kernel` | `solution/kernels.py` | fp32 silu * up и его backward над слитым `[gate \| up]` | 1,4 + 2,3% |
| `lm_head_ce_kernel` | `solution/kernels.py` | fp32-логиты, cross-entropy, entropy и accuracy TRL, loss строк для rewind, градиент логитов | 0,8% |
| **Всего** | | | **18,2%** (порог 5%) |

Ideas taken from other kernels, with links (we wrote the code ourselves): Liger-Kernel fused
RMSNorm и fused linear cross-entropy (https://github.com/linkedin/Liger-Kernel), Unsloth
(https://github.com/unslothai/unsloth), Cut Cross-Entropy (https://arxiv.org/abs/2411.09009),
online softmax (https://arxiv.org/abs/1805.02867), QLoRA NF4 / double quantization
(https://arxiv.org/abs/2305.14314), activation recomputation (https://arxiv.org/abs/1604.06174),
ZeRO-Offload / ZeRO-Infinity weight offloading (https://arxiv.org/abs/2101.06840), FlashAttention
varlen API (https://arxiv.org/abs/2205.14135; у нас - встроенное в PyTorch ядро с `cu_seqlens`).

## Results on Kaggle T4 (from `runs/result.json` and `runs/checks.json` of the notebook above)
`run_task.py --repeats 2` (база, решение, решение, база в одной сессии), публичные строки, seed 1.

| | Base Soup | Ours | Ratio |
|---|---|---|---|
| Time of `soup train`, s | [финал] | [финал] | [финал]x speedup |
| Peak GPU memory, MiB | 14 345 | [финал] | [финал]x base/ours |
| Eval loss (public rows) | [финал] | [финал] | [финал]% vs base |

- [ ] All 10 checks of `harness/check_rules.py` passed (if not: which failed and why) - [финал]
- [x] README.md rewritten for our solution: what we did, how to run it, results, limits

Дополнительно (Brev T4, те же версии пакетов, не официальный замер): на срезах, похожих на
скрытые (строки 20000 и 10000, seed 3 и 11), все проверки проходят, memory 10,9-11,25x, eval loss
в пределах +0,011%; на стресс-срезе из батчей 4 x 512 токенов база падает с OOM, решение
проходит с пиком 1 293 MiB. Таблица - в README, раздел "Срезы, похожие на скрытые".

## Presentation covers
Какие операции ускоряли и почему (паддинг, fp32-копии активаций, логиты на весь словарь, веса на
GPU всё обучение), как устроены кернелы (деквантизация NF4 под Turing, слитые RMSNorm + LoRA,
RoPE, SwiGLU, голова с online logsumexp), откуда 11x по памяти, результаты против базы (скорость,
память, качество), ограничения и дальнейшие шаги.

## Limitations
См. README.md, "Ограничения и что не получилось": решение заточено под Qwen2 + NF4 + LoRA на q/v в
fp16 (иначе исходный код Soup); GEMM cuBLAS ~69% работы GPU и свой GEMM на Triton для T4 не
получился; вне цикла обучения выигрыш небольшой (~5 с патчами запуска); пик ниже загрузки модели в
transformers не опускается.
