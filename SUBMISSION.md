# Submission: TODO (заполняет команда) название команды

Слитый шаг обучения Qwen2 + NF4 + LoRA на своих Triton-кернелах: без паддинга, с пересчётом слоя в
backward. NF4-веса слоёв и эмбеддинг / `lm_head` с загрузки модели лежат в pinned-памяти CPU и
подкачиваются на GPU по слою и кусками словаря. Подробно - `README.md`.

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
`5c93f07` на T4 (Brev), публичные строки. Финальные доли - в `runs/checks.json` ноутбука.

| Kernel (`@triton.jit` name) | File | What it replaces | Share of the training time (check 4) |
|---|---|---|---|
| `nf4_dequant_kernel` | `solution/kernels.py` | деквантизация NF4 в bitsandbytes (absmax + 4 бита, 2 кернела); у нас один проход, один запуск на [q\|k\|v] и один на [gate\|up], побитно как bnb | 8,0% |
| `rms_fwd_kernel` | `solution/kernels.py` | fp32 RMSNorm (5-6 операций PyTorch), residual add, касты в fp16, dropout входов LoRA | 2,4% |
| `rms_bwd_kernel` | `solution/kernels.py` | backward RMSNorm, backward dropout LoRA, суммы градиентов | 1,2% |
| `qkv_rope_fwd_kernel` | `solution/kernels.py` | разрезка q/k/v, fp32-касты, прибавка LoRA, `apply_rotary_pos_emb`, `repeat_kv` | 1,0% |
| `qkv_rope_bwd_kernel` | `solution/kernels.py` | backward RoPE, backward `repeat_kv`, сборка градиентов под один GEMM | 0,4% |
| `glu_fwd_kernel`, `glu_bwd_kernel` | `solution/kernels.py` | fp32 silu * up и его backward над слитым `[gate \| up]` | 1,3 + 2,2% |
| `lm_head_piece_kernel`, `lm_head_finish_kernel` | `solution/kernels.py` | fp32-логиты, cross-entropy, entropy и accuracy TRL, loss строк для rewind, градиент логитов - по кускам словаря, пока вес `lm_head` стримится с хоста | 0,55 + 0,01% |
| `lm_head_ce_kernel` | `solution/kernels.py` | то же за один проход по словарю, если вес `lm_head` на GPU (`OFFLOAD_HEAD = False`) | - |
| **Всего** | | | **17,1%** (порог 5%) |

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

Дополнительно (Brev T4, те же версии пакетов, не официальный замер; код `5c93f07` и `180c8e4`): на
публичных строках и на срезах, похожих на скрытые (другие строки, seed 3, 7, 11, 17), проверки
проходят, memory 17,2-17,6x (пик 813-835 MiB), eval loss в пределах +-0,015%; на стресс-срезе из батчей
4 x 512 токенов база падает с OOM, решение проходит с пиком 873 MiB. Таблица - в README, раздел
"Срезы, похожие на скрытые".

## Presentation covers
Какие операции ускоряли и почему (паддинг, fp32-копии активаций, логиты на весь словарь, веса на
GPU всё обучение), как устроены кернелы (деквантизация NF4 под Turing, слитые RMSNorm + LoRA,
RoPE, SwiGLU, голова кусками словаря с online logsumexp), откуда 17x по памяти, результаты против базы (скорость,
память, качество), ограничения и дальнейшие шаги.

## Limitations
См. README.md, "Ограничения и что не получилось": решение заточено под Qwen2 + NF4 + LoRA на q/v в
fp16 (иначе исходный код Soup); GEMM cuBLAS ~69% работы GPU и свой GEMM на Triton для T4 не
получился; вне цикла обучения выигрыш небольшой (~6 с патчами запуска по сумме их A/B); из пика
813 MiB ~165 MiB - вне аллокатора PyTorch (CUDA-контекст, модули кернелов, cuBLAS) и ~150 MiB -
слоты весов слоя и буферы головы; выгрузка головы зависит от PCIe (на ~6,3 ГБ/с стоит ~2,4 с).
