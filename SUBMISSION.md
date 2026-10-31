# Submission: TODO (заполняет команда) название команды

Слитый шаг обучения Qwen2 + NF4 + LoRA на своих Triton-кернелах: без паддинга, с пересчётом слоя в
backward. NF4-веса слоёв и эмбеддинг / `lm_head` с загрузки модели лежат в pinned-памяти CPU и
подкачиваются на GPU по слою и кусками словаря; входы слоёв для backward тоже ждут на хосте.
Подробно - `README.md`.

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
Доля во времени обучения - как в проверке 4 (CUDA events harness, шаги 3-64); Kaggle T4, коммит
`6b1cc09`, публичные строки, меньший из двух запусков `--repeats 2` (на Brev T4 - 16,7-17,2%).

| Kernel (`@triton.jit` name) | File | What it replaces | Share of the training time (check 4) |
|---|---|---|---|
| `nf4_dequant_kernel` | `solution/kernels.py` | деквантизация NF4 в bitsandbytes (absmax + 4 бита, 2 кернела); у нас один проход, один запуск на [q\|k\|v] и один на [gate\|up], побитно как bnb | 7,5% |
| `rms_fwd_kernel` | `solution/kernels.py` | fp32 RMSNorm (5-6 операций PyTorch), residual add, касты в fp16, dropout входов LoRA | 2,5% |
| `rms_bwd_kernel` | `solution/kernels.py` | backward RMSNorm, backward dropout LoRA, суммы градиентов | 1,2% |
| `qkv_rope_fwd_kernel` | `solution/kernels.py` | разрезка q/k/v, fp32-касты, прибавка LoRA, `apply_rotary_pos_emb`, `repeat_kv` | 0,95% |
| `qkv_rope_bwd_kernel` | `solution/kernels.py` | backward RoPE, backward `repeat_kv`, сборка градиентов под один GEMM | 0,37% |
| `glu_fwd_kernel`, `glu_bwd_kernel` | `solution/kernels.py` | fp32 silu * up и его backward над слитым `[gate \| up]` | 1,2 + 2,1% |
| `lm_head_piece_kernel`, `lm_head_finish_kernel` | `solution/kernels.py` | fp32-логиты, cross-entropy, entropy и accuracy TRL, loss строк для rewind, градиент логитов - по кускам словаря, пока вес `lm_head` стримится с хоста | 0,56 + 0,01% |
| `lm_head_ce_kernel` | `solution/kernels.py` | то же за один проход по словарю, если вес `lm_head` на GPU (`OFFLOAD_HEAD = False`) | - |
| **Всего** | | | **16,4%** (порог 5%) |

Ideas taken from other kernels, with links (we wrote the code ourselves): Liger-Kernel fused
RMSNorm и fused linear cross-entropy (https://github.com/linkedin/Liger-Kernel), Unsloth
(https://github.com/unslothai/unsloth), Cut Cross-Entropy (https://arxiv.org/abs/2411.09009),
online softmax (https://arxiv.org/abs/1805.02867), QLoRA NF4 / double quantization
(https://arxiv.org/abs/2305.14314), activation recomputation (https://arxiv.org/abs/1604.06174),
ZeRO-Offload / ZeRO-Infinity weight offloading (https://arxiv.org/abs/2101.06840), FlashAttention
varlen API (https://arxiv.org/abs/2205.14135; у нас - встроенное в PyTorch ядро с `cu_seqlens`).

## Results on Kaggle T4 (from `runs/result.json` and `runs/checks.json` of the notebook above)
`run_task.py --repeats 2` (база, решение, решение, база в одной сессии), публичные строки, seed 1:
цифры таблицы - из `runs_r2/result.json` и `runs_r2/checks.json` дополнительной ячейки (README,
"Как запустить", шаг 5); шаблонный прогон с одним запуском - `runs/result.json` и `runs/checks.json`.

| | Base Soup | Ours | Ratio |
|---|---|---|---|
| Time of `soup train`, s (mean of 2 runs) | 79,3 | 48,7 | 1,628x speedup |
| Training loop, 64 steps, s | 57,4 | 28,1 | 2,04x |
| Peak GPU memory, MiB | 14 345 | 635 | 22,59x base/ours |
| Eval loss (public rows) | 2,04965 | 2,04954 | -0,005% vs base |

Коммит `6b1cc09` (Kaggle, 31.10). Шаблонный прогон того же ноутбука (один запуск): 1,805x / 22,59x
(медленная база, 88,5 с).

- [x] All 10 checks of `harness/check_rules.py` passed: все 10 и в прогоне `--repeats 2`, и в
  шаблонном (проверка 10 - 79 тестов Soup passed)
- [x] README.md rewritten for our solution: what we did, how to run it, results, limits

Дополнительно (Brev T4, те же версии пакетов, не официальный замер; код с выгрузкой входов слоёв,
`0223a36` = `6b1cc09`): на публичных строках и на 5 срезах, похожих на скрытые (другие строки,
seed 3, 29, 37, 71, 379), проверки проходят, пик 635 MiB (memory 22,6x), на хвосте с батчами до
1 954 токенов - 615 MiB (23,4x), eval loss в пределах +-0,013%; на стресс-срезе из батчей 4 x 512
токенов база падает с OOM, решение проходит с пиком 613 MiB. Таблица - в README, раздел "Срезы,
похожие на скрытые".

## Presentation covers
Какие операции ускоряли и почему (паддинг, fp32-копии активаций, логиты на весь словарь, веса на
GPU всё обучение), как устроены кернелы (деквантизация NF4 под Turing, слитые RMSNorm + LoRA,
RoPE, SwiGLU, голова кусками словаря с online logsumexp), откуда 22x по памяти, результаты против базы (скорость,
память, качество), ограничения и дальнейшие шаги.

## Limitations
См. README.md, "Ограничения и что не получилось": решение заточено под Qwen2 + NF4 + LoRA на q/v в
fp16 (иначе исходный код Soup); GEMM cuBLAS ~69% работы GPU и свой GEMM на Triton для T4 не
получился; вне цикла обучения выигрыш небольшой (~7 с патчами запуска по сумме их A/B); из пика
635 MiB ~167 MiB - вне аллокатора PyTorch (CUDA-контекст, модули кернелов, cuBLAS) и ~150 MiB -
слоты весов слоя и буферы головы; выгрузки головы и входов слоёв зависят от PCIe (на Kaggle,
~13 ГБ/с, - в пределах разброса запусков, на ~6,3 ГБ/с - ~2,4 с и ~0,7 с); на хосте ~1,6 ГБ pinned-памяти.
