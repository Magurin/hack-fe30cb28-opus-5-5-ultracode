# Submission: TODO (заполняет команда) название команды

Слитый шаг обучения Qwen2 + NF4 + LoRA на своих Triton-кернелах: без паддинга, с пересчётом слоя в
backward. NF4-веса слоёв и эмбеддинг / `lm_head` с загрузки модели лежат в pinned-памяти CPU и
подкачиваются на GPU по слою и кусками словаря; входы слоёв и выходы GEMM `[gate | up]` для
backward тоже ждут на хосте (последние - вместо пересчёта в backward), а на быстрой шине (от 9 ГБ/с,
как на Kaggle) - и тензоры attention (q, k/v, выход attention, `o_proj`), так что backward
пересчитывает из forward слоя только две RMSNorm. Пик памяти - 463 MiB на Kaggle (A/B).
Подробно - `README.md`.

## Links
- Repository, code of the measurement: https://github.com/BAITC-Hacks/hack-fe30cb28-opus-5-5-ultracode/commit/bfff00c
  (`solution_sha256` `9efa8ec1`). Since then `main` also has the late warm-up with NF4 loaded straight
  into pinned memory (`feca479`, `43765db`, `838827f`): -0.4..-0.9 s out of the loop in Kaggle A/Bs;
  the official measurement of the current `main` will replace this
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
`8c584c0`, публичные строки, меньший из двух запусков `--repeats 2` (у кода `5f9b2ef` на Brev T4 -
13,9-15,0%: другие процессор и шина).

| Kernel (`@triton.jit` name) | File | What it replaces | Share of the training time (check 4) |
|---|---|---|---|
| `nf4_dequant_kernel` | `solution/kernels.py` | деквантизация NF4 в bitsandbytes (absmax + 4 бита, 2 кернела); у нас один проход, один запуск на [q\|k\|v] и один на [gate\|up], побитно как bnb | 9,5% |
| `rms_fwd_kernel` | `solution/kernels.py` | fp32 RMSNorm (5-6 операций PyTorch), residual add перед MLP, касты в fp16, dropout входов LoRA (маска бит в бит как у базы) | 2,7% |
| `rms_bwd_kernel` | `solution/kernels.py` | backward RMSNorm, backward dropout LoRA (те же маски), суммы градиентов | 1,5% |
| `qkv_rope_fwd_kernel` | `solution/kernels.py` | разрезка q/k/v, fp32-касты, прибавка LoRA, `apply_rotary_pos_emb`, `repeat_kv` | 1,25% |
| `qkv_rope_bwd_kernel` | `solution/kernels.py` | backward RoPE, backward `repeat_kv`, сборка градиентов под один GEMM | 0,45% |
| `glu_fwd_kernel`, `glu_bwd_kernel` | `solution/kernels.py` | fp32 silu * up и его backward над слитым `[gate \| up]` | 1,5 + 2,7% |
| `lm_head_piece_kernel`, `lm_head_finish_kernel` | `solution/kernels.py` | fp32-логиты, cross-entropy, entropy и accuracy TRL, loss строк для rewind, градиент логитов - по кускам словаря, пока вес `lm_head` стримится с хоста | 0,70 + 0,01% |
| `lm_head_ce_kernel` | `solution/kernels.py` | то же за один проход по словарю, если вес `lm_head` на GPU (`OFFLOAD_HEAD = False`) | - |
| **Всего** | | | **18,0%** (`bfff00c`, меньший из 2 запусков r2; шаблонный прогон - 18,2%; порог 5%) |

Ideas taken from other kernels, with links (we wrote the code ourselves): Liger-Kernel fused
RMSNorm и fused linear cross-entropy (https://github.com/linkedin/Liger-Kernel), Unsloth
(https://github.com/unslothai/unsloth), Cut Cross-Entropy (https://arxiv.org/abs/2411.09009),
online softmax (https://arxiv.org/abs/1805.02867), QLoRA NF4 / double quantization
(https://arxiv.org/abs/2305.14314), activation recomputation (https://arxiv.org/abs/1604.06174),
ZeRO-Offload / ZeRO-Infinity weight offloading (https://arxiv.org/abs/2101.06840), FlashAttention
varlen API (https://arxiv.org/abs/2205.14135; у нас - встроенное в PyTorch ядро с `cu_seqlens`).

## Results on Kaggle T4 (from `runs_r2/result.json` and `runs_r2/checks.json` of the notebook above)
`run_task.py --repeats 2` (база, решение, решение, база в одной сессии), публичные строки, seed 1 -
последняя ячейка ноутбука (README, "Как запустить", шаг 5); шаблонный прогон с одним запуском -
`runs/result.json` и `runs/checks.json`.

| | Base Soup | Ours | Ratio |
|---|---|---|---|
| Time of `soup train`, s (mean of 2 runs) | 80,26 | 44,22 | 1,815x speedup |
| Training loop, 64 steps, s | 58,62 | 25,57 | 2,29x |
| Peak GPU memory, MiB | 14 345 | 463 | 30,98x base/ours |
| Eval loss (public rows) | 2,04964 | 2,04971 | +0,003% vs base |

Код `bfff00c` (Kaggle, 31.10, версия 24 ноутбука `magurin/soup-kernel-run`; шаблонный прогон той же
версии - 1,768x / 30,98x). Прошлые официальные замеры тем же ноутбуком (r2, все 10 проверок):
`7962b1d` - 1,777x / 30,98x (463 MiB, `522e22c0`, до быстрого выхода, шаблон 1,972x при холодной базе),
`a123fba` - 1,8448x / 26,418x (пик 543 MiB, `solution_sha256` `f302c7e6`, медленная сессия, шаблон
1,964x), `9d31eea` - 1,8322x / 25,48x (563 MiB, `9155f08c`, шаблон 2,005x), `8c584c0` - 1,824x /
24,61x (583 MiB). Эти три ещё токенизировали своим кодом без `datasets.map`; организаторы позже
запретили это ([t.me/souptasterschat/4879](https://t.me/souptasterschat/4879),
[t.me/souptasterschat/5045](https://t.me/souptasterschat/5045), п. 2), и финальный код токенизирует
снова `map` Soup (~2-3 с вне цикла).

- [x] All 10 checks of `harness/check_rules.py` passed: да (`bfff00c`, версия 24: и `--repeats 2`, и шаблонный прогон; во всех прошлых официальных замерах
  - все 10 и в `--repeats 2`, и в шаблонном; проверка 10 - 79 тестов Soup passed)
- [x] README.md rewritten for our solution: what we did, how to run it, results, limits

Дополнительно (Brev T4 - арендованные T4 с теми же версиями пакетов, не официальный замер): 15
новых срезов по 256 строк со всего набора (новые seed, батчи до 1 955 токенов) на коде до масок
dropout базы - все проверки проходят, пик 577-601 MiB (memory 23,88-24,82x), eval -0,017..+0,015%;
ещё 4 длинных среза с масками базы (2 из них на `5f9b2ef`) - все проверки, кривая loss
0,01% в среднем (на уровне разброса двух прогонов самой базы), eval -0,001..+0,002%, пик 579-599 MiB.
На стресс-срезе из батчей 4 x 512 токенов база падает с OOM, решение проходит (на текущем коде - с пиком 499 MiB).
Текущий код `70d0e89` (`AttnStream` включён принудительно, на Brev шина 6,3 ГБ/с ниже порога 9 ГБ/с):
14 срезов, 21 прогон с базой и тестами Soup - все 10 проверок, кривая loss 0,01% в среднем, максимум
по шагам до 0,10%, eval -0,002..+0,002%, пик 461-481 MiB (ещё 10 прогонов - 461-501), стресс 499.
Таблицы - в README, раздел "Срезы, похожие на скрытые".

## Presentation covers
Какие операции ускоряли и почему (паддинг, fp32-копии активаций, логиты на весь словарь, веса на
GPU всё обучение), как устроены кернелы (деквантизация NF4 под Turing, слитые RMSNorm + LoRA,
RoPE, SwiGLU, голова кусками словаря с online logsumexp), откуда ~31x по памяти, результаты против базы (скорость,
память, качество), ограничения и дальнейшие шаги.

## Limitations
См. README.md, "Ограничения и что не получилось": решение заточено под Qwen2 + NF4 + LoRA на q/v в
fp16 (иначе исходный код Soup); GEMM cuBLAS ~69% работы GPU и свой GEMM на Triton для T4 не
получился; вне цикла обучения выигрыш небольшой (~11 с патчами запуска по сумме A/B отдельных
патчей, из них -1,8 с - выход через `os._exit` после всех обработчиков выхода, разрешение
[t.me/souptasterschat/5199](https://t.me/souptasterschat/5199), и ~-0,9 с - прогрев в конце `apply()`,
дешёвый ключ кэша Triton ([t.me/souptasterschat/2922](https://t.me/souptasterschat/2922), п. 2) и NF4
загрузчиком прямо в pinned-буфер стриминга; своя токенизация без `datasets.map`
запрещена и убрана); ниже ~461 MiB пик не опускается:
~113-115 MiB - вне аллокатора PyTorch (CUDA-контекст ~65, модуль кастов PyTorch, cuBLAS), а резерв
аллокатора 346 MiB держит фрагментация (настройки аллокатора харнесса менять нельзя); выгрузки
зависят от шины PCIe (на Kaggle - в пределах разброса запусков, на Brev с вдвое более медленной
шиной голова и входы стоят ~3 с), `[gate | up]` на хосте выгоден только при быстрой шине (Kaggle
-1,7..-2,6 с, Brev +~5 с), тензоры attention решение держит на хосте только при шине от 9 ГБ/с;
на хосте ~4,2 ГБ pinned-памяти.
