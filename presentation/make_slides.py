"""The presentation: 10 slides drawn with matplotlib into presentation/slides.pdf (16:9).

    python presentation/make_slides.py        # needs only matplotlib

Every number is the README's (Kaggle T4, commit aafb9a7, `run_task.py --repeats 2`, unless a slide
says otherwise). The team's name goes in TEAM below.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

OUT = Path(__file__).resolve().parent / "slides.pdf"
TEAM = "[название команды]"
REPO = "github.com/BAITC-Hacks/hack-fe30cb28-opus-5-5-ultracode"
NOTEBOOK = "kaggle.com/code/magurin/soup-kernel-challenge-opus-ultracode"

INK, MUTED, ACCENT, LIGHT, BASE = "#1b1f24", "#6a737d", "#1f6feb", "#dbe7fb", "#b0b7c0"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 13, "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
                     "axes.spines.top": False, "axes.spines.right": False})


def ru(x, digits=0):
    """A number the Russian way: 14 345, 1,705."""
    s = f"{x:,.{digits}f}".replace(",", " ").replace(".", ",")
    return s


def slide(pdf, title, number, draw):
    fig = plt.figure(figsize=(13.33, 7.5))
    fig.patch.set_facecolor("white")
    fig.text(0.05, 0.92, title, fontsize=26, weight="bold", color=INK, va="top")
    if title:
        fig.add_artist(plt.Line2D([0.05, 0.95], [0.84, 0.84], color=LIGHT, lw=2))
    fig.text(0.95, 0.03, f"{number} / 10", fontsize=10, color=MUTED, ha="right")
    fig.text(0.05, 0.03, "Soup Kernel Challenge, Decentrathon 6.0", fontsize=10, color=MUTED)
    draw(fig)
    pdf.savefig(fig)
    plt.close(fig)


def line_height(size):
    """Height of one text line in figure fractions (a 7.5 in tall slide)."""
    return size * 1.35 / (7.5 * 72)


def para(fig, x, y, text, width_chars, size=14, color=INK, weight=None):
    """A wrapped paragraph from (x, y) down; returns the y below it."""
    import textwrap
    wrapped = "\n".join(textwrap.fill(p, width_chars) for p in text.split("\n"))
    fig.text(x, y, wrapped, fontsize=size, color=color, va="top", linespacing=1.3, weight=weight)
    return y - line_height(size) * (wrapped.count("\n") + 1)


def bullets(fig, lines, x=0.05, y=0.78, width_chars=58, size=15):
    """A list of wrapped items from (x, y) down; returns the y below it."""
    import textwrap
    for line in lines:
        wrapped = textwrap.fill(line, width_chars, subsequent_indent="  ")
        fig.text(x, y, "- " + wrapped, fontsize=size, color=INK, va="top", linespacing=1.3)
        y -= line_height(size) * (wrapped.count("\n") + 1) + 0.022
    return y


# ------------------------------------------------------------------------------------------------
def s1_title(fig):
    fig.texts[0].set_text("")  # the title slide has its own layout
    fig.text(0.05, 0.80, "14 ГБ -> 0,6 ГБ", fontsize=54, weight="bold", color=ACCENT)
    fig.text(0.05, 0.70, "Обучение Qwen2.5-1.5B (NF4 + LoRA) на Kaggle T4 на своих Triton-кернелах",
             fontsize=22, color=INK)
    fig.text(0.05, 0.64, f"Команда: {TEAM}", fontsize=16, color=MUTED)
    for i, (big, small) in enumerate([("1,698x", "быстрее вся команда soup train\n(84,6 -> 49,8 с)"),
                                      ("23,87x", "меньше пик памяти GPU\n(14 345 -> 601 MiB)"),
                                      ("10 / 10", "проверок check_rules.py,\neval loss -0,006%")]):
        x = 0.05 + i * 0.31
        fig.add_artist(FancyBboxPatch((x, 0.22), 0.28, 0.30, boxstyle="round,pad=0.01",
                                      transform=fig.transFigure, fc=LIGHT, ec="none"))
        fig.text(x + 0.02, 0.43, big, fontsize=40, weight="bold", color=ACCENT)
        fig.text(x + 0.02, 0.29, small, fontsize=14, color=INK, linespacing=1.4)
    para(fig, 0.05, 0.15, "Тот же адаптер на тех же батчах: модель, данные, LoRA, оптимизатор и выходы "
             "Soup не меняются. Kaggle T4, --repeats 2, коммит aafb9a7.", 110, size=12, color=MUTED)


def s2_base(fig):
    ax = fig.add_axes([0.70, 0.26, 0.26, 0.50])
    parts = [("Активации 28 слоёв\n(в основном fp32)", 9.5), ("Логиты (4, T, 151 936)\nв нескольких копиях", 3.5),
             ("Модель: NF4 +\nэмбеддинг в fp32", 1.5)]
    ax.barh([p[0] for p in parts][::-1], [p[1] for p in parts][::-1], color=[BASE, BASE, ACCENT])
    for i, (_, v) in enumerate(parts[::-1]):
        ax.text(v + 0.2, i, f"~{ru(v, 1)} ГБ", va="center", fontsize=13)
    ax.set_xlim(0, 12)
    ax.set_xlabel("ГБ на пике (порядок величины, разбор кода)")
    ax.set_title("Пик базы: 14 345 MiB (nvidia-smi)", fontsize=14, loc="left")
    bullets(fig, ["База: soup train 84,6 с, из них цикл 64 шагов 62,8 с; пик 14 345 из 15 ГБ.",
                  "bitsandbytes возвращает выход каждой проекции в fp32: gate/up MLP (токены x 8 960) "
                  "хранятся для backward в fp32.",
                  "peft поднимает эмбеддинг / lm_head (151 936 x 1 536) в fp32.",
                  "Паддинг: ~205 реальных токенов на строку при длине батча ~380.",
                  "Логиты на весь словарь: fp16, fp32 для loss, градиент, копия для accuracy."],
            width_chars=50)
    para(fig, 0.05, 0.11, "Память съедают не веса, а то, что модули HF хранят между собой: менять надо весь шаг.",
         110, size=14, color=ACCENT, weight="bold")


def box(fig, x, y, w, h, text, fc=LIGHT, size=12):
    fig.add_artist(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.006", transform=fig.transFigure,
                                  fc=fc, ec="none"))
    fig.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size, color=INK,
             linespacing=1.3)


def arrow(fig, x0, y0, x1, y1):
    fig.add_artist(matplotlib.patches.FancyArrowPatch((x0, y0), (x1, y1), transform=fig.transFigure,
                                                      arrowstyle="-|>", mutation_scale=14, color=MUTED))


def s3_idea(fig):
    y = 0.58
    steps = [("батч 4 x L\nупаковка без паддинга\n(N токенов, cu_seqlens)", 0.05),
             ("28 x _Layer\n(autograd.Function)\nнаши кернелы + cuBLAS", 0.29),
             ("голова кусками\nсловаря: loss,\nэнтропия, accuracy", 0.53),
             ("backward 27 -> 0\nпересчёт по плану\nпод потолком памяти", 0.77)]
    for text, x in steps:
        box(fig, x, y, 0.19, 0.18, text)
    for (_, x0), (_, x1) in zip(steps, steps[1:]):
        arrow(fig, x0 + 0.19, y + 0.09, x1, y + 0.09)
    lanes = [("CPU -> GPU, по слою вперёд: NF4-веса слоя (3 слота по 23 MiB)", 0.44),
             ("GPU -> CPU -> GPU: входы слоёв (кольцо из 3 слотов) и выходы [gate | up]", 0.37),
             ("CPU -> GPU: куски lm_head по 8 192 строки (3 буфера по 24 MiB)", 0.30)]
    for text, ly in lanes:
        box(fig, 0.05, ly, 0.90, 0.055, text, fc="#f1f3f5", size=12)
    y = para(fig, 0.05, 0.24, "GEMM - cuBLAS, всё между ними - наши кернелы. На T4 tl.dot в Triton 3.6 без "
             "тензорных ядер: ~1 TFLOPS против 16-34 у cuBLAS.", 110, size=14)
    para(fig, 0.05, y - 0.02, "Те же округления до fp16 в тех же местах, fp32 для residual, норм, RoPE, softmax "
         "и loss -> кривая loss в 0,01-0,05% от базы.", 110, size=14)


def s4_kernels(fig):
    ax = fig.add_axes([0.17, 0.16, 0.36, 0.62])
    k = [("nf4_dequant", 7.8), ("rms_fwd", 2.7), ("glu_bwd", 2.0), ("rms_bwd", 1.3), ("glu_fwd", 1.1),
         ("qkv_rope_fwd", 0.88), ("lm_head_piece", 0.70), ("qkv_rope_bwd", 0.34), ("lm_head_finish", 0.01)]
    ax.barh([n for n, _ in k][::-1], [v for _, v in k][::-1], color=ACCENT)
    for i, (_, v) in enumerate(k[::-1]):
        ax.text(v + 0.1, i, f"{ru(v, 2 if v < 1 else 1)}%", va="center", fontsize=12)
    ax.set_xlim(0, 9.5)
    ax.set_xlabel("доля времени обучения (проверка 4, шаги 3-64)")
    ax.set_title("Наши кернелы: вместе 16,8% (порог 5%)", fontsize=14, loc="left")
    bullets(fig, ["Ни одного tl.dot; пример шаблона удалён.",
                  "rms_fwd за один проход: residual add, RMSNorm в fp32, каст в fp16 и входы обеих "
                  "LoRA с dropout; маска - функция seed и позиции, в backward генерируется заново.",
                  "Шаг на GPU: GEMM cuBLAS ~69% работы; GPU занят 82-88%: щели между мелкими "
                  "запусками ~20 мс/шаг, граница шага тренера HF 8-15 мс.",
                  "Идеи (код наш): Liger-Kernel, Unsloth, Cut Cross-Entropy, QLoRA, online softmax, "
                  "ZeRO-Offload - ссылки в README."],
            x=0.57, width_chars=40, size=14)


def s5_nf4(fig):
    ax = fig.add_axes([0.07, 0.18, 0.38, 0.56])
    ax.bar(["bitsandbytes", "наш кернел", "копия fp16\n(предел полосы)"], [137, 212, 232],
           color=[BASE, ACCENT, "#e1e4e8"])
    for i, v in enumerate([137, 212, 232]):
        ax.text(i, v + 4, f"{v} ГБ/с", ha="center", fontsize=13)
    ax.set_ylim(0, 260)
    ax.set_title("Деквантизация NF4 на T4", fontsize=14, loc="left")
    bullets(fig, ["Узкое место было не в полосе памяти, а в загрузках таблицы на каждый элемент "
                  "(~100 мс на шаг).",
                  "Тайл из блоков, уровни NF4 из констант, absmax двойной квантизации побитно как в "
                  "bitsandbytes: ~100 -> 36 мс на шаг.",
                  "Один запуск на [q|k|v] и один на [gate|up] вместо двух кернелов bnb на вес.",
                  "23 варианта перепробовали; остановились у предела полосы.",
                  "Speedup всего решения 1,39x -> 1,53x от одного этого кернела."],
            x=0.52, width_chars=44, size=14)


def s6_head(fig):
    n = 19
    for j in range(n):
        x = 0.05 + j * 0.047
        fc = ACCENT if j < 3 else LIGHT
        fig.add_artist(FancyBboxPatch((x, 0.62), 0.038, 0.10, boxstyle="round,pad=0.001",
                                      transform=fig.transFigure, fc=fc, ec="none"))
    fig.text(0.05, 0.75, "вес lm_head 151 936 x 1 536 (445 MiB) в pinned-памяти CPU: 19 кусков по 8 192 строки",
             fontsize=13, color=INK)
    fig.text(0.05, 0.58, "на GPU одновременно 3 буфера по 24 MiB; копии начинаются ещё на последних 3 слоях forward",
             fontsize=13, color=ACCENT)
    bullets(fig, ["На каждый кусок: GEMM lm_head (cuBLAS) -> lm_head_piece: online-состояние строки "
                  "(максимум, сумма exp, argmax) и накопление градиента с пересчётом при росте максимума.",
                  "В конце lm_head_finish: loss, энтропия, accuracy и градиент. Логиты на весь словарь не "
                  "существуют ни в какой момент.",
                  "Строки промпта (~37% токенов) получают только энтропию и argmax: без backward-GEMM "
                  "через матрицу 151 936 x 1 536.",
                  "Loss, энтропия, accuracy и loss строк для rewind - в том же проходе, значения как у "
                  "базы (проверка 5). Честно: против головы на GPU loss совпадает до ~1e-7, не побитно."],
            y=0.50, width_chars=100, size=14)


def s7_waterfall(fig):
    ax = fig.add_axes([0.07, 0.37, 0.88, 0.40])
    steps = [("база", 14345, BASE), ("слитый шаг\nбез паддинга*", 3355, LIGHT),
             ("пересчёт\nслоя*", 1468, LIGHT), ("NF4-веса\nна CPU", 1385, ACCENT),
             ("слоты после\nосвобождения", 1275, ACCENT), ("NF4 на хост\nпри загрузке", 1233, ACCENT),
             ("голова\nна CPU", 813, ACCENT), ("входы слоёв\nна CPU", 735, ACCENT),
             ("потолок шага\n420 MiB", 635, ACCENT), ("[gate | up]\nна CPU (Kaggle)", 639, ACCENT),
             ("стек CUDA\n(Kaggle)", 601, ACCENT)]
    xs = range(len(steps))
    ax.bar(xs, [v for _, v, _ in steps], color=[c for _, _, c in steps], width=0.7)
    ax.set_yscale("log")
    ax.set_ylim(400, 25000)
    ax.set_xticks(list(xs), [n for n, _, _ in steps], fontsize=9)
    for x, (_, v, _) in zip(xs, steps):
        ax.text(x, v * 1.08, ru(v), ha="center", fontsize=12, weight="bold" if x in (0, len(steps) - 1) else None)
    ax.set_ylabel("пик, MiB (лог. шкала)")
    ax.set_title("Пик памяти GPU по шагам работы (nvidia-smi; * - allocated на RTX 4060)", fontsize=14, loc="left")
    y = para(fig, 0.05, 0.265, "На GPU нет ни одного замороженного веса целиком и ни одного отложенного входа "
             "слоя: нормы, LoRA, 3 слота весов слоя, 3 буфера головы, кольца и активации текущего слоя.",
             115, size=13)
    para(fig, 0.05, y - 0.015, "Шаг сам прогнозирует свой пик и тратит запас под потолком 420 MiB на хранение "
         "вместо пересчёта. Пик не зависит от данных (Brev, со стеком CUDA): 597-599 MiB на публичных, "
         "579-599 на хвосте с длинными батчами, 573 на батчах 4 x 512.",
         115, size=13)


def s8_results(fig):
    rows = [("", "База", "Решение", "Отношение"),
            ("soup train, с", "84,6", "49,8", "1,698x"),
            ("цикл 64 шагов, с", "62,8", "29,9", "2,10x"),
            ("медиана шага, с", "0,981", "0,413", "2,3x"),
            ("пик GPU, MiB", "14 345", "601", "23,87x"),
            ("eval loss", "2,04965", "2,04953", "-0,006%"),
            ("проверки", "", "", "10 / 10")]
    for i, row in enumerate(rows):
        y = 0.74 - i * 0.065
        for j, cell in enumerate(row):
            fig.text(0.05 + [0, 0.17, 0.27, 0.37][j], y, cell, fontsize=14,
                     weight="bold" if i == 0 or j == 3 else None, color=ACCENT if j == 3 and i else INK)
    fig.text(0.05, 0.30, "Kaggle T4, --repeats 2 (база, решение, решение, база), aafb9a7;\n"
             "медиана - запуск 1. Сессии Kaggle различаются (база 79-88 с):\nнаши r2 - 1,598-1,712x, последние три - 1,698-1,712x.",
             fontsize=12, color=MUTED, linespacing=1.4, va="top")
    para(fig, 0.05, 0.17, "Срезы, похожие на скрытые (Brev T4): 10/10 проверок; со стеком CUDA пик "
         "579-599 MiB; на батчах 4 x 512 база падает с OOM, решение - 573 MiB.", 62, size=12)
    ax = fig.add_axes([0.56, 0.20, 0.38, 0.52])
    versions = [3, 4, 5, "6-8", 9, 10, 11, 12, 13, 14, 15]
    s = [1.42, 1.53, 1.57, 1.56, 1.60, 1.61, 1.624, 1.628, 1.705, 1.712, 1.698]
    m = [11.25, 11.25, 11.23, 11.23, 11.23, 11.23, 17.64, 22.59, 22.45, 22.45, 23.87]
    xs = range(len(versions))
    ax.plot(xs, s, marker="o", color=ACCENT, lw=2)
    ax.set_ylabel("speedup s (r2)", color=ACCENT)
    ax.set_ylim(1.3, 1.8)
    ax2 = ax.twinx()
    ax2.plot(xs, m, marker="s", color=MUTED, lw=2)
    ax2.set_ylabel("memory m", color=MUTED)
    ax2.set_ylim(0, 25)
    ax2.spines["right"].set_visible(True)
    ax.set_xticks(list(xs), [str(v) for v in versions])
    ax.set_xlabel("версия (история версий в README)")
    ax.set_title("Kaggle T4 по версиям: s и m", fontsize=14, loc="left")
    ax.text(10, 1.698 + 0.02, "1,698", ha="center", color=ACCENT, fontsize=11)
    ax2.text(10, 23.87 + 0.9, "23,87", ha="center", color=MUTED, fontsize=11)


def s9_failed(fig):
    bullets(fig, ["Свой GEMM на Triton: tl.dot на sm_75 без тензорных ядер - ~1 TFLOPS против 16-34 у cuBLAS.",
                  "int8-GEMM (разрешён): в forward [gate | up] - первый loss до 1,29% при допуске 1%; только в "
                  "backward - точность как у fp16 и -58 MiB, но на Kaggle медленнее на 1,6-2,3 с (квантизация "
                  "упирается в медленный CPU).",
                  "Загрузка модели в фоновом потоке: 54,8 с против 54,3 - Python-код загрузки делит GIL.",
                  "CUDA Graphs: длина батча меняется каждый шаг, корзины стоят 3-7% лишних GEMM.",
                  "Голова с резидентными на GPU кусками (4 и 8 по 24 MiB): цикл не быстрее, пик выше.",
                  "Потолок шага ниже 420 MiB: 340 -> 555 MiB, но цикл +0,52 с; ниже 555 экономный шаг не опускается.",
                  "Случайный fingerprint для datasets.map (-2 с): организаторы сочли изменением данных - убрали."],
            width_chars=105, size=15)
    para(fig, 0.05, 0.16, "Каждую гипотезу меряли A/B в одной сессии T4 и оставляли только подтверждённое. "
             "Полный журнал - в README.", 90, size=14, color=ACCENT, weight="bold")


def s10_limits(fig):
    ax = fig.add_axes([0.66, 0.36, 0.29, 0.38])
    ranges = [("база", 960, 1115, BASE), ("решение", 770, 830, ACCENT), ("GEMM [gate | up]\nна горячем GPU", 680, 680, MUTED)]
    for i, (name, lo, hi, c) in enumerate(ranges):
        ax.plot([lo, hi], [i, i], color=c, lw=10, solid_capstyle="round", marker="o", ms=10)
        ax.text(hi + 20, i, f"{lo}-{hi}" if hi != lo else f"~{lo}", va="center", fontsize=12)
    ax.set_yticks(range(len(ranges)), [r[0] for r in ranges])
    ax.set_xlim(600, 1250)
    ax.invert_yaxis()
    ax.set_xlabel("частота SM под нагрузкой, МГц")
    ax.set_title("T4 на Kaggle - всегда в пределе 70 Вт", fontsize=13, loc="left")
    fig.text(0.62, 0.13, "Внутри запуска решения ~860 -> ~750 МГц,\nот запуска к запуску цикл 26,75 -> 28,22 с;\n"
             "r2 одним ноутбуком - от 1,598x до 1,712x.", fontsize=11, color=MUTED, linespacing=1.4)
    y = bullets(fig, ["Заточено под Qwen2 + NF4 + LoRA на q/v в fp16 (T4); остальное - исходный код Soup.",
                      "GEMM ~69% работы GPU и упираются в cuBLAS; вне цикла (~20 с) наш выигрыш ~7 с.",
                      "Выгрузки зависят от PCIe: на Kaggle (~13 ГБ/с) голова и входы - в пределах разброса, "
                      "на медленной шине ~2,4 с и ~0,7 с.",
                      "Loss не побитно равен варианту с головой на GPU (~1e-7); длинные батчи +9% цикла."],
                width_chars=56, size=14)
    para(fig, 0.05, y - 0.02, "Дальше: GEMM на тензорных ядрах T4 через mma.sync внутри Triton; bf16-путь "
         "для Ampere; LoRA на всех проекциях.", 54, size=14, color=ACCENT)
    fig.text(0.05, 0.075, f"{REPO}    {NOTEBOOK}", fontsize=11, color=MUTED)


SLIDES = [("", s1_title), ("Задача и база: куда уходят 14 ГБ", s2_base),
          ("Идея: заменить весь шаг, сохранив математику базы", s3_idea),
          ("Наши кернелы", s4_kernels), ("Кернел крупным планом: деквантизация NF4 под Turing", s5_nf4),
          ("Голова кусками словаря, вес на CPU", s6_head),
          ("Откуда 24x по памяти", s7_waterfall), ("Результаты против базы", s8_results),
          ("Что не сработало", s9_failed), ("Ограничения и дальнейшие шаги", s10_limits)]

if __name__ == "__main__":
    with PdfPages(OUT) as pdf:
        for i, (title, draw) in enumerate(SLIDES, 1):
            slide(pdf, title, i, draw)
    print("written", OUT)
