"""Figures of the README (docs/img/*.svg, a light and a dark variant of each, each on its own
background so it stays readable whichever GitHub theme shows it).

Numbers: the official Kaggle T4 measurement of bfff00c (version 24 of magurin/soup-kernel-run,
--repeats 2) and the memory waterfall of the README (nvidia-smi peaks on a T4).
Run: python docs/make_figures.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = Path(__file__).parent / "img"
THEMES = {
    "light": dict(bg="#ffffff", text="#1f2328", muted="#59636e", grid="#d1d9e0", base="#8c959f", base2="#d1d9e0",
                  ours="#1a7f37", ours2="#7ee2a8"),
    "dark": dict(bg="#0d1117", text="#e6edf3", muted="#9198a1", grid="#3d444d", base="#6e7681", base2="#3d444d",
                 ours="#3fb950", ours2="#1f6f3a"),
}
plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "path", "font.size": 11})

BASE_LOOP, BASE_WALL = 58.62, 80.26
OURS_LOOP, OURS_WALL = 25.57, 44.22
BASE_PEAK, OURS_PEAK = 14345, 463

WATERFALL = [
    ("База Soup 0.77.0", 14345),
    ("Слитый шаг, пересчёт слоя, fp16-эмбеддинг", 1761),
    ("NF4-веса слоёв в pinned-памяти CPU", 1385),
    ("Слоты после загрузки, ленивый возврат весов", 1275),
    ("Эмбеддинг и lm_head на CPU, голова кусками", 813),
    ("Входы слоёв для backward на CPU", 735),
    ("Потолок шага 420 MiB", 635),
    ("Стек потоков CUDA 16 байт", 601),
    ("Рабочая область cuBLAS :16:8", 583),
    ("Тензоры attention на CPU (AttnStream)", 563),
    ("2 слота весов слоя вместо 3", 543),
    ("Предвыборка головы в 2 буфера из 3", 523),
    ("[gate | up] половинами, потолок шага 340", 463),
]


def num(x, digits=1):
    s = f"{x:,.{digits}f}".replace(",", " ").replace(".", ",")
    return s


def style(ax, t):
    ax.set_facecolor(t["bg"])
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(t["grid"])
    ax.tick_params(colors=t["muted"], length=0)
    ax.xaxis.grid(True, color=t["grid"], linewidth=0.8)
    ax.set_axisbelow(True)


def hero(name, t):
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 3.3), gridspec_kw={"width_ratios": [1.25, 1]})
    fig.patch.set_facecolor(t["bg"])
    rows = ["Решение", "База"]
    loops, walls = [OURS_LOOP, BASE_LOOP], [OURS_WALL, BASE_WALL]
    a.barh(rows, loops, color=[t["ours"], t["base"]], height=0.56, label="цикл обучения, 64 шага")
    a.barh(rows, [w - l for w, l in zip(walls, loops)], left=loops, color=[t["ours2"], t["base2"]],
           height=0.56, label="вне цикла: импорт, загрузка, датасет, сохранение")
    for y, w in enumerate(walls):
        a.text(w + 1.2, y, f"{num(w)} с", va="center", color=t["text"], fontsize=12, fontweight="bold")
    a.text(OURS_WALL + 16, 0, "в 1,815 раза быстрее", va="center", color=t["ours"], fontsize=11)
    a.set_xlim(0, 100)
    a.set_title("Время всей команды soup train, с", loc="left", color=t["text"], fontsize=12.5,
                fontweight="bold", pad=10)
    a.legend(loc="lower right", bbox_to_anchor=(1.0, -0.30), ncol=2, frameon=False, fontsize=9,
             labelcolor=t["muted"], handlelength=1.2)
    style(a, t)
    a.tick_params(axis="y", colors=t["text"], labelsize=12)

    b.barh(rows, [OURS_PEAK, BASE_PEAK], color=[t["ours"], t["base"]], height=0.56)
    b.text(BASE_PEAK - 250, 1, f"{num(BASE_PEAK, 0)} MiB", va="center", ha="right", color="white",
           fontsize=12, fontweight="bold")
    b.text(OURS_PEAK + 250, 0, f"{OURS_PEAK} MiB  ·  в 30,98 раза меньше", va="center",
           color=t["text"], fontsize=12, fontweight="bold")
    b.set_xlim(0, 15500)
    b.set_xticks([0, 5000, 10000, 15000], ["0", "5 000", "10 000", "15 000"])
    b.set_title("Пик памяти GPU (nvidia-smi), MiB", loc="left", color=t["text"], fontsize=12.5,
                fontweight="bold", pad=10)
    style(b, t)
    b.tick_params(axis="y", colors=t["text"], labelsize=12)
    fig.text(0.01, -0.01, "Kaggle T4, Soup 0.77.0, официальный замер bfff00c: run_task.py --repeats 2, "
             "среднее двух запусков; eval loss +0,003%, все 10 проверок check_rules.py",
             color=t["muted"], fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / f"hero-{name}.svg", bbox_inches="tight", facecolor=t["bg"], pad_inches=0.25)
    plt.close(fig)


def waterfall(name, t):
    labels = [w[0] for w in WATERFALL][::-1]
    vals = [w[1] for w in WATERFALL][::-1]
    fig, ax = plt.subplots(figsize=(11, 5.6))
    fig.patch.set_facecolor(t["bg"])
    colors = [t["ours"] if i == 0 else (t["base"] if i == len(vals) - 1 else t["ours2"])
              for i in range(len(vals))]
    ax.barh(labels, vals, color=colors, height=0.66)
    ax.set_xscale("log")
    ax.set_xlim(300, 30000)
    for y, v in enumerate(vals):
        ax.text(v * 1.06, y, f"{num(v, 0)}", va="center", color=t["text"], fontsize=10.5,
                fontweight="bold" if y in (0, len(vals) - 1) else "normal")
    ax.set_xticks([500, 1000, 2000, 5000, 10000, 20000], ["500", "1 000", "2 000", "5 000", "10 000", "20 000"])
    ax.set_title("Откуда 31x по памяти: пик nvidia-smi после каждого шага, MiB (лог. шкала)", loc="left",
                 color=t["text"], fontsize=12.5, fontweight="bold", pad=10)
    style(ax, t)
    ax.tick_params(axis="y", colors=t["text"], labelsize=10.5)
    fig.text(0.01, -0.03, "T4: шаги до 635 MiB мерили на Brev, дальше - Kaggle A/B; база и итог 463 MiB - "
             "официальные замеры на Kaggle", color=t["muted"], fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / f"memory-{name}.svg", bbox_inches="tight", facecolor=t["bg"], pad_inches=0.25)
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    for name, t in THEMES.items():
        hero(name, t)
        waterfall(name, t)
    print("written", sorted(p.name for p in OUT.iterdir()))
