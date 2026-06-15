"""
Rapor sekillerini uretir:
  - Sekil 4.2: Karpuz bazli LOWO dogruluk dagilimi (19 katlama) bar grafigi
  - Sekil 4.3: RFC iki sinifli (Yenir/Yenmez) konfuzyon matrisi

Gercek verilerle calisir: data/processed/*.npy uzerinden 2-sinif LOWO'yu
yeniden kosarak her katlamanin (karpuzun) dogrulugunu toplar; konfuzyon
matrisini de hesaplar (data/models/binary_lowo_results.json ile birebir
uyumludur).

Cikti: rapor/figures/sekil_4_2_lowo_bar.png, rapor/figures/sekil_4_3_cm.png
"""
import os
import sys
import numpy as np
from pathlib import Path

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

# --- Tema ---
GREEN = "#1B7F3A"
GREEN_L = "#7CB342"
RED = "#E53935"
SLATE = "#5F6B5C"
GRID = "#E3E8DC"
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.edgecolor": SLATE,
    "axes.linewidth": 0.8,
    "figure.dpi": 200,
})

OUT = ROOT / "rapor" / "figures"
OUT.mkdir(parents=True, exist_ok=True)


def load_data():
    X = np.load(str(ROOT / "data/processed/X_features.npy"))
    y = np.load(str(ROOT / "data/processed/y_labels.npy"))
    g = np.load(str(ROOT / "data/processed/groups.npy"))
    return X, y, g


def run_lowo(X, y_bin, groups):
    from sklearn.model_selection import LeaveOneGroupOut
    from sklearn.metrics import confusion_matrix
    from backend.module_e.classifier import RipenessClassifier

    logo = LeaveOneGroupOut()
    rows = []
    all_true, all_pred = [], []
    for tr, te in logo.split(X, y_bin, groups):
        held = int(np.unique(groups[te])[0])
        clf = RipenessClassifier(model_type="rfc")
        clf.train(X[tr], y_bin[tr])
        y_pred, _ = clf.predict(X[te])
        acc = float(np.mean(y_pred == y_bin[te]))
        # held-out karpuzun gercek sinifi (LOWO'da tek sinif olur)
        true_cls = int(np.round(np.mean(y_bin[te])))  # 0 Yenmez / 1 Yenir
        rows.append((held, acc, true_cls))
        all_true.extend(y_bin[te].tolist())
        all_pred.extend(y_pred.tolist())
        print(f"  Fold karpuz={held:2d} sinif={true_cls} acc={acc:.3f}")
    cm = confusion_matrix(all_true, all_pred, labels=[0, 1])
    mean_acc = float(np.mean([r[1] for r in rows]))
    return rows, cm, mean_acc


def fig_lowo_bar(rows, mean_acc):
    rows_sorted = sorted(rows, key=lambda r: r[0])
    karpuz = [str(r[0]) for r in rows_sorted]
    accs = [r[1] * 100 for r in rows_sorted]
    true_cls = [r[2] for r in rows_sorted]
    colors = [GREEN if c == 1 else RED for c in true_cls]

    fig, ax = plt.subplots(figsize=(10, 4.6))
    bars = ax.bar(karpuz, accs, color=colors, edgecolor="white",
                  linewidth=0.6, width=0.74, zorder=3)
    ax.axhline(mean_acc * 100, color=SLATE, linestyle="--", linewidth=1.3,
               zorder=4, label=f"Ortalama = %{mean_acc*100:.1f}")

    for b, a in zip(bars, accs):
        ax.text(b.get_x() + b.get_width() / 2, a + 1.5, f"{a:.0f}",
                ha="center", va="bottom", fontsize=8, color=SLATE)

    ax.set_ylim(0, 108)
    ax.set_ylabel("LOWO Dogrulugu (%)", fontsize=11)
    ax.set_xlabel("Held-out Karpuz (Katlama)", fontsize=11)
    ax.set_title("Karpuz Bazli LOWO Dogruluk Dagilimi (19 Katlama) — RFC, 2-Sinif",
                 fontsize=12, color=GREEN, fontweight="bold", pad=12)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)

    from matplotlib.patches import Patch
    legend_items = [
        Patch(facecolor=GREEN, label="Held-out: Yenir (Olgun)"),
        Patch(facecolor=RED, label="Held-out: Yenmez (Ham/Gecmis)"),
        plt.Line2D([0], [0], color=SLATE, linestyle="--",
                   label=f"Ortalama = %{mean_acc*100:.1f}"),
    ]
    ax.legend(handles=legend_items, loc="lower center", ncol=3,
              frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.30))
    fig.tight_layout()
    p = OUT / "sekil_4_2_lowo_bar.png"
    fig.savefig(str(p), bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {p}")


def fig_confusion(cm):
    # cm rows = gercek [Yenmez, Yenir], cols = tahmin [Yenmez, Yenir]
    labels = ["Yenmez\n(Ham/Gecmis)", "Yenir\n(Olgun)"]
    total = cm.sum()
    fig, ax = plt.subplots(figsize=(5.6, 5.0))

    # normalize (satir bazli) renk icin
    cm_norm = cm / cm.sum(axis=1, keepdims=True)
    im = ax.imshow(cm_norm, cmap="Greens", vmin=0, vmax=1)

    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("Tahmin Edilen Sinif", fontsize=11)
    ax.set_ylabel("Gercek Sinif", fontsize=11)
    ax.set_title("RFC Iki Sinifli Konfuzyon Matrisi (LOWO, n=4671)",
                 fontsize=12, color=GREEN, fontweight="bold", pad=14)

    for i in range(2):
        for j in range(2):
            n = int(cm[i, j])
            pct = 100.0 * n / total
            txt_color = "white" if cm_norm[i, j] > 0.55 else "#1A1A1A"
            ax.text(j, i - 0.08, f"{n}", ha="center", va="center",
                    fontsize=18, fontweight="bold", color=txt_color)
            ax.text(j, i + 0.20, f"%{pct:.1f}", ha="center", va="center",
                    fontsize=10, color=txt_color)

    ax.set_xticks([-0.5, 0.5, 1.5], minor=True)
    ax.set_yticks([-0.5, 0.5, 1.5], minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="minor", length=0)

    acc = (cm[0, 0] + cm[1, 1]) / total
    fig.text(0.5, 0.02, f"Toplam dogruluk (mikro) = %{acc*100:.1f}   |   "
             f"Yenir F1 = 0.67   Yenmez F1 = 0.54",
             ha="center", fontsize=9, color=SLATE)
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    p = OUT / "sekil_4_3_cm.png"
    fig.savefig(str(p), bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {p}")


def main():
    X, y, g = load_data()
    y_bin = np.where(y == 1, 1, 0).astype(np.int64)
    print("LOWO yeniden kosuluyor (RFC, 2-sinif)...")
    rows, cm, mean_acc = run_lowo(X, y_bin, g)
    print(f"\nOrtalama LOWO = %{mean_acc*100:.2f}")
    print(f"Konfuzyon matrisi:\n{cm}")
    fig_lowo_bar(rows, mean_acc)
    fig_confusion(cm)


if __name__ == "__main__":
    main()
