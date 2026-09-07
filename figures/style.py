import os
from pathlib import Path
ROOT = Path(os.environ.get('SPARSE_CODING_ROOT', Path(__file__).resolve().parents[1]))
from pathlib import Path
import matplotlib as mpl


HERE = ROOT / "figmain"
MAIN = HERE / "main"
EXT = HERE / "extended"

# Palette sampled from fig1_core_concept_v1.png. Colors have fixed semantics.
PASSIVE = "#18B8C8"
ACTIVE = "#F5823A"
MODEL = "#7651A2"
BIO = "#355F7F"
CONTROL = "#A9AFB6"
INK = "#202428"
MID = "#68737D"
LIGHT = "#DCE1E5"
PALE_PASSIVE = "#DDF4F6"
PALE_ACTIVE = "#FCE9DD"


def setup():
    mpl.rcParams.update({
        "font.family": "Arial",
        "font.size": 7.5,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.linewidth": .75,
        "lines.linewidth": 1.35,
        "xtick.major.width": .65,
        "ytick.major.width": .65,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.dpi": 400,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "text.color": INK,
        "axes.edgecolor": INK,
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
    })


def clean(ax, grid=False):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if grid:
        ax.grid(axis="y", color=LIGHT, lw=.45, zorder=0)


def panel(ax, letter, x=-.16, y=1.06):
    writer = ax.text2D if hasattr(ax, "text2D") else ax.text
    writer(x, y, letter, transform=ax.transAxes, fontsize=11,
           fontweight="bold", ha="left", va="bottom")


def save(fig, name, folder=MAIN):
    folder.mkdir(parents=True, exist_ok=True)
    for ext in ("svg", "pdf", "png"):
        fig.savefig(folder / f"{name}.{ext}", bbox_inches="tight", facecolor="white")
