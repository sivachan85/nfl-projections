"""Generate the charts used in the README."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

PROC_DIR = Path(__file__).parent.parent / "data" / "processed"
FIG_DIR = Path(__file__).parent.parent / "figures"


def calibration_chart(df: pd.DataFrame, model: str = "proj_roll5") -> None:
    """Projected vs actual, binned in 10-yard buckets."""
    sub = df.dropna(subset=[model]).copy()
    sub["bin"] = (sub[model] // 10) * 10
    grouped = sub.groupby("bin").agg(
        actual=("receiving_yards", "mean"), n=("receiving_yards", "size"))
    grouped = grouped[grouped.n >= 50]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(grouped.index, grouped.actual, marker="o", label="Actual average")
    lim = [0, grouped.index.max()]
    ax.plot(lim, lim, "--", color="gray", label="Perfect calibration")
    ax.set_xlabel("Projected receiving yards")
    ax.set_ylabel("Actual receiving yards")
    ax.set_title("Projections drift above actuals at the high end")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "calibration.png", dpi=140)
    plt.close(fig)


def bias_chart(df: pd.DataFrame, model: str = "proj_roll5") -> None:
    """Mean signed error by projected volume bucket."""
    sub = df.dropna(subset=[model]).copy()
    sub["volume"] = pd.cut(sub[model], [-1, 20, 40, 60, 1000],
                           labels=["0-20", "20-40", "40-60", "60+"])
    sub["err"] = sub[model] - sub["receiving_yards"]
    bias = sub.groupby("volume", observed=True)["err"].mean()

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(bias.index.astype(str), bias.values)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Projected volume bucket (yards)")
    ax.set_ylabel("Mean error (projected - actual)")
    ax.set_title("Systematic bias by projected volume")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "bias_by_volume.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    FIG_DIR.mkdir(exist_ok=True)
    d = pd.read_parquet(PROC_DIR / "wr_projections.parquet")
    calibration_chart(d)
    bias_chart(d)
    print("charts written to figures/")
