"""Evaluate baseline projections: overall error and error by segment."""
from pathlib import Path
import numpy as np
import pandas as pd

PROC_DIR = Path(__file__).parent.parent / "data" / "processed"
MODELS = ["proj_roll3", "proj_roll5", "proj_roll8", "proj_career"]


def errors(df: pd.DataFrame, model: str) -> pd.Series:
    """MAE, RMSE and bias for one projection column."""
    sub = df.dropna(subset=[model])
    err = sub[model] - sub["receiving_yards"]
    return pd.Series({
        "n": len(sub),
        "MAE": err.abs().mean(),
        "RMSE": np.sqrt((err ** 2).mean()),
        "bias": err.mean(),
    })


def compare(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({m: errors(df, m) for m in MODELS}).T


def by_volume(df: pd.DataFrame, model="proj_roll5") -> pd.DataFrame:
    """Error broken out by how much the model projected."""
    sub = df.dropna(subset=[model]).copy()
    sub["volume"] = pd.cut(sub[model], [-1, 20, 40, 60, 1000],
                           labels=["0-20", "20-40", "40-60", "60+"])
    sub["err"] = sub[model] - sub["receiving_yards"]
    return sub.groupby("volume", observed=True).agg(
        n=("err", "size"),
        MAE=("err", lambda e: e.abs().mean()),
        bias=("err", "mean"))


def by_week(df: pd.DataFrame, model="proj_roll5") -> pd.DataFrame:
    sub = df.dropna(subset=[model]).copy()
    sub["err"] = sub[model] - sub["receiving_yards"]
    return sub.groupby("week").agg(
        n=("err", "size"),
        MAE=("err", lambda e: e.abs().mean())).head(8)


if __name__ == "__main__":
    d = pd.read_parquet(PROC_DIR / "wr_projections.parquet")
    print("=== model comparison ===")
    print(compare(d).round(2).to_string())
    print("\n=== error by projected volume ===")
    print(by_volume(d).round(2).to_string())
    print("\n=== MAE by week of season ===")
    print(by_week(d).round(2).to_string())
