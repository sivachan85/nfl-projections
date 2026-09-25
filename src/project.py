"""Build baseline projections using only prior-week data."""
from pathlib import Path
import pandas as pd

PROC_DIR = Path(__file__).parent.parent / "data" / "processed"


def add_projections(df: pd.DataFrame, windows=(3, 5, 8)) -> pd.DataFrame:
    """Add rolling-average and career-average projections.

    Every projection uses .shift(1) so the value for week N depends
    only on weeks before N. This is the leakage guard.
    """
    df = df.sort_values(["player_id", "season", "week"]).copy()
    g = df.groupby("player_id", group_keys=False)["receiving_yards"]

    for w in windows:
        df[f"proj_roll{w}"] = g.transform(
            lambda s: s.shift(1).rolling(w, min_periods=2).mean()
        )

    df["proj_career"] = g.transform(
        lambda s: s.shift(1).expanding(min_periods=2).mean()
    )
    return df


if __name__ == "__main__":
    d = pd.read_parquet(PROC_DIR / "wr_weeks.parquet")
    d = add_projections(d)
    d.to_parquet(PROC_DIR / "wr_projections.parquet", index=False)

    cols = ["player", "season", "week", "receiving_yards",
            "proj_roll3", "proj_roll5", "proj_career"]
    print(d[d.player == "Larry Fitzgerald"][cols].head(8).to_string(index=False))
    print("\nnon-null proj_roll3:", d.proj_roll3.notna().sum())
