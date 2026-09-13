"""Build a clean WR receiving-yards dataset from raw nflverse files."""
from pathlib import Path
import duckdb
import pandas as pd

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
PROC_DIR = Path(__file__).parent.parent / "data" / "processed"

QUERY = """
SELECT
    player_id,
    player_display_name AS player,
    recent_team          AS team,
    opponent_team        AS opponent,
    season,
    week,
    targets,
    receptions,
    receiving_yards
FROM read_parquet(?)
WHERE position = 'WR'
  AND season_type = 'REG'
  AND opponent_team IS NOT NULL
ORDER BY player_id, season, week
"""


def build() -> pd.DataFrame:
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    pattern = str(RAW_DIR / "player_stats_*.parquet")

    con = duckdb.connect()
    df = con.execute(QUERY, [pattern]).df()

    df["receiving_yards"] = df["receiving_yards"].fillna(0)
    df["targets"] = df["targets"].fillna(0)

    out = PROC_DIR / "wr_weeks.parquet"
    df.to_parquet(out, index=False)
    print(f"{len(df):,} WR player-weeks -> {out.name}")
    print(f"seasons: {df.season.min()}-{df.season.max()}, players: {df.player_id.nunique():,}")
    return df


if __name__ == "__main__":
    d = build()
    print(d.head(8).to_string(index=False))