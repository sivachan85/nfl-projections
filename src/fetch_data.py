"""Fetch NFL weekly player stats from nflverse and cache locally."""
from pathlib import Path
import pandas as pd

BASE_URL = "https://github.com/nflverse/nflverse-data/releases/download/player_stats"
RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
SEASONS = range(2018, 2025)


def fetch_season(season: int, force: bool = False) -> pd.DataFrame:
    """Download one season, using the local cache if it already exists."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    local_path = RAW_DIR / f"player_stats_{season}.parquet"

    if local_path.exists() and not force:
        return pd.read_parquet(local_path)

    url = f"{BASE_URL}/player_stats_{season}.parquet"
    print(f"  {season}: downloading")
    df = pd.read_parquet(url)
    df.to_parquet(local_path, index=False)
    return df


def fetch_all(force: bool = False) -> pd.DataFrame:
    """Fetch every season and stack them into one DataFrame."""
    frames = [fetch_season(s, force=force) for s in SEASONS]
    combined = pd.concat(frames, ignore_index=True)
    print(f"{len(combined):,} player-weeks across {len(SEASONS)} seasons")
    return combined


if __name__ == "__main__":
    fetch_all()