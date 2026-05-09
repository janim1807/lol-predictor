"""
Loader for Oracle's Elixir professional LoL match data.
Download CSV files from: https://oracleselixir.com/tools/downloads

Place CSVs in:  src/data/oracleselixir/

Performance notes
-----------------
build_game_df() uses vectorised pivot operations instead of per-game iteration.
load_and_cache() wraps the full pipeline with a Parquet cache keyed to the
source CSV mtimes — subsequent loads are ~30x faster (~0.5s vs ~20s).
"""

import re
import pandas as pd
import numpy as np
from pathlib import Path

ROLES = ["top", "jng", "mid", "bot", "sup"]

# League tier: reflects competitive strength & how meaningful Elo diffs are.
# 1 = world-class (LCK, LPL)
# 2 = major (LEC, LTA N)
# 3 = regional (everything else named)
# 4 = default / unknown
LEAGUE_TIER: dict[str, int] = {
    # Tier 1
    "LCK": 1, "LPL": 1,
    # Tier 2
    "LEC": 2, "LTA N": 2, "LTA": 2,
    # Tier 3
    "LCKC": 3, "LDL": 3, "LFL": 3, "NACL": 3, "LCP": 3,
    "LTA S": 3, "VCS": 3, "LJL": 3, "PCS": 3, "TCL": 3,
    "CBLOL": 3, "EBL": 3, "NLC": 3, "LFL2": 3, "LRS": 3,
    "LRN": 3, "LAS": 3, "LIT": 3, "HLL": 3, "RL": 3,
    "AL": 3, "HC": 3, "HM": 3, "HW": 3, "CD": 3,
    "EM": 3, "LPLOL": 3, "ROL": 3, "NEXO": 3, "LVP SL": 3,
    "FST": 3, "CT": 3, "PRM": 3, "KeSPA": 3, "CCWS": 3,
    # International events — Worlds/MSI are tier 1 (best teams per region)
    "WLDs": 1, "MSI": 1,
    # Other international events — tier 3 (mixed-region, limited cross-data)
    "EWC": 3, "ASI": 3,
    "Asia Master": 3, "DCup": 3, "NACL": 3,
}


# ── Raw CSV loading ────────────────────────────────────────────────────────────

def load_raw(path, complete_only: bool = True) -> pd.DataFrame:
    """Load one or more Oracle's Elixir CSV files.

    Args:
        path:          path to a single .csv file, or a directory of .csv files
        complete_only: drop rows where datacompleteness != 'complete'
    """
    path = Path(path)
    if path.is_dir():
        dfs = [pd.read_csv(f, low_memory=False) for f in sorted(path.glob("*.csv"))]
        if not dfs:
            raise FileNotFoundError(f"No CSV files found in {path}")
        df = pd.concat(dfs, ignore_index=True)
    else:
        df = pd.read_csv(path, low_memory=False)

    if complete_only and "datacompleteness" in df.columns:
        df = df[df["datacompleteness"] == "complete"].reset_index(drop=True)

    return df


# ── Fast wide-format builder ───────────────────────────────────────────────────

def build_game_df(raw: pd.DataFrame, since_year: int = 2020) -> pd.DataFrame:
    """Convert Oracle's Elixir long format (12 rows/game) to wide format (1 row/game).

    Vectorised: uses pivot_table instead of per-game Python iteration.
    ~10-15x faster than the previous groupby loop on 62k games.

    Output columns:
        gameid, patch, league, date,
        blue_team, red_team, blue_first_pick,
        blue_{top,jng,mid,bot,sup},         red_{top,jng,mid,bot,sup},
        blue_{top,jng,mid,bot,sup}_player,  red_{top,jng,mid,bot,sup}_player,
        blue_pick_order_{top,…},            red_pick_order_{top,…},
        blue_ban{1-5},                      red_ban{1-5},
        blue_win
    """
    if "year" in raw.columns:
        raw = raw[raw["year"] >= since_year]

    players = raw[raw["position"].str.lower().isin(ROLES)].copy()
    players["position"] = players["position"].str.lower()
    teams   = raw[raw["position"].str.lower() == "team"].copy() \
              if "position" in raw.columns else pd.DataFrame()

    # ── Step 1: game-level metadata ───────────────────────────────────────────
    meta_cols = [c for c in ["patch", "league", "date", "playoffs"] if c in players.columns]
    game_meta = (players.groupby("gameid", sort=False)[meta_cols]
                        .first()
                        .reset_index())
    game_meta["date"] = pd.to_datetime(game_meta["date"], errors="coerce")
    if "playoffs" in game_meta.columns:
        game_meta.rename(columns={"playoffs": "is_playoffs"}, inplace=True)
        game_meta["is_playoffs"] = game_meta["is_playoffs"].fillna(0).astype(int)
    else:
        game_meta["is_playoffs"] = 0

    # ── Step 2: team names + result per side ──────────────────────────────────
    for side, prefix in [("Blue", "blue"), ("Red", "red")]:
        side_df = players[players["side"] == side]
        agg = (side_df.groupby("gameid", sort=False)
                      .agg(**{
                          f"{prefix}_team":   ("teamname", "first"),
                          f"{prefix}_result": ("result",   "first"),
                      })
                      .reset_index())
        game_meta = game_meta.merge(agg, on="gameid", how="left")

    game_meta["blue_win"] = game_meta["blue_result"].fillna(0).astype(int)
    game_meta.drop(columns=["blue_result", "red_result"], inplace=True, errors="ignore")

    # ── Step 3: firstPick ─────────────────────────────────────────────────────
    if "firstPick" in players.columns:
        fp = (players[players["side"] == "Blue"]
              .groupby("gameid", sort=False)["firstPick"]
              .first())
        game_meta["blue_first_pick"] = game_meta["gameid"].map(fp).fillna(-1).astype(int)
    else:
        game_meta["blue_first_pick"] = -1

    # ── Step 4: champion + player pivot ──────────────────────────────────────
    # Create slot column: "blue_top", "red_mid", etc.
    players["slot"] = players["side"].str.lower() + "_" + players["position"]

    val_cols = [c for c in ["champion", "playername"] if c in players.columns]
    wide = players.pivot_table(
        index="gameid", columns="slot", values=val_cols, aggfunc="first"
    )
    # Flatten multi-level columns:
    #   ("champion",  "blue_top") → "blue_top"
    #   ("playername","blue_top") → "blue_top_player"
    wide.columns = [
        slot if stat == "champion" else f"{slot}_player"
        for stat, slot in wide.columns
    ]
    wide = wide.reset_index()
    game_meta = game_meta.merge(wide, on="gameid", how="inner")

    # ── Step 5: bans from team rows ───────────────────────────────────────────
    if not teams.empty:
        for side, prefix in [("Blue", "blue"), ("Red", "red")]:
            side_teams = teams[teams["side"] == side].set_index("gameid")
            for i in range(1, 6):
                col = f"ban{i}"
                game_meta[f"{prefix}_ban{i}"] = (
                    game_meta["gameid"].map(side_teams[col])
                    if col in side_teams.columns else None
                )
    else:
        for prefix in ["blue", "red"]:
            for i in range(1, 6):
                game_meta[f"{prefix}_ban{i}"] = None

    # ── Step 6: pick order (vectorised map) ───────────────────────────────────
    if not teams.empty:
        avail_picks = [f"pick{i}" for i in range(1, 6) if f"pick{i}" in teams.columns]
        if avail_picks:
            tm_long = (teams[["gameid", "side"] + avail_picks]
                       .melt(id_vars=["gameid", "side"],
                              value_vars=avail_picks,
                              var_name="pick_num",
                              value_name="champion")
                       .dropna(subset=["champion"]))
            tm_long["order"] = tm_long["pick_num"].str[4:].astype(int) - 1
            # Build (gameid, side, champion) → order lookup as a Series
            tm_long["key"] = (tm_long["gameid"].astype(str) + "|"
                               + tm_long["side"] + "|"
                               + tm_long["champion"].astype(str))
            pick_lookup = tm_long.set_index("key")["order"]

            for role in ROLES:
                for side, cap_side in [("blue", "Blue"), ("red", "Red")]:
                    champ_col = f"{side}_{role}"
                    if champ_col not in game_meta.columns:
                        game_meta[f"{side}_pick_order_{role}"] = -1
                        continue
                    keys = (game_meta["gameid"].astype(str) + "|"
                            + cap_side + "|"
                            + game_meta[champ_col].fillna("").astype(str))
                    game_meta[f"{side}_pick_order_{role}"] = (
                        keys.map(pick_lookup).fillna(-1).astype(int)
                    )
    else:
        for role in ROLES:
            for side in ["blue", "red"]:
                game_meta[f"{side}_pick_order_{role}"] = -1

    # ── Step 7: league tier ───────────────────────────────────────────────────
    game_meta["league_tier"] = game_meta["league"].map(LEAGUE_TIER).fillna(4).astype(int)

    # ── Step 8: drop games missing any champion pick ──────────────────────────
    pick_cols = [f"blue_{r}" for r in ROLES] + [f"red_{r}" for r in ROLES]
    game_meta = game_meta.dropna(subset=pick_cols)

    return game_meta.sort_values("date").reset_index(drop=True)


# ── Parquet cache ──────────────────────────────────────────────────────────────

def load_and_cache(
    csv_dir:       str | Path,
    cache_path:    str | Path | None = None,
    complete_only: bool = True,
    since_year:    int  = 2020,
    force_rebuild: bool = False,
) -> pd.DataFrame:
    """Load game_df with a Parquet cache.

    On first call (or when source CSVs are newer than the cache):
      reads CSVs → build_game_df() → saves Parquet → returns DataFrame

    On subsequent calls:
      loads Parquet directly (~0.5s instead of ~20s)

    Args:
        csv_dir:       directory containing OraclesElixir CSV files
        cache_path:    where to save/load the Parquet file.
                       Defaults to <csv_dir>/../game_df_cache.parquet
        complete_only: passed to load_raw
        since_year:    passed to build_game_df
        force_rebuild: ignore cache and rebuild from CSVs
    """
    csv_dir    = Path(csv_dir)
    cache_path = Path(cache_path) if cache_path else csv_dir.parent / "game_df_cache.parquet"

    csv_files  = sorted(csv_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {csv_dir}")

    # Cache is valid if it exists and is newer than all source CSVs
    if not force_rebuild and cache_path.exists():
        cache_mtime = cache_path.stat().st_mtime
        newest_csv  = max(f.stat().st_mtime for f in csv_files)
        if cache_mtime >= newest_csv:
            df = pd.read_parquet(cache_path)
            df["date"] = pd.to_datetime(df["date"])
            print(f"Loaded game_df from cache ({len(df):,} games) — {cache_path.name}")
            return df

    print(f"Building game_df from {len(csv_files)} CSV file(s)…")
    raw    = load_raw(csv_dir, complete_only=complete_only)
    df     = build_game_df(raw, since_year=since_year)
    df.to_parquet(cache_path, index=False)
    print(f"Built and cached {len(df):,} games → {cache_path}")
    return df


# ── Vocabularies ───────────────────────────────────────────────────────────────

def build_vocabularies(game_df: pd.DataFrame):
    """Build champion, team, and patch → integer index mappings.

    Index 0 is reserved for unknown/padding in all three.

    Returns:
        champ2idx:  dict[str, int]
        team2idx:   dict[str, int]
        patch2idx:  dict[str, int]
    """
    pick_cols = [f"blue_{r}" for r in ROLES] + [f"red_{r}" for r in ROLES]
    ban_cols  = [f"blue_ban{i}" for i in range(1, 6)] + [f"red_ban{i}" for i in range(1, 6)]
    all_champs = pd.concat([game_df[c] for c in pick_cols + ban_cols
                             if c in game_df.columns]).dropna().unique()
    champ2idx = {c: i + 1 for i, c in enumerate(sorted(all_champs))}

    all_teams = pd.concat([game_df["blue_team"], game_df["red_team"]]).dropna().unique()
    team2idx  = {t: i + 1 for i, t in enumerate(sorted(all_teams))}

    def _norm_patch(p):
        m = re.match(r"(\d+\.\d+)", str(p))
        return m.group(1) if m else None

    all_patches = game_df["patch"].map(_norm_patch).dropna().unique()
    patch2idx   = {p: i + 1 for i, p in enumerate(sorted(all_patches))}

    return champ2idx, team2idx, patch2idx
