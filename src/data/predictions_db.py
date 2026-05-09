"""SQLite store for saved match predictions.

Predictions are stored before games are played; results are auto-filled
by reconcile_results() after refreshing the OraclesElixir data.
"""

import json
import sqlite3
import pandas as pd
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

DB_PATH = Path(__file__).parent.parent.parent / "models" / "predictions.db"


def _connect(path: Path = DB_PATH) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    return conn


def init_db(path: Path = DB_PATH):
    """Create tables if they don't exist, and migrate missing columns."""
    with _connect(path) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                saved_at        TEXT    NOT NULL,
                match_date      TEXT,
                league          TEXT,
                tournament      TEXT,
                blue_team       TEXT    NOT NULL,
                red_team        TEXT    NOT NULL,
                is_playoffs     INTEGER NOT NULL DEFAULT 0,
                best_of         INTEGER NOT NULL DEFAULT 1,
                elo_prob        REAL    NOT NULL,
                xgb_prob        REAL    NOT NULL,
                lgbm_prob       REAL,
                gnn_prob        REAL    NOT NULL,
                tabpfn_prob     REAL,
                ensemble_prob   REAL    NOT NULL,
                bookmaker_prob  REAL,
                series_prob     REAL,
                score_dist      TEXT,
                actual_blue_win INTEGER,
                resolved_at     TEXT,
                actual_score    TEXT
            )
        """)
        # Migrate existing databases that lack columns added after initial release
        existing = {r[1] for r in conn.execute("PRAGMA table_info(predictions)")}
        for col, typedef in [
            ("lgbm_prob",     "REAL"),
            ("tabpfn_prob",   "REAL"),
            ("best_of",       "INTEGER NOT NULL DEFAULT 1"),
            ("series_prob",   "REAL"),
            ("score_dist",    "TEXT"),
            ("actual_score",  "TEXT"),
            ("bookmaker_prob","REAL"),
        ]:
            if col.split()[0] not in existing:
                conn.execute(f"ALTER TABLE predictions ADD COLUMN {col}")
        conn.commit()


def save_prediction(
    blue_team:      str,
    red_team:       str,
    elo_prob:       float,
    xgb_prob:       float,
    gnn_prob:       float,
    ensemble_prob:  float,
    lgbm_prob:      Optional[float] = None,
    tabpfn_prob:    Optional[float] = None,
    bookmaker_prob: Optional[float] = None,
    series_prob:    Optional[float] = None,
    score_dist:     Optional[list]  = None,
    best_of:        int             = 1,
    match_date:     Optional[str]   = None,
    league:         Optional[str]   = None,
    tournament:     Optional[str]   = None,
    is_playoffs:    bool            = False,
    path:           Path            = DB_PATH,
) -> int:
    """Insert a prediction. Returns the new row id."""
    init_db(path)
    now        = datetime.now(timezone.utc).isoformat()
    dist_json  = json.dumps(score_dist) if score_dist is not None else None
    with _connect(path) as conn:
        cur = conn.execute("""
            INSERT INTO predictions
                (saved_at, match_date, league, tournament,
                 blue_team, red_team, is_playoffs, best_of,
                 elo_prob, xgb_prob, lgbm_prob, gnn_prob, tabpfn_prob,
                 ensemble_prob, bookmaker_prob, series_prob, score_dist)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (now, match_date, league, tournament,
              blue_team, red_team, int(is_playoffs), int(best_of),
              round(elo_prob, 4), round(xgb_prob, 4),
              round(lgbm_prob, 4)      if lgbm_prob      is not None else None,
              round(gnn_prob, 4),
              round(tabpfn_prob, 4)    if tabpfn_prob    is not None else None,
              round(ensemble_prob, 4),
              round(bookmaker_prob, 4) if bookmaker_prob is not None else None,
              round(series_prob, 4)    if series_prob    is not None else None,
              dist_json))
        conn.commit()
        return cur.lastrowid


def delete_prediction(pred_id: int, path: Path = DB_PATH):
    """Permanently delete a single prediction row."""
    with _connect(path) as conn:
        conn.execute("DELETE FROM predictions WHERE id = ?", (pred_id,))
        conn.commit()


def set_result(pred_id: int, blue_won: int, actual_score: Optional[str] = None,
               path: Path = DB_PATH):
    """Manually mark the result for a prediction (1=blue won, 0=red won).

    actual_score: optional score string, e.g. "2-1" (blue perspective).
    """
    now = datetime.now(timezone.utc).isoformat()
    with _connect(path) as conn:
        conn.execute("""
            UPDATE predictions
               SET actual_blue_win = ?, actual_score = ?, resolved_at = ?
             WHERE id = ?
        """, (int(blue_won), actual_score, now, pred_id))
        conn.commit()


def reconcile_results(game_df: pd.DataFrame, path: Path = DB_PATH, window_days: int = 3):
    """
    Auto-fill results for unresolved predictions by matching against game_df.

    Matches on (blue_team, red_team) where the game date in game_df falls
    within `window_days` of the prediction's match_date.

    Call this after downloading fresh OraclesElixir data.

    Returns number of predictions resolved.
    """
    init_db(path)
    with _connect(path) as conn:
        rows = conn.execute("""
            SELECT id, match_date, blue_team, red_team, saved_at
              FROM predictions
             WHERE actual_blue_win IS NULL
        """).fetchall()

    if not rows:
        return 0

    resolved = 0
    now = datetime.now(timezone.utc).isoformat()

    for row in rows:
        blue, red = row["blue_team"], row["red_team"]
        pred_date  = row["match_date"]
        saved_at   = row["saved_at"]

        # Normalise pred_date to a tz-naive timestamp for window comparisons.
        pd_ts = None
        if pred_date:
            try:
                pd_ts = pd.Timestamp(pred_date).tz_localize(None).normalize()
            except TypeError:
                pd_ts = pd.Timestamp(pred_date).normalize()

        # Search both side assignments — sometimes the scheduled blue/red differs
        # from what actually happened in game_df (LEC/LCK can swap sides in Bo3).
        # flipped=True means blue/red were reversed; we invert blue_win accordingly.
        found = False
        for flipped, b, r in [(False, blue, red), (True, red, blue)]:
            mask = (game_df["blue_team"] == b) & (game_df["red_team"] == r)
            candidates = game_df[mask].copy()
            if candidates.empty:
                continue

            # Date window — anchor to match_date if available.
            # The saved_at guard is only applied when match_date is absent, to
            # prevent matching a future prediction to an old historical game.
            if pd_ts is not None:
                try:
                    candidates = candidates[
                        (candidates["date"] >= pd_ts - pd.Timedelta(days=1)) &
                        (candidates["date"] <= pd_ts + pd.Timedelta(days=window_days))
                    ]
                except Exception:
                    pass
            elif saved_at:
                # Fallback: no match_date — only take games after the prediction was saved
                try:
                    saved_ts = pd.Timestamp(saved_at).tz_localize(None).normalize()
                    candidates = candidates[candidates["date"] >= saved_ts]
                except Exception:
                    pass

            if candidates.empty:
                continue

            match    = candidates.sort_values("date").iloc[-1]
            blue_won = int(match["blue_win"])
            if flipped:
                blue_won = 1 - blue_won  # invert: our "blue" was actually red in game_df

            with _connect(path) as conn:
                conn.execute("""
                    UPDATE predictions
                       SET actual_blue_win = ?, resolved_at = ?
                     WHERE id = ?
                """, (blue_won, now, row["id"]))
                conn.commit()
            resolved += 1
            found = True
            break

    return resolved


def load_predictions(path: Path = DB_PATH) -> pd.DataFrame:
    """Load all predictions as a DataFrame, newest first."""
    init_db(path)
    with _connect(path) as conn:
        df = pd.read_sql_query(
            "SELECT * FROM predictions ORDER BY saved_at DESC",
            conn,
        )
    return df


def prediction_stats(df: pd.DataFrame) -> dict:
    """
    Compute performance statistics over resolved predictions.

    Returns dict with:
        total, resolved, accuracy, brier_score,
        ece (expected calibration error),
        per_model_accuracy (elo/xgb/gnn/ensemble)
    """
    import numpy as np

    resolved = df[df["actual_blue_win"].notna()].copy()
    if resolved.empty:
        return {"total": len(df), "resolved": 0}

    y = resolved["actual_blue_win"].values.astype(float)

    stats: dict = {
        "total":    len(df),
        "resolved": len(resolved),
    }

    for model in ["elo", "xgb", "lgbm", "gnn", "tabpfn", "ensemble", "bookmaker"]:
        prob_col = f"{model}_prob"
        if prob_col not in resolved.columns or resolved[prob_col].isna().all():
            continue
        probs = resolved[prob_col].dropna()
        y_sub = y[resolved[prob_col].notna()]
        preds = (probs > 0.5).astype(float)
        stats[f"{model}_accuracy"] = float((preds == y_sub).mean())
        stats[f"{model}_brier"]    = float(np.mean((probs.values - y_sub) ** 2))

    # ECE for ensemble
    probs = resolved["ensemble_prob"].values
    bins  = np.linspace(0, 1, 11)
    ece   = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (probs >= lo) & (probs < hi)
        if mask.sum() == 0:
            continue
        ece += mask.sum() * abs(y[mask].mean() - probs[mask].mean())
    stats["ece"] = float(ece / len(resolved))

    return stats
