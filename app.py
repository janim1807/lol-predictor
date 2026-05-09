"""
LoL Pro Match Prediction Dashboard
===================================
Run:  streamlit run app.py
"""

import sys
import os
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Load .env from project root before anything else
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass  # python-dotenv not installed — use shell env vars instead

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import numpy as np
import pandas as pd
import torch
import joblib
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

from data.oracleselixir import load_raw, build_game_df, build_vocabularies, load_and_cache
from data.leaguepedia import fetch_schedule as _fetch_schedule_raw
from data.odds_api import fetch_match_odds, list_lol_sport_keys


@st.cache_data(ttl=300, show_spinner=False)
def fetch_schedule(date: str, leagues: tuple | None) -> list[dict]:
    return _fetch_schedule_raw(date=date, leagues=list(leagues) if leagues else None)
from data.predictions_db import (
    save_prediction, load_predictions, reconcile_results,
    prediction_stats, set_result, delete_prediction, init_db,
)
from models.elo import EloSystem, PlayerEloSystem, LeagueEloSystem
from models.match_gnn import build_match_graph, PreDraftPredictor

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT     = Path(__file__).parent
SAVE_DIR = ROOT / "models" / "predraft"
DATA_DIR = ROOT / "src" / "data" / "oracleselixir"

FEATURE_COLS = [
    "elo_diff",
    "blue_ewma_wr", "red_ewma_wr",
    "is_playoffs",
    "blue_elo_momentum", "red_elo_momentum",
    "blue_top_elo", "blue_jng_elo", "blue_mid_elo", "blue_bot_elo", "blue_sup_elo",
    "red_top_elo",  "red_jng_elo",  "red_mid_elo",  "red_bot_elo",  "red_sup_elo",
    "league_tier",
    "blue_games_conf", "red_games_conf",
]
FEATURE_LABELS = [
    "Elo diff", "Blue EWMA WR", "Red EWMA WR", "Playoffs",
    "Blue Elo momentum", "Red Elo momentum",
    "Blue TOP elo", "Blue JNG elo", "Blue MID elo", "Blue BOT elo", "Blue SUP elo",
    "Red TOP elo",  "Red JNG elo",  "Red MID elo",  "Red BOT elo",  "Red SUP elo",
    "League tier",
    "Blue games (conf)", "Red games (conf)",
]

# Mirror of src/data/oracleselixir.py LEAGUE_TIER
LEAGUE_TIER: dict[str, int] = {
    "LCK": 1, "LPL": 1,
    "LEC": 2, "LTA N": 2, "LTA": 2,
    "LCKC": 3, "LDL": 3, "LFL": 3, "NACL": 3, "LCP": 3,
    "LTA S": 3, "VCS": 3, "LJL": 3, "PCS": 3, "TCL": 3,
    "CBLOL": 3, "EBL": 3, "NLC": 3, "LFL2": 3, "LRS": 3,
    "LRN": 3, "LAS": 3, "LIT": 3, "HLL": 3, "RL": 3,
    "AL": 3, "HC": 3, "HM": 3, "HW": 3, "CD": 3,
    "EM": 3, "LPLOL": 3, "ROL": 3, "NEXO": 3, "LVP SL": 3,
    "FST": 3, "CT": 3, "PRM": 3, "KeSPA": 3, "CCWS": 3,
    "WLDs": 1, "MSI": 1,
    "EWC": 3, "ASI": 3,
    "Asia Master": 3, "DCup": 3,
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LoL Match Predictor",
    page_icon="⚔️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Design system ─────────────────────────────────────────────────────────────
# Palette
C_BG        = "#0d1117"
C_CARD      = "#161b22"
C_CARD2     = "#1c2333"
C_BORDER    = "#30363d"
C_BLUE      = "#1f6feb"
C_BLUE_SOFT = "#388bfd"
C_RED       = "#da3633"
C_RED_SOFT  = "#f85149"
C_GOLD      = "#d29922"
C_TEXT      = "#e6edf3"
C_MUTED     = "#7d8590"
C_GREEN     = "#3fb950"

st.markdown(f"""
<style>
/* ── Base ────────────────────────────────────────────────────────────────── */
html, body, [class*="css"] {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
}}
.stApp {{ background-color: {C_BG}; }}

/* ── Hide default Streamlit chrome ──────────────────────────────────────── */
#MainMenu, footer, header {{ visibility: hidden; }}
.block-container {{ padding-top: 1.5rem; padding-bottom: 2rem; }}

/* ── Tabs ────────────────────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {{
    gap: 4px;
    background: {C_CARD};
    border-radius: 10px;
    padding: 4px;
    border: 1px solid {C_BORDER};
}}
.stTabs [data-baseweb="tab"] {{
    border-radius: 8px;
    color: {C_MUTED};
    font-weight: 500;
    padding: 8px 20px;
    border: none !important;
    background: transparent !important;
}}
.stTabs [aria-selected="true"] {{
    background: {C_CARD2} !important;
    color: {C_TEXT} !important;
}}

/* ── Buttons ─────────────────────────────────────────────────────────────── */
.stButton > button {{
    background: {C_CARD2};
    color: {C_TEXT};
    border: 1px solid {C_BORDER};
    border-radius: 8px;
    font-weight: 500;
    transition: all 0.15s ease;
}}
.stButton > button:hover {{
    background: {C_BORDER};
    border-color: {C_BLUE_SOFT};
    color: {C_TEXT};
}}
.stButton > button[kind="primary"] {{
    background: {C_BLUE};
    border-color: {C_BLUE};
    color: white;
}}
.stButton > button[kind="primary"]:hover {{
    background: {C_BLUE_SOFT};
    border-color: {C_BLUE_SOFT};
}}

/* ── Inputs ──────────────────────────────────────────────────────────────── */
.stSelectbox > div > div,
.stMultiSelect > div > div,
.stDateInput > div > div {{
    background: {C_CARD} !important;
    border: 1px solid {C_BORDER} !important;
    border-radius: 8px !important;
    color: {C_TEXT} !important;
}}

/* ── DataFrames ──────────────────────────────────────────────────────────── */
.stDataFrame {{ border-radius: 10px; overflow: hidden; }}
.stDataFrame [data-testid="stDataFrameGlideDataEditor"] {{
    background: {C_CARD} !important;
}}

/* ── Expander ────────────────────────────────────────────────────────────── */
.streamlit-expanderHeader {{
    background: {C_CARD2} !important;
    border-radius: 8px !important;
    color: {C_TEXT} !important;
    border: 1px solid {C_BORDER} !important;
}}
.streamlit-expanderContent {{
    background: {C_CARD} !important;
    border: 1px solid {C_BORDER} !important;
    border-top: none !important;
    border-radius: 0 0 8px 8px !important;
}}

/* ── Custom components ───────────────────────────────────────────────────── */
.page-header {{
    background: linear-gradient(135deg, #0d1117 0%, #161b22 50%, #0d1117 100%);
    border: 1px solid {C_BORDER};
    border-radius: 16px;
    padding: 28px 36px;
    margin-bottom: 24px;
    position: relative;
    overflow: hidden;
}}
.page-header::before {{
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, {C_BLUE}, {C_GOLD}, {C_RED});
}}
.page-header h1 {{
    font-size: 2rem;
    font-weight: 700;
    color: {C_TEXT};
    margin: 0 0 4px 0;
    letter-spacing: -0.5px;
}}
.page-header p {{
    color: {C_MUTED};
    margin: 0;
    font-size: 0.9rem;
}}

.match-card {{
    background: {C_CARD};
    border: 1px solid {C_BORDER};
    border-radius: 16px;
    padding: 28px;
    margin-bottom: 20px;
    position: relative;
    overflow: hidden;
}}
.match-card::before {{
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, {C_BLUE} 0%, {C_RED} 100%);
}}

.team-name-blue {{
    font-size: 1.5rem;
    font-weight: 700;
    color: {C_BLUE_SOFT};
    letter-spacing: -0.3px;
}}
.team-name-red {{
    font-size: 1.5rem;
    font-weight: 700;
    color: {C_RED_SOFT};
    letter-spacing: -0.3px;
    text-align: right;
}}
.vs-badge {{
    background: {C_CARD2};
    border: 1px solid {C_BORDER};
    border-radius: 50%;
    width: 44px;
    height: 44px;
    display: flex;
    align-items: center;
    justify-content: center;
    margin: auto;
    font-size: 0.75rem;
    font-weight: 700;
    color: {C_MUTED};
    letter-spacing: 1px;
}}

.prob-container {{
    background: {C_CARD2};
    border: 1px solid {C_BORDER};
    border-radius: 12px;
    padding: 20px;
    margin: 16px 0;
}}
.prob-label {{
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: {C_MUTED};
    margin-bottom: 12px;
    text-align: center;
}}
.prob-bar-wrap {{
    display: flex;
    border-radius: 8px;
    overflow: hidden;
    height: 52px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.4);
}}
.prob-bar-b {{
    background: linear-gradient(90deg, #0c2461, {C_BLUE});
    display: flex;
    align-items: center;
    justify-content: center;
    color: white;
    font-size: 1.3rem;
    font-weight: 800;
    min-width: 40px;
    transition: width 0.4s ease;
}}
.prob-bar-r {{
    background: linear-gradient(90deg, {C_RED}, #7b0000);
    display: flex;
    align-items: center;
    justify-content: center;
    color: white;
    font-size: 1.3rem;
    font-weight: 800;
    min-width: 40px;
    transition: width 0.4s ease;
}}
.prob-teams {{
    display: flex;
    justify-content: space-between;
    margin-top: 8px;
}}
.prob-team-label {{
    font-size: 0.78rem;
    color: {C_MUTED};
}}

.model-chip {{
    background: {C_CARD2};
    border: 1px solid {C_BORDER};
    border-radius: 10px;
    padding: 12px 8px;
    text-align: center;
}}
.model-chip .chip-label {{
    font-size: 0.68rem;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    color: {C_MUTED};
    font-weight: 600;
    margin-bottom: 4px;
}}
.model-chip .chip-value {{
    font-size: 1.1rem;
    font-weight: 700;
    color: {C_TEXT};
}}
.model-chip .chip-value.favor-blue {{ color: {C_BLUE_SOFT}; }}
.model-chip .chip-value.favor-red  {{ color: {C_RED_SOFT}; }}
.model-chip .chip-value.neutral    {{ color: {C_GOLD}; }}

.stat-card {{
    background: {C_CARD};
    border: 1px solid {C_BORDER};
    border-radius: 12px;
    padding: 16px 20px;
    text-align: center;
}}
.stat-card .s-label {{
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    color: {C_MUTED};
    font-weight: 600;
    margin-bottom: 6px;
}}
.stat-card .s-value {{
    font-size: 1.4rem;
    font-weight: 700;
    color: {C_TEXT};
    line-height: 1;
}}
.stat-card .s-sub {{
    font-size: 0.75rem;
    color: {C_MUTED};
    margin-top: 4px;
}}

.section-header {{
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    color: {C_MUTED};
    font-weight: 600;
    margin: 20px 0 12px 0;
    display: flex;
    align-items: center;
    gap: 8px;
}}
.section-header::after {{
    content: '';
    flex: 1;
    height: 1px;
    background: {C_BORDER};
}}

.winner-badge {{
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(63,185,80,0.12);
    border: 1px solid rgba(63,185,80,0.3);
    color: {C_GREEN};
    border-radius: 20px;
    padding: 4px 12px;
    font-size: 0.8rem;
    font-weight: 600;
}}

.roster-row {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 6px 0;
    border-bottom: 1px solid {C_BORDER};
}}
.roster-row:last-child {{ border-bottom: none; }}
.role-badge {{
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.8px;
    background: {C_CARD2};
    border: 1px solid {C_BORDER};
    border-radius: 4px;
    padding: 2px 6px;
    color: {C_MUTED};
    min-width: 36px;
    text-align: center;
    text-transform: uppercase;
}}
.player-elo {{
    font-size: 0.75rem;
    color: {C_GOLD};
    font-weight: 600;
}}

.league-badge {{
    display: inline-block;
    background: rgba(31,111,235,0.15);
    border: 1px solid rgba(31,111,235,0.3);
    color: {C_BLUE_SOFT};
    border-radius: 6px;
    padding: 2px 8px;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.5px;
}}

.save-btn-saved {{
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(63,185,80,0.1);
    border: 1px solid rgba(63,185,80,0.25);
    color: {C_GREEN};
    border-radius: 8px;
    padding: 6px 14px;
    font-size: 0.82rem;
    font-weight: 600;
}}
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _chart_layout(**kwargs):
    """Shared Plotly layout for consistent dark theming."""
    base = dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=C_CARD2,
        font=dict(color=C_TEXT, family="-apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif"),
        margin=dict(l=0, r=0, t=36, b=0),
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor=C_BORDER),
        xaxis=dict(gridcolor=C_BORDER, zerolinecolor=C_BORDER, linecolor=C_BORDER),
        yaxis=dict(gridcolor=C_BORDER, zerolinecolor=C_BORDER, linecolor=C_BORDER),
    )
    base.update(kwargs)
    return base


def stat_card(label: str, value: str, sub: str = "", color: str = "") -> str:
    val_style = f"color:{color};" if color else ""
    return f"""
    <div class="stat-card">
      <div class="s-label">{label}</div>
      <div class="s-value" style="{val_style}">{value}</div>
      {"" if not sub else f'<div class="s-sub">{sub}</div>'}
    </div>"""


def model_chip(label: str, prob: float) -> str:
    cls = "favor-blue" if prob > 0.52 else ("favor-red" if prob < 0.48 else "neutral")
    return f"""
    <div class="model-chip">
      <div class="chip-label">{label}</div>
      <div class="chip-value {cls}">{prob:.1%}</div>
    </div>"""


# ── Cached loading ────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading data & models…")
def load_everything():
    cache_path = ROOT / "src" / "data" / "game_df_cache.parquet"
    game_df = load_and_cache(
        csv_dir    = DATA_DIR,
        cache_path = cache_path,
    )
    if "is_playoffs" not in game_df.columns:
        raw = load_raw(str(DATA_DIR), complete_only=True)
        if "playoffs" in raw.columns:
            playoff_map = raw.drop_duplicates("gameid").set_index("gameid")["playoffs"]
            game_df["is_playoffs"] = game_df["gameid"].map(playoff_map).fillna(0).astype(int)
        else:
            game_df["is_playoffs"] = 0
        game_df.to_parquet(cache_path, index=False)
    game_df = game_df.sort_values("date").reset_index(drop=True)

    elo        = EloSystem()
    player_elo = PlayerEloSystem()
    league_elo = LeagueEloSystem()

    # Restore league_elo rolling window from saved pickle (if present)
    elo_pkl = SAVE_DIR / "elo_ratings.pkl"
    if elo_pkl.exists():
        import pickle
        with open(elo_pkl, "rb") as _f:
            _saved = pickle.load(_f)
        for _lg, _entries in _saved.get("league_elo_updates", {}).items():
            from collections import deque
            league_elo._updates[_lg] = deque(
                [(pd.Timestamp(_dt), _delta) for _dt, _delta in _entries]
            )

    game_df = elo.process_and_annotate(game_df, min_games=5,
                                        player_elo=player_elo, league_elo=league_elo)

    _, team2idx, _ = build_vocabularies(game_df)
    xgb = joblib.load(SAVE_DIR / "xgb_model.joblib")

    # LightGBM (optional — only present after retraining with new features)
    lgbm = None
    lgbm_path = SAVE_DIR / "lgbm_model.joblib"
    if lgbm_path.exists():
        lgbm = joblib.load(lgbm_path)

    # Meta-learner (optional)
    meta = None
    meta_path = SAVE_DIR / "meta_learner.joblib"
    if meta_path.exists():
        meta = joblib.load(meta_path)

    # TabPFN (optional — loaded lazily to avoid import errors if not installed)
    tabpfn = None
    tabpfn_path = SAVE_DIR / "tabpfn_model.joblib"
    if tabpfn_path.exists():
        try:
            tabpfn = joblib.load(tabpfn_path)
        except Exception:
            pass

    full_graph = build_match_graph(game_df, team2idx, halflife_days=180, max_repeats=5)
    full_graph = full_graph.to(device)
    state = torch.load(SAVE_DIR / "gnn_weights.pt", map_location=device, weights_only=True)
    ckpt_emb = state["encoder.team_emb.weight"]          # [ckpt_size, emb_dim]
    ckpt_num_teams = ckpt_emb.shape[0] - 1               # -1 for padding row
    current_num_teams = len(team2idx)

    if current_num_teams != ckpt_num_teams:
        # New teams appeared in fresh CSV data — expand the embedding table.
        # Existing team rows are copied verbatim; new rows are zero-initialised.
        # GNN is slightly less accurate for new teams until you retrain.
        new_size = current_num_teams + 1                  # +1 for padding
        expanded = torch.zeros(new_size, ckpt_emb.shape[1], dtype=ckpt_emb.dtype)
        expanded[:ckpt_emb.shape[0]] = ckpt_emb
        state["encoder.team_emb.weight"] = expanded
        _new_team_count = current_num_teams - ckpt_num_teams
        st.warning(
            f"⚠️ {_new_team_count} new team(s) detected since last GNN training. "
            "Their embeddings are zero-initialised — GNN predictions for these teams "
            "will be less accurate. Run **Retrain on full data** in the Data Status tab.",
        )

    gnn = PreDraftPredictor(num_teams=current_num_teams, emb_dim=128, out_dim=64,
                             hidden=128, dropout=0.2).to(device)
    gnn.load_state_dict(state)
    gnn.eval()
    with torch.no_grad():
        team_embeddings = gnn.encode_all_teams(full_graph)

    try:
        import shap
        # Use LightGBM for SHAP if available, else XGBoost
        explainer = shap.TreeExplainer(lgbm if lgbm is not None else xgb)
    except ImportError:
        explainer = None

    return elo, player_elo, game_df, team2idx, xgb, lgbm, tabpfn, meta, gnn, team_embeddings, explainer


# ── Prediction ────────────────────────────────────────────────────────────────

def _build_feats(elo, player_elo, game_df, blue, red, is_playoffs, league=""):
    roster_b = player_elo.current_roster(blue, game_df)
    roster_r = player_elo.current_roster(red,  game_df)
    roles = ["top", "jng", "mid", "bot", "sup"]
    b_role_elos = [player_elo.rating(roster_b.get(r)) for r in roles]
    r_role_elos = [player_elo.rating(roster_r.get(r)) for r in roles]
    tier = float(LEAGUE_TIER.get(league, 4))
    blue_conf = min(elo._games.get(blue, 0) / 200.0, 1.0)
    red_conf  = min(elo._games.get(red,  0) / 200.0, 1.0)
    values = [
        elo.effective_rating(blue) - elo.effective_rating(red),
        elo.ewma_win_rate(blue),
        elo.ewma_win_rate(red),
        int(is_playoffs),
        elo.elo_momentum(blue),
        elo.elo_momentum(red),
        *b_role_elos,
        *r_role_elos,
        tier,
        blue_conf,
        red_conf,
    ]
    return pd.DataFrame([values], columns=FEATURE_COLS).astype(np.float32)


def _series_prob(p_game: float, best_of: int) -> float:
    """P(win series) from per-game win probability, assuming game independence.

    Bo3 (first to 2): P(2-0) + P(2-1) = 3p² - 2p³
    Bo5 (first to 3): P(3-0) + P(3-1) + P(3-2) = p³(1 + 3q + 6q²)  where q=1-p
    """
    p = float(np.clip(p_game, 1e-6, 1 - 1e-6))
    q = 1 - p
    if best_of == 3:
        return 3*p**2 - 2*p**3
    elif best_of == 5:
        return p**3 * (1 + 3*q + 6*q**2)
    else:  # Bo1 or unknown
        return p


def _score_distribution(p_game: float, best_of: int) -> list[tuple[str, float, bool]]:
    """Return (score_label, probability, blue_wins) for every possible series outcome.

    Uses the negative binomial: the winner always takes the last game, so in
    a W-L series P = C(W+L-2, L) * p^W * q^L   (W = wins_needed for winner).

    Args:
        p_game:  side-agnostic per-game win probability for the blue team
        best_of: 1, 3, or 5

    Returns:
        List of (label, prob, blue_wins) sorted by prob descending.
        E.g. [("2-0", 0.36, True), ("2-1", 0.29, True), ("1-2", 0.19, False), ("0-2", 0.16, False)]
    """
    from math import comb
    p = float(np.clip(p_game, 1e-6, 1 - 1e-6))
    q = 1 - p
    wins_needed = (best_of + 1) // 2

    results = []
    for blue_w in range(wins_needed + 1):
        for red_w in range(wins_needed + 1):
            # Exactly one side reaches wins_needed; the other has fewer
            if blue_w == wins_needed and red_w < wins_needed:
                # Blue wins: blue wins last game; before that (W-1) blue wins + red_w losses
                prob = comb(wins_needed - 1 + red_w, red_w) * (p ** wins_needed) * (q ** red_w)
                results.append((f"{blue_w}-{red_w}", prob, True))
            elif red_w == wins_needed and blue_w < wins_needed:
                # Red wins: red wins last game
                prob = comb(wins_needed - 1 + blue_w, blue_w) * (q ** wins_needed) * (p ** blue_w)
                results.append((f"{blue_w}-{red_w}", prob, False))

    results.sort(key=lambda x: -x[1])
    return results


def _ensemble(elo_p, xgb_p, lgbm_p, gnn_p, tabpfn_p, meta, xgb_fallback):
    if meta is not None:
        meta_in = np.array([[
            elo_p,
            xgb_p,
            lgbm_p   if lgbm_p   is not None else xgb_fallback,
            gnn_p,
            tabpfn_p if tabpfn_p is not None else xgb_fallback,
        ]])
        return float(meta.predict_proba(meta_in)[0, 1])
    return float(np.mean([p for p in [elo_p, xgb_p, lgbm_p, gnn_p, tabpfn_p] if p is not None]))


def predict_game(elo, player_elo, game_df, xgb, lgbm, tabpfn, meta,
                 gnn, team_embeddings, team2idx,
                 blue, red, is_playoffs=False, best_of=1, league=""):
    """Predict match outcome.

    Returns per-model blue-side game probabilities, a side-agnostic game win
    probability (average of blue and red side), and a series win probability
    computed from the best-of format.
    """
    # ── Blue-side prediction (team `blue` plays blue) ─────────────────────────
    feats    = _build_feats(elo, player_elo, game_df, blue, red, is_playoffs, league)
    elo_prob = float(elo.win_probability(blue, red))
    xgb_prob = float(xgb.predict_proba(feats)[0, 1])
    lgbm_prob   = float(lgbm.predict_proba(feats)[0, 1])   if lgbm   is not None else None
    try:
        tabpfn_prob = float(tabpfn.predict_proba(feats)[0, 1]) if tabpfn is not None else None
    except Exception:
        tabpfn_prob = None

    bi  = torch.tensor([team2idx.get(blue, 0)], dtype=torch.long, device=device)
    ri  = torch.tensor([team2idx.get(red,  0)], dtype=torch.long, device=device)
    ctx = torch.tensor(feats.values, dtype=torch.float, device=device)
    with torch.no_grad():
        gnn_prob = float(gnn(team_embeddings, bi, ri, ctx).item())

    ensemble = _ensemble(elo_prob, xgb_prob, lgbm_prob, gnn_prob, tabpfn_prob, meta, xgb_prob)

    # ── Red-side prediction (team `blue` plays red, sides swapped) ────────────
    feats_s   = _build_feats(elo, player_elo, game_df, red, blue, is_playoffs, league)
    xgb_s     = float(xgb.predict_proba(feats_s)[0, 1])
    lgbm_s    = float(lgbm.predict_proba(feats_s)[0, 1]) if lgbm   is not None else None
    try:
        tabpfn_s  = float(tabpfn.predict_proba(feats_s)[0, 1]) if tabpfn is not None else None
    except Exception:
        tabpfn_s = None
    bi_s = torch.tensor([team2idx.get(red,  0)], dtype=torch.long, device=device)
    ri_s = torch.tensor([team2idx.get(blue, 0)], dtype=torch.long, device=device)
    ctx_s = torch.tensor(feats_s.values, dtype=torch.float, device=device)
    with torch.no_grad():
        gnn_s = float(gnn(team_embeddings, bi_s, ri_s, ctx_s).item())
    elo_s    = float(elo.win_probability(red, blue))
    ens_s    = _ensemble(elo_s, xgb_s, lgbm_s, gnn_s, tabpfn_s, meta, xgb_s)

    # P(blue wins as red side) = 1 - P(red wins as blue side)
    p_game       = (ensemble + (1 - ens_s)) / 2
    series_prob  = _series_prob(p_game, best_of)
    score_dist   = _score_distribution(p_game, best_of)

    return {
        "elo": elo_prob, "xgb": xgb_prob, "lgbm": lgbm_prob,
        "gnn": gnn_prob, "tabpfn": tabpfn_prob,
        "ensemble":    ensemble,     # P(blue wins) when assigned blue side
        "p_game":      p_game,       # side-agnostic P(blue team wins a game)
        "series_prob": series_prob,  # P(blue team wins the series)
        "score_dist":  score_dist,   # [(label, prob, blue_wins), ...]
        "best_of":     best_of,
        "feats": feats,
    }


def _team_known(team: str, team2idx: dict, elo) -> bool:
    """True if the team exists in the GNN vocabulary AND has Elo history."""
    return team in team2idx and team in elo._ratings


def _unknown_team_warnings(blue: str, red: str, team2idx: dict, elo) -> list[str]:
    """Return a list of human-readable warning strings for unrecognised teams."""
    warnings = []
    if not _team_known(blue, team2idx, elo):
        warnings.append(
            f"**{blue}** is not in the training data — GNN will use a blank embedding "
            f"and Elo will use the default rating. Prediction quality is reduced."
        )
    if not _team_known(red, team2idx, elo):
        warnings.append(
            f"**{red}** is not in the training data — GNN will use a blank embedding "
            f"and Elo will use the default rating. Prediction quality is reduced."
        )
    return warnings


def shap_values_for(explainer, feats):
    if explainer is None:
        return None
    try:
        sv = explainer.shap_values(feats)
        if isinstance(sv, list):
            sv = sv[1]
        return sv[0]
    except Exception:
        return None


# ── Prediction card ───────────────────────────────────────────────────────────

def render_prediction_card(
    blue, red, result, explainer,
    label="", is_playoffs=False,
    player_elo=None, game_df=None,
    match_date=None, league=None, tournament=None,
    side_known=True, team2idx=None, elo=None,
    bookmaker_prob=None,
):
    elo_p    = result["elo"]
    xgb_p    = result["xgb"]
    lgbm_p   = result.get("lgbm")
    gnn_p    = result["gnn"]
    tabpfn_p = result.get("tabpfn")
    ens_p    = result["ensemble"]       # blue-side game prob
    p_game   = result.get("p_game", ens_p)        # side-agnostic game prob
    ser_p    = result.get("series_prob", ens_p)   # series win prob
    best_of  = result.get("best_of", 1)
    feats    = result["feats"]

    # Headline probability: series win prob (or game prob for Bo1)
    blue_pct = int(round(ser_p * 100))
    red_pct  = 100 - blue_pct
    favor    = blue if ser_p >= 0.5 else red
    conf     = max(ser_p, 1 - ser_p)

    # Build league/time/format badge
    badge_html = ""
    if league:
        badge_html += f'<span class="league-badge">{league}</span> '
    if best_of > 1:
        badge_html += f'<span class="league-badge" style="background:rgba(56,139,253,0.1);border-color:rgba(56,139,253,0.3);color:{C_BLUE_SOFT}">BO{best_of}</span> '
    if is_playoffs:
        badge_html += f'<span class="league-badge" style="background:rgba(210,153,34,0.15);border-color:rgba(210,153,34,0.3);color:{C_GOLD}">PLAYOFFS</span> '
    if match_date:
        try:
            dt = datetime.fromisoformat(match_date)
            badge_html += f'<span style="font-size:0.75rem;color:{C_MUTED}">{dt.strftime("%H:%M UTC")}</span>'
        except Exception:
            pass

    st.markdown(f'<div class="match-card">', unsafe_allow_html=True)

    if badge_html:
        st.markdown(f'<div style="margin-bottom:16px">{badge_html}</div>', unsafe_allow_html=True)

    # Unknown team warnings
    if team2idx is not None and elo is not None:
        for w in _unknown_team_warnings(blue, red, team2idx, elo):
            st.warning(w)

    # Side-unknown note for upcoming games
    if not side_known:
        st.caption("⚠️ Side assignment unknown — probabilities are averaged across both sides.")

    # Team header
    col_b, col_vs, col_r = st.columns([5, 1, 5])
    with col_b:
        team_ratings = elo._ratings if elo else {}
        elo_blue_str = f"{team_ratings[blue]:.0f}" if blue in team_ratings else "—"
        st.markdown(
            f'<div class="team-name-blue">{blue}</div>'
            f'<div style="font-size:0.8rem;color:{C_MUTED};margin-top:2px">'
            f'Team Elo {elo_blue_str}'
            f'</div>',
            unsafe_allow_html=True,
        )
    with col_vs:
        st.markdown(f'<div style="height:100%;display:flex;align-items:center;justify-content:center;padding-top:8px"><div class="vs-badge">VS</div></div>', unsafe_allow_html=True)
    with col_r:
        elo_red_str = f"{team_ratings[red]:.0f}" if red in team_ratings else "—"
        st.markdown(
            f'<div class="team-name-red">{red}</div>'
            f'<div style="font-size:0.8rem;color:{C_MUTED};margin-top:2px;text-align:right">'
            f'Team Elo {elo_red_str}'
            f'</div>',
            unsafe_allow_html=True,
        )

    # Series win probability bar (headline)
    bar_label = f"Series Win Probability (Bo{best_of})" if best_of > 1 else "Win Probability"
    st.markdown(f"""
    <div class="prob-container">
      <div class="prob-label">{bar_label}</div>
      <div class="prob-bar-wrap">
        <div class="prob-bar-b" style="width:{blue_pct}%">{blue_pct}%</div>
        <div class="prob-bar-r" style="width:{red_pct}%">{red_pct}%</div>
      </div>
      <div class="prob-teams">
        <span class="prob-team-label">← {blue}</span>
        <span class="winner-badge">▶ {favor} favoured · {conf:.0%}</span>
        <span class="prob-team-label">{red} →</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Score distribution + per-game breakdown (only for series)
    if best_of > 1:
        score_dist = result.get("score_dist", [])
        pg_b = int(round(p_game * 100))
        pg_r = 100 - pg_b

        # Score probability chips
        if score_dist:
            chips_html = ""
            for label, prob, blue_wins in score_dist:
                color  = C_BLUE_SOFT if blue_wins else C_RED_SOFT
                bg     = "rgba(56,139,253,0.08)" if blue_wins else "rgba(248,81,73,0.08)"
                border = "rgba(56,139,253,0.25)" if blue_wins else "rgba(248,81,73,0.25)"
                chips_html += (
                    f'<span style="display:inline-block;padding:3px 10px;margin:3px 4px 3px 0;'
                    f'border-radius:6px;border:1px solid {border};background:{bg};'
                    f'font-size:0.8rem;color:{color};font-weight:600;">'
                    f'{label} <span style="font-weight:400;color:{C_MUTED}">{prob:.0%}</span></span>'
                )
            st.markdown(
                f'<div style="margin:10px 0 4px 0">'
                f'<div style="font-size:0.72rem;color:{C_MUTED};margin-bottom:4px;text-transform:uppercase;'
                f'letter-spacing:0.05em">Score probabilities</div>'
                f'{chips_html}</div>',
                unsafe_allow_html=True,
            )

        # Per-game line
        side_note = f" · Blue-side: <b style='color:{C_TEXT}'>{ens_p:.0%}</b>" if side_known else ""
        st.markdown(
            f'<div style="font-size:0.78rem;color:{C_MUTED};margin:4px 0 8px 0">'
            f'Per-game (side-agnostic): '
            f'<b style="color:{C_BLUE_SOFT}">{pg_b}%</b> {blue} · '
            f'<b style="color:{C_RED_SOFT}">{pg_r}%</b> {red}'
            f'{side_note}</div>',
            unsafe_allow_html=True,
        )

    # Model chips
    chip_label = "Model breakdown (blue-side game)" if side_known else "Model breakdown (side-averaged game)"
    st.markdown(f'<div class="section-header">{chip_label}</div>', unsafe_allow_html=True)
    all_chips = [
        ("Elo",        elo_p),
        ("XGBoost",    xgb_p),
        ("LightGBM",   lgbm_p),
        ("GNN",        gnn_p),
        ("TabPFN",     tabpfn_p),
        ("Ensemble",   ens_p),
        ("Bookmaker",  bookmaker_prob),
    ]
    active_chips = [(n, p) for n, p in all_chips if p is not None]
    chip_cols = st.columns(len(active_chips))
    for col, (name, prob) in zip(chip_cols, active_chips):
        col.markdown(model_chip(name, prob), unsafe_allow_html=True)

    st.markdown("<div style='margin-top:16px'></div>", unsafe_allow_html=True)

    # Save button
    save_key = f"saved_{blue}_{red}_{match_date or ''}"
    if st.session_state.get(save_key):
        st.markdown('<div class="save-btn-saved">✓ Saved to history</div>', unsafe_allow_html=True)
    else:
        if st.button("💾 Save prediction", key=f"save_btn_{blue}_{red}_{match_date or ''}"):
            save_prediction(
                blue_team=blue, red_team=red,
                elo_prob=result["elo"], xgb_prob=result["xgb"],
                lgbm_prob=result.get("lgbm"), tabpfn_prob=result.get("tabpfn"),
                gnn_prob=result["gnn"], ensemble_prob=result["ensemble"],
                bookmaker_prob=bookmaker_prob,
                series_prob=result.get("series_prob"), best_of=result.get("best_of", 1),
                score_dist=result.get("score_dist"),
                match_date=match_date, league=league, tournament=tournament,
                is_playoffs=is_playoffs,
            )
            st.session_state[save_key] = True
            st.rerun()

    # Roster + SHAP (expanders)
    exp_col1, exp_col2 = st.columns(2)

    with exp_col1:
        if player_elo is not None and game_df is not None:
            with st.expander("🧑‍💻 Rosters (last known)"):
                rc = st.columns(2)
                for ci, (team, side_color) in enumerate([(blue, C_BLUE_SOFT), (red, C_RED_SOFT)]):
                    roster = player_elo.current_roster(team, game_df)
                    with rc[ci]:
                        st.markdown(f'<div style="color:{side_color};font-weight:700;font-size:0.85rem;margin-bottom:8px">{team}</div>', unsafe_allow_html=True)
                        for role, player in roster.items():
                            p_elo = player_elo.rating(player) if player else 1500
                            st.markdown(f"""
                            <div class="roster-row">
                              <span class="role-badge">{role}</span>
                              <span style="font-size:0.82rem;color:{C_TEXT}">{player or "Unknown"}</span>
                              <span class="player-elo">{p_elo:.0f}</span>
                            </div>""", unsafe_allow_html=True)

    with exp_col2:
        with st.expander("📊 Feature importance (SHAP)"):
            sv = shap_values_for(explainer, feats)
            if sv is not None:
                shap_df = pd.DataFrame({"Feature": FEATURE_LABELS, "SHAP": sv}).sort_values("SHAP")
                colors  = [C_RED_SOFT if v < 0 else C_BLUE_SOFT for v in shap_df["SHAP"]]
                fig = go.Figure(go.Bar(
                    x=shap_df["SHAP"], y=shap_df["Feature"],
                    orientation="h",
                    marker_color=colors,
                    marker_line_width=0,
                    text=[f"{v:+.4f}" for v in shap_df["SHAP"]],
                    textposition="outside",
                    textfont=dict(size=10, color=C_MUTED),
                ))
                fig.update_layout(**_chart_layout(height=240, title="XGBoost SHAP values"))
                fig.update_xaxes(showgrid=False)
                st.plotly_chart(fig, use_container_width=True,
                                key=f"shap_{blue}_{red}_{label}")
            else:
                feat_df = pd.DataFrame({"Feature": FEATURE_LABELS,
                                         "Value": [f"{v:.4f}" for v in feats[0]]})
                st.dataframe(feat_df, hide_index=True, use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)


# ── Team stats ────────────────────────────────────────────────────────────────

def render_team_stats(elo, player_elo, game_df, team):
    elo_val  = elo.rating(team)
    games    = elo._games.get(team, 0)
    blue_wr  = elo.blue_side_win_rate(team)
    red_wr   = elo.red_side_win_rate(team)
    ewma_wr  = elo.ewma_win_rate(team)
    momentum = elo.elo_momentum(team)

    all_teams = pd.Series(elo._ratings).sort_values(ascending=False)
    rank = (all_teams.index.tolist().index(team) + 1) if team in all_teams.index else "?"

    # Stat cards row
    cards = [
        stat_card("Elo Rating",   f"{elo_val:.0f}", color=C_GOLD),
        stat_card("Global Rank",  f"#{rank}"),
        stat_card("Games Played", f"{games:,}"),
        stat_card("EWMA Win Rate",f"{ewma_wr:.0%}",
                  color=C_GREEN if ewma_wr > 0.5 else C_RED_SOFT),
        stat_card("Blue-side WR", f"{blue_wr:.0%}"),
        stat_card("Red-side WR",  f"{red_wr:.0%}"),
        stat_card("Elo Momentum", f"{momentum:+.0f}",
                  sub="last 5 games",
                  color=C_GREEN if momentum > 0 else C_RED_SOFT),
    ]
    cols = st.columns(len(cards))
    for col, card in zip(cols, cards):
        col.markdown(card, unsafe_allow_html=True)

    st.markdown("<div style='margin-top:16px'></div>", unsafe_allow_html=True)

    # Player roster
    if player_elo is not None:
        roster = player_elo.current_roster(team, game_df)
        valid  = {r: p for r, p in roster.items() if p}
        if valid:
            st.markdown('<div class="section-header">Current roster (last known)</div>', unsafe_allow_html=True)
            p_cols = st.columns(len(valid))
            role_colors = {"top": "#e67e22", "jng": "#27ae60", "mid": "#3498db",
                           "bot": "#9b59b6", "sup": "#e74c3c"}
            for col, (role, player) in zip(p_cols, valid.items()):
                p_elo = player_elo.rating(player)
                col.markdown(f"""
                <div class="stat-card">
                  <div class="s-label" style="color:{role_colors.get(role, C_MUTED)}">{role.upper()}</div>
                  <div style="font-size:0.9rem;font-weight:600;color:{C_TEXT};margin:4px 0">{player}</div>
                  <div style="font-size:0.78rem;color:{C_GOLD};font-weight:600">{p_elo:.0f}</div>
                </div>""", unsafe_allow_html=True)

    # Elo history chart
    team_games = game_df[(game_df["blue_team"] == team) | (game_df["red_team"] == team)].copy()
    if len(team_games) > 0:
        st.markdown('<div class="section-header">Elo history</div>', unsafe_allow_html=True)
        elo_history = []
        wins = []
        for _, row in team_games.iterrows():
            if row["blue_team"] == team:
                elo_history.append({"date": row["date"], "elo": row["blue_elo"]})
                wins.append(row["blue_win"] == 1)
            else:
                elo_history.append({"date": row["date"], "elo": row["red_elo"]})
                wins.append(row["blue_win"] == 0)
        hist_df = pd.DataFrame(elo_history).sort_values("date")

        fig = go.Figure()
        # Shaded area under the line
        fig.add_trace(go.Scatter(
            x=hist_df["date"], y=hist_df["elo"],
            fill="tozeroy",
            fillcolor=f"rgba(31,111,235,0.06)",
            line=dict(color=C_BLUE, width=2),
            mode="lines",
            name="Elo",
            hovertemplate="%{x|%Y-%m-%d}<br>Elo: %{y:.0f}<extra></extra>",
        ))
        layout = _chart_layout(height=280)
        layout["yaxis"]["title"] = "Elo"
        layout["xaxis"]["title"] = ""
        fig.update_layout(**layout)
        st.plotly_chart(fig, use_container_width=True, key=f"elo_hist_{team}")

    # Recent form
    st.markdown('<div class="section-header">Recent form — last 10 games</div>', unsafe_allow_html=True)
    recent = team_games.tail(10)
    form_html = '<div style="display:flex;gap:6px;flex-wrap:wrap;margin-bottom:16px">'
    for _, row in recent.iterrows():
        won = (row["blue_team"] == team and row["blue_win"] == 1) or \
              (row["red_team"]  == team and row["blue_win"] == 0)
        opp  = row["red_team"] if row["blue_team"] == team else row["blue_team"]
        side = "B" if row["blue_team"] == team else "R"
        bg   = "rgba(63,185,80,0.15)" if won else "rgba(218,54,51,0.15)"
        bc   = "rgba(63,185,80,0.4)"  if won else "rgba(218,54,51,0.4)"
        tc   = C_GREEN if won else C_RED_SOFT
        lbl  = "W" if won else "L"
        form_html += f"""
        <div title="{lbl} vs {opp} ({side})" style="background:{bg};border:1px solid {bc};
             border-radius:8px;width:42px;height:42px;display:flex;align-items:center;
             justify-content:center;font-weight:800;font-size:1rem;color:{tc};
             cursor:default">{lbl}</div>"""
    form_html += "</div>"
    st.markdown(form_html, unsafe_allow_html=True)

    # H2H table
    st.markdown('<div class="section-header">Head-to-head vs top opponents</div>', unsafe_allow_html=True)
    opponents = pd.concat([
        game_df[game_df["blue_team"] == team]["red_team"],
        game_df[game_df["red_team"]  == team]["blue_team"],
    ]).value_counts().head(8)

    h2h_rows = []
    for opp in opponents.index:
        w1, g1 = elo._h2h.get((team, opp), [0, 0])
        w2, g2 = elo._h2h.get((opp, team),  [0, 0])
        total = g1 + g2
        wins  = w1 + (g2 - w2)
        wr    = wins / total if total else 0.5
        h2h_rows.append({
            "Opponent": opp, "Games": total, "Wins": wins, "Losses": total - wins,
            "Win Rate": f"{wr:.0%}",
        })
    if h2h_rows:
        st.dataframe(pd.DataFrame(h2h_rows), hide_index=True, use_container_width=True)


# ── Players tab ──────────────────────────────────────────────────────────────

ROLE_COLORS = {
    "top": "#e67e22", "jng": "#27ae60", "mid": "#3498db",
    "bot": "#9b59b6", "sup": "#e74c3c",
}


@st.cache_data(show_spinner=False)
def _build_player_leaderboard(_player_elo, min_games: int = 20) -> pd.DataFrame:
    """Build a ranked DataFrame of all players with >= min_games."""
    rows = []
    for player, elo_val in _player_elo._ratings.items():
        games = _player_elo._games.get(player, 0)
        if games >= min_games:
            rows.append({"Player": player, "Elo": round(elo_val, 1), "Games": games})
    df = pd.DataFrame(rows).sort_values("Elo", ascending=False).reset_index(drop=True)
    df.insert(0, "Rank", range(1, len(df) + 1))
    return df


def _render_players_tab(player_elo, game_df):
    all_players = sorted(player_elo._ratings.keys())

    # ── Leaderboard section ─────────────────────────────────────────────────
    st.markdown('<div class="section-header">Player Elo leaderboard — top 50</div>',
                unsafe_allow_html=True)

    lb_df = _build_player_leaderboard(player_elo, min_games=20)

    ldr_c1, ldr_c2 = st.columns([1, 2])
    with ldr_c1:
        st.dataframe(lb_df.head(50), hide_index=True, use_container_width=True, height=520)
    with ldr_c2:
        top30 = lb_df.head(30)
        fig_lb = go.Figure(go.Bar(
            x=top30["Elo"].iloc[::-1],
            y=top30["Player"].iloc[::-1],
            orientation="h",
            marker=dict(
                color=top30["Elo"].iloc[::-1],
                colorscale=[[0, C_RED], [0.4, C_GOLD], [1, C_BLUE_SOFT]],
                line_width=0,
            ),
            text=top30["Elo"].iloc[::-1].apply(lambda v: f"{v:.0f}"),
            textposition="outside",
            textfont=dict(size=10, color=C_MUTED),
            customdata=top30["Games"].iloc[::-1],
            hovertemplate="%{y}<br>Elo: %{x:.0f}<br>Games: %{customdata}<extra></extra>",
        ))
        layout_lb = _chart_layout(height=520)
        layout_lb["xaxis"]["range"] = [
            max(1400, top30["Elo"].min() - 60),
            top30["Elo"].max() + 100,
        ]
        fig_lb.add_vline(x=1500, line_dash="dot", line_color=C_MUTED, opacity=0.5,
                         annotation_text="1500", annotation_font_color=C_MUTED,
                         annotation_position="bottom right")
        fig_lb.update_layout(**layout_lb)
        st.plotly_chart(fig_lb, use_container_width=True)

    # ── Player profile ──────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Player profile</div>', unsafe_allow_html=True)

    # Default to top-ranked player with enough games
    default_player = lb_df.iloc[0]["Player"] if not lb_df.empty else (all_players[0] if all_players else None)
    if not default_player:
        st.info("No player data available yet.")
        return

    # Sort player list by Elo so top players appear first
    players_by_elo = lb_df["Player"].tolist() + [
        p for p in all_players if p not in lb_df["Player"].values
    ]
    selected_player = st.selectbox(
        "Search player", players_by_elo,
        index=0,
        label_visibility="collapsed",
        placeholder="Search player…",
    )

    if not selected_player:
        return

    p_elo  = player_elo.rating(selected_player)
    p_games = player_elo._games.get(selected_player, 0)
    info   = player_elo.player_info(selected_player, game_df)
    hist   = player_elo.elo_history(selected_player)

    # Rank
    rank_row = lb_df[lb_df["Player"] == selected_player]
    rank_val = int(rank_row["Rank"].iloc[0]) if not rank_row.empty else "?"

    # Current team + role
    team_str = info.get("team") or "Unknown"
    role_str = (info.get("role") or "?").upper()
    last_date = info.get("last_date")
    last_date_str = pd.Timestamp(last_date).strftime("%Y-%m-%d") if last_date is not None else "?"

    role_color = ROLE_COLORS.get((info.get("role") or "").lower(), C_MUTED)

    # Stat cards
    p_cards = [
        stat_card("Player Elo",   f"{p_elo:.0f}", color=C_GOLD),
        stat_card("Global Rank",  f"#{rank_val}"),
        stat_card("Games Played", f"{p_games:,}"),
        stat_card("Current Team", team_str,
                  sub=f'<span style="color:{role_color};font-weight:700">{role_str}</span>'),
        stat_card("Last Seen",    last_date_str),
    ]
    p_cols = st.columns(len(p_cards))
    for col, card in zip(p_cols, p_cards):
        col.markdown(card, unsafe_allow_html=True)

    st.markdown("<div style='margin-top:16px'></div>", unsafe_allow_html=True)

    # Elo history chart
    if hist:
        st.markdown('<div class="section-header">Elo history</div>', unsafe_allow_html=True)
        hist_df = pd.DataFrame(hist, columns=["date", "elo"]).sort_values("date")
        # Append current Elo as last point
        hist_df = pd.concat([
            hist_df,
            pd.DataFrame([{"date": pd.Timestamp("today"), "elo": p_elo}]),
        ], ignore_index=True)

        fig_hist = go.Figure()
        fig_hist.add_trace(go.Scatter(
            x=hist_df["date"], y=hist_df["elo"],
            fill="tozeroy",
            fillcolor="rgba(31,111,235,0.06)",
            line=dict(color=C_BLUE, width=2),
            mode="lines",
            hovertemplate="%{x|%Y-%m-%d}<br>Elo: %{y:.0f}<extra></extra>",
        ))
        # Peak marker
        peak_idx = hist_df["elo"].idxmax()
        fig_hist.add_trace(go.Scatter(
            x=[hist_df.loc[peak_idx, "date"]],
            y=[hist_df.loc[peak_idx, "elo"]],
            mode="markers+text",
            marker=dict(color=C_GOLD, size=10, symbol="star"),
            text=[f"Peak {hist_df.loc[peak_idx, 'elo']:.0f}"],
            textposition="top center",
            textfont=dict(color=C_GOLD, size=11),
            showlegend=False,
        ))
        layout_hist = _chart_layout(height=280)
        layout_hist["yaxis"]["title"] = "Elo"
        fig_hist.update_layout(**layout_hist)
        st.plotly_chart(fig_hist, use_container_width=True)
    else:
        st.caption("No Elo history recorded for this player (history is built at load time).")

    # Recent games table
    st.markdown('<div class="section-header">Recent games</div>', unsafe_allow_html=True)
    player_cols = [(s, r) for s in ["blue", "red"] for r in ["top", "jng", "mid", "bot", "sup"]]
    game_rows = []
    for side, role in player_cols:
        col = f"{side}_{role}_player"
        if col not in game_df.columns:
            continue
        matches = game_df[game_df[col] == selected_player].copy()
        matches["_side"] = side
        matches["_role"] = role
        game_rows.append(matches)

    if game_rows:
        p_games_df = pd.concat(game_rows).drop_duplicates(subset=["gameid"] if "gameid" in game_df.columns else None)
        p_games_df = p_games_df.sort_values("date", ascending=False).head(20)

        disp_rows = []
        for _, row in p_games_df.iterrows():
            side = row["_side"]
            won  = (side == "blue" and row["blue_win"] == 1) or \
                   (side == "red"  and row["blue_win"] == 0)
            team     = row["blue_team"] if side == "blue" else row["red_team"]
            opponent = row["red_team"]  if side == "blue" else row["blue_team"]
            disp_rows.append({
                "Date":     pd.Timestamp(row["date"]).strftime("%Y-%m-%d"),
                "Team":     team,
                "Opponent": opponent,
                "Side":     side.capitalize(),
                "Role":     row["_role"].upper(),
                "Result":   "Win" if won else "Loss",
            })

        if disp_rows:
            rg_df = pd.DataFrame(disp_rows)
            st.dataframe(rg_df, hide_index=True, use_container_width=True)
    else:
        st.caption("No game data found for this player.")

    # Compare with teammates / role peers
    st.markdown('<div class="section-header">Top players by role</div>', unsafe_allow_html=True)
    selected_role = (info.get("role") or "").lower()
    role_options  = ["top", "jng", "mid", "bot", "sup"]
    default_role_idx = role_options.index(selected_role) if selected_role in role_options else 2
    role_filter = st.selectbox(
        "Role", role_options,
        index=default_role_idx,
        format_func=lambda r: r.upper(),
        label_visibility="collapsed",
        key="player_role_filter",
    )

    # Find all players who ever played this role
    role_players: set[str] = set()
    for side in ["blue", "red"]:
        col = f"{side}_{role_filter}_player"
        if col in game_df.columns:
            role_players.update(game_df[col].dropna().unique())

    role_lb = (
        lb_df[lb_df["Player"].isin(role_players)]
        .head(20)
        .copy()
        .reset_index(drop=True)
    )
    role_lb.insert(0, "Role Rank", range(1, len(role_lb) + 1))
    # Highlight selected player
    if not role_lb.empty:
        st.dataframe(role_lb, hide_index=True, use_container_width=True)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # Page header
    st.markdown(f"""
    <div class="page-header">
      <h1>⚔️ LoL Pro Match Predictor</h1>
      <p>Pre-draft win probability · Team Elo · Player Elo · XGBoost · LightGBM · GNN · TabPFN · Meta-learner stack</p>
    </div>
    """, unsafe_allow_html=True)

    with st.spinner("Loading data & models…"):
        elo, player_elo, game_df, team2idx, xgb, lgbm, tabpfn, meta, gnn, team_embeddings, explainer = load_everything()

    all_teams = sorted(team2idx.keys())

    # ── Data staleness banner ─────────────────────────────────────────────────
    latest_date = pd.to_datetime(game_df["date"]).max()
    days_stale  = (datetime.now(timezone.utc).replace(tzinfo=None) - latest_date.to_pydatetime().replace(tzinfo=None)).days
    if days_stale > 14:
        st.warning(
            f"⚠️ Training data is **{days_stale} days old** (last game: {latest_date.strftime('%Y-%m-%d')}). "
            f"Download fresh CSVs from [Oracle's Elixir](https://oracleselixir.com/tools/downloads) "
            f"and place them in `src/data/oracleselixir/`, then restart the app to retrain.",
            icon="⚠️",
        )
    elif days_stale > 7:
        st.info(
            f"ℹ️ Training data is {days_stale} days old (last game: {latest_date.strftime('%Y-%m-%d')}). "
            f"Consider updating Oracle's Elixir CSVs soon.",
        )

    init_db()
    n_resolved = reconcile_results(game_df)
    if n_resolved:
        st.toast(f"✅ Auto-resolved {n_resolved} prediction result(s) from new data.")

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
        ["📅  Upcoming Games", "🎮  Custom Match", "📊  Team Explorer", "👤  Players", "📋  History", "🗄️  Data Status"]
    )

    # ── Tab 1: Upcoming Games ─────────────────────────────────────────────────
    with tab1:
        st.markdown('<div class="section-header">Schedule</div>', unsafe_allow_html=True)

        c1, c2, c3, c4 = st.columns([2, 3, 1, 1])
        with c1:
            tomorrow = (datetime.now(timezone.utc) + timedelta(days=1)).date()
            selected_date = st.date_input("Date", value=tomorrow, label_visibility="collapsed")
        with c2:
            league_choices = ["LCK", "LPL", "LEC", "LTA", "LCP", "LCKC", "LDL", "NACL"]
            selected_leagues = st.multiselect(
                "Leagues", league_choices, default=["LEC", "LCK", "LPL"],
                label_visibility="collapsed",
                placeholder="Select leagues…",
            )
        with c3:
            fetch_btn = st.button("🔄 Fetch games", use_container_width=True)
        with c4:
            is_playoffs_tab1 = st.checkbox("Playoffs", value=False)

        if "schedule" not in st.session_state:
            st.session_state.schedule = []
        if "predictions" not in st.session_state:
            st.session_state.predictions = {}

        if fetch_btn:
            with st.spinner("Fetching from Leaguepedia…"):
                try:
                    games = fetch_schedule(
                        date=selected_date.strftime("%Y-%m-%d"),
                        leagues=tuple(selected_leagues) if selected_leagues else None,
                    )
                    st.session_state.schedule = games
                    st.session_state.predictions = {}
                    if games:
                        st.success(f"Found {len(games)} game(s) — click **Predict all** to run models.")
                    else:
                        st.warning("No games found. Try adjusting the date or leagues.")
                except RuntimeError as e:
                    st.error(f"Leaguepedia API error: {e}")
                    st.session_state.schedule = []

        schedule = st.session_state.schedule

        if schedule:
            sched_df = pd.DataFrame([{
                "Time (UTC)": g["datetime_utc"].strftime("%H:%M") if g["datetime_utc"] else "TBD",
                "League":     g["league"],
                "Blue":       g["team1"],
                "Red":        g["team2"],
                "Format":     f"BO{g['best_of']}",
            } for g in schedule])
            st.dataframe(sched_df, hide_index=True, use_container_width=True)

            if st.button("⚡  Predict all games", type="primary"):
                preds = {}
                bk_odds = {}
                bar = st.progress(0)
                err_placeholder = st.empty()
                _odds_key = st.session_state.get("odds_api_key", "").strip()
                try:
                    for i, g in enumerate(schedule):
                        blue, red = g["team1"], g["team2"]
                        preds[f"{blue}_vs_{red}"] = predict_game(
                            elo, player_elo, game_df, xgb, lgbm, tabpfn, meta,
                            gnn, team_embeddings, team2idx,
                            blue, red, is_playoffs=is_playoffs_tab1,
                            best_of=g.get("best_of", 1),
                            league=g.get("league", ""),
                        )
                        if _odds_key:
                            try:
                                bk_odds[f"{blue}_vs_{red}"] = fetch_match_odds(
                                    blue, red, api_key=_odds_key
                                )
                            except Exception:
                                pass
                        bar.progress((i + 1) / len(schedule))
                except Exception as e:
                    err_placeholder.error(f"Prediction error: {e}")
                    import traceback
                    st.code(traceback.format_exc())
                bar.empty()
                st.session_state.predictions = preds
                st.session_state.bookmaker_odds = bk_odds

        if "bookmaker_odds" not in st.session_state:
            st.session_state.bookmaker_odds = {}

        preds = st.session_state.predictions
        if preds and schedule:
            st.markdown('<div class="section-header">Predictions</div>', unsafe_allow_html=True)
            for g in schedule:
                blue, red = g["team1"], g["team2"]
                key = f"{blue}_vs_{red}"
                if key in preds:
                    render_prediction_card(
                        blue, red, preds[key], explainer,
                        label=g["tournament"].split("/")[0] if g["tournament"] else "",
                        is_playoffs=is_playoffs_tab1,
                        player_elo=player_elo, game_df=game_df,
                        match_date=g["datetime_utc"].isoformat() if g.get("datetime_utc") else None,
                        league=g.get("league"), tournament=g.get("tournament"),
                        side_known=False, team2idx=team2idx, elo=elo,
                        bookmaker_prob=st.session_state.bookmaker_odds.get(key),
                    )
        elif not schedule:
            st.markdown(f"""
            <div style="text-align:center;padding:60px 20px;color:{C_MUTED}">
              <div style="font-size:2.5rem;margin-bottom:12px">📅</div>
              <div style="font-size:1rem;font-weight:600;color:{C_TEXT}">No games loaded</div>
              <div style="font-size:0.85rem;margin-top:4px">Select a date and leagues, then click Fetch</div>
            </div>""", unsafe_allow_html=True)

    # ── Tab 2: Custom Match ───────────────────────────────────────────────────
    with tab2:
        st.markdown('<div class="section-header">Select teams</div>', unsafe_allow_html=True)

        ca, cb, cc, cd, ce = st.columns([4, 4, 2, 2, 1])
        with ca:
            blue_team = st.selectbox(
                "Blue side", all_teams,
                index=all_teams.index("T1") if "T1" in all_teams else 0,
                label_visibility="collapsed",
            )
            st.markdown(f'<div style="font-size:0.72rem;color:{C_BLUE_SOFT};font-weight:600;margin-top:-8px;margin-bottom:8px">🔵 BLUE SIDE</div>', unsafe_allow_html=True)
        with cb:
            red_team = st.selectbox(
                "Red side", all_teams,
                index=all_teams.index("Gen.G") if "Gen.G" in all_teams else 1,
                label_visibility="collapsed",
            )
            st.markdown(f'<div style="font-size:0.72rem;color:{C_RED_SOFT};font-weight:600;margin-top:-8px;margin-bottom:8px">🔴 RED SIDE</div>', unsafe_allow_html=True)
        with cc:
            best_of_tab2 = st.selectbox("Format", [1, 3, 5],
                                         format_func=lambda x: f"Bo{x}",
                                         index=1, key="bo2",
                                         label_visibility="collapsed")
        with cd:
            league_tab2 = st.selectbox(
                "League", ["LCK", "LPL", "LEC", "LTA N", "LCKC", "LFL",
                           "NACL", "LCP", "WLDs", "MSI", "Other"],
                index=0, key="lg2", label_visibility="collapsed",
            )
            league_tab2 = "" if league_tab2 == "Other" else league_tab2
        with ce:
            predict_btn = st.button("⚡", type="primary", use_container_width=True, help="Predict")

        if predict_btn:
            if blue_team == red_team:
                st.error("Select two different teams.")
            else:
                with st.spinner("Running models…"):
                    result = predict_game(
                        elo, player_elo, game_df, xgb, lgbm, tabpfn, meta,
                        gnn, team_embeddings, team2idx,
                        blue_team, red_team, is_playoffs=is_playoffs_tab2,
                        best_of=best_of_tab2, league=league_tab2,
                    )
                render_prediction_card(
                    blue_team, red_team, result, explainer,
                    is_playoffs=is_playoffs_tab2,
                    player_elo=player_elo, game_df=game_df,
                    side_known=True, team2idx=team2idx, elo=elo,
                )

    # ── Tab 3: Team Explorer ──────────────────────────────────────────────────
    with tab3:
        teams_by_elo = sorted(all_teams, key=lambda t: elo.rating(t), reverse=True)
        selected_team = st.selectbox(
            "Search team", teams_by_elo,
            index=teams_by_elo.index("T1") if "T1" in teams_by_elo else 0,
            label_visibility="collapsed",
        )
        if selected_team:
            render_team_stats(elo, player_elo, game_df, selected_team)

        # Leaderboard
        st.markdown('<div class="section-header">Elo leaderboard — top 30</div>', unsafe_allow_html=True)
        game_counts = pd.concat([game_df["blue_team"], game_df["red_team"]]).value_counts()
        rankings = (
            pd.Series(elo._ratings, name="Elo").reset_index()
            .rename(columns={"index": "Team"})
            .assign(Games=lambda df: df["Team"].map(game_counts).fillna(0).astype(int))
        )
        rankings = rankings[rankings["Games"] >= 20].sort_values("Elo", ascending=False).head(30)
        rankings["Elo"] = rankings["Elo"].round(1)
        rankings.insert(0, "Rank", range(1, len(rankings) + 1))

        ldr_col1, ldr_col2 = st.columns([1, 2])
        with ldr_col1:
            st.dataframe(rankings, hide_index=True, use_container_width=True, height=500)
        with ldr_col2:
            fig = go.Figure(go.Bar(
                x=rankings["Elo"].iloc[::-1],
                y=rankings["Team"].iloc[::-1],
                orientation="h",
                marker=dict(
                    color=rankings["Elo"].iloc[::-1],
                    colorscale=[[0, C_RED], [0.4, C_GOLD], [1, C_BLUE_SOFT]],
                    line_width=0,
                ),
                text=rankings["Elo"].iloc[::-1].apply(lambda v: f"{v:.0f}"),
                textposition="outside",
                textfont=dict(size=10, color=C_MUTED),
            ))
            layout = _chart_layout(height=500)
            layout["xaxis"]["range"] = [rankings["Elo"].min() - 50, rankings["Elo"].max() + 80]
            fig.update_layout(**layout)
            fig.add_vline(x=1500, line_dash="dot", line_color=C_MUTED, opacity=0.5,
                          annotation_text="1500", annotation_font_color=C_MUTED,
                          annotation_position="bottom right")
            st.plotly_chart(fig, use_container_width=True)

    # ── Tab 4: Players ────────────────────────────────────────────────────────
    with tab4:
        _render_players_tab(player_elo, game_df)

    # ── Tab 5: Prediction History ─────────────────────────────────────────────
    with tab5:
        hist_df = load_predictions()

        if hist_df.empty:
            st.markdown(f"""
            <div style="text-align:center;padding:60px 20px;color:{C_MUTED}">
              <div style="font-size:2.5rem;margin-bottom:12px">📋</div>
              <div style="font-size:1rem;font-weight:600;color:{C_TEXT}">No saved predictions yet</div>
              <div style="font-size:0.85rem;margin-top:4px">Save predictions using the 💾 button on any game card</div>
            </div>""", unsafe_allow_html=True)
        else:
            resolved = hist_df[hist_df["actual_blue_win"].notna()]
            pending  = hist_df[hist_df["actual_blue_win"].isna()]

            def _cell(text, color=None):
                c = color or C_TEXT
                return f"<span style='font-size:0.85rem;color:{c}'>{text}</span>"

            if not resolved.empty:
                stats = prediction_stats(hist_df)
                y_res = resolved["actual_blue_win"].values.astype(float)
                n_res = len(resolved)

                # ── Summary stat cards ────────────────────────────────────────
                st.markdown('<div class="section-header">Performance summary</div>', unsafe_allow_html=True)
                s_cards = [
                    stat_card("Total saved",       str(stats["total"])),
                    stat_card("Resolved",          str(stats["resolved"])),
                    stat_card("Ensemble accuracy", f"{stats.get('ensemble_accuracy',0):.1%}",
                              color=C_GREEN if stats.get("ensemble_accuracy",0) > 0.55 else C_RED_SOFT),
                    stat_card("Ensemble Brier",    f"{stats.get('ensemble_brier', 0):.4f}",
                              sub="lower is better (random=0.25)"),
                    stat_card("Calibration (ECE)", f"{stats.get('ece',0):.3f}",
                              sub="lower is better"),
                ]
                sc = st.columns(len(s_cards))
                for col, card in zip(sc, s_cards):
                    col.markdown(card, unsafe_allow_html=True)

                st.markdown("<div style='margin-top:16px'></div>", unsafe_allow_html=True)

                # ── Charts row: accuracy + reliability diagram ─────────────
                ch1, ch2 = st.columns(2)
                with ch1:
                    model_accs = {k: v for k, v in {
                        "Elo":        stats.get("elo_accuracy"),
                        "XGBoost":    stats.get("xgb_accuracy"),
                        "LightGBM":   stats.get("lgbm_accuracy"),
                        "GNN":        stats.get("gnn_accuracy"),
                        "TabPFN":     stats.get("tabpfn_accuracy"),
                        "Ensemble":   stats.get("ensemble_accuracy"),
                        "Bookmaker":  stats.get("bookmaker_accuracy"),
                    }.items() if v is not None}
                    bar_colors = [C_GREEN if v > 0.55 else C_GOLD if v > 0.50 else C_RED_SOFT
                                  for v in model_accs.values()]
                    fig_acc = go.Figure(go.Bar(
                        x=list(model_accs.keys()), y=list(model_accs.values()),
                        marker_color=bar_colors, marker_line_width=0,
                        text=[f"{v:.1%}" for v in model_accs.values()],
                        textposition="outside",
                        textfont=dict(size=12, color=C_TEXT),
                    ))
                    layout_acc = _chart_layout(height=280, title="Accuracy by model")
                    layout_acc["yaxis"]["range"] = [0.45, 1.0]
                    layout_acc["yaxis"]["tickformat"] = ".0%"
                    fig_acc.add_hline(y=0.5, line_dash="dot", line_color=C_MUTED, opacity=0.5)
                    fig_acc.update_layout(**layout_acc)
                    st.plotly_chart(fig_acc, use_container_width=True)

                with ch2:
                    # Multi-model reliability diagram
                    # Each model gets its own calibration curve on one plot.
                    # Points below the diagonal = overconfident.
                    # Marker size = number of predictions in that bin.
                    _cal_models = [
                        ("Elo",       "elo_prob",       "#888888"),
                        ("XGBoost",   "xgb_prob",       "#e07b39"),
                        ("LightGBM",  "lgbm_prob",      "#3ab073"),
                        ("GNN",       "gnn_prob",       "#4c8ef7"),
                        ("TabPFN",    "tabpfn_prob",    "#a64ca6"),
                        ("Ensemble",  "ensemble_prob",  C_RED_SOFT),
                        ("Bookmaker", "bookmaker_prob", "#f0c040"),
                    ]
                    bins = np.linspace(0, 1, 11)
                    fig_cal = go.Figure()
                    fig_cal.add_trace(go.Scatter(
                        x=[0, 1], y=[0, 1], mode="lines",
                        line=dict(dash="dot", color=C_MUTED, width=1),
                        name="Perfect", showlegend=True,
                    ))
                    for m_name, m_col, m_color in _cal_models:
                        if m_col not in resolved.columns:
                            continue
                        col_data = resolved[m_col].dropna()
                        if col_data.empty:
                            continue
                        probs_m = resolved.loc[col_data.index, m_col].values
                        y_m     = y_res[col_data.index] if len(col_data) < n_res else y_res
                        cal_x, cal_y, cal_n = [], [], []
                        for lo, hi in zip(bins[:-1], bins[1:]):
                            mask = (probs_m >= lo) & (probs_m < hi)
                            if mask.sum() >= 2:
                                cal_x.append(float((lo + hi) / 2))
                                cal_y.append(float(y_m[mask].mean()))
                                cal_n.append(int(mask.sum()))
                        if cal_x:
                            fig_cal.add_trace(go.Scatter(
                                x=cal_x, y=cal_y, mode="markers+lines",
                                marker=dict(size=[max(6, n // 3) for n in cal_n],
                                            color=m_color, opacity=0.85),
                                line=dict(color=m_color, width=1.5),
                                name=m_name,
                                text=[f"n={n}" for n in cal_n],
                                hovertemplate="%{text}<br>pred=%{x:.2f} actual=%{y:.2f}<extra>" + m_name + "</extra>",
                            ))
                    layout_cal = _chart_layout(height=280, title="Reliability diagram — all models")
                    layout_cal["xaxis"]["title"] = "Predicted probability"
                    layout_cal["yaxis"]["title"] = "Actual win rate"
                    layout_cal["xaxis"]["range"] = [0, 1]
                    layout_cal["yaxis"]["range"] = [0, 1]
                    fig_cal.update_layout(**layout_cal)
                    st.plotly_chart(fig_cal, use_container_width=True)
                    if n_res < 50:
                        st.caption(f"⚠️ Only {n_res} resolved predictions — calibration curve is noisy below ~50 games.")

                # ── Brier score breakdown ─────────────────────────────────────
                # Brier = calibration loss + refinement loss.
                # Lower is better; a coin-flip predictor scores 0.25.
                brier_rows = {k: v for k, v in {
                    "Elo":       stats.get("elo_brier"),
                    "XGBoost":   stats.get("xgb_brier"),
                    "LightGBM":  stats.get("lgbm_brier"),
                    "GNN":       stats.get("gnn_brier"),
                    "TabPFN":    stats.get("tabpfn_brier"),
                    "Ensemble":  stats.get("ensemble_brier"),
                    "Bookmaker": stats.get("bookmaker_brier"),
                }.items() if v is not None}
                if brier_rows:
                    st.markdown('<div class="section-header">Brier scores</div>', unsafe_allow_html=True)
                    st.caption("Brier = mean squared error of probabilities. Lower is better. Random = 0.25, perfect = 0.00.")
                    b_cols = st.columns(len(brier_rows))
                    for col, (name, brier) in zip(b_cols, brier_rows.items()):
                        color = C_GREEN if brier < 0.22 else C_GOLD if brier < 0.24 else C_RED_SOFT
                        col.markdown(stat_card(name, f"{brier:.4f}", color=color), unsafe_allow_html=True)

                # ── Model comparison table (Brier Skill Score vs Elo baseline) ─
                elo_brier = stats.get("elo_brier")
                if elo_brier and len(brier_rows) > 1:
                    st.markdown('<div class="section-header">Model comparison vs Elo baseline</div>', unsafe_allow_html=True)
                    st.caption("Brier Skill Score = 1 − (model Brier / Elo Brier). Positive = better than Elo.")
                    _cmp_models = [
                        ("Elo (baseline)", "elo"),
                        ("XGBoost",        "xgb"),
                        ("LightGBM",       "lgbm"),
                        ("GNN",            "gnn"),
                        ("TabPFN",         "tabpfn"),
                        ("Ensemble",       "ensemble"),
                        ("Bookmaker",      "bookmaker"),
                    ]
                    cmp_rows = []
                    for label, key in _cmp_models:
                        acc   = stats.get(f"{key}_accuracy")
                        brier = stats.get(f"{key}_brier")
                        if acc is None or brier is None:
                            continue
                        if key == "elo":
                            bss_str = "— (baseline)"
                        else:
                            bss = 1.0 - brier / elo_brier
                            bss_str = f"{bss:+.1%}"
                        cmp_rows.append({"Model": label, "Accuracy": f"{acc:.1%}", "Brier": f"{brier:.4f}", "Skill vs Elo": bss_str})
                    if cmp_rows:
                        st.dataframe(
                            pd.DataFrame(cmp_rows),
                            hide_index=True, use_container_width=True,
                        )

            # Resolved table
            if not resolved.empty:
                st.markdown('<div class="section-header">Resolved predictions</div>', unsafe_allow_html=True)
                # Header row — Bookmaker column shown only if any row has it
                _has_bk = "bookmaker_prob" in resolved.columns and resolved["bookmaker_prob"].notna().any()
                if _has_bk:
                    _res_widths = [1.2, 0.8, 2, 2, 2.2, 2.2, 0.5, 1.3, 1.3, 1.3, 1.3, 0.8]
                    _res_labels = ["Date", "League", "Blue", "Red", "Predicted", "Result", "✓", "Elo", "XGB", "Ens", "Book", ""]
                else:
                    _res_widths = [1.2, 0.8, 2, 2, 2.2, 2.2, 0.5, 1.5, 1.5, 1.5, 0.8]
                    _res_labels = ["Date", "League", "Blue", "Red", "Predicted", "Result", "✓", "Elo", "XGB", "Ens", ""]
                hc = st.columns(_res_widths)
                for i, label in enumerate(_res_labels):
                    hc[i].markdown(
                        f"<span style='font-size:0.75rem;color:{C_MUTED};font-weight:600'>{label}</span>",
                        unsafe_allow_html=True)
                for _, r in resolved.iterrows():
                    correct = (r["ensemble_prob"] > 0.5) == bool(r["actual_blue_win"])
                    date_str = pd.to_datetime(r["match_date"], errors="coerce")
                    date_str = date_str.strftime("%Y-%m-%d") if pd.notna(date_str) else "?"
                    score_str = f" {r['actual_score']}" if pd.notna(r.get("actual_score")) and r.get("actual_score") else ""
                    result_str = (f"🔵 Blue{score_str}" if r["actual_blue_win"] == 1 else f"🔴 Red{score_str}")
                    pred_str   = (f"🔵 {r['ensemble_prob']:.0%}" if r["ensemble_prob"] >= 0.5
                                  else f"🔴 {1-r['ensemble_prob']:.0%}")
                    rc = st.columns(_res_widths)
                    rc[0].markdown(_cell(date_str, C_MUTED), unsafe_allow_html=True)
                    rc[1].markdown(_cell(r.get("league") or ""), unsafe_allow_html=True)
                    rc[2].markdown(_cell(r["blue_team"], C_BLUE_SOFT), unsafe_allow_html=True)
                    rc[3].markdown(_cell(r["red_team"], C_RED_SOFT), unsafe_allow_html=True)
                    rc[4].markdown(_cell(pred_str), unsafe_allow_html=True)
                    rc[5].markdown(_cell(result_str), unsafe_allow_html=True)
                    rc[6].markdown(_cell("✅" if correct else "❌"), unsafe_allow_html=True)
                    rc[7].markdown(_cell(f"{r['elo_prob']:.0%}", C_MUTED), unsafe_allow_html=True)
                    rc[8].markdown(_cell(f"{r['xgb_prob']:.0%}", C_MUTED), unsafe_allow_html=True)
                    rc[9].markdown(_cell(f"{r['ensemble_prob']:.0%}"), unsafe_allow_html=True)
                    if _has_bk:
                        bk_val = r.get("bookmaker_prob")
                        rc[10].markdown(_cell(f"{bk_val:.0%}" if pd.notna(bk_val) else "—", C_MUTED), unsafe_allow_html=True)
                        if rc[11].button("🗑", key=f"del_res_{r['id']}", help="Delete this prediction"):
                            delete_prediction(int(r["id"]))
                            st.rerun()
                    else:
                        if rc[10].button("🗑", key=f"del_res_{r['id']}", help="Delete this prediction"):
                            delete_prediction(int(r["id"]))
                            st.rerun()

            # Pending
            if not pending.empty:
                st.markdown(f'<div class="section-header">Pending ({len(pending)})</div>', unsafe_allow_html=True)
                pend_disp = pending.copy()
                pend_disp["Predicted"] = pend_disp["ensemble_prob"].apply(
                    lambda p: f"🔵 Blue {p:.0%}" if p >= 0.5 else f"🔴 Red {1-p:.0%}"
                )
                # Show predicted score distribution (top outcome only) if stored
                def _top_score(dist_json):
                    try:
                        dist = json.loads(dist_json) if dist_json else None
                        if dist:
                            top = max(dist, key=lambda x: x[1])
                            return f"{top[0]} ({top[1]:.0%})"
                    except Exception:
                        pass
                    return "—"
                pend_disp["Score dist"] = pend_disp.get("score_dist", pd.Series(dtype=str)).apply(_top_score) if "score_dist" in pend_disp.columns else "—"
                pend_disp["Date"] = pd.to_datetime(pend_disp["match_date"], errors="coerce").dt.strftime("%Y-%m-%d %H:%M").fillna("?")
                # Header row
                _pend_widths = [0.6, 1.2, 1, 2, 2, 2, 1.5, 0.8]
                ph = st.columns(_pend_widths)
                for label, col in zip(["ID", "Date", "League", "Blue", "Red", "Predicted", "Score", ""], ph):
                    col.markdown(
                        f"<span style='font-size:0.75rem;color:{C_MUTED};font-weight:600'>{label}</span>",
                        unsafe_allow_html=True)
                for _, r in pend_disp.iterrows():
                    pc = st.columns(_pend_widths)
                    pc[0].markdown(_cell(str(int(r["id"])), C_MUTED), unsafe_allow_html=True)
                    pc[1].markdown(_cell(r["Date"], C_MUTED), unsafe_allow_html=True)
                    pc[2].markdown(_cell(r.get("league") or "—"), unsafe_allow_html=True)
                    pc[3].markdown(_cell(r["blue_team"], C_BLUE_SOFT), unsafe_allow_html=True)
                    pc[4].markdown(_cell(r["red_team"], C_RED_SOFT), unsafe_allow_html=True)
                    pc[5].markdown(_cell(r["Predicted"]), unsafe_allow_html=True)
                    pc[6].markdown(_cell(r["Score dist"], C_MUTED), unsafe_allow_html=True)
                    if pc[7].button("🗑", key=f"del_pend_{r['id']}", help="Delete this prediction"):
                        delete_prediction(int(r["id"]))
                        st.rerun()

                col_m1, col_m2, col_m3, col_m4 = st.columns([1, 2, 1, 1])
                with col_m1:
                    pred_id = st.number_input("Prediction ID", min_value=1, step=1, label_visibility="collapsed")
                with col_m2:
                    winner_side = st.radio("Who won?", ["Blue side", "Red side"],
                                           horizontal=True, label_visibility="collapsed")
                with col_m3:
                    # Score input — options depend on best_of for that prediction
                    sel_pred = pending[pending["id"] == pred_id]
                    bo = int(sel_pred["best_of"].iloc[0]) if not sel_pred.empty and "best_of" in sel_pred.columns else 1
                    if bo == 3:
                        score_opts = ["2-0", "2-1", "1-2", "0-2"]
                    elif bo == 5:
                        score_opts = ["3-0", "3-1", "3-2", "2-3", "1-3", "0-3"]
                    else:
                        score_opts = ["1-0", "0-1"]
                    actual_score = st.selectbox("Score", score_opts, label_visibility="collapsed")
                with col_m4:
                    if st.button("Mark result", type="primary", use_container_width=True):
                        set_result(int(pred_id), 1 if winner_side == "Blue side" else 0,
                                   actual_score=actual_score)
                        st.rerun()

            col_r1, _ = st.columns([1, 3])
            with col_r1:
                if st.button("🔄 Re-check results from data", use_container_width=True):
                    n = reconcile_results(game_df)
                    st.success(f"Resolved {n} prediction(s)." if n else "No new results found.")
                    st.rerun()


    # ── Tab 6: Data Status ────────────────────────────────────────────────────
    with tab6:
        st.markdown('<div class="section-header">Training data</div>', unsafe_allow_html=True)

        today = datetime.now(timezone.utc).replace(tzinfo=None)
        latest_game = pd.to_datetime(game_df["date"]).max()
        oldest_game = pd.to_datetime(game_df["date"]).min()
        days_stale  = (today - latest_game.to_pydatetime()).days
        stale_color = C_RED_SOFT if days_stale > 14 else (C_GOLD if days_stale > 7 else C_GREEN)

        sc = st.columns(5)
        sc[0].markdown(stat_card("Total games",   f"{len(game_df):,}"),        unsafe_allow_html=True)
        sc[1].markdown(stat_card("Date range",    f"{oldest_game.strftime('%Y-%m-%d')} → {latest_game.strftime('%Y-%m-%d')}"), unsafe_allow_html=True)
        sc[2].markdown(stat_card("Days since last update", str(days_stale), color=stale_color), unsafe_allow_html=True)
        sc[3].markdown(stat_card("Teams tracked", str(len(team2idx))),          unsafe_allow_html=True)
        sc[4].markdown(stat_card("Leagues covered", str(game_df["league"].nunique())), unsafe_allow_html=True)

        # ── Games per week chart ──────────────────────────────────────────────
        st.markdown('<div class="section-header">Games per week</div>', unsafe_allow_html=True)
        gdf = game_df.copy()
        gdf["week"] = pd.to_datetime(gdf["date"]).dt.to_period("W").dt.start_time
        weekly = gdf.groupby("week").size().reset_index(name="games")
        # Highlight the last 8 weeks
        cutoff = weekly["week"].max() - pd.Timedelta(weeks=8)
        weekly["color"] = weekly["week"].apply(lambda w: C_BLUE if w >= cutoff else C_MUTED)
        fig_w = go.Figure(go.Bar(
            x=weekly["week"], y=weekly["games"],
            marker_color=weekly["color"],
            hovertemplate="%{x|%Y-%m-%d}<br>Games: %{y}<extra></extra>",
        ))
        fig_w.update_layout(**_chart_layout(height=260))
        fig_w.update_xaxes(title_text="")
        fig_w.update_yaxes(title_text="Games")
        st.plotly_chart(fig_w, use_container_width=True, key="data_weekly_games")

        # ── Per-league coverage ───────────────────────────────────────────────
        st.markdown('<div class="section-header">Coverage by league</div>', unsafe_allow_html=True)
        league_stats = (
            game_df.groupby("league")
            .agg(
                games  =("gameid", "count"),
                first  =("date",   "min"),
                latest =("date",   "max"),
            )
            .reset_index()
            .sort_values("games", ascending=False)
        )
        league_stats["days_stale"] = (today - pd.to_datetime(league_stats["latest"]).dt.to_pydatetime()).apply(lambda d: d.days if hasattr(d, "days") else int(d.total_seconds() // 86400))
        league_stats["first"]  = pd.to_datetime(league_stats["first"]).dt.strftime("%Y-%m-%d")
        league_stats["latest"] = pd.to_datetime(league_stats["latest"]).dt.strftime("%Y-%m-%d")

        def _stale_badge(d):
            if d > 60:  return f'<span style="color:{C_MUTED}">{d}d</span>'
            if d > 14:  return f'<span style="color:{C_RED_SOFT}">{d}d</span>'
            if d > 7:   return f'<span style="color:{C_GOLD}">{d}d</span>'
            return f'<span style="color:{C_GREEN}">{d}d</span>'

        league_stats["staleness"] = league_stats["days_stale"].apply(_stale_badge)

        tbl_html = '<table style="width:100%;border-collapse:collapse;font-size:0.83rem">'
        tbl_html += f'<tr style="color:{C_MUTED};border-bottom:1px solid #333">'
        for col in ["League", "Games", "First game", "Latest game", "Days stale"]:
            tbl_html += f'<th style="padding:6px 10px;text-align:left">{col}</th>'
        tbl_html += "</tr>"
        for _, row in league_stats.iterrows():
            tbl_html += f'<tr style="border-bottom:1px solid #222">'
            tbl_html += f'<td style="padding:6px 10px;font-weight:600">{row["league"]}</td>'
            tbl_html += f'<td style="padding:6px 10px">{row["games"]:,}</td>'
            tbl_html += f'<td style="padding:6px 10px;color:{C_MUTED}">{row["first"]}</td>'
            tbl_html += f'<td style="padding:6px 10px">{row["latest"]}</td>'
            tbl_html += f'<td style="padding:6px 10px">{row["staleness"]}</td>'
            tbl_html += "</tr>"
        tbl_html += "</table>"
        st.markdown(tbl_html, unsafe_allow_html=True)

        # ── CSV source files ──────────────────────────────────────────────────
        st.markdown('<div class="section-header" style="margin-top:24px">Source CSV files</div>', unsafe_allow_html=True)
        csv_files = sorted(DATA_DIR.glob("*.csv"))
        if csv_files:
            csv_rows = []
            for f in csv_files:
                if f.suffix == ".csv" and not f.name.endswith(".tmp"):
                    stat = f.stat()
                    mtime = datetime.fromtimestamp(stat.st_mtime)
                    age_d = (today - mtime).days
                    csv_rows.append({
                        "File":     f.name,
                        "Size":     f"{stat.st_size / 1e6:.1f} MB",
                        "Modified": mtime.strftime("%Y-%m-%d %H:%M"),
                        "Age":      f"{age_d}d ago",
                    })
            if csv_rows:
                st.dataframe(pd.DataFrame(csv_rows), hide_index=True, use_container_width=True)

        # ── Model files ───────────────────────────────────────────────────────
        st.markdown('<div class="section-header">Saved model files</div>', unsafe_allow_html=True)
        model_files = [
            ("xgb_model.joblib",    "XGBoost"),
            ("lgbm_model.joblib",   "LightGBM"),
            ("gnn_weights.pt",      "GNN"),
            ("tabpfn_model.joblib", "TabPFN"),
            ("meta_learner.joblib", "Meta-learner"),
            ("elo_ratings.pkl",     "Elo state"),
        ]
        model_rows = []
        for fname, label in model_files:
            fpath = SAVE_DIR / fname
            if fpath.exists():
                stat  = fpath.stat()
                mtime = datetime.fromtimestamp(stat.st_mtime)
                age_d = (today - mtime).days
                model_rows.append({
                    "Model":    label,
                    "File":     fname,
                    "Size":     f"{stat.st_size / 1e6:.2f} MB",
                    "Saved":    mtime.strftime("%Y-%m-%d %H:%M"),
                    "Age":      f"{age_d}d ago",
                })
            else:
                model_rows.append({
                    "Model": label, "File": fname,
                    "Size": "—", "Saved": "—", "Age": "missing",
                })
        if model_rows:
            mdf = pd.DataFrame(model_rows)
            st.dataframe(mdf, hide_index=True, use_container_width=True)

        # ── Parquet cache ─────────────────────────────────────────────────────
        cache_path = ROOT / "src" / "data" / "game_df_cache.parquet"
        if cache_path.exists():
            cs   = cache_path.stat()
            cmtime = datetime.fromtimestamp(cs.st_mtime)
            st.caption(
                f"Parquet cache: `{cache_path.name}` — "
                f"{cs.st_size / 1e6:.1f} MB — "
                f"last rebuilt {cmtime.strftime('%Y-%m-%d %H:%M')} "
                f"({(today - cmtime).days}d ago)"
            )

        # ── Bookmaker odds (The Odds API) ─────────────────────────────────────
        st.markdown('<div class="section-header">Bookmaker odds (The Odds API)</div>', unsafe_allow_html=True)
        st.markdown(
            "Enter your API key from [the-odds-api.com](https://the-odds-api.com) (free tier: 500 req/month). "
            "When set, bookmaker implied probabilities are fetched automatically when you click **Predict all games**."
        )
        _saved_key = st.session_state.get("odds_api_key", os.environ.get("ODDS_API_KEY", ""))
        ok_col1, ok_col2, ok_col3 = st.columns([3, 1, 2])
        with ok_col1:
            new_odds_key = st.text_input(
                "API key", value=_saved_key,
                type="password", label_visibility="collapsed",
                placeholder="Paste your Odds API key here…",
            )
        with ok_col2:
            if st.button("Save key", use_container_width=True):
                st.session_state["odds_api_key"] = new_odds_key.strip()
                st.success("Saved.")
        with ok_col3:
            if new_odds_key.strip():
                if st.button("🔍 Check active LoL sports", use_container_width=True):
                    active = list_lol_sport_keys(new_odds_key.strip())
                    if active:
                        st.success("Active LoL keys: " + ", ".join(s["key"] for s in active))
                    else:
                        st.warning("No active LoL sports found (may be off-season or key invalid).")

        # ── Retrain on full data ──────────────────────────────────────────────
        st.markdown('<div class="section-header">Retrain on full data</div>', unsafe_allow_html=True)
        st.markdown(
            "Trains all models on **100 % of available data** (no held-out test split). "
            "Use this for live predictions after you are satisfied with model accuracy from the notebook. "
            "Takes ~10–30 min depending on GPU."
        )
        rt_col1, rt_col2 = st.columns([2, 3])
        with rt_col1:
            retrain_btn = st.button("🚀 Retrain on full data", type="primary", use_container_width=True)
        with rt_col2:
            st.caption(
                "Launches `retrain_full.py` in the background. "
                "Progress is printed to the terminal. "
                "Restart the app when it finishes to load the new models."
            )

        if retrain_btn:
            import subprocess
            log_path = ROOT / "retrain_full.log"
            try:
                proc = subprocess.Popen(
                    [sys.executable, str(ROOT / "retrain_full.py")],
                    stdout=open(log_path, "w"),
                    stderr=subprocess.STDOUT,
                    cwd=str(ROOT),
                )
                st.success(
                    f"Training started (PID {proc.pid}). "
                    f"Tail the log: `tail -f {log_path.name}` "
                    f"or check the terminal. Restart the app when done."
                )
            except Exception as e:
                st.error(f"Failed to launch training: {e}")

        # ── Refresh instructions ──────────────────────────────────────────────
        with st.expander("How to refresh data"):
            st.markdown(f"""
1. Download the latest Oracle's Elixir CSVs or run:
   ```
   uv run python refresh_data.py
   ```
2. Restart the app — the Parquet cache rebuilds automatically when CSVs are newer.
3. Retrain models (only needed every few months or after a major patch):
   - **Full training (recommended for live use):** click **Retrain on full data** above,
     or run `uv run python retrain_full.py` in a terminal.
   - **Development / evaluation:** open `Notebooks/PreDraft_Training.ipynb`,
     set `FORCE_RETRAIN = True`, run all cells.
""")


if __name__ == "__main__":
    main()
