"""
retrain_full.py — Retrain all models on the COMPLETE dataset (no held-out test split).

Use this for live/production predictions after you are satisfied with model accuracy
from the evaluation notebook.  The regular notebook trains on 70 % and evaluates on
the remaining 30 % — that split exists purely for model development.  Here we use
everything.

Meta-learner strategy (no free lunch — we still need OOF to avoid leakage):
  • XGBoost / LightGBM  : 5-fold TimeSeriesSplit OOF  → fast, no info leak
  • GNN                 : last 15 % as val for early-stop + OOF predictions
  • TabPFN              : last 20 % as OOF (TabPFN is a single forward pass, no epochs)
  • Elo                 : already zero-leakage (annotated chronologically)
  After OOF collection every base model is REFIT on 100 % of data.
  The meta LogisticRegression is trained on the OOF stack.

Usage:
    uv run python retrain_full.py
    # or from the dashboard Data Status tab → "Retrain on full data"
"""

import sys, os, pickle, time
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, roc_auc_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from data.oracleselixir import load_and_cache, load_raw, build_vocabularies
from models.elo import EloSystem, PlayerEloSystem, LeagueEloSystem
from models.match_gnn import build_match_graph, PreDraftPredictor

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT     = Path(__file__).parent
SAVE_DIR = ROOT / "models" / "predraft"
DATA_DIR = ROOT / "src" / "data" / "oracleselixir"
CACHE    = ROOT / "src" / "data" / "game_df_cache.parquet"

SAVE_DIR.mkdir(parents=True, exist_ok=True)

ELO_PATH    = SAVE_DIR / "elo_ratings.pkl"
XGB_PATH    = SAVE_DIR / "xgb_model.joblib"
LGBM_PATH   = SAVE_DIR / "lgbm_model.joblib"
TABPFN_PATH = SAVE_DIR / "tabpfn_model.joblib"
META_PATH   = SAVE_DIR / "meta_learner.joblib"
GNN_PATH    = SAVE_DIR / "gnn_weights.pt"
VOC_PATH    = SAVE_DIR / "team2idx.pkl"
CKPT_PATH   = SAVE_DIR / "_full_train_ckpt.pt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

# GNN hyper-params — mirror the training notebook
EPOCHS   = 150
PATIENCE = 20
LR       = 3e-4

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _t(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def flip_augment(X: np.ndarray, y: np.ndarray):
    """Swap blue/red features, negate elo_diff, invert label."""
    Xf = X.copy()
    Xf[:, 0]     = -X[:, 0]
    Xf[:, 1]     = X[:, 2];  Xf[:, 2]  = X[:, 1]   # ewma_wr
    Xf[:, 4]     = X[:, 5];  Xf[:, 5]  = X[:, 4]   # momentum
    Xf[:, 6:11]  = X[:, 11:16]                       # role elos
    Xf[:, 11:16] = X[:, 6:11]
    Xf[:, 17]    = X[:, 18];  Xf[:, 18] = X[:, 17]  # games_conf
    return np.vstack([X, Xf]), np.concatenate([y, 1 - y])


def df_to_tensors(df, team2idx, device, augment=False):
    bi  = [team2idx.get(t, 0) for t in df["blue_team"]]
    ri  = [team2idx.get(t, 0) for t in df["red_team"]]
    ctx = df[FEATURE_COLS].values.astype(np.float32)
    lbl = df["blue_win"].values.astype(np.float32)
    if augment:
        ctx_f = ctx.copy()
        ctx_f[:, 0]    = -ctx[:, 0]
        ctx_f[:, 1]    = ctx[:, 2];  ctx_f[:, 2]  = ctx[:, 1]   # ewma_wr
        ctx_f[:, 4]    = ctx[:, 5];  ctx_f[:, 5]  = ctx[:, 4]   # momentum
        ctx_f[:, 6:11] = ctx[:, 11:16]                            # role elos
        ctx_f[:, 11:16]= ctx[:, 6:11]
        ctx_f[:, 17]   = ctx[:, 18];  ctx_f[:, 18] = ctx[:, 17]  # games_conf
        bi  = bi  + [team2idx.get(t, 0) for t in df["red_team"]]
        ri  = ri  + [team2idx.get(t, 0) for t in df["blue_team"]]
        ctx = np.vstack([ctx, ctx_f])
        lbl = np.concatenate([lbl, 1 - lbl])
    return (
        torch.tensor(bi,  dtype=torch.long,  device=device),
        torch.tensor(ri,  dtype=torch.long,  device=device),
        torch.tensor(ctx, dtype=torch.float, device=device),
        torch.tensor(lbl, dtype=torch.float, device=device),
    )


def train_gnn(train_df, val_df, team2idx, graph, model, optimizer, scheduler):
    """Train GNN with early stopping. Returns best state_dict."""
    best_loss, patience_count = float("inf"), 0
    best_state = None
    train_eval = train_df[~train_df["cold_start"]].reset_index(drop=True)
    val_eval   = val_df[~val_df["cold_start"]].reset_index(drop=True)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        with torch.set_grad_enabled(True):
            emb = model.encode_all_teams(graph)
        bi, ri, ctx, lbl = df_to_tensors(train_eval, team2idx, device, augment=True)
        total, n = 0.0, len(lbl)
        for s in range(0, n, 128):
            e, r, c, y = bi[s:s+128], ri[s:s+128], ctx[s:s+128], lbl[s:s+128]
            prob = model(emb, e, r, c)
            loss = F.binary_cross_entropy(prob, y)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            emb  = model.encode_all_teams(graph)
            total += loss.item() * len(y)
        scheduler.step()

        model.eval()
        with torch.no_grad():
            emb  = model.encode_all_teams(graph)
            bi_v, ri_v, ctx_v, lbl_v = df_to_tensors(val_eval, team2idx, device)
            probs_v = model(emb, bi_v, ri_v, ctx_v)
            val_loss = F.binary_cross_entropy(probs_v, lbl_v).item()

        if epoch % 10 == 0 or epoch == 1:
            _t(f"  GNN epoch {epoch:3d}  train_loss={total/n:.4f}  val_loss={val_loss:.4f}"
               + ("  ← best" if val_loss < best_loss else ""))

        if val_loss < best_loss:
            best_loss, patience_count = val_loss, 0
            raw = model._orig_mod if hasattr(model, "_orig_mod") else model
            best_state = {k: v.clone() for k, v in raw.state_dict().items()}
        else:
            patience_count += 1
            if patience_count >= PATIENCE:
                _t(f"  Early stop at epoch {epoch}  (best val_loss={best_loss:.4f})")
                break

    return best_state


def gnn_predict_df(df, model, graph, team2idx):
    model.eval()
    eval_df = df[~df["cold_start"]].reset_index(drop=True) if "cold_start" in df.columns else df
    bi  = torch.tensor([team2idx.get(t, 0) for t in eval_df["blue_team"]], dtype=torch.long,  device=device)
    ri  = torch.tensor([team2idx.get(t, 0) for t in eval_df["red_team"]],  dtype=torch.long,  device=device)
    ctx = torch.tensor(eval_df[FEATURE_COLS].values.astype(np.float32),    dtype=torch.float, device=device)
    with torch.no_grad():
        emb  = model.encode_all_teams(graph)
        prob = model(emb, bi, ri, ctx).cpu().numpy()
    return eval_df, prob


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    _t(f"Device: {device}")
    _t("=" * 60)
    _t("Step 1/7 — Load & cache data")
    _t("=" * 60)

    game_df = load_and_cache(csv_dir=DATA_DIR, cache_path=CACHE, force_rebuild=False)
    if "is_playoffs" not in game_df.columns:
        raw = load_raw(str(DATA_DIR), complete_only=True)
        if "playoffs" in raw.columns:
            playoff_map = raw.drop_duplicates("gameid").set_index("gameid")["playoffs"]
            game_df["is_playoffs"] = game_df["gameid"].map(playoff_map).fillna(0).astype(int)
        else:
            game_df["is_playoffs"] = 0
        game_df.to_parquet(CACHE, index=False)
    game_df = game_df.sort_values("date").reset_index(drop=True)
    _t(f"  {len(game_df):,} games  {game_df['date'].min().date()} → {game_df['date'].max().date()}")

    _t("=" * 60)
    _t("Step 2/7 — Elo annotation (zero-leakage)")
    _t("=" * 60)
    elo        = EloSystem()
    player_elo = PlayerEloSystem()
    league_elo = LeagueEloSystem()
    game_df = elo.process_and_annotate(game_df, min_games=5,
                                        player_elo=player_elo, league_elo=league_elo)
    _, team2idx, _ = build_vocabularies(game_df)
    full_eval = game_df[~game_df["cold_start"]].reset_index(drop=True)
    n = len(full_eval)
    _t(f"  Evaluable games: {n:,}  (cold-start excluded: {game_df['cold_start'].sum():,})")

    X_full = full_eval[FEATURE_COLS].values.astype(np.float32)
    y_full = full_eval["blue_win"].values

    # OOF arrays (filled progressively)
    oof_xgb    = np.zeros(n)
    oof_lgbm   = np.zeros(n)
    oof_gnn    = np.full(n, np.nan)
    oof_tabpfn = np.full(n, np.nan)
    oof_elo    = full_eval["elo_win_prob"].values.copy()  # already zero-leakage

    # ---------------------------------------------------------------------------
    _t("=" * 60)
    _t("Step 3/7 — XGBoost & LightGBM  (5-fold TimeSeriesSplit OOF)")
    _t("=" * 60)
    tss = TimeSeriesSplit(n_splits=5)
    for fold, (tr_idx, val_idx) in enumerate(tss.split(X_full), 1):
        X_tr, y_tr = flip_augment(X_full[tr_idx], y_full[tr_idx])
        X_vl, y_vl = X_full[val_idx], y_full[val_idx]

        xgb_fold = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                                  subsample=0.8, eval_metric="logloss", verbosity=0)
        xgb_fold.fit(X_tr, y_tr, eval_set=[(X_vl, y_vl)], verbose=False)
        oof_xgb[val_idx] = xgb_fold.predict_proba(X_vl)[:, 1]

        lgbm_fold = LGBMClassifier(n_estimators=500, max_depth=5, learning_rate=0.03,
                                    num_leaves=31, subsample=0.8, colsample_bytree=0.8,
                                    min_child_samples=20, verbose=-1)
        lgbm_fold.fit(X_tr, y_tr, eval_set=[(X_vl, y_vl)])
        oof_lgbm[val_idx] = lgbm_fold.predict_proba(X_vl)[:, 1]

        _t(f"  Fold {fold}/5  val_size={len(val_idx):,}"
           f"  xgb_auc={roc_auc_score(y_vl, oof_xgb[val_idx]):.4f}"
           f"  lgbm_auc={roc_auc_score(y_vl, oof_lgbm[val_idx]):.4f}")

    _t(f"  XGB  OOF AUC: {roc_auc_score(y_full, oof_xgb):.4f}")
    _t(f"  LGBM OOF AUC: {roc_auc_score(y_full, oof_lgbm):.4f}")

    # ---------------------------------------------------------------------------
    _t("=" * 60)
    _t("Step 4/7 — GNN  (last 15 % as val for early stop + OOF)")
    _t("=" * 60)
    n_gnn_val  = int(n * 0.15)
    gnn_tr_df  = full_eval.iloc[:-n_gnn_val].reset_index(drop=True)
    gnn_val_df = full_eval.iloc[-n_gnn_val:].reset_index(drop=True)

    # Build graph on the GNN training slice
    # (use full game_df indices so team vocab is consistent)
    tr_game_ids = set(gnn_tr_df["gameid"]) if "gameid" in gnn_tr_df.columns else None
    tr_graph_df = game_df[game_df["gameid"].isin(tr_game_ids)] if tr_game_ids else gnn_tr_df
    gnn_tr_graph = build_match_graph(tr_graph_df, team2idx, halflife_days=180, max_repeats=5).to(device)

    gnn_model = PreDraftPredictor(num_teams=len(team2idx), emb_dim=128,
                                   out_dim=64, hidden=128, dropout=0.2).to(device)
    gnn_model = torch.compile(gnn_model)
    optimizer  = torch.optim.AdamW(gnn_model.parameters(), lr=LR, weight_decay=1e-3)
    scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)

    best_state = train_gnn(gnn_tr_df, gnn_val_df, team2idx, gnn_tr_graph,
                            gnn_model, optimizer, scheduler)
    raw_gnn = gnn_model._orig_mod if hasattr(gnn_model, "_orig_mod") else gnn_model
    raw_gnn.load_state_dict(best_state)

    # OOF predictions on the val slice
    _, gnn_val_proba = gnn_predict_df(gnn_val_df, gnn_model, gnn_tr_graph, team2idx)
    oof_gnn[-len(gnn_val_proba):] = gnn_val_proba
    valid_gnn = ~np.isnan(oof_gnn)
    _t(f"  GNN OOF AUC (last 15%): {roc_auc_score(y_full[valid_gnn], oof_gnn[valid_gnn]):.4f}")

    # ---------------------------------------------------------------------------
    _t("=" * 60)
    _t("Step 5/7 — TabPFN  (last 20 % OOF, then full refit)")
    _t("=" * 60)
    TABPFN_AVAILABLE = False
    try:
        from tabpfn import TabPFNClassifier
        n_tabpfn_val  = int(n * 0.20)
        X_tabpfn_tr   = X_full[:-n_tabpfn_val]
        y_tabpfn_tr   = y_full[:-n_tabpfn_val]
        X_tabpfn_val  = X_full[-n_tabpfn_val:]
        y_tabpfn_val  = y_full[-n_tabpfn_val:]

        tabpfn_fold = TabPFNClassifier(device=str(device))
        tabpfn_fold.fit(X_tabpfn_tr, y_tabpfn_tr)
        oof_tabpfn[-n_tabpfn_val:] = tabpfn_fold.predict_proba(X_tabpfn_val)[:, 1]
        valid_tp = ~np.isnan(oof_tabpfn)
        _t(f"  TabPFN OOF AUC (last 20%): {roc_auc_score(y_full[valid_tp], oof_tabpfn[valid_tp]):.4f}")
        TABPFN_AVAILABLE = True
    except Exception as e:
        _t(f"  TabPFN unavailable ({e}) — skipping.")

    # ---------------------------------------------------------------------------
    _t("=" * 60)
    _t("Step 6/7 — Fit meta-learner on OOF stack, then refit all base models on 100 %")
    _t("=" * 60)

    # Build meta OOF stack — use valid rows (GNN + TabPFN only have partial OOF)
    valid_meta = valid_gnn.copy()
    meta_cols  = [oof_elo, oof_xgb, oof_lgbm, oof_gnn]
    if TABPFN_AVAILABLE:
        valid_meta &= valid_tp
        meta_cols.append(oof_tabpfn)

    meta_X = np.column_stack([c[valid_meta] for c in meta_cols])
    meta_y = y_full[valid_meta]
    meta   = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs")
    meta.fit(meta_X, meta_y)
    model_names = ["elo", "xgb", "lgbm", "gnn"] + (["tabpfn"] if TABPFN_AVAILABLE else [])
    _t(f"  Meta weights: { {n: round(float(c), 3) for n, c in zip(model_names, meta.coef_[0])} }")

    # ── Refit XGBoost on full data ────────────────────────────────────────────
    _t("  Refitting XGBoost on 100 % …")
    X_aug, y_aug = flip_augment(X_full, y_full)
    xgb = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                         subsample=0.8, eval_metric="logloss", verbosity=0)
    xgb.fit(X_aug, y_aug, verbose=False)

    # ── Refit LightGBM on full data ───────────────────────────────────────────
    _t("  Refitting LightGBM on 100 % …")
    lgbm_full = pd.DataFrame(X_aug, columns=FEATURE_COLS)
    lgbm = LGBMClassifier(n_estimators=500, max_depth=5, learning_rate=0.03,
                           num_leaves=31, subsample=0.8, colsample_bytree=0.8,
                           min_child_samples=20, verbose=-1)
    lgbm.fit(lgbm_full, y_aug)

    # ── Refit GNN on full data ────────────────────────────────────────────────
    _t("  Refitting GNN on 100 % …")
    full_graph   = build_match_graph(game_df, team2idx, halflife_days=180, max_repeats=5).to(device)
    gnn_full     = PreDraftPredictor(num_teams=len(team2idx), emb_dim=128,
                                      out_dim=64, hidden=128, dropout=0.2).to(device)
    gnn_full     = torch.compile(gnn_full)
    opt_full     = torch.optim.AdamW(gnn_full.parameters(), lr=LR, weight_decay=1e-3)
    sch_full     = torch.optim.lr_scheduler.CosineAnnealingLR(opt_full, T_max=EPOCHS, eta_min=1e-5)
    # Use last 10 % as val for early stopping only (no OOF needed here)
    n_val_full   = int(n * 0.10)
    full_gnn_val = full_eval.iloc[-n_val_full:].reset_index(drop=True)
    full_gnn_tr  = full_eval.reset_index(drop=True)   # train on ALL, early stop on last 10 %
    best_full    = train_gnn(full_gnn_tr, full_gnn_val, team2idx, full_graph,
                              gnn_full, opt_full, sch_full)
    raw_gnn_full = gnn_full._orig_mod if hasattr(gnn_full, "_orig_mod") else gnn_full
    raw_gnn_full.load_state_dict(best_full)

    # ── Refit TabPFN on full data ─────────────────────────────────────────────
    tabpfn = None
    if TABPFN_AVAILABLE:
        _t("  Refitting TabPFN on 100 % …")
        try:
            from tabpfn import TabPFNClassifier
            tabpfn = TabPFNClassifier(device=str(device))
            tabpfn.fit(X_full, y_full)
        except Exception as e:
            _t(f"  TabPFN refit failed: {e}")
            tabpfn = None

    # ---------------------------------------------------------------------------
    _t("=" * 60)
    _t("Step 7/7 — Save all models")
    _t("=" * 60)
    torch.save(best_full, GNN_PATH)
    with open(VOC_PATH, "wb") as f:
        pickle.dump(team2idx, f)
    with open(ELO_PATH, "wb") as f:
        pickle.dump({
            "ratings":        elo._ratings,
            "games":          elo._games,
            "blue_wins":      dict(elo._blue_wins),
            "blue_games":     dict(elo._blue_games),
            "red_wins":       dict(elo._red_wins),
            "red_games":      dict(elo._red_games),
            "h2h":            dict(elo._h2h),
            "elo_hist":       {k: list(v) for k, v in elo._elo_hist.items()},
            "player_ratings": dict(player_elo._ratings),
            "league_elo_updates": {
                lg: [(str(dt), delta) for dt, delta in entries]
                for lg, entries in league_elo._updates.items()
            },
        }, f)
    joblib.dump(xgb,  XGB_PATH)
    joblib.dump(lgbm, LGBM_PATH)
    joblib.dump(meta, META_PATH)
    if tabpfn is not None:
        joblib.dump(tabpfn, TABPFN_PATH)
    CKPT_PATH.unlink(missing_ok=True)

    _t(f"  GNN        → {GNN_PATH}")
    _t(f"  team2idx   → {VOC_PATH}")
    _t(f"  Elo state  → {ELO_PATH}")
    _t(f"  XGBoost    → {XGB_PATH}")
    _t(f"  LightGBM   → {LGBM_PATH}")
    _t(f"  Meta       → {META_PATH}")
    if tabpfn is not None:
        _t(f"  TabPFN     → {TABPFN_PATH}")
    _t("Done. Restart the Streamlit app to load the new models.")


if __name__ == "__main__":
    main()
