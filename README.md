# LoL Pro Match Predictor

Pre-draft win probability for professional League of Legends matches — based on team identity and historical performance, no in-game data needed.

## Models

| Model | Description |
|---|---|
| **Elo** | Custom rating system with tiered starts (LCK 1620, LPL 1600, LEC 1530…), variable K-factors, season decay, and a rolling league Elo calibrated from international results |
| **XGBoost / LightGBM** | Gradient-boosted trees on Elo-derived features |
| **GNN** | GATv2 graph network over the team match-history graph with recency-weighted edges |
| **Ensemble** | Stacked meta-learner (LogisticRegression) over OOF predictions from all base models |

## Setup

```bash
uv sync
cp .env.example .env          # add your ODDS_API_KEY if you have one
uv run python refresh_data.py # download latest Oracle's Elixir CSVs
streamlit run app.py
```

## Training

```bash
uv run python retrain_full.py
```

Trains all models on the full dataset using OOF stacking to avoid leakage. Saves artifacts to `models/predraft/`.

## Dashboard

| Tab | Description |
|---|---|
| Tomorrow's Games | Fetch upcoming matches from Leaguepedia and run all predictions |
| Custom Match | Predict any matchup with team dropdowns |
| Team Explorer | Elo rankings, recent form, head-to-head stats |
| History | Saved predictions with accuracy, Brier score, and calibration vs. bookmaker |
| Data Status | Refresh CSVs, retrain models, configure API keys |

## Data

Match data from [Oracle's Elixir](https://oracleselixir.com/tools/downloads) (2020–present), ~62k games. Update via `refresh_data.py` — configure the Google Drive file ID at the top of the script when OE publishes a new version.

## Notebooks

`Notebooks/` contains the original training and analysis notebooks:
- `PreDraft_Training.ipynb` — model development with train/test split
- `LivePrediction.ipynb` — quick manual predictions
- `Betting_Analysis.ipynb` — calibration and edge analysis
