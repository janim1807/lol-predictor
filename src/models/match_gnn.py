"""
GNN over the match history graph for pre-draft win prediction.

Graph: one node per team, bidirectional edges per match.
Recency weighting via edge repetition — recent matches get up to max_repeats
copies so they carry more weight during neighbourhood aggregation.

Context features (per match): elo_diff, elo_win_prob, blue/red_recent_wr,
is_playoffs, blue/red_elo_momentum.
"""

import math
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATv2Conv


def build_match_graph(
    game_df,
    team2idx: dict,
    halflife_days: float = 180.0,
    max_repeats: int = 5,
) -> Data:
    """Build a recency-weighted team interaction graph. Returns PyG Data."""
    n_nodes  = len(team2idx) + 1
    src, dst = [], []

    df = game_df.sort_values("date").reset_index(drop=True)
    latest = df["date"].max()
    decay_lambda = math.log(2) / halflife_days  # λ for exp decay

    for _, row in df.iterrows():
        blue_idx = team2idx.get(row["blue_team"], 0)
        red_idx  = team2idx.get(row["red_team"],  0)
        if blue_idx == 0 or red_idx == 0:
            continue

        days_ago = (latest - row["date"]).days if pd.notna(row["date"]) else 0
        weight   = math.exp(-decay_lambda * max(days_ago, 0))
        repeats  = max(1, round(weight * max_repeats))

        for _ in range(repeats):
            src += [blue_idx, red_idx]
            dst += [red_idx,  blue_idx]

    edge_index = torch.tensor([src, dst], dtype=torch.long)
    x_indices  = torch.arange(n_nodes, dtype=torch.long)

    return Data(x_indices=x_indices, edge_index=edge_index)


class MatchGraphEncoder(nn.Module):
    """GATv2 encoder: embedding → N × (GATv2Conv + residual + LayerNorm) → projection."""

    def __init__(
        self,
        num_teams: int,
        emb_dim:   int = 128,
        out_dim:   int = 64,
        heads:     int = 8,
        layers:    int = 3,
        dropout:   float = 0.1,
    ):
        super().__init__()
        self.team_emb = nn.Embedding(num_teams + 1, emb_dim, padding_idx=0)

        self.convs  = nn.ModuleList()
        self.norms  = nn.ModuleList()
        self.res_projs = nn.ModuleList()

        for i in range(layers):
            in_dim  = emb_dim if i == 0 else emb_dim
            # GATv2 with concat=False averages heads → output is emb_dim
            self.convs.append(GATv2Conv(in_dim, emb_dim // heads, heads=heads,
                                        concat=True, dropout=dropout))
            self.norms.append(nn.LayerNorm(emb_dim))
            # residual projection only needed if dims change (they don't here)
            self.res_projs.append(nn.Identity())

        self.proj    = nn.Linear(emb_dim, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_indices: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.team_emb(x_indices)

        for conv, norm, res_proj in zip(self.convs, self.norms, self.res_projs):
            residual = res_proj(x)
            x = conv(x, edge_index)         # [N, emb_dim]
            x = self.dropout(F.gelu(x))
            x = norm(x + residual)          # residual connection + LayerNorm

        return self.proj(x)                 # [num_teams, out_dim]


class PreDraftPredictor(nn.Module):
    """Predicts match outcome before champion select.

    Pipeline:
      1. Run MatchGraphEncoder once per forward pass → team embedding table
      2. Look up blue_team and red_team embeddings
      3. Concatenate with context features (Elo, recent form, playoffs flag)
      4. MLP → sigmoid win probability for blue side

    Context feature order (context_dim must match):
      [elo_diff, blue_ewma_wr, red_ewma_wr, is_playoffs,
       blue_elo_momentum, red_elo_momentum,
       blue_player_elo_avg, red_player_elo_avg]
    """

    CONTEXT_DIM = 19

    def __init__(
        self,
        num_teams: int,
        emb_dim: int = 64,
        out_dim: int = 32,
        hidden: int = 64,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.encoder = MatchGraphEncoder(num_teams, emb_dim=emb_dim, out_dim=out_dim)
        in_dim = out_dim * 2 + self.CONTEXT_DIM
        self.classifier = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def encode_all_teams(self, graph: Data) -> torch.Tensor:
        """Run GNN on the full match graph → [num_teams, out_dim]."""
        return self.encoder(graph.x_indices, graph.edge_index)

    def forward(
        self,
        team_embeddings: torch.Tensor,  # precomputed from encode_all_teams()
        blue_idx: torch.Tensor,         # [batch]
        red_idx:  torch.Tensor,         # [batch]
        context:  torch.Tensor,         # [batch, CONTEXT_DIM]
    ) -> torch.Tensor:
        blue_emb = team_embeddings[blue_idx]
        red_emb  = team_embeddings[red_idx]
        x = torch.cat([blue_emb, red_emb, context], dim=-1)
        return torch.sigmoid(self.classifier(x)).view(-1)
