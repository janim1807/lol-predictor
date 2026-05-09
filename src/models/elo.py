"""
Elo rating system for professional LoL teams.

Matches are processed chronologically; each row is annotated with
pre-match Elo so there is no data leakage.

League structure note: major regions (LCK/LPL/LEC/LTA/LCP) only meet at
MSI and Worlds — cross-region comparisons are only meaningful there.
Regional leagues (PRM, LCKC, LDL, …) are self-contained. Tiered starting
ratings + season decay anchored to each league's baseline keep the gaps
meaningful across seasons.

League Elo: a rolling 1-year window of international results blends 20%
into each team's effective rating, mirroring Riot's Global Power Rankings.

K-factors: K=12 regular season, K=20 regional playoffs / international
groups, K=32 international knockouts. Provisional teams (< 50 games) get
K=40/25 to converge faster.
"""

import math
from collections import defaultdict, deque

import pandas as pd


DEFAULT_RATING    = 1500.0
BLUE_ADVANTAGE    = 15.0   # blue side wins ~52-53% in pro play ≈ +15 Elo
RECENT_WINDOW     = 10     # games used for rolling win rate (display only)
MOMENTUM_WINDOW   = 5      # games used for Elo momentum (trend)
EWMA_ALPHA        = 0.25   # EWMA smoothing for win rate (α=0.25 → ~7 game half-life)
SEASON_DECAY      = 0.4    # fraction pulled back to league anchor each year

K_REGIONAL_REG   = 12
K_REGIONAL_PO    = 20
K_INTL_REG       = 20
K_INTL_PO        = 32

K_PROVISIONAL_1  = 40   # games 1–20
K_PROVISIONAL_2  = 25   # games 21–50
PROVISIONAL_1_THRESHOLD = 20
PROVISIONAL_2_THRESHOLD = 50

LEAGUE_ELO_WEIGHT  = 0.20   # 80% team + 20% league (mirrors official GPR)
LEAGUE_ELO_WINDOW  = 365    # rolling 1-year window; rosters turn over yearly
INTERNATIONAL_LEAGUES = {"WLDs", "MSI"}

# Starting Elo by league. Major regions are seeded by historical international
# performance; regional leagues hover 1370–1400 (internal differences matter,
# not absolute values vs. major regions).
LEAGUE_TIERS: dict[str, float] = {
    "LCK":    1620,
    "LPL":    1600,
    "LEC":    1530,
    "LTA":    1490,   # Americas (merged 2025: LCS + LLA + CBLOL + LAS)
    "LCP":    1480,   # Pacific (merged 2025: PCS + VCS + LJL + LCO)
    # Pre-2025 names
    "LCS":    1510,
    "EULCS":  1530,
    "NALCS":  1510,
    "LLA":    1460,
    "LAS":    1450,
    "CBLOL":  1460,
    "PCS":    1470,
    "VCS":    1460,
    "LJL":    1450,
    "LCO":    1440,
    "GPL":    1440,
    # International events
    "WLDs":   1500,
    "MSI":    1500,
    "EWC":    1500,
    "MSC":    1500,
    "IEM":    1500,
    "KeSPA":  1500,
    "First Stand": 1500,
    # Academy / second-division
    "LCKC":   1400,
    "LDL":    1400,
    "NACL":   1400,
    "LCSA":   1390,
    "CBLOLA": 1390,
    # EMEA regional leagues
    "EUM":    1420,
    "EM":     1420,
    "LFL":    1390,
    "PRM":    1385,
    "NLC":    1385,
    "LVP SL": 1385,
    "EBL":    1380,
    "LPLOL":  1380,
    "TCL":    1400,
    "HM":     1375,
    "UL":     1375,
    "GLL":    1375,
    "LFL2":   1360,
    "CDF":    1365,
    "LRS":    1365,
    "NEXO":   1370,
    "LMF":    1370,   # Liga Máster Flow (LatAm regional)
    "UPL":    1370,
    "LJLC":   1375,   # LJL Challengers
    "LVP CL": 1370,
}

DEFAULT_LEAGUE_RATING = 1370   # anything not listed → treat as minor regional


def league_start_rating(league: str) -> float:
    return LEAGUE_TIERS.get(league, DEFAULT_LEAGUE_RATING)


def _k_factor(league: str, is_playoffs: bool,
              blue_games: int = 999, red_games: int = 999) -> float:
    """Return the K-factor for a match.

    Provisional boost: new teams get higher K so their ratings converge faster.
    The boost is applied if *either* team is still in their provisional window —
    a new team playing an established team should still move quickly.
    """
    min_games = min(blue_games, red_games)
    if min_games <= PROVISIONAL_1_THRESHOLD:
        provisional_k = K_PROVISIONAL_1
    elif min_games <= PROVISIONAL_2_THRESHOLD:
        provisional_k = K_PROVISIONAL_2
    else:
        provisional_k = None

    if league in INTERNATIONAL_LEAGUES:
        base_k = K_INTL_PO if is_playoffs else K_INTL_REG
    else:
        base_k = K_REGIONAL_PO if is_playoffs else K_REGIONAL_REG

    return max(base_k, provisional_k) if provisional_k is not None else base_k


class LeagueEloSystem:
    """Tracks per-region Elo from international results only (Worlds/MSI).

    Rolling window of window_days (default 365); older results expire so
    recent form dominates. Baseline = 1500; positive delta = above average.
    """

    def __init__(
        self,
        window_days: int   = LEAGUE_ELO_WINDOW,
        k_regular:   float = K_INTL_REG,
        k_playoffs:  float = K_INTL_PO,
        baseline:    float = DEFAULT_RATING,
    ):
        self.window_days = window_days
        self.k_regular   = k_regular
        self.k_playoffs  = k_playoffs
        self.baseline    = baseline
        # league → deque of (date: pd.Timestamp, delta: float)
        self._updates: dict[str, deque] = defaultdict(deque)

    def _expire(self, league: str, current_date: pd.Timestamp):
        """Drop updates older than window_days from the front of the deque."""
        cutoff = current_date - pd.Timedelta(days=self.window_days)
        dq = self._updates[league]
        while dq and dq[0][0] < cutoff:
            dq.popleft()

    def elo(self, league: str, current_date: pd.Timestamp) -> float:
        """Current rolling league Elo: baseline + sum of unexpired deltas."""
        self._expire(league, current_date)
        return self.baseline + sum(d for _, d in self._updates[league])

    def update(
        self,
        blue_league: str,
        red_league:  str,
        blue_won:    int,
        date:        pd.Timestamp,
        is_playoffs: bool = False,
    ):
        """Update after an international game result."""
        k = self.k_playoffs if is_playoffs else self.k_regular
        blue_elo = self.elo(blue_league, date)
        red_elo  = self.elo(red_league,  date)
        exp      = 1.0 / (1.0 + math.pow(10, -(blue_elo - red_elo) / 400.0))
        delta    = k * (blue_won - exp)
        self._updates[blue_league].append((date,  delta))
        self._updates[red_league].append( (date, -delta))

    def snapshot(self, current_date: pd.Timestamp) -> dict[str, float]:
        """Return {league: elo} for all leagues that have received updates."""
        return {lg: self.elo(lg, current_date) for lg in self._updates}


class EloSystem:
    def __init__(
        self,
        default:       float = DEFAULT_RATING,
        blue_bonus:    float = BLUE_ADVANTAGE,
        recent_window: int   = RECENT_WINDOW,
        season_decay:  float = SEASON_DECAY,
    ):
        self.default       = default
        self.blue_bonus    = blue_bonus
        self.recent_window = recent_window
        self.season_decay  = season_decay

        self._ratings: dict[str, float] = {}
        self._games:   dict[str, int]   = defaultdict(int)
        self._recent:  dict[str, deque] = defaultdict(lambda: deque(maxlen=recent_window))
        self._team_league: dict[str, str] = {}   # first league seen per team
        self._elo_hist: dict[str, deque] = defaultdict(lambda: deque(maxlen=MOMENTUM_WINDOW))

        self._ewma_wr: dict[str, float] = {}

        self._blue_wins:  dict[str, int] = defaultdict(int)
        self._blue_games: dict[str, int] = defaultdict(int)
        self._red_wins:   dict[str, int] = defaultdict(int)
        self._red_games:  dict[str, int] = defaultdict(int)
        self._h2h: dict[tuple, list] = defaultdict(lambda: [0, 0])

        # Attached after process_and_annotate — used by win_probability()
        self.league_elo: "LeagueEloSystem | None" = None
        self._latest_date: "pd.Timestamp | None"  = None

    def _init_team(self, team: str, league: str):
        """Seed a new team's rating from their league tier (once only)."""
        if team not in self._ratings:
            self._ratings[team] = league_start_rating(league)
            self._team_league[team] = league

    def rating(self, team: str) -> float:
        return self._ratings.get(team, self.default)

    def effective_rating(
        self,
        team:        str,
        league_elo:  "LeagueEloSystem | None" = None,
        date:        "pd.Timestamp | None"    = None,
    ) -> float:
        """Blended rating: (1-w)*team_elo + w*league_elo.

        Falls back to pure team Elo if league_elo or date is not provided,
        or if the team's league has no international history yet.
        """
        r = self._ratings.get(team, self.default)
        le = league_elo or self.league_elo
        dt = date or self._latest_date
        if le is None or dt is None:
            return r
        league = self._team_league.get(team, "")
        if not league:
            return r
        lr = le.elo(league, dt)
        return (1.0 - LEAGUE_ELO_WEIGHT) * r + LEAGUE_ELO_WEIGHT * lr

    def games_played(self, team: str) -> int:
        return self._games[team]

    def recent_win_rate(self, team: str) -> float:
        hist = self._recent[team]
        if not hist:
            return 0.5
        return sum(hist) / len(hist)

    def ewma_win_rate(self, team: str) -> float:
        """Exponentially weighted moving average win rate.

        More recent games contribute more than older ones.
        α=0.25 gives a half-life of ~7 games (2-3 weeks of play).
        Returns 0.5 until the team has played at least one game.
        """
        return self._ewma_wr.get(team, 0.5)

    def elo_momentum(self, team: str) -> float:
        """Elo change over the last MOMENTUM_WINDOW games (pre-match values).

        Positive = trending up, negative = trending down.
        Returns 0.0 if the team hasn't played MOMENTUM_WINDOW games yet.
        """
        hist = self._elo_hist[team]
        if len(hist) < MOMENTUM_WINDOW:
            return 0.0
        return self._ratings.get(team, self.default) - hist[0]

    def blue_side_win_rate(self, team: str) -> float:
        g = self._blue_games[team]
        return self._blue_wins[team] / g if g else 0.5

    def red_side_win_rate(self, team: str) -> float:
        g = self._red_games[team]
        return self._red_wins[team] / g if g else 0.5

    def h2h_win_rate(self, blue: str, red: str) -> float:
        """Historical win rate for blue team when facing red team on blue side."""
        wins, games = self._h2h[(blue, red)]
        return wins / games if games else 0.5

    def win_probability(
        self,
        blue_team:  str,
        red_team:   str,
        league_elo: "LeagueEloSystem | None" = None,
        date:       "pd.Timestamp | None"    = None,
    ) -> float:
        """Expected win probability for blue side (includes blue-side bonus).

        Uses blended effective Elo if league_elo is provided (or attached via
        self.league_elo). Falls back to pure team Elo otherwise.
        """
        blue_r = self.effective_rating(blue_team,  league_elo, date)
        red_r  = self.effective_rating(red_team,   league_elo, date)
        diff   = blue_r + self.blue_bonus - red_r
        return 1.0 / (1.0 + math.pow(10, -diff / 400.0))

    def update(
        self,
        blue_team:   str,
        red_team:    str,
        blue_won:    int,
        k:           float,
        league_elo:  "LeagueEloSystem | None" = None,
        date:        "pd.Timestamp | None"    = None,
    ):
        # Record pre-game Elo for momentum tracking (before the update)
        self._elo_hist[blue_team].append(self._ratings.get(blue_team, self.default))
        self._elo_hist[red_team].append( self._ratings.get(red_team,  self.default))

        exp   = self.win_probability(blue_team, red_team, league_elo, date)
        delta = k * (blue_won - exp)
        self._ratings[blue_team] = self._ratings.get(blue_team, self.default) + delta
        self._ratings[red_team]  = self._ratings.get(red_team,  self.default) - delta
        self._games[blue_team] += 1
        self._games[red_team]  += 1
        self._recent[blue_team].append(blue_won)
        self._recent[red_team].append(1 - blue_won)

        alpha = EWMA_ALPHA
        self._ewma_wr[blue_team] = alpha * blue_won       + (1 - alpha) * self._ewma_wr.get(blue_team, 0.5)
        self._ewma_wr[red_team]  = alpha * (1 - blue_won) + (1 - alpha) * self._ewma_wr.get(red_team,  0.5)

        self._blue_games[blue_team] += 1
        self._blue_wins[blue_team]  += blue_won
        self._red_games[red_team]   += 1
        self._red_wins[red_team]    += (1 - blue_won)
        self._h2h[(blue_team, red_team)][0] += blue_won
        self._h2h[(blue_team, red_team)][1] += 1

    def _apply_season_decay(self):
        """Pull ratings toward their league anchor at year boundaries.

        Uses each team's league-tier baseline as the anchor (not a flat 1500)
        so LCK teams decay toward 1620 and PRM teams toward 1385 — the
        cross-region gap is preserved across seasons even without direct games.
        """
        for team in list(self._ratings):
            anchor = league_start_rating(self._team_league.get(team, ""))
            self._ratings[team] = (
                anchor + (self._ratings[team] - anchor) * (1.0 - self.season_decay)
            )

    def process_and_annotate(
        self,
        game_df:    pd.DataFrame,
        min_games:  int = 5,
        player_elo: "PlayerEloSystem | None"   = None,
        league_elo: "LeagueEloSystem | None"   = None,
    ) -> pd.DataFrame:
        """Annotate game_df with pre-match Elo features (no data leakage).

        Adds: blue_elo, red_elo, elo_diff, elo_win_prob, blue/red_recent_wr,
        blue/red_ewma_wr, blue/red_elo_momentum, blue/red_side_wr, h2h_wr,
        blue/red_games, blue/red_games_conf, cold_start.
        Season decay is applied at year boundaries.
        """
        df = game_df.sort_values("date").reset_index(drop=True)
        records = []
        current_year = None

        for row in df.itertuples(index=False):
            date   = row.date
            year   = date.year if pd.notna(date) else None
            league = getattr(row, "league", "") or ""
            is_po  = bool(getattr(row, "is_playoffs", 0))

            if year and current_year and year != current_year:
                self._apply_season_decay()
                if player_elo is not None:
                    player_elo._apply_season_decay()
            current_year = year

            blue, red = row.blue_team, row.red_team
            self._init_team(blue, league)
            self._init_team(red,  league)

            rec = {
                "blue_elo":          self.effective_rating(blue, league_elo, date),
                "red_elo":           self.effective_rating(red,  league_elo, date),
                "elo_diff":          self.effective_rating(blue, league_elo, date)
                                   - self.effective_rating(red,  league_elo, date),
                "elo_win_prob":      self.win_probability(blue, red, league_elo, date),
                "blue_recent_wr":    self.recent_win_rate(blue),
                "red_recent_wr":     self.recent_win_rate(red),
                "blue_ewma_wr":      self.ewma_win_rate(blue),
                "red_ewma_wr":       self.ewma_win_rate(red),
                "blue_elo_momentum": self.elo_momentum(blue),
                "red_elo_momentum":  self.elo_momentum(red),
                "blue_side_wr":      self.blue_side_win_rate(blue),
                "red_side_wr":       self.red_side_win_rate(red),
                "h2h_wr":            self.h2h_win_rate(blue, red),
                "blue_games":        self.games_played(blue),
                "red_games":         self.games_played(red),
                # Confidence in Elo accuracy: saturates at 1.0 after ~200 games.
                # Low value flags new/provisional teams to the model.
                "blue_games_conf":   min(self.games_played(blue) / 200.0, 1.0),
                "red_games_conf":    min(self.games_played(red)  / 200.0, 1.0),
                "cold_start":        (
                    self.games_played(blue) < min_games or
                    self.games_played(red)  < min_games
                ),
            }

            if player_elo is not None:
                blue_players = [
                    getattr(row, f"blue_{r}_player", None) for r in ["top", "jng", "mid", "bot", "sup"]
                ]
                red_players = [
                    getattr(row, f"red_{r}_player",  None) for r in ["top", "jng", "mid", "bot", "sup"]
                ]
                rec["blue_player_elo_avg"] = player_elo.team_avg_elo(blue_players)
                rec["red_player_elo_avg"]  = player_elo.team_avg_elo(red_players)
                for role, bp, rp in zip(
                    ["top", "jng", "mid", "bot", "sup"], blue_players, red_players
                ):
                    rec[f"blue_{role}_elo"] = player_elo.rating(bp)
                    rec[f"red_{role}_elo"]  = player_elo.rating(rp)

            records.append(rec)
            blue_won = int(row.blue_win)

            k = _k_factor(league, is_po,
                          blue_games=self._games[blue],
                          red_games=self._games[red])
            self.update(blue, red, blue_won, k=k, league_elo=league_elo, date=date)

            if player_elo is not None:
                player_elo.update(blue_players, red_players, blue_won, k=k, date=date)

            # Update rolling league Elo from international matches
            if league_elo is not None and league in INTERNATIONAL_LEAGUES:
                blue_league = self._team_league.get(blue, "")
                red_league  = self._team_league.get(red,  "")
                if blue_league and red_league and blue_league != red_league:
                    league_elo.update(blue_league, red_league, blue_won,
                                      date=date, is_playoffs=is_po)

        # Attach for live prediction calls
        self.league_elo   = league_elo
        self._latest_date = df["date"].max() if len(df) else None

        features = pd.DataFrame(records)
        return pd.concat([df.reset_index(drop=True), features], axis=1)


class PlayerEloSystem:
    """Individual player Elo ratings that persist across team changes.

    All 5 players on the winning team get +delta; losers get -delta.
    A strong player on a weak team converges toward team outcomes — intentional,
    since the team wins or loses as a unit.
    """

    SEASON_DECAY = 0.18  # lighter than team decay (skill is more persistent than team form)

    def __init__(self, default: float = DEFAULT_RATING):
        self.default   = default
        self._ratings: dict[str, float] = {}
        self._games:   dict[str, int]   = defaultdict(int)
        self._elo_hist: dict[str, list] = defaultdict(list)  # player -> [(date, elo), ...]

    def rating(self, player: str | None) -> float:
        if not player or not isinstance(player, str):
            return self.default
        return self._ratings.get(player, self.default)

    def team_avg_elo(self, players: list) -> float:
        """Mean Elo of a list of player names (None/NaN treated as default)."""
        valid = [p for p in players if p and isinstance(p, str)]
        if not valid:
            return self.default
        return sum(self.rating(p) for p in valid) / len(valid)

    def _apply_season_decay(self):
        """Pull all player ratings 18% toward 1500 at year boundaries."""
        for p in list(self._ratings):
            self._ratings[p] = (
                self.default + (self._ratings[p] - self.default) * (1.0 - self.SEASON_DECAY)
            )

    def update(self, blue_players: list, red_players: list, blue_won: int,
               k: float = K_REGIONAL_REG, date=None):
        """Update all 10 players after a game result.

        Args:
            k: K-factor for this match (use _k_factor(league, is_playoffs) from caller).
               Defaults to K_REGIONAL_REG=12 for backward compatibility.
        """
        blue_avg = self.team_avg_elo(blue_players)
        red_avg  = self.team_avg_elo(red_players)
        exp   = 1.0 / (1.0 + math.pow(10, -(blue_avg - red_avg) / 400.0))
        delta = k * (blue_won - exp)

        for p in blue_players:
            if p and isinstance(p, str):
                if date is not None:
                    self._elo_hist[p].append((date, self.rating(p)))
                self._ratings[p] = self.rating(p) + delta
                self._games[p]  += 1
        for p in red_players:
            if p and isinstance(p, str):
                if date is not None:
                    self._elo_hist[p].append((date, self.rating(p)))
                self._ratings[p] = self.rating(p) - delta
                self._games[p]  += 1

    def top_players_by_elo(self, n: int = 20) -> list[tuple[str, float]]:
        """Return top N players sorted by current Elo (for display)."""
        return sorted(self._ratings.items(), key=lambda x: -x[1])[:n]

    def elo_history(self, player: str) -> list[tuple]:
        """Return [(date, pre_game_elo), ...] for a player, sorted by date."""
        return sorted(self._elo_hist.get(player, []), key=lambda x: x[0])

    def player_info(self, player: str, game_df) -> dict:
        """
        Return dict with current team, role, and last game date for a player.
        Scans all player columns in game_df for the most recent appearance.
        """
        roles = ["top", "jng", "mid", "bot", "sup"]
        sides = ["blue", "red"]
        result = {"team": None, "role": None, "last_date": None}
        best_date = None

        for side in sides:
            for role in roles:
                col = f"{side}_{role}_player"
                if col not in game_df.columns:
                    continue
                matches = game_df[game_df[col] == player]
                if matches.empty:
                    continue
                last = matches.sort_values("date").iloc[-1]
                if best_date is None or last["date"] > best_date:
                    best_date = last["date"]
                    result["role"]      = role
                    result["last_date"] = last["date"]
                    team_col = "blue_team" if side == "blue" else "red_team"
                    result["team"] = last.get(team_col)

        return result

    def current_roster(self, team: str, game_df) -> dict[str, str | None]:
        """
        Infer the most recent known roster for a team from game_df.

        Returns: {role: player_name} for top/jng/mid/bot/sup.
        Falls back to None if player columns aren't in game_df.
        """
        player_cols = [f"blue_{r}_player" for r in ["top", "jng", "mid", "bot", "sup"]]
        if not any(c in game_df.columns for c in player_cols):
            return {r: None for r in ["top", "jng", "mid", "bot", "sup"]}

        all_games = (
            game_df[game_df["blue_team"].eq(team) | game_df["red_team"].eq(team)]
            .sort_values("date")
        )
        if all_games.empty:
            return {r: None for r in ["top", "jng", "mid", "bot", "sup"]}

        last = all_games.iloc[-1]
        roster = {}
        prefix = "blue" if last["blue_team"] == team else "red"
        for role in ["top", "jng", "mid", "bot", "sup"]:
            col = f"{prefix}_{role}_player"
            roster[role] = last.get(col) if col in last.index else None
        return roster
