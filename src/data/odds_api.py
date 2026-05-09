"""
Fetch bookmaker implied win probabilities from The Odds API.
https://the-odds-api.com — free tier: 500 requests/month.

Set ODDS_API_KEY in .env or enter it in the Dashboard → Data Status tab.
"""

import os
import requests
from difflib import SequenceMatcher
from typing import Optional

ODDS_API_BASE = "https://api.the-odds-api.com/v4"

# The Odds API organises LoL by tournament — Worlds/MSI get their own keys;
# regional leagues fall under the generic esports key.
LOL_SPORT_KEYS = [
    "esports_lol",
    "esports_lol_worlds",
    "esports_lol_msi",
]



def _similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower().strip(), b.lower().strip()).ratio()


def _remove_vig(p1: float, p2: float) -> tuple[float, float]:
    """Normalise two raw implied probabilities to sum to 1.0."""
    total = p1 + p2
    if total <= 0:
        return 0.5, 0.5
    return p1 / total, p2 / total


def _decimal_to_prob(odds: float) -> float:
    return 1.0 / odds if odds and odds > 1 else 0.5


def fetch_match_odds(
    team1: str,
    team2: str,
    api_key: Optional[str] = None,
    sport_keys: Optional[list[str]] = None,
) -> Optional[float]:
    """
    Return P(team1 wins) as bookmaker consensus implied probability (vig removed).
    Returns None if no odds found or the API key is missing/invalid.

    Matches team names by fuzzy string similarity, so minor spelling differences
    between OraclesElixir names and The Odds API names are handled automatically.
    """
    key = api_key or os.environ.get("ODDS_API_KEY", "").strip()
    if not key:
        return None

    keys_to_try = sport_keys or LOL_SPORT_KEYS

    for sport_key in keys_to_try:
        try:
            resp = requests.get(
                f"{ODDS_API_BASE}/sports/{sport_key}/odds/",
                params={
                    "apiKey":     key,
                    "regions":    "eu,uk,us",
                    "markets":    "h2h",
                    "oddsFormat": "decimal",
                },
                timeout=10,
            )
        except requests.RequestException:
            continue

        if resp.status_code == 401:
            return None           # bad key — stop trying
        if resp.status_code == 404:
            continue              # sport not active today
        if not resp.ok:
            continue

        events = resp.json()
        if not isinstance(events, list):
            continue

        for event in events:
            home_raw = event.get("home_team", "")
            away_raw = event.get("away_team", "")

            sim_t1_home = _similarity(team1, home_raw)
            sim_t1_away = _similarity(team1, away_raw)
            sim_t2_home = _similarity(team2, home_raw)
            sim_t2_away = _similarity(team2, away_raw)

            # Both teams must match above threshold (in either order)
            THRESH = 0.55
            t1_home = sim_t1_home >= THRESH and sim_t2_away >= THRESH
            t1_away = sim_t1_away >= THRESH and sim_t2_home >= THRESH

            if not (t1_home or t1_away):
                continue

            # Collect h2h prices across bookmakers
            probs_t1: list[float] = []
            for bk in event.get("bookmakers", []):
                for market in bk.get("markets", []):
                    if market.get("key") != "h2h":
                        continue
                    outcomes = market.get("outcomes", [])
                    if len(outcomes) < 2:
                        continue
                    # outcome order matches home/away
                    p_home = _decimal_to_prob(outcomes[0]["price"])
                    p_away = _decimal_to_prob(outcomes[1]["price"])
                    p_home, p_away = _remove_vig(p_home, p_away)
                    probs_t1.append(p_home if t1_home else p_away)

            if probs_t1:
                return round(sum(probs_t1) / len(probs_t1), 4)

    return None


def list_lol_sport_keys(api_key: Optional[str] = None) -> list[dict]:
    """Return all active sport keys from The Odds API that mention lol/esport."""
    key = api_key or os.environ.get("ODDS_API_KEY", "").strip()
    if not key:
        return []
    try:
        resp = requests.get(
            f"{ODDS_API_BASE}/sports/",
            params={"apiKey": key, "all": "false"},
            timeout=10,
        )
        resp.raise_for_status()
        return [
            s for s in resp.json()
            if "lol" in s.get("key", "").lower()
            or "esport" in s.get("key", "").lower()
        ]
    except Exception:
        return []
