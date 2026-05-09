"""
Fetch upcoming pro LoL matches from the official LoL Esports API (primary)
with Leaguepedia Cargo API as fallback.

LoL Esports API
---------------
Endpoint: https://esports-api.lolesports.com/persisted/gw/getSchedule
API key:  publicly distributed with the lolesports.com frontend bundle.
No authentication workflow required — the key is embedded in the site JS.
No rate limiting observed in normal usage.

Leaguepedia fallback
--------------------
Uses the MediaWiki Cargo query endpoint at lol.fandom.com.
Rate-limited aggressively (~1 req/min per IP); only used if the
LoL Esports API fails or returns no results.

Usage:
    from data.leaguepedia import fetch_schedule
    games = fetch_schedule(date="2026-05-04", leagues=["LEC", "LCK"])
    # Returns list of dicts: {team1, team2, best_of, tournament, datetime_utc, league}
"""

import time
import requests
import pandas as pd
from datetime import datetime, timedelta, timezone
from typing import Optional


# ---------------------------------------------------------------------------
# LoL Esports API — league IDs (from getLeagues endpoint)
# ---------------------------------------------------------------------------
LOLESPORTS_API_KEY  = "0TvQnueqKa5mxJntVWt0w4LpLfEkrV1Ta8rQBb9Z"
LOLESPORTS_SCHEDULE = "https://esports-api.lolesports.com/persisted/gw/getSchedule"

LEAGUE_IDS: dict[str, str] = {
    "LCK":   "98767991310872058",
    "LPL":   "98767991314006698",
    "LEC":   "98767991302996019",
    "LCS":   "98767991299243165",   # old NA slot (pre-2025)
    "LTA":   "113470291645289904",  # LTA North (main NA, 2025+)
    "LTA N": "113470291645289904",
    "LTA S": "113475181634818701",
    "LCP":   "113476371197627891",
    "NACL":  "109511549831443335",
    "LCKC":  "98767991335774713",
    "WLDs":  "98767975604431411",
    "MSI":   "98767991325878492",
    "PCS":   "104366947889790212",
    "VCS":   "107213827295848783",
    "LJL":   "98767991349978712",
    "CBLOL": "98767991332355509",
    "LLA":   "101382741235120470",
}

# ---------------------------------------------------------------------------
# Team name mapping: LoL Esports API name → OraclesElixir name
# ---------------------------------------------------------------------------
LOLESPORTS_TO_OE: dict[str, str] = {
    # LCK
    "T1":                            "T1",
    "Gen.G Esports":                 "Gen.G",
    "Gen.G":                         "Gen.G",
    "Hanwha Life Esports":           "Hanwha Life Esports",
    "KT Rolster":                    "KT Rolster",
    "kt Rolster":                    "KT Rolster",     # API variant (lowercase kt)
    "Dplus KIA":                     "Dplus Kia",
    "DRX":                           "DRX",
    "Kiwoom DRX":                    "Kiwoom DRX",
    "KIWOOM DRX":                    "Kiwoom DRX",     # API variant (all-caps)
    "NONGSHIM RED FORCE":            "Nongshim RedForce",
    "Nongshim RedForce":             "Nongshim RedForce",
    "BNK FEARX":                     "BNK FEARX",
    "OKSavingsBank BRION":           "OKSavingsBank BRION",
    "OK BRION":                      "OKSavingsBank BRION",
    "DN SOOPers":                    "DN SOOPers",
    "Liiv SANDBOX":                  "Liiv SANDBOX",
    "HANJIN BRION":                  "HANJIN BRION",
    # LPL — OE canonical name is the team's base name (no city prefix, capital-D EDward)
    "Bilibili Gaming":               "Bilibili Gaming",
    "BILIBILI GAMING":               "Bilibili Gaming",  # API variant (all-caps)
    "JDG Intel Esports Club":        "JD Gaming",
    "JD Gaming":                     "JD Gaming",
    "Beijing JDG Esports":           "JD Gaming",        # city-branded API name (2025+)
    "Weibo Gaming":                  "Weibo Gaming",
    "WeiboGaming":                   "Weibo Gaming",     # API variant (no space)
    "Top Esports":                   "Top Esports",
    "TOP ESPORTS":                   "Top Esports",      # API variant (all-caps)
    "Edward Gaming":                 "EDward Gaming",
    "EDward Gaming":                 "EDward Gaming",
    "EDWARD GAMING":                 "EDward Gaming",    # API variant (all-caps)
    "LNG Esports":                   "LNG Esports",
    "Suzhou LNG Esports":            "LNG Esports",      # city-branded API name (2025+)
    "Royal Never Give Up":           "Royal Never Give Up",
    "FunPlus Phoenix":               "FunPlus Phoenix",
    "Invictus Gaming":               "Invictus Gaming",
    "NIP":                           "NIP",
    "Ninjas in Pyjamas":             "Ninjas in Pyjamas",
    "Shenzhen NINJAS IN PYJAMAS":    "Ninjas in Pyjamas", # city-branded API name (2025+)
    "Oh My God":                     "Oh My God",
    "ThunderTalk Gaming":            "ThunderTalk Gaming",
    "THUNDER TALK GAMING":           "ThunderTalk Gaming", # API variant (all-caps + space)
    "Ultra Prime":                   "Ultra Prime",
    "LGD Gaming":                    "LGD Gaming",
    "LGD GAMING":                    "LGD Gaming",       # API variant (all-caps)
    "Team WE":                       "Team WE",
    "Xi'an Team WE":                 "Team WE",          # city-branded API name (2025+)
    # LEC
    "G2 Esports":                    "G2 Esports",
    "Fnatic":                        "Fnatic",
    "Team Vitality":                 "Team Vitality",
    "Team BDS":                      "Team BDS",
    "Karmine Corp":                  "Karmine Corp",
    "SK Gaming":                     "SK Gaming",
    "Excel Esports":                 "Excel Esports",
    "MAD Lions KOI":                 "MAD Lions KOI",
    "Rogue":                         "Rogue",
    "GiantX":                        "GiantX",
    "GIANTX":                        "GiantX",          # API variant (all-caps)
    "Team Heretics":                 "Team Heretics",
    # LTA / NA
    "Cloud9":                        "Cloud9",
    "Team Liquid":                   "Team Liquid",
    "100 Thieves":                   "100 Thieves",
    "FlyQuest":                      "FlyQuest",
    "NRG":                           "NRG",
    "Dignitas":                      "Dignitas",
    "Evil Geniuses":                 "Evil Geniuses",
    "Golden Guardians":              "Golden Guardians",
    "Immortals":                     "Immortals",
    "Shopify Rebellion":             "Shopify Rebellion",
    # LCP / Pacific
    "PSG Talon":                     "PSG Talon",
    "CTBC Flying Oyster":            "CTBC Flying Oyster",
    "GAM Esports":                   "GAM Esports",
    "DetonatioN FocusMe":            "DetonatioN FocusMe",
}

# ---------------------------------------------------------------------------
# Leaguepedia fallback — team name mapping
# ---------------------------------------------------------------------------
LEAGUEPEDIA_TO_OE: dict[str, str] = {
    "T1": "T1", "Gen.G": "Gen.G",
    "Hanwha Life Esports": "Hanwha Life Esports",
    "KT Rolster": "KT Rolster", "DRX": "DRX",
    "Nongshim RedForce": "Nongshim RedForce",
    "BNK FearX": "BNK FEARX", "OK BRION": "OKSavingsBank BRION",
    "Bilibili Gaming": "Bilibili Gaming",
    "JDG Intel Esports Club": "JD Gaming", "JD Gaming": "JD Gaming",
    "Weibo Gaming": "Weibo Gaming",
    "Top Esports": "Top Esports", "Edward Gaming": "EDward Gaming",
    "EDward Gaming": "EDward Gaming",
    "LNG Esports": "LNG Esports", "NIP": "NIP",
    "Invictus Gaming": "Invictus Gaming",
    "Oh My God": "Oh My God", "ThunderTalk Gaming": "ThunderTalk Gaming",
    "Ultra Prime": "Ultra Prime", "LGD Gaming": "LGD Gaming",
    "Team WE": "Team WE",
    "G2 Esports": "G2 Esports", "Fnatic": "Fnatic",
    "Team Vitality": "Team Vitality", "Team BDS": "Team BDS",
    "Karmine Corp": "Karmine Corp", "SK Gaming": "SK Gaming",
    "Excel Esports": "Excel Esports", "MAD Lions KOI": "MAD Lions KOI",
    "Rogue": "Rogue",
    "Cloud9": "Cloud9", "Team Liquid": "Team Liquid",
    "100 Thieves": "100 Thieves", "FlyQuest": "FlyQuest",
    "NRG": "NRG", "Dignitas": "Dignitas",
    "Evil Geniuses": "Evil Geniuses",
    "Golden Guardians": "Golden Guardians",
    "Immortals": "Immortals", "Shopify Rebellion": "Shopify Rebellion",
    "PSG Talon": "PSG Talon",
    "CTBC Flying Oyster": "CTBC Flying Oyster",
    "GAM Esports": "GAM Esports",
    "DetonatioN FocusMe": "DetonatioN FocusMe",
}


def normalize_team(name: str) -> str:
    """Map a raw team name to OraclesElixir name (tries both mappings)."""
    return LOLESPORTS_TO_OE.get(name) or LEAGUEPEDIA_TO_OE.get(name, name)


# ---------------------------------------------------------------------------
# Primary: LoL Esports API
# ---------------------------------------------------------------------------

def _fetch_lolesports(
    date:    str,
    leagues: Optional[list[str]],
    days:    int,
) -> list[dict]:
    """Fetch schedule from the official LoL Esports API."""
    start_dt = datetime.strptime(date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end_dt   = start_dt + timedelta(days=days)
    headers  = {"x-api-key": LOLESPORTS_API_KEY}

    ids_to_query = (
        {LEAGUE_IDS[lg] for lg in leagues if lg in LEAGUE_IDS}
        if leagues else set(LEAGUE_IDS.values())
    )
    if not ids_to_query:
        return []

    results: list[dict] = []
    seen_ids: set[str]  = set()

    for league_id in ids_to_query:
        try:
            resp = requests.get(
                LOLESPORTS_SCHEDULE,
                params={"hl": "en-US", "leagueId": league_id},
                headers=headers,
                timeout=10,
            )
            resp.raise_for_status()
            events = resp.json().get("data", {}).get("schedule", {}).get("events", [])
        except Exception:
            continue

        for event in events:
            if event.get("type") != "match":
                continue

            start_str = event.get("startTime", "")
            try:
                dt_utc = datetime.fromisoformat(start_str.replace("Z", "+00:00"))
            except (ValueError, AttributeError):
                continue

            if not (start_dt <= dt_utc < end_dt):
                continue

            match    = event.get("match", {})
            match_id = match.get("id", "")
            if match_id in seen_ids:
                continue
            seen_ids.add(match_id)

            teams = match.get("teams", [])
            if len(teams) < 2:
                continue

            team1_raw = teams[0].get("name", "")
            team2_raw = teams[1].get("name", "")
            if not team1_raw or not team2_raw:
                continue

            strategy   = match.get("strategy", {})
            best_of    = int(strategy.get("count", 1))
            league     = event.get("league", {}).get("name", "Unknown")
            block      = event.get("blockName", "")
            tournament = f"{league}/{block}" if block else league

            results.append({
                "team1":        _normalize_lolesports(team1_raw),
                "team2":        _normalize_lolesports(team2_raw),
                "team1_raw":    team1_raw,
                "team2_raw":    team2_raw,
                "best_of":      best_of,
                "tournament":   tournament,
                "league":       league,
                "datetime_utc": dt_utc,
            })

    results.sort(key=lambda x: x["datetime_utc"])
    return results


def _normalize_lolesports(name: str) -> str:
    return LOLESPORTS_TO_OE.get(name, name)


# ---------------------------------------------------------------------------
# Fallback: Leaguepedia
# ---------------------------------------------------------------------------

def _infer_league(overview_page: str) -> str:
    if not overview_page:
        return "Unknown"
    top = overview_page.split("/")[0].strip()
    normalizations = {
        "LoL EMEA Championship": "LEC",
        "League of Legends EMEA Championship": "LEC",
        "LCK Champions Korea": "LCK",
        "League Champions Korea": "LCK",
    }
    return normalizations.get(top, top)


def _fetch_leaguepedia(
    date:    str,
    leagues: Optional[list[str]],
    days:    int,
) -> list[dict]:
    """Fallback: fetch schedule from Leaguepedia Cargo API."""
    start_dt  = datetime.strptime(date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end_dt    = start_dt + timedelta(days=days)
    start_str = start_dt.strftime("%Y-%m-%d %H:%M:%S")
    end_str   = end_dt.strftime("%Y-%m-%d %H:%M:%S")

    params = {
        "action":   "cargoquery",
        "tables":   "MatchSchedule",
        "fields":   "DateTime_UTC,Team1,Team2,BestOf,OverviewPage,Tab",
        "where":    f"DateTime_UTC >= '{start_str}' AND DateTime_UTC < '{end_str}'",
        "order_by": "DateTime_UTC ASC",
        "limit":    "100",
        "format":   "json",
    }

    last_exc = None
    for attempt in range(3):
        try:
            resp = requests.get(
                "https://lol.fandom.com/api.php",
                params=params,
                timeout=15,
                headers={"User-Agent": "LoLPredictorDashboard/1.0"},
            )
            resp.raise_for_status()
            data = resp.json()
            if "error" in data:
                err = data["error"]
                if err.get("code") == "ratelimited":
                    wait = 15 * (attempt + 1)
                    last_exc = RuntimeError(f"Leaguepedia rate limit: {err.get('info', '')}")
                    time.sleep(wait)
                    continue
                raise RuntimeError(f"Leaguepedia API error: {err.get('info', err)}")
            break
        except requests.RequestException as e:
            last_exc = RuntimeError(f"Leaguepedia request failed: {e}")
            time.sleep(5 * (attempt + 1))
    else:
        raise last_exc

    results = []
    for item in data.get("cargoquery", []):
        r     = item.get("title", {})
        team1 = r.get("Team1", "").strip()
        team2 = r.get("Team2", "").strip()
        if not team1 or not team2:
            continue

        overview = r.get("OverviewPage", "")
        league   = _infer_league(overview)
        if leagues and not any(league.startswith(lg) for lg in leagues):
            continue

        dt_str = r.get("DateTime UTC", r.get("DateTime_UTC", ""))
        try:
            dt_utc = datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
        except (ValueError, TypeError):
            dt_utc = None

        try:
            best_of = int(r.get("BestOf", "1") or "1")
        except ValueError:
            best_of = 1

        results.append({
            "team1":        LEAGUEPEDIA_TO_OE.get(team1, team1),
            "team2":        LEAGUEPEDIA_TO_OE.get(team2, team2),
            "team1_raw":    team1,
            "team2_raw":    team2,
            "best_of":      best_of,
            "tournament":   overview,
            "league":       league,
            "datetime_utc": dt_utc,
        })

    return results


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_schedule(
    date:    Optional[str] = None,
    leagues: Optional[list[str]] = None,
    days:    int = 1,
) -> list[dict]:
    """
    Fetch scheduled matches for the given date range.

    Tries the official LoL Esports API first (fast, no rate limits).
    Falls back to Leaguepedia if the primary source fails.

    Args:
        date:    "YYYY-MM-DD". Defaults to tomorrow UTC.
        leagues: Short league names to filter, e.g. ["LEC", "LCK"].
                 None = all leagues in LEAGUE_IDS.
        days:    Number of days to fetch starting from `date`.

    Returns:
        List of dicts: team1, team2, best_of, tournament, datetime_utc, league
        Sorted chronologically.
    """
    if date is None:
        date = (datetime.now(timezone.utc) + timedelta(days=1)).strftime("%Y-%m-%d")

    # Primary: LoL Esports API
    try:
        results = _fetch_lolesports(date, leagues, days)
        if results:
            return results
    except Exception:
        pass

    # Fallback: Leaguepedia
    return _fetch_leaguepedia(date, leagues, days)


def fetch_schedule_as_df(
    date:    Optional[str] = None,
    leagues: Optional[list[str]] = None,
    days:    int = 1,
) -> pd.DataFrame:
    """Same as fetch_schedule but returns a DataFrame."""
    rows = fetch_schedule(date=date, leagues=leagues, days=days)
    if not rows:
        return pd.DataFrame(columns=["team1", "team2", "best_of", "tournament",
                                      "league", "datetime_utc"])
    return pd.DataFrame(rows)
