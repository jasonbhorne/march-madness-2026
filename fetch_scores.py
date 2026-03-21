"""
Fetch NCAA tournament scores from ESPN API and update tournament_state.json.

Usage:
  python fetch_scores.py                    # Fetch all tournament days so far
  python fetch_scores.py --date 20260320    # Fetch a specific date
  python fetch_scores.py --dry-run          # Show what would be updated without saving

No API key needed. ESPN's public scoreboard endpoint.
"""

import json
import urllib.request
import argparse
from datetime import date, timedelta

STATE_FILE = "/Users/hornej/Documents/Research/march-madness-2026/tournament_state.json"
PICKS_FILE = "/Users/hornej/Downloads/bracket_picks.json"

ESPN_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/scoreboard?dates={date}&groups=100&limit=100"

# ESPN uses full names; our bracket uses short names. Map mismatches here.
ESPN_TO_BRACKET = {
    "Connecticut Huskies": "Connecticut",
    "UConn Huskies": "Connecticut",
    "Michigan State Spartans": "Michigan St.",
    "Ohio State Buckeyes": "Ohio St.",
    "Iowa State Cyclones": "Iowa St.",
    "North Dakota State Bison": "North Dakota St.",
    "Saint Mary's Gaels": "Saint Mary's",
    "St. John's Red Storm": "St. John's",
    "North Carolina Tar Heels": "North Carolina",
    "North Carolina State Wolfpack": "North Carolina St.",
    "Miami Hurricanes": "Miami FL",
    "Miami (OH) RedHawks": "Miami OH",
    "McNeese Cowboys": "McNeese St.",
    "Kennesaw State Owls": "Kennesaw St.",
    "Utah State Aggies": "Utah St.",
    "LIU Sharks": "LIU Brooklyn",
    "Prairie View A&M Panthers": "Prairie View A&M",
    "Texas A&M Aggies": "Texas A&M",
    "South Florida Bulls": "South Florida",
    "Tennessee State Tigers": "Tennessee St.",
    "Cal Baptist Lancers": "Cal Baptist",
    "Wright State Raiders": "Wright St.",
    "High Point Panthers": "High Point",
    "Northern Iowa Panthers": "Northern Iowa",
    "Hawai'i Rainbow Warriors": "Hawaii",
    "Hawaii Rainbow Warriors": "Hawaii",
    "NC State Wolfpack": "North Carolina St.",
    "SMU Mustangs": "SMU",
    "UMBC Retrievers": "UMBC",
    "Lehigh Mountain Hawks": "Lehigh",
    "LIU Brooklyn Sharks": "LIU Brooklyn",
    "Queens Royals": "Queens",
}

# Tournament dates: First Four through Championship
TOURNAMENT_DATES = [
    "20260317", "20260318",  # First Four
    "20260319", "20260320",  # Round of 64
    "20260321", "20260322",  # Round of 32
    "20260326", "20260327",  # Sweet 16
    "20260328", "20260329",  # Elite Eight
    "20260404", "20260405",  # Final Four
    "20260407",              # Championship
]

ROUND_EXPECTED_GAMES = {
    "Round of 64": 32,
    "Round of 32": 16,
    "Sweet 16": 8,
    "Elite Eight": 4,
    "Final Four": 2,
    "Championship": 1,
}

ROUND_ORDER = [
    "Play-In", "Round of 64", "Round of 32",
    "Sweet 16", "Elite Eight", "Final Four", "Championship"
]


def espn_name_to_bracket(display_name):
    """Convert ESPN displayName to our bracket_picks.json name."""
    if display_name in ESPN_TO_BRACKET:
        return ESPN_TO_BRACKET[display_name]
    # Strip mascot: "Duke Blue Devils" -> "Duke"
    # But handle multi-word cities: "Iowa State Cyclones" already mapped above
    # For most teams, the first word(s) before common mascot words work.
    # Safer: check against known bracket names.
    return None


def load_bracket_teams():
    """Load all team names from bracket_picks.json for fuzzy matching."""
    with open(PICKS_FILE) as f:
        picks = json.load(f)
    teams = set()
    for rnd in picks.values():
        for g in rnd:
            teams.add(g["winner"])
            teams.add(g["loser"])
    return teams


def match_espn_team(display_name, bracket_teams):
    """Match an ESPN team name to a bracket team name."""
    # Direct map first
    mapped = espn_name_to_bracket(display_name)
    if mapped and mapped in bracket_teams:
        return mapped

    # Try stripping the last word (mascot)
    parts = display_name.rsplit(" ", 1)
    if len(parts) == 2 and parts[0] in bracket_teams:
        return parts[0]

    # Try common abbreviation patterns
    for bt in bracket_teams:
        if display_name.startswith(bt + " "):
            return bt
        if bt.replace("St.", "State") in display_name:
            return bt

    return None


def fetch_date(date_str):
    """Fetch scores for a given date from ESPN API."""
    url = ESPN_URL.format(date=date_str)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        data = json.loads(resp.read().decode())
    return data


def parse_games(data, bracket_teams):
    """Parse ESPN response into game results."""
    results = []
    events = data.get("events", [])

    for event in events:
        status = event.get("status", {}).get("type", {}).get("name", "")
        if status != "STATUS_FINAL":
            continue

        # Check it's a tournament game
        notes = event.get("notes", [])
        is_tourney = any("NCAA" in n.get("headline", "") or "March Madness" in n.get("headline", "")
                         for n in notes)
        if not is_tourney and event.get("season", {}).get("type", 0) != 3:
            # season type 3 = postseason
            competitions = event.get("competitions", [{}])
            if competitions:
                groups = competitions[0].get("groups", {})
                if not groups:
                    continue

        competitions = event.get("competitions", [{}])
        if not competitions:
            continue

        comp = competitions[0]
        competitors = comp.get("competitors", [])
        if len(competitors) != 2:
            continue

        # Determine winner
        teams_data = []
        for c in competitors:
            team_info = c.get("team", {})
            display_name = team_info.get("displayName", "")
            score = int(c.get("score", "0"))
            seed = c.get("curatedRank", {}).get("current", 99)
            winner_flag = c.get("winner", False)
            bracket_name = match_espn_team(display_name, bracket_teams)

            teams_data.append({
                "espn_name": display_name,
                "bracket_name": bracket_name,
                "score": score,
                "seed": seed,
                "is_winner": winner_flag,
            })

        winner = next((t for t in teams_data if t["is_winner"]), None)
        loser = next((t for t in teams_data if not t["is_winner"]), None)

        if not winner or not loser:
            # Fallback: higher score wins
            teams_data.sort(key=lambda x: -x["score"])
            winner, loser = teams_data[0], teams_data[1]

        if not winner["bracket_name"] or not loser["bracket_name"]:
            print(f"  WARNING: Could not match team names:")
            print(f"    {winner['espn_name']} -> {winner['bracket_name']}")
            print(f"    {loser['espn_name']} -> {loser['bracket_name']}")
            continue

        # Determine round from notes (if available)
        round_name = None
        for n in notes:
            headline = n.get("headline", "")
            if "1st Round" in headline:
                round_name = "Round of 64"
            elif "2nd Round" in headline:
                round_name = "Round of 32"
            elif "Sweet 16" in headline or "Regional Semifinal" in headline:
                round_name = "Sweet 16"
            elif "Elite Eight" in headline or "Elite 8" in headline or "Regional Final" in headline:
                round_name = "Elite Eight"
            elif "Final Four" in headline or "National Semifinal" in headline:
                round_name = "Final Four"
            elif "Championship" in headline or "National Championship" in headline:
                round_name = "Championship"

        # If notes are empty, flag for date-based round assignment later
        results.append({
            "winner": winner["bracket_name"],
            "winner_seed": winner["seed"],
            "loser": loser["bracket_name"],
            "loser_seed": loser["seed"],
            "winner_score": winner["score"],
            "loser_score": loser["score"],
            "round": round_name,
        })

    return results


def main():
    parser = argparse.ArgumentParser(description="Fetch NCAA tournament scores from ESPN")
    parser.add_argument("--date", help="Specific date to fetch (YYYYMMDD)")
    parser.add_argument("--dry-run", action="store_true", help="Show results without saving")
    args = parser.parse_args()

    bracket_teams = load_bracket_teams()
    print(f"Loaded {len(bracket_teams)} bracket teams")

    # Determine which dates to fetch
    if args.date:
        dates = [args.date]
    else:
        today = date.today().strftime("%Y%m%d")
        dates = [d for d in TOURNAMENT_DATES if d <= today]

    # Map dates to rounds for fallback when ESPN notes are empty
    DATE_TO_ROUND = {
        "20260317": "Play-In", "20260318": "Play-In",
        "20260319": "Round of 64", "20260320": "Round of 64",
        "20260321": "Round of 32", "20260322": "Round of 32",
        "20260326": "Sweet 16", "20260327": "Sweet 16",
        "20260328": "Elite Eight", "20260329": "Elite Eight",
        "20260404": "Final Four", "20260405": "Final Four",
        "20260407": "Championship",
    }

    all_results = []
    for d in dates:
        print(f"\nFetching {d}...")
        try:
            data = fetch_date(d)
            games = parse_games(data, bracket_teams)
            # Assign round from date if notes were empty
            fallback_round = DATE_TO_ROUND.get(d)
            for g in games:
                if g["round"] is None and fallback_round:
                    g["round"] = fallback_round
            print(f"  Found {len(games)} completed tournament games")
            all_results.extend(games)
        except Exception as e:
            print(f"  Error: {e}")

    if not all_results:
        print("\nNo completed tournament games found.")
        return

    # Group by round
    by_round = {}
    for g in all_results:
        rnd = g.get("round", "Unknown")
        if rnd not in by_round:
            by_round[rnd] = []
        # Strip score fields for state file
        by_round[rnd].append({
            "winner": g["winner"],
            "winner_seed": g["winner_seed"],
            "loser": g["loser"],
            "loser_seed": g["loser_seed"],
        })

    # Print summary
    print(f"\n{'='*60}")
    print(f"  RESULTS SUMMARY")
    print(f"{'='*60}")
    for rnd in ROUND_ORDER:
        if rnd in by_round:
            games = by_round[rnd]
            print(f"\n  {rnd} ({len(games)} games):")
            for g in games:
                print(f"    ({g['winner_seed']:>2}) {g['winner']:<22} over ({g['loser_seed']:>2}) {g['loser']}")

    # Compare with predictions
    with open(PICKS_FILE) as f:
        picks = json.load(f)

    correct = 0
    wrong = 0
    for rnd, games in by_round.items():
        if rnd not in picks:
            continue
        predicted_winners = {g["winner"] for g in picks[rnd]}
        for g in games:
            if g["winner"] in predicted_winners:
                correct += 1
            else:
                wrong += 1
                print(f"  UPSET: ({g['winner_seed']}) {g['winner']} beat our pick in {rnd}")

    total = correct + wrong
    if total > 0:
        print(f"\n  Model accuracy so far: {correct}/{total} ({correct/total*100:.1f}%)")

    if args.dry_run:
        print("\n  (dry run, not saving)")
        return

    # Update state file
    with open(STATE_FILE) as f:
        state = json.load(f)

    state["actual_results"] = by_round
    state["last_updated"] = str(date.today())

    # Mark completed rounds
    state["completed_rounds"] = []
    for rnd in ROUND_ORDER:
        if rnd in by_round:
            expected = ROUND_EXPECTED_GAMES.get(rnd, 0)
            if len(by_round[rnd]) >= expected > 0:
                state["completed_rounds"].append(rnd)

    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)
    print(f"\n  State saved to {STATE_FILE}")


if __name__ == "__main__":
    main()
