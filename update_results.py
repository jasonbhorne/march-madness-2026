"""
Update tournament_state.json with actual game results.

Usage:
  python update_results.py --round "Round of 64" --results "Duke>Siena,Ohio St.>TCU"
  python update_results.py --round r64 --interactive
  python update_results.py --show

Shorthand round names: playin, r64, r32, sweet16, elite8, f4, championship
"""

import json
import argparse
import sys

STATE_FILE = "/Users/hornej/Documents/Research/march-madness-2026/tournament_state.json"
PICKS_FILE = "/Users/hornej/Documents/Research/march-madness-2026/bracket_picks.json"

ROUND_MAP = {
    "playin": "Play-In",
    "r64": "Round of 64",
    "r32": "Round of 32",
    "sweet16": "Sweet 16",
    "elite8": "Elite Eight",
    "f4": "Final Four",
    "championship": "Championship",
}

ROUND_ORDER = [
    "Play-In", "Round of 64", "Round of 32",
    "Sweet 16", "Elite Eight", "Final Four", "Championship"
]


def resolve_round(name):
    if name in ROUND_ORDER:
        return name
    return ROUND_MAP.get(name.lower().replace(" ", "").replace("-", ""), name)


def load_state():
    with open(STATE_FILE) as f:
        return json.load(f)


def save_state(state):
    from datetime import date
    state["last_updated"] = str(date.today())
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)
    print(f"State saved to {STATE_FILE}")


def load_picks():
    with open(PICKS_FILE) as f:
        return json.load(f)


def show_state():
    state = load_state()
    print(f"Last updated: {state['last_updated']}")
    print(f"Completed rounds: {state['completed_rounds'] or '(none)'}")
    for rnd, results in state.get("actual_results", {}).items():
        print(f"\n{rnd} ({len(results)} games):")
        for g in results:
            print(f"  {g['winner']} (#{g['winner_seed']}) over {g['loser']} (#{g['loser_seed']})")


def parse_results_string(results_str, round_name):
    """Parse 'Duke>Siena,Ohio St.>TCU' into result dicts.
    Seed lookup comes from bracket_picks.json."""
    picks = load_picks()

    # Build a seed lookup from all rounds in picks
    seed_lookup = {}
    for rnd_games in picks.values():
        for g in rnd_games:
            seed_lookup[g["winner"]] = g["winner_seed"]
            seed_lookup[g["loser"]] = g["loser_seed"]

    results = []
    for matchup in results_str.split(","):
        matchup = matchup.strip()
        if ">" not in matchup:
            print(f"Skipping invalid matchup (use Winner>Loser): '{matchup}'")
            continue
        winner, loser = [t.strip() for t in matchup.split(">", 1)]

        w_seed = seed_lookup.get(winner)
        l_seed = seed_lookup.get(loser)
        if w_seed is None:
            print(f"WARNING: '{winner}' not found in bracket_picks.json. Enter seed manually.")
            w_seed = int(input(f"  Seed for {winner}: "))
        if l_seed is None:
            print(f"WARNING: '{loser}' not found in bracket_picks.json. Enter seed manually.")
            l_seed = int(input(f"  Seed for {loser}: "))

        results.append({
            "winner": winner,
            "winner_seed": w_seed,
            "loser": loser,
            "loser_seed": l_seed,
        })

    return results


def interactive_mode(round_name):
    """Walk through expected matchups from bracket_picks.json and ask for winners."""
    picks = load_picks()
    if round_name not in picks:
        print(f"Round '{round_name}' not found in bracket_picks.json.")
        print(f"Available: {list(picks.keys())}")
        return []

    expected = picks[round_name]
    results = []
    print(f"\n{round_name}: {len(expected)} games")
    print("For each matchup, enter 1 for the predicted winner, 2 for the upset, or skip (s).\n")

    for i, g in enumerate(expected, 1):
        w, l = g["winner"], g["loser"]
        ws, ls = g["winner_seed"], g["loser_seed"]
        prob = g.get("prob", "?")

        print(f"Game {i}/{len(expected)}: ({ws}) {w} vs ({ls}) {l}  [predicted: {w} at {prob}%]")
        while True:
            choice = input("  Winner [1/2/s]: ").strip().lower()
            if choice == "1":
                results.append({
                    "winner": w, "winner_seed": ws,
                    "loser": l, "loser_seed": ls,
                })
                print(f"  -> {w} wins")
                break
            elif choice == "2":
                results.append({
                    "winner": l, "winner_seed": ls,
                    "loser": w, "loser_seed": ws,
                })
                print(f"  -> {l} wins (upset!)")
                break
            elif choice == "s":
                print("  -> skipped")
                break
            else:
                print("  Enter 1, 2, or s")

    return results


def main():
    parser = argparse.ArgumentParser(description="Update tournament state with actual results")
    parser.add_argument("--round", "-r", help="Round name (e.g., r64, sweet16, 'Round of 64')")
    parser.add_argument("--results", help="Comma-separated results: 'Winner>Loser,Winner>Loser'")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive mode")
    parser.add_argument("--show", "-s", action="store_true", help="Show current state")
    parser.add_argument("--mark-complete", action="store_true",
                        help="Mark the round as complete (auto if all games entered)")
    args = parser.parse_args()

    if args.show:
        show_state()
        return

    if not args.round:
        parser.error("--round is required (unless using --show)")

    round_name = resolve_round(args.round)
    print(f"Round: {round_name}")

    state = load_state()
    existing = state.get("actual_results", {}).get(round_name, [])
    if existing:
        print(f"  (already has {len(existing)} results recorded)")

    if args.interactive:
        new_results = interactive_mode(round_name)
    elif args.results:
        new_results = parse_results_string(args.results, round_name)
    else:
        parser.error("Provide --results or --interactive")

    if not new_results:
        print("No results to add.")
        return

    # Merge: replace existing results for this round
    if "actual_results" not in state:
        state["actual_results"] = {}

    if round_name in state["actual_results"] and not args.interactive:
        # Append mode for --results flag
        state["actual_results"][round_name].extend(new_results)
        print(f"Appended {len(new_results)} results (total: {len(state['actual_results'][round_name])})")
    else:
        state["actual_results"][round_name] = new_results
        print(f"Set {len(new_results)} results for {round_name}")

    # Auto-mark complete based on expected game count
    picks = load_picks()
    expected_count = len(picks.get(round_name, []))
    actual_count = len(state["actual_results"][round_name])

    if args.mark_complete or actual_count >= expected_count > 0:
        if round_name not in state["completed_rounds"]:
            state["completed_rounds"].append(round_name)
            # Keep in order
            state["completed_rounds"] = [r for r in ROUND_ORDER if r in state["completed_rounds"]]
            print(f"Marked {round_name} as complete ({actual_count}/{expected_count} games)")

    save_state(state)


if __name__ == "__main__":
    main()
