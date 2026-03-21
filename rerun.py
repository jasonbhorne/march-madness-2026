"""
Rerun bracket predictions with actual results locked in.

Loads tournament_state.json, uses actual results for completed rounds,
and re-predicts remaining rounds using the same ML model.

Usage:
  python rerun.py                    # auto-detect next round from state
  python rerun.py --round sweet16    # predict from a specific round
  python rerun.py --dry-run          # show what would happen without writing files
"""

import json
import argparse
import sys
import os

# Import everything we need from predict_bracket
from predict_bracket import (
    load_team_data, load_matchups, build_training_data, train_model,
    get_2026_bracket, get_team_features, predict_matchup,
    _get_prob, ALL_FEATURE_NAMES, generate_html_bracket,
    simulate_bracket, run_deterministic_bracket, print_results,
    DATA_DIR, OUT_DIR,
)

STATE_FILE = os.path.join(OUT_DIR, "tournament_state.json")
PICKS_FILE = os.path.join(OUT_DIR, "bracket_picks.json")

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


def detect_next_round(state):
    """Figure out the next round to predict based on completed rounds."""
    completed = set(state.get("completed_rounds", []))
    for rnd in ROUND_ORDER:
        if rnd not in completed:
            return rnd
    return None


def build_team_cache_from_bracket(model, feature_cols, kb, resume, cbb, play_in_matchups, r64_matchups):
    """Build team_cache the same way simulate_bracket does."""
    team_cache = {}
    all_teams = set()
    for t1, t2 in play_in_matchups + r64_matchups:
        all_teams.add((t1['TEAM'], t1['TEAM NO'], t1['SEED']))
        all_teams.add((t2['TEAM'], t2['TEAM NO'], t2['SEED']))

    for name, team_no, seed in all_teams:
        feats = get_team_features(name, team_no, 2026, kb, resume, cbb)
        team_cache[name] = {'name': name, 'seed': seed, 'feats': feats, 'team_no': team_no}

    return team_cache


def advance_with_actuals(state, team_cache, play_in_matchups, r64_matchups):
    """Use actual results to determine who advanced through completed rounds.
    Returns the bracket state as a list of team dicts ready for the next round,
    plus a dict of round_name -> [(winner, loser, prob_or_None)] for completed rounds."""

    actual = state.get("actual_results", {})
    locked_results = {}  # round_name -> [(winner_dict, loser_dict, prob)]

    # Start with play-in resolution
    play_in_winners = {}
    if "Play-In" in actual:
        pi_results = []
        for g in actual["Play-In"]:
            w_name, l_name = g["winner"], g["loser"]
            w = team_cache.get(w_name)
            l = team_cache.get(l_name)
            if w and l:
                play_in_winners[w_name] = w
                play_in_winners[l_name] = w
                pi_results.append((w, l, None))
        locked_results["Play-In"] = pi_results

    # Build initial R64 bracket (same logic as run_deterministic_bracket)
    bracket = []
    for t1_data, t2_data in r64_matchups:
        t1 = team_cache.get(t1_data['TEAM'])
        t2 = team_cache.get(t2_data['TEAM'])
        if t1 and t1['name'] in play_in_winners:
            t1 = play_in_winners[t1['name']]
        if t2 and t2['name'] in play_in_winners:
            t2 = play_in_winners[t2['name']]
        bracket.append(t1)
        bracket.append(t2)

    round_names_by_size = {64: 'Round of 64', 32: 'Round of 32', 16: 'Sweet 16',
                           8: 'Elite Eight', 4: 'Final Four', 2: 'Championship'}

    current = list(bracket)
    round_size = 64

    completed_set = set(state.get("completed_rounds", []))

    while len(current) > 1:
        rname = round_names_by_size.get(round_size, f"Round of {round_size}")

        if rname in completed_set and rname in actual:
            # Use actual results to determine advancers
            actual_winners = {g["winner"] for g in actual[rname]}
            rnd_results = []
            next_round = []

            for i in range(0, len(current) - 1, 2):
                t1, t2 = current[i], current[i + 1]
                if t1 is None or t2 is None:
                    next_round.append(t1 or t2)
                    continue

                if t1['name'] in actual_winners:
                    winner, loser = t1, t2
                elif t2['name'] in actual_winners:
                    winner, loser = t2, t1
                else:
                    # Can't find this matchup in actuals, skip
                    print(f"  WARNING: No actual result for {t1['name']} vs {t2['name']} in {rname}")
                    next_round.append(t1)  # fallback
                    continue

                rnd_results.append((winner, loser, None))
                next_round.append(winner)

            locked_results[rname] = rnd_results
            current = next_round
            round_size //= 2
        else:
            # This round and all subsequent are not yet played
            break

    return current, round_size, locked_results


def predict_remaining(model, feature_cols, current_bracket, round_size):
    """Predict from current bracket state forward through the championship."""
    round_names = {64: 'Round of 64', 32: 'Round of 32', 16: 'Sweet 16',
                   8: 'Elite Eight', 4: 'Final Four', 2: 'Championship'}

    predicted_results = {}
    current = list(current_bracket)

    while len(current) > 1:
        rname = round_names.get(round_size, f"Round of {round_size}")
        rnd_results = []
        next_round = []

        for i in range(0, len(current) - 1, 2):
            t1, t2 = current[i], current[i + 1]
            if t1 is None or t2 is None:
                next_round.append(t1 or t2)
                continue

            prob = _get_prob(model, feature_cols, t1, t2)
            if prob >= 0.5:
                winner, loser, wp = t1, t2, prob
            else:
                winner, loser, wp = t2, t1, 1 - prob

            rnd_results.append((winner, loser, wp))
            next_round.append(winner)

        predicted_results[rname] = rnd_results
        current = next_round
        round_size //= 2

    champion = current[0] if current else None
    return predicted_results, champion


def merge_results(locked, predicted):
    """Combine locked actual results and predicted results into one dict."""
    merged = {}
    for rname in ROUND_ORDER:
        if rname in locked:
            merged[rname] = locked[rname]
        elif rname in predicted:
            merged[rname] = predicted[rname]
    return merged


def build_bracket_picks_json(merged_results, locked_rounds):
    """Convert merged results to bracket_picks.json format."""
    picks = {}
    locked_set = set(locked_rounds)

    for rname in ROUND_ORDER:
        if rname not in merged_results:
            continue
        games = []
        for winner, loser, wp in merged_results[rname]:
            game = {
                "winner": winner["name"],
                "winner_seed": winner["seed"],
                "loser": loser["name"],
                "loser_seed": loser["seed"],
            }
            if wp is not None:
                game["prob"] = round(wp * 100, 1)
            if rname in locked_set:
                game["actual"] = True
            games.append(game)
        picks[rname] = games

    return picks


def generate_rerun_html(merged_results, champion, advancement, champion_counts,
                        team_cache, n_sims, feat_imp, cv_accuracy, locked_rounds):
    """Generate HTML with visual distinction for actual vs predicted results."""

    locked_set = set(locked_rounds)

    # Build rounds_data for the HTML
    rounds_data = {}
    for rname in ['Round of 64', 'Round of 32', 'Sweet 16',
                  'Elite Eight', 'Final Four', 'Championship']:
        if rname in merged_results:
            games = []
            for winner, loser, wp in merged_results[rname]:
                games.append({
                    'winner': winner['name'], 'winner_seed': winner['seed'],
                    'loser': loser['name'], 'loser_seed': loser['seed'],
                    'prob': round(wp * 100, 1) if wp is not None else None,
                    'actual': rname in locked_set,
                })
            rounds_data[rname] = games

    # Top championship odds
    champ_sorted = sorted(champion_counts.items(), key=lambda x: -x[1])[:15]
    champ_odds = [{'team': t, 'seed': team_cache[t]['seed'],
                   'pct': round(c / n_sims * 100, 1)} for t, c in champ_sorted]

    # Final four odds
    f4 = {}
    for name, counts in advancement.items():
        f4_count = counts.get(4, 0)
        if f4_count > 0:
            f4[name] = f4_count / n_sims * 100
    f4_sorted = sorted(f4.items(), key=lambda x: -x[1])[:15]
    f4_odds = [{'team': t, 'seed': team_cache[t]['seed'],
                'pct': round(p, 1)} for t, p in f4_sorted]

    # Feature importance
    feat_data = [{'name': n.replace('_DIFF', ''), 'importance': round(v * 100, 2)}
                 for n, v in feat_imp[:15]]

    completed_str = ", ".join(locked_rounds) if locked_rounds else "None"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>March Madness 2026 - Updated Bracket Predictions</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

  :root {{
    --bg: #0a0e17;
    --card: #131a2b;
    --card-border: #1e2a42;
    --accent: #f97316;
    --accent2: #3b82f6;
    --accent3: #10b981;
    --text: #e2e8f0;
    --text-dim: #64748b;
    --winner: #f97316;
    --prob-high: #10b981;
    --prob-mid: #f59e0b;
    --prob-low: #ef4444;
    --actual-bg: rgba(16, 185, 129, 0.08);
    --actual-border: rgba(16, 185, 129, 0.3);
  }}

  * {{ margin: 0; padding: 0; box-sizing: border-box; }}

  body {{
    font-family: 'Inter', -apple-system, sans-serif;
    background: var(--bg);
    color: var(--text);
    line-height: 1.5;
  }}

  .hero {{
    text-align: center;
    padding: 4rem 2rem 2rem;
    background: linear-gradient(135deg, #0a0e17 0%, #1a1040 50%, #0a0e17 100%);
    border-bottom: 1px solid var(--card-border);
  }}

  .hero h1 {{
    font-size: 3rem;
    font-weight: 900;
    background: linear-gradient(135deg, var(--accent), var(--accent2));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.5rem;
  }}

  .hero .subtitle {{
    font-size: 1.1rem;
    color: var(--text-dim);
    margin-bottom: 1.5rem;
  }}

  .hero .model-badge {{
    display: inline-block;
    background: var(--card);
    border: 1px solid var(--card-border);
    border-radius: 2rem;
    padding: 0.5rem 1.5rem;
    font-size: 0.85rem;
    color: var(--accent3);
  }}

  .update-banner {{
    background: var(--actual-bg);
    border: 1px solid var(--actual-border);
    border-radius: 0.75rem;
    padding: 1rem 1.5rem;
    margin: 1.5rem 0;
    font-size: 0.9rem;
    display: flex;
    align-items: center;
    gap: 0.75rem;
  }}

  .update-banner .icon {{ font-size: 1.2rem; }}

  .container {{
    max-width: 1200px;
    margin: 0 auto;
    padding: 2rem;
  }}

  .section-title {{
    font-size: 1.5rem;
    font-weight: 800;
    margin: 2.5rem 0 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 2px solid var(--accent);
    display: inline-block;
  }}

  .champion-card {{
    background: linear-gradient(135deg, #1a0f00 0%, #2a1800 100%);
    border: 2px solid var(--accent);
    border-radius: 1rem;
    padding: 2rem;
    text-align: center;
    margin: 2rem 0;
    position: relative;
    overflow: hidden;
  }}

  .champion-card::before {{
    content: '';
    position: absolute;
    top: -50%; left: -50%;
    width: 200%; height: 200%;
    background: radial-gradient(circle, rgba(249,115,22,0.1) 0%, transparent 60%);
  }}

  .champion-card .label {{
    font-size: 0.85rem;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    color: var(--accent);
    margin-bottom: 0.5rem;
  }}

  .champion-card .team {{
    font-size: 2.5rem;
    font-weight: 900;
    color: var(--accent);
  }}

  .champion-card .seed {{
    font-size: 1rem;
    color: var(--text-dim);
    margin-top: 0.25rem;
  }}

  .odds-grid {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 2rem;
    margin: 1rem 0;
  }}

  @media (max-width: 768px) {{
    .odds-grid {{ grid-template-columns: 1fr; }}
    .hero h1 {{ font-size: 2rem; }}
  }}

  .odds-card {{
    background: var(--card);
    border: 1px solid var(--card-border);
    border-radius: 0.75rem;
    padding: 1.5rem;
  }}

  .odds-card h3 {{
    font-size: 1.1rem;
    font-weight: 700;
    margin-bottom: 1rem;
    color: var(--accent2);
  }}

  .odds-row {{
    display: flex;
    align-items: center;
    padding: 0.35rem 0;
    border-bottom: 1px solid rgba(255,255,255,0.05);
  }}

  .odds-row:last-child {{ border-bottom: none; }}
  .odds-seed {{ width: 2rem; font-size: 0.75rem; color: var(--text-dim); text-align: center; }}
  .odds-team {{ flex: 1; font-weight: 600; font-size: 0.9rem; }}
  .odds-pct {{ width: 3.5rem; text-align: right; font-weight: 700; font-size: 0.9rem; }}

  .odds-bar-wrap {{
    width: 120px; height: 8px;
    background: rgba(255,255,255,0.05);
    border-radius: 4px; margin-left: 0.75rem; overflow: hidden;
  }}

  .odds-bar {{ height: 100%; border-radius: 4px; }}
  .odds-bar.champ {{ background: linear-gradient(90deg, var(--accent), #fb923c); }}
  .odds-bar.f4 {{ background: linear-gradient(90deg, var(--accent2), #60a5fa); }}

  .bracket-round {{
    background: var(--card);
    border: 1px solid var(--card-border);
    border-radius: 0.75rem;
    padding: 1.5rem;
    margin: 1.5rem 0;
  }}

  .bracket-round.actual-round {{
    border-color: var(--actual-border);
    background: linear-gradient(135deg, var(--card) 0%, rgba(16, 185, 129, 0.04) 100%);
  }}

  .bracket-round h3 {{
    font-size: 1.1rem;
    font-weight: 700;
    margin-bottom: 1rem;
    color: var(--accent2);
  }}

  .bracket-round h3 .badge {{
    display: inline-block;
    font-size: 0.7rem;
    font-weight: 600;
    padding: 0.15rem 0.5rem;
    border-radius: 1rem;
    margin-left: 0.5rem;
    vertical-align: middle;
  }}

  .badge-actual {{
    background: rgba(16, 185, 129, 0.15);
    color: var(--accent3);
  }}

  .badge-predicted {{
    background: rgba(59, 130, 246, 0.15);
    color: var(--accent2);
  }}

  .matchup {{
    display: flex;
    align-items: center;
    padding: 0.4rem 0.75rem;
    margin: 0.25rem 0;
    border-radius: 0.5rem;
    background: rgba(255,255,255,0.02);
  }}

  .matchup:hover {{ background: rgba(255,255,255,0.05); }}
  .matchup .winner {{ color: var(--winner); font-weight: 700; }}
  .matchup .loser {{ color: var(--text-dim); }}

  .matchup.actual-game {{
    background: var(--actual-bg);
    border-left: 3px solid var(--accent3);
  }}

  .matchup .m-seed {{ width: 1.8rem; font-size: 0.75rem; text-align: center; }}
  .matchup .m-team {{ width: 180px; font-size: 0.85rem; }}
  .matchup .m-vs {{ width: 40px; text-align: center; font-size: 0.7rem; color: var(--text-dim); }}

  .matchup .m-prob {{
    font-size: 0.8rem; font-weight: 600; margin-left: auto;
    padding: 0.15rem 0.5rem; border-radius: 1rem;
    font-variant-numeric: tabular-nums;
  }}

  .matchup .m-actual {{
    font-size: 0.7rem; font-weight: 600; margin-left: auto;
    padding: 0.15rem 0.5rem; border-radius: 1rem;
    background: rgba(16, 185, 129, 0.15); color: var(--accent3);
  }}

  .prob-high {{ background: rgba(16,185,129,0.15); color: var(--prob-high); }}
  .prob-mid {{ background: rgba(245,158,11,0.15); color: var(--prob-mid); }}
  .prob-low {{ background: rgba(239,68,68,0.15); color: var(--prob-low); }}

  .feat-row {{ display: flex; align-items: center; padding: 0.4rem 0; }}
  .feat-name {{ width: 200px; font-size: 0.85rem; font-weight: 500; }}
  .feat-bar-wrap {{
    flex: 1; height: 12px;
    background: rgba(255,255,255,0.05);
    border-radius: 6px; overflow: hidden; margin: 0 1rem;
  }}
  .feat-bar {{
    height: 100%;
    background: linear-gradient(90deg, var(--accent2), var(--accent3));
    border-radius: 6px;
  }}
  .feat-val {{ width: 50px; text-align: right; font-size: 0.8rem; color: var(--text-dim); }}

  .methodology {{
    background: var(--card);
    border: 1px solid var(--card-border);
    border-radius: 0.75rem;
    padding: 2rem; margin: 2rem 0;
    font-size: 0.9rem; line-height: 1.8;
  }}
  .methodology h3 {{ color: var(--accent2); margin-bottom: 1rem; }}
  .methodology ul {{ padding-left: 1.5rem; margin: 0.5rem 0; }}
  .methodology li {{ margin: 0.3rem 0; }}

  .footer {{
    text-align: center; padding: 2rem;
    color: var(--text-dim); font-size: 0.8rem;
    border-top: 1px solid var(--card-border); margin-top: 3rem;
  }}
  .footer a {{ color: var(--accent2); text-decoration: none; }}
</style>
</head>
<body>

<div class="hero">
  <h1>March Madness 2026</h1>
  <p class="subtitle">Updated Bracket Predictions</p>
  <span class="model-badge">Stacked Ensemble &middot; {cv_accuracy:.1f}% accuracy &middot; {n_sims:,} simulations</span>
</div>

<div class="container">

  <div class="update-banner">
    <span class="icon">&#9989;</span>
    <div>
      <strong>Bracket updated with real results.</strong>
      Completed rounds: {completed_str}. Remaining rounds are ML predictions.
    </div>
  </div>

  <div class="champion-card">
    <div class="label">Predicted Champion</div>
    <div class="team">{champion['name']}</div>
    <div class="seed">#{champion['seed']} Seed &middot; {round(champion_counts.get(champion['name'], 0) / n_sims * 100, 1)}% probability</div>
  </div>

  <div class="odds-grid">
    <div class="odds-card">
      <h3>Championship Odds (Updated)</h3>
      {''.join(f"""<div class="odds-row">
        <span class="odds-seed">{o['seed']}</span>
        <span class="odds-team">{o['team']}</span>
        <span class="odds-pct">{o['pct']}%</span>
        <div class="odds-bar-wrap"><div class="odds-bar champ" style="width:{min(o['pct'] / champ_odds[0]['pct'] * 100, 100) if champ_odds else 0}%"></div></div>
      </div>""" for o in champ_odds)}
    </div>
    <div class="odds-card">
      <h3>Final Four Odds (Updated)</h3>
      {''.join(f"""<div class="odds-row">
        <span class="odds-seed">{o['seed']}</span>
        <span class="odds-team">{o['team']}</span>
        <span class="odds-pct">{o['pct']}%</span>
        <div class="odds-bar-wrap"><div class="odds-bar f4" style="width:{min(o['pct'] / f4_odds[0]['pct'] * 100, 100) if f4_odds else 0}%"></div></div>
      </div>""" for o in f4_odds)}
    </div>
  </div>

  <div class="section-title">Bracket Results</div>
"""

    # Add each round
    for rname in ['Round of 64', 'Round of 32', 'Sweet 16',
                  'Elite Eight', 'Final Four', 'Championship']:
        if rname not in rounds_data:
            continue
        is_actual = rounds_data[rname][0]['actual'] if rounds_data[rname] else False
        round_class = "bracket-round actual-round" if is_actual else "bracket-round"
        badge = '<span class="badge badge-actual">ACTUAL</span>' if is_actual else '<span class="badge badge-predicted">PREDICTED</span>'

        html += f'  <div class="{round_class}"><h3>{rname} {badge}</h3>\n'
        for g in rounds_data[rname]:
            if g['actual']:
                html += f"""    <div class="matchup actual-game">
      <span class="m-seed winner">{g['winner_seed']}</span>
      <span class="m-team winner">{g['winner']}</span>
      <span class="m-vs">over</span>
      <span class="m-seed loser">{g['loser_seed']}</span>
      <span class="m-team loser">{g['loser']}</span>
      <span class="m-actual">FINAL</span>
    </div>\n"""
            else:
                prob_class = 'prob-high' if g['prob'] >= 75 else ('prob-mid' if g['prob'] >= 60 else 'prob-low')
                html += f"""    <div class="matchup">
      <span class="m-seed winner">{g['winner_seed']}</span>
      <span class="m-team winner">{g['winner']}</span>
      <span class="m-vs">over</span>
      <span class="m-seed loser">{g['loser_seed']}</span>
      <span class="m-team loser">{g['loser']}</span>
      <span class="m-prob {prob_class}">{g['prob']}%</span>
    </div>\n"""
        html += '  </div>\n'

    # Feature importance
    max_imp = feat_data[0]['importance'] if feat_data else 1
    html += """
  <div class="section-title">What Matters Most</div>
  <div class="bracket-round">
    <h3>Top Features by Importance</h3>
"""
    for f in feat_data:
        html += f"""    <div class="feat-row">
      <span class="feat-name">{f['name']}</span>
      <div class="feat-bar-wrap"><div class="feat-bar" style="width:{f['importance']/max_imp*100}%"></div></div>
      <span class="feat-val">{f['importance']}%</span>
    </div>\n"""

    html += f"""  </div>

  <div class="footer">
    Updated with actual tournament results. Built with Python, scikit-learn, and too much coffee.<br>
    <a href="https://jasonhorne.org">jasonhorne.org</a> &middot; March 2026
  </div>

</div>
</body>
</html>"""

    return html


def main():
    parser = argparse.ArgumentParser(description="Rerun bracket with actual results locked in")
    parser.add_argument("--round", "-r", help="Start predictions from this round (e.g., sweet16)")
    parser.add_argument("--dry-run", action="store_true", help="Show plan without writing files")
    parser.add_argument("--no-sim", action="store_true", help="Skip Monte Carlo simulation (faster)")
    args = parser.parse_args()

    state = load_state()
    completed = state.get("completed_rounds", [])

    if args.round:
        target = resolve_round(args.round)
        print(f"Target round: {target}")
    else:
        target = detect_next_round(state)
        if target is None:
            print("All rounds complete. Nothing to predict.")
            return
        print(f"Auto-detected next round: {target}")

    print(f"Completed rounds: {completed or '(none)'}")
    for rnd in completed:
        n_games = len(state.get("actual_results", {}).get(rnd, []))
        print(f"  {rnd}: {n_games} actual results")

    if args.dry_run:
        print("\nDry run. Would predict from", target, "onward.")
        return

    # Train model (same as original)
    print("\n[1/5] Loading data...")
    kb, resume, cbb = load_team_data()
    games = load_matchups()

    print("\n[2/5] Building training features...")
    train_df = build_training_data(games, kb, resume, cbb)

    print("\n[3/5] Training model...")
    model, feature_cols, feat_imp, cv_accuracy = train_model(train_df)

    print("\n[4/5] Loading bracket and applying actual results...")
    play_in_matchups, r64_matchups = get_2026_bracket()

    # Build team cache
    team_cache = build_team_cache_from_bracket(
        model, feature_cols, kb, resume, cbb, play_in_matchups, r64_matchups
    )

    # Advance through completed rounds using actuals
    remaining_bracket, next_round_size, locked_results = advance_with_actuals(
        state, team_cache, play_in_matchups, r64_matchups
    )

    print(f"  Locked {len(locked_results)} rounds with actual results")
    print(f"  Remaining bracket: {len(remaining_bracket)} teams, next round size = {next_round_size}")

    # Predict remaining rounds
    predicted_results, champion = predict_remaining(
        model, feature_cols, remaining_bracket, next_round_size
    )

    # Merge
    merged = merge_results(locked_results, predicted_results)

    print(f"\n[5/5] Generating outputs...")

    # Run simulation for updated odds (uses actuals for completed rounds via the same state)
    N_SIMS = 10000
    if args.no_sim:
        # Fake minimal sim data
        advancement = {}
        champion_counts = {champion['name']: N_SIMS} if champion else {}
        print("  Skipped simulation (--no-sim)")
    else:
        print(f"  Running {N_SIMS:,} simulations...")
        advancement, champion_counts, _ = simulate_bracket(
            model, feature_cols, kb, resume, cbb, play_in_matchups, r64_matchups, n_sims=N_SIMS
        )

    # Save bracket_picks.json
    picks = build_bracket_picks_json(merged, completed)
    with open(PICKS_FILE, 'w') as f:
        json.dump(picks, f, indent=2)
    print(f"  Updated {PICKS_FILE}")

    # Generate HTML
    html = generate_rerun_html(
        merged, champion, advancement, champion_counts,
        team_cache, N_SIMS, feat_imp, cv_accuracy, completed
    )
    html_path = os.path.join(OUT_DIR, "march_madness_2026.html")
    with open(html_path, 'w') as f:
        f.write(html)
    print(f"  Updated {html_path}")

    # Print bracket summary
    print(f"\n{'=' * 60}")
    for rname in ROUND_ORDER:
        if rname not in merged:
            continue
        tag = "ACTUAL" if rname in set(completed) else "PREDICTED"
        print(f"\n  {rname} [{tag}]:")
        for winner, loser, wp in merged[rname]:
            prob_str = f"({wp:.1%})" if wp is not None else "(actual)"
            print(f"    ({winner['seed']:>2}) {winner['name']:<22} over "
                  f"({loser['seed']:>2}) {loser['name']:<22} {prob_str}")

    if champion:
        print(f"\n  {'=' * 55}")
        print(f"  PREDICTED CHAMPION: ({champion['seed']}) {champion['name']}")
        print(f"  {'=' * 55}")

    print("\nDone!")


if __name__ == "__main__":
    main()
