"""
March Madness 2026 - Round of 32 Predictions (Colab Edition)
=============================================================
Fetches actual Round of 64 results from ESPN, locks them in,
then re-predicts Round of 32 through Championship.

Run in Google Colab:
  1. Mount Google Drive (cell below handles this)
  2. Data should be in: My Drive/March Madness 2026/data/
  3. Outputs save to:   My Drive/March Madness 2026/

Blog: jasonhorne.org
"""

# ─── Colab Setup ──────────────────────────────────────────────────────────
from google.colab import drive
drive.mount('/content/drive')

DATA_DIR = "/content/drive/MyDrive/March Madness 2026/data"
OUT_DIR = "/content/drive/MyDrive/March Madness 2026"

import os
csv_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.csv')]
print(f"Found {len(csv_files)} CSV files in {DATA_DIR}")
if len(csv_files) == 0:
    raise FileNotFoundError(f"No CSV files found in {DATA_DIR}. Check your Drive path.")

# ─── Imports ──────────────────────────────────────────────────────────────
import pandas as pd
import numpy as np
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    GradientBoostingClassifier, RandomForestClassifier,
    StackingClassifier, HistGradientBoostingClassifier
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import brier_score_loss
import json
import urllib.request
from datetime import date
import warnings
warnings.filterwarnings('ignore')

# ─── 1. Features ─────────────────────────────────────────────────────────

KB_FEATURES = [
    'KADJ O', 'KADJ D', 'KADJ EM', 'BADJ EM', 'BADJ O', 'BADJ D',
    'BARTHAG', 'KADJ T', 'WIN%',
    'EFG%', 'EFG%D', 'FTR', 'FTRD', 'TOV%', 'TOV%D', 'OREB%', 'DREB%',
    '2PT%', '2PT%D', '3PT%', '3PT%D', 'FT%',
    '2PTR', '3PTR',
    'BLK%', 'AST%', 'OP AST%',
    'PPPO', 'PPPD',
    'ELITE SOS', 'WAB', 'EXP', 'TALENT', 'AVG HGT', 'EFF HGT',
]

RESUME_FEATURES = [
    'ELO', 'NET RPI', 'Q1 W', 'Q2 W', 'Q1 PLUS Q2 W',
    'Q3 Q4 L', 'PLUS 500', 'R SCORE',
]

def engineer_features(row):
    feats = {}
    feats['NET_EFF'] = row.get('KADJ O', 0) - row.get('KADJ D', 0)
    kadj_o = row.get('KADJ O', 0)
    kadj_d = row.get('KADJ D', 0)
    feats['OFF_DEF_RATIO'] = kadj_o / max(kadj_d, 1)
    feats['FOUR_FACTORS_O'] = (
        0.4 * row.get('EFG%', 0) + 0.25 * (100 - row.get('TOV%', 0)) +
        0.20 * row.get('OREB%', 0) + 0.15 * row.get('FTR', 0)
    )
    feats['FOUR_FACTORS_D'] = (
        0.4 * (100 - row.get('EFG%D', 50)) + 0.25 * row.get('TOV%D', 0) +
        0.20 * (100 - row.get('OP OREB%', 50)) + 0.15 * (100 - row.get('FTRD', 0))
    )
    feats['SHOOT_VERSATILITY'] = row.get('2PT%', 0) * 0.5 + row.get('3PT%', 0) * 0.5
    feats['DEF_PRESSURE'] = row.get('BLK%', 0) + row.get('TOV%D', 0)
    feats['BALL_SECURITY'] = 100 - row.get('TOV%', 0)
    feats['REB_MARGIN'] = row.get('OREB%', 0) + row.get('DREB%', 0) - 100
    q12 = row.get('Q1 PLUS Q2 W', 0)
    total_w = row.get('W', 1)
    feats['QUALITY_WIN_PCT'] = q12 / max(total_w, 1)
    feats['BAD_LOSS_RATE'] = row.get('Q3 Q4 L', 0)
    return feats

ENGINEERED_NAMES = [
    'NET_EFF', 'OFF_DEF_RATIO', 'FOUR_FACTORS_O', 'FOUR_FACTORS_D',
    'SHOOT_VERSATILITY', 'DEF_PRESSURE', 'BALL_SECURITY', 'REB_MARGIN',
    'QUALITY_WIN_PCT', 'BAD_LOSS_RATE'
]

ALL_FEATURE_NAMES = KB_FEATURES + RESUME_FEATURES + ENGINEERED_NAMES

# ─── 2. Load data ────────────────────────────────────────────────────────

def load_team_data():
    kb = pd.read_csv(f"{DATA_DIR}/KenPom Barttorvik.csv")
    print(f"KenPom Barttorvik: {kb.shape[0]} team-seasons, years {kb['YEAR'].min()}-{kb['YEAR'].max()}")
    resume = pd.read_csv(f"{DATA_DIR}/Resumes.csv")
    print(f"Resumes: {resume.shape[0]} team-seasons")
    cbb_frames = []
    for year in range(13, 27):
        full_year = 2000 + year
        try:
            df = pd.read_csv(f"{DATA_DIR}/cbb{year}.csv", encoding='utf-8-sig')
            df['YEAR'] = full_year
            cbb_frames.append(df)
        except FileNotFoundError:
            continue
    cbb = pd.concat(cbb_frames, ignore_index=True) if cbb_frames else pd.DataFrame()
    print(f"CBB stats: {cbb.shape[0]} team-seasons")
    return kb, resume, cbb

def load_matchups():
    df = pd.read_csv(f"{DATA_DIR}/Tournament Matchups.csv")
    games = []
    for year in df['YEAR'].unique():
        for rnd in df[df['YEAR'] == year]['CURRENT ROUND'].unique():
            rnd_df = df[(df['YEAR'] == year) & (df['CURRENT ROUND'] == rnd)].sort_values('BY YEAR NO')
            teams = rnd_df.to_dict('records')
            for i in range(0, len(teams) - 1, 2):
                t1, t2 = teams[i], teams[i + 1]
                if pd.isna(t1.get('SCORE')) or pd.isna(t2.get('SCORE')):
                    continue
                if t1['ROUND'] < t2['ROUND']:
                    winner, loser = t1, t2
                elif t2['ROUND'] < t1['ROUND']:
                    winner, loser = t2, t1
                elif t1['SCORE'] > t2['SCORE']:
                    winner, loser = t1, t2
                else:
                    winner, loser = t2, t1
                games.append({
                    'YEAR': year, 'ROUND': rnd,
                    'WINNER': winner['TEAM'], 'WINNER_SEED': winner['SEED'],
                    'WINNER_NO': winner['TEAM NO'],
                    'LOSER': loser['TEAM'], 'LOSER_SEED': loser['SEED'],
                    'LOSER_NO': loser['TEAM NO'],
                    'WIN_SCORE': winner['SCORE'], 'LOSE_SCORE': loser['SCORE'],
                    'MARGIN': winner['SCORE'] - loser['SCORE'],
                })
    games_df = pd.DataFrame(games)
    print(f"Parsed {len(games_df)} tournament games across {games_df['YEAR'].nunique()} years")
    return games_df

# ─── 3. Feature building ─────────────────────────────────────────────────

def get_team_features(team_name, team_no, year, kb, resume, cbb):
    feats = {}
    kb_match = kb[(kb['YEAR'] == year) & (kb['TEAM NO'] == team_no)]
    if len(kb_match) == 0:
        kb_match = kb[(kb['YEAR'] == year) & (kb['TEAM'] == team_name)]
    if len(kb_match) > 0:
        row = kb_match.iloc[0]
        for f in KB_FEATURES:
            if f in row.index:
                feats[f] = row[f]
        feats['W'] = row.get('W', 0)
        feats['OP OREB%'] = row.get('OP OREB%', 0)
        feats.update(engineer_features({**feats, **row.to_dict()}))
    else:
        cbb_match = cbb[(cbb['YEAR'] == year) & (cbb['TEAM'] == team_name)]
        if len(cbb_match) > 0:
            row = cbb_match.iloc[0]
            cbb_map = {
                'ADJOE': 'KADJ O', 'ADJDE': 'KADJ D', 'BARTHAG': 'BARTHAG',
                'EFG_O': 'EFG%', 'EFG_D': 'EFG%D', 'TOR': 'TOV%', 'TORD': 'TOV%D',
                'ORB': 'OREB%', 'DRB': 'DREB%', 'FTR': 'FTR', 'FTRD': 'FTRD',
                '2P_O': '2PT%', '2P_D': '2PT%D', '3P_O': '3PT%', '3P_D': '3PT%D',
                'ADJ_T': 'KADJ T', 'WAB': 'WAB',
            }
            for src, dst in cbb_map.items():
                if src in row.index:
                    feats[dst] = row[src]
            feats['W'] = row.get('W', 0)
            feats.update(engineer_features(feats))
    res_match = resume[(resume['YEAR'] == year) & (resume['TEAM NO'] == team_no)]
    if len(res_match) == 0:
        res_match = resume[(resume['YEAR'] == year) & (resume['TEAM'] == team_name)]
    if len(res_match) > 0:
        row = res_match.iloc[0]
        for f in RESUME_FEATURES:
            if f in row.index:
                feats[f] = row[f]
        if 'Q1 PLUS Q2 W' in feats:
            feats['QUALITY_WIN_PCT'] = feats['Q1 PLUS Q2 W'] / max(feats.get('W', 1), 1)
        if 'Q3 Q4 L' in feats:
            feats['BAD_LOSS_RATE'] = feats['Q3 Q4 L']
    return feats if feats else None

def build_training_data(games, kb, resume, cbb):
    rows = []
    matched = 0
    unmatched = 0
    for _, game in games.iterrows():
        year = game['YEAR']
        w_feats = get_team_features(game['WINNER'], game['WINNER_NO'], year, kb, resume, cbb)
        l_feats = get_team_features(game['LOSER'], game['LOSER_NO'], year, kb, resume, cbb)
        if w_feats is None or l_feats is None:
            unmatched += 1
            continue
        row = {'YEAR': year, 'ROUND': game['ROUND'], 'MARGIN': game['MARGIN']}
        row['SEED_DIFF'] = game['WINNER_SEED'] - game['LOSER_SEED']
        row['SEED_SUM'] = game['WINNER_SEED'] + game['LOSER_SEED']
        row['SEED_PRODUCT'] = game['WINNER_SEED'] * game['LOSER_SEED']
        for feat in ALL_FEATURE_NAMES:
            w_val = w_feats.get(feat)
            l_val = l_feats.get(feat)
            if w_val is not None and l_val is not None:
                row[f'{feat}_DIFF'] = w_val - l_val
        rows.append(row)
        matched += 1
    print(f"  Matched: {matched}, Unmatched: {unmatched}")
    df = pd.DataFrame(rows)
    print(f"  Built training set: {len(df)} games with features")
    return df

# ─── 4. Train model ──────────────────────────────────────────────────────

def train_model(train_df):
    feature_cols = ['SEED_DIFF', 'SEED_SUM', 'SEED_PRODUCT']
    feature_cols += [f'{f}_DIFF' for f in ALL_FEATURE_NAMES if f'{f}_DIFF' in train_df.columns]
    nan_pcts = train_df[feature_cols].isna().mean()
    good_cols = [c for c in feature_cols if nan_pcts[c] < 0.3]
    dropped_cols = set(feature_cols) - set(good_cols)
    if dropped_cols:
        print(f"  Dropped {len(dropped_cols)} features with >30% NaN")
    feature_cols = good_cols
    train_clean = train_df[feature_cols].fillna(0)
    X = train_clean.values
    X_flip = -X.copy()
    sum_idx = feature_cols.index('SEED_SUM') if 'SEED_SUM' in feature_cols else None
    prod_idx = feature_cols.index('SEED_PRODUCT') if 'SEED_PRODUCT' in feature_cols else None
    if sum_idx is not None:
        X_flip[:, sum_idx] = X[:, sum_idx]
    if prod_idx is not None:
        X_flip[:, prod_idx] = X[:, prod_idx]
    X_full = np.vstack([X, X_flip])
    y_full = np.concatenate([np.ones(len(X)), np.zeros(len(X))])
    rng = np.random.RandomState(42)
    idx = rng.permutation(len(X_full))
    X_full, y_full = X_full[idx], y_full[idx]
    print(f"  Training on {len(X_full)} samples, {len(feature_cols)} features")

    base_estimators = [
        ('lr', Pipeline([('scaler', StandardScaler()), ('lr', LogisticRegression(C=0.5, max_iter=2000))])),
        ('gb', GradientBoostingClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                                          subsample=0.8, min_samples_leaf=10, random_state=42)),
        ('rf', RandomForestClassifier(n_estimators=300, max_depth=6, min_samples_leaf=10, random_state=42)),
        ('hgb', HistGradientBoostingClassifier(max_iter=300, max_depth=5, learning_rate=0.05,
                                                min_samples_leaf=20, random_state=42)),
    ]

    model = StackingClassifier(
        estimators=base_estimators,
        final_estimator=LogisticRegression(C=1.0, max_iter=1000),
        cv=3, stack_method='predict_proba', passthrough=False, n_jobs=-1
    )

    X_tr, X_te, y_tr, y_te = train_test_split(X_full, y_full, test_size=0.2, random_state=42, stratify=y_full)
    model.fit(X_tr, y_tr)
    holdout_acc = model.score(X_te, y_te)
    holdout_proba = model.predict_proba(X_te)[:, 1]
    holdout_brier = brier_score_loss(y_te, holdout_proba)
    print(f"  Holdout Accuracy: {holdout_acc:.4f}")
    print(f"  Holdout Brier Score: {holdout_brier:.4f}")
    model.fit(X_full, y_full)

    gb_model = model.estimators_[1]
    importances = gb_model.feature_importances_
    feat_imp = sorted(zip(feature_cols, importances), key=lambda x: -x[1])
    print(f"\n  Top 15 Feature Importances:")
    for name, imp in feat_imp[:15]:
        bar = "\u2588" * int(imp * 200)
        print(f"  {name:<30} {imp:>10.4f} {bar}")

    return model, feature_cols, feat_imp, holdout_acc * 100

# ─── 5. Predict matchup ──────────────────────────────────────────────────

def predict_matchup(model, feature_cols, t1_feats, t2_feats, seed1, seed2):
    row = {}
    row['SEED_DIFF'] = seed1 - seed2
    row['SEED_SUM'] = seed1 + seed2
    row['SEED_PRODUCT'] = seed1 * seed2
    for feat in ALL_FEATURE_NAMES:
        col = f'{feat}_DIFF'
        if col in feature_cols:
            v1 = t1_feats.get(feat, 0) if t1_feats.get(feat) is not None else 0
            v2 = t2_feats.get(feat, 0) if t2_feats.get(feat) is not None else 0
            row[col] = v1 - v2
    X = np.array([[row.get(f, 0) for f in feature_cols]])
    return model.predict_proba(X)[0][1]

def _get_prob(model, feature_cols, t1, t2):
    if t1['feats'] is not None and t2['feats'] is not None:
        return predict_matchup(model, feature_cols, t1['feats'], t2['feats'], t1['seed'], t2['seed'])
    else:
        prob = 0.5 + (t2['seed'] - t1['seed']) * 0.03
        return max(0.05, min(0.95, prob))

# ─── 6. Bracket and simulation ───────────────────────────────────────────

def get_2026_bracket():
    df = pd.read_csv(f"{DATA_DIR}/Tournament Matchups.csv")
    r64 = df[(df['YEAR'] == 2026) & (df['CURRENT ROUND'] == 64)].sort_values('BY YEAR NO', ascending=False)
    play_in_matchups = []
    r64_list = r64.to_dict('records')
    r64_matchups = []
    for i in range(0, len(r64_list) - 1, 2):
        r64_matchups.append((r64_list[i], r64_list[i + 1]))
    print(f"  Round of 64 matchups: {len(r64_matchups)}")
    return play_in_matchups, r64_matchups

def build_team_cache(kb, resume, cbb, play_in_matchups, r64_matchups):
    team_cache = {}
    all_teams = set()
    for t1, t2 in play_in_matchups + r64_matchups:
        all_teams.add((t1['TEAM'], t1['TEAM NO'], t1['SEED']))
        all_teams.add((t2['TEAM'], t2['TEAM NO'], t2['SEED']))
    for name, team_no, seed in all_teams:
        feats = get_team_features(name, team_no, 2026, kb, resume, cbb)
        team_cache[name] = {'name': name, 'seed': seed, 'feats': feats, 'team_no': team_no}
    missing = [name for name, data in team_cache.items() if data['feats'] is None]
    if missing:
        print(f"  WARNING: Missing features for {len(missing)} teams: {missing[:5]}")
    return team_cache

# ─── 7. ESPN Score Fetcher ────────────────────────────────────────────────

ESPN_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/scoreboard?dates={date}&groups=100&limit=100"

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
    "NC State Wolfpack": "North Carolina St.",
    "Miami Hurricanes": "Miami FL",
    "Miami (OH) RedHawks": "Miami OH",
    "McNeese Cowboys": "McNeese St.",
    "Kennesaw State Owls": "Kennesaw St.",
    "Utah State Aggies": "Utah St.",
    "LIU Sharks": "LIU Brooklyn",
    "LIU Brooklyn Sharks": "LIU Brooklyn",
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
    "SMU Mustangs": "SMU",
    "UMBC Retrievers": "UMBC",
    "Lehigh Mountain Hawks": "Lehigh",
    "Queens Royals": "Queens",
}

def match_espn_team(display_name, bracket_teams):
    if display_name in ESPN_TO_BRACKET:
        mapped = ESPN_TO_BRACKET[display_name]
        if mapped in bracket_teams:
            return mapped
    parts = display_name.rsplit(" ", 1)
    if len(parts) == 2 and parts[0] in bracket_teams:
        return parts[0]
    for bt in bracket_teams:
        if display_name.startswith(bt + " "):
            return bt
        if bt.replace("St.", "State") in display_name:
            return bt
    return None

def fetch_espn_scores(bracket_teams):
    """Fetch actual Round of 64 results from ESPN."""
    R64_DATES = ["20260319", "20260320"]
    # Also grab R32 dates in case some games have been played
    R32_DATES = ["20260321", "20260322"]

    DATE_TO_ROUND = {
        "20260319": "Round of 64", "20260320": "Round of 64",
        "20260321": "Round of 32", "20260322": "Round of 32",
    }

    all_results = {}

    for d in R64_DATES + R32_DATES:
        print(f"  Fetching {d}...")
        try:
            url = ESPN_URL.format(date=d)
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode())

            events = data.get("events", [])
            for event in events:
                status = event.get("status", {}).get("type", {}).get("name", "")
                if status != "STATUS_FINAL":
                    continue

                # Check tournament game
                notes = event.get("notes", [])
                is_tourney = any("NCAA" in n.get("headline", "") or "March Madness" in n.get("headline", "")
                                 for n in notes)
                if not is_tourney and event.get("season", {}).get("type", 0) != 3:
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

                teams_data = []
                for c in competitors:
                    team_info = c.get("team", {})
                    display_name = team_info.get("displayName", "")
                    score = int(c.get("score", "0"))
                    seed = c.get("curatedRank", {}).get("current", 99)
                    winner_flag = c.get("winner", False)
                    bracket_name = match_espn_team(display_name, bracket_teams)
                    teams_data.append({
                        "espn_name": display_name, "bracket_name": bracket_name,
                        "score": score, "seed": seed, "is_winner": winner_flag,
                    })

                winner = next((t for t in teams_data if t["is_winner"]), None)
                loser = next((t for t in teams_data if not t["is_winner"]), None)
                if not winner or not loser:
                    teams_data.sort(key=lambda x: -x["score"])
                    winner, loser = teams_data[0], teams_data[1]

                if not winner["bracket_name"] or not loser["bracket_name"]:
                    print(f"    WARNING: Could not match: {winner['espn_name']} vs {loser['espn_name']}")
                    continue

                # Determine round from notes
                round_name = None
                for n in notes:
                    headline = n.get("headline", "")
                    if "1st Round" in headline:
                        round_name = "Round of 64"
                    elif "2nd Round" in headline:
                        round_name = "Round of 32"
                    elif "Sweet 16" in headline or "Regional Semifinal" in headline:
                        round_name = "Sweet 16"

                if round_name is None:
                    round_name = DATE_TO_ROUND.get(d, "Round of 64")

                if round_name not in all_results:
                    all_results[round_name] = []

                all_results[round_name].append({
                    "winner": winner["bracket_name"],
                    "winner_seed": winner["seed"],
                    "loser": loser["bracket_name"],
                    "loser_seed": loser["seed"],
                    "winner_score": winner["score"],
                    "loser_score": loser["score"],
                })

        except Exception as e:
            print(f"    Error: {e}")

    return all_results

# ─── 8. Lock actuals and re-predict ──────────────────────────────────────

def advance_with_actuals(actual_results, team_cache, r64_matchups):
    """Use actual R64 results to build the R32 bracket, then predict from there."""
    actual_r64_winners = set()
    if "Round of 64" in actual_results:
        for g in actual_results["Round of 64"]:
            actual_r64_winners.add(g["winner"])

    # Build initial bracket from matchups
    bracket = []
    for t1_data, t2_data in r64_matchups:
        t1 = team_cache.get(t1_data['TEAM'])
        t2 = team_cache.get(t2_data['TEAM'])
        bracket.append(t1)
        bracket.append(t2)

    # Advance through R64 using actual winners
    r32_bracket = []
    r64_locked = []
    for i in range(0, len(bracket) - 1, 2):
        t1, t2 = bracket[i], bracket[i + 1]
        if t1 is None or t2 is None:
            r32_bracket.append(t1 or t2)
            continue

        if t1['name'] in actual_r64_winners:
            winner, loser = t1, t2
        elif t2['name'] in actual_r64_winners:
            winner, loser = t2, t1
        else:
            # No actual result for this game yet, shouldn't happen if R64 is done
            print(f"  WARNING: No R64 result for {t1['name']} vs {t2['name']}, using model")
            prob = _get_prob(model, feature_cols, t1, t2)
            if prob >= 0.5:
                winner, loser = t1, t2
            else:
                winner, loser = t2, t1

        r64_locked.append((winner, loser, None))
        r32_bracket.append(winner)

    return r32_bracket, r64_locked

def predict_from_bracket(model, feature_cols, current_bracket, start_round_size):
    """Predict from a given bracket state forward."""
    round_names = {32: 'Round of 32', 16: 'Sweet 16', 8: 'Elite Eight',
                   4: 'Final Four', 2: 'Championship'}
    predicted = {}
    current = list(current_bracket)
    round_size = start_round_size

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
        predicted[rname] = rnd_results
        current = next_round
        round_size //= 2

    champion = current[0] if current else None
    return predicted, champion

def simulate_from_r32(model, feature_cols, r32_bracket, actual_r32_results, n_sims=10000):
    """Monte Carlo from R32 onward, with any completed R32 games locked in."""
    team_names = list(set(t['name'] for t in r32_bracket if t is not None))

    # Pre-compute pairwise probabilities
    prob_cache = {}
    X_batch = []
    pairs = []
    for i, n1 in enumerate(team_names):
        for n2 in team_names[i+1:]:
            t1 = next(t for t in r32_bracket if t is not None and t['name'] == n1)
            t2 = next(t for t in r32_bracket if t is not None and t['name'] == n2)
            row_vals = []
            for f in feature_cols:
                if f == 'SEED_DIFF':
                    row_vals.append(t1['seed'] - t2['seed'])
                elif f == 'SEED_SUM':
                    row_vals.append(t1['seed'] + t2['seed'])
                elif f == 'SEED_PRODUCT':
                    row_vals.append(t1['seed'] * t2['seed'])
                else:
                    feat_name = f.replace('_DIFF', '')
                    v1 = (t1['feats'] or {}).get(feat_name, 0) or 0
                    v2 = (t2['feats'] or {}).get(feat_name, 0) or 0
                    row_vals.append(v1 - v2)
            X_batch.append(row_vals)
            pairs.append((n1, n2))

    if X_batch:
        X_batch = np.array(X_batch)
        probs = model.predict_proba(X_batch)[:, 1]
        for idx, (n1, n2) in enumerate(pairs):
            prob_cache[(n1, n2)] = probs[idx]
            prob_cache[(n2, n1)] = 1.0 - probs[idx]

    # Build set of locked R32 winners (if any R32 games already played)
    r32_locked_winners = set()
    if actual_r32_results:
        for g in actual_r32_results:
            r32_locked_winners.add(g["winner"])

    def get_prob_cached(n1, n2):
        if n1 == n2:
            return 0.5
        return prob_cache.get((n1, n2), 0.5)

    advancement = {name: {} for name in team_names}
    champion_counts = {}
    rng = np.random.RandomState(42)
    # Max games from R32 onward: 16 + 8 + 4 + 2 + 1 = 31
    all_randoms = rng.random((n_sims, 40))

    print(f"  Running {n_sims:,} simulations from Round of 32...")
    for sim in range(n_sims):
        rand_idx = 0
        randoms = all_randoms[sim]
        current = [t['name'] for t in r32_bracket if t is not None]

        round_size = 32
        while len(current) > 1:
            next_round = []
            rnd_key = round_size // 2
            for i in range(0, len(current) - 1, 2):
                n1, n2 = current[i], current[i + 1]

                # If this is R32 and we have actual results, lock them in
                if round_size == 32 and r32_locked_winners:
                    if n1 in r32_locked_winners:
                        winner = n1
                    elif n2 in r32_locked_winners:
                        winner = n2
                    else:
                        prob = get_prob_cached(n1, n2)
                        winner = n1 if randoms[rand_idx] < prob else n2
                else:
                    prob = get_prob_cached(n1, n2)
                    winner = n1 if randoms[rand_idx] < prob else n2

                rand_idx += 1
                next_round.append(winner)
                advancement[winner][rnd_key] = advancement[winner].get(rnd_key, 0) + 1

            current = next_round
            round_size //= 2

        champ = current[0]
        champion_counts[champ] = champion_counts.get(champ, 0) + 1

    return advancement, champion_counts

# ─── 9. Accuracy report ──────────────────────────────────────────────────

def compare_predictions(actual_results, original_picks_path):
    """Compare actual results against original model predictions."""
    try:
        with open(original_picks_path) as f:
            original_picks = json.load(f)
    except FileNotFoundError:
        print("  Original picks file not found, skipping comparison")
        return None

    report = {}
    for rnd, actual_games in actual_results.items():
        if rnd not in original_picks:
            continue
        predicted_winners = {g["winner"] for g in original_picks[rnd]}
        correct = 0
        wrong = 0
        upsets = []
        for g in actual_games:
            if g["winner"] in predicted_winners:
                correct += 1
            else:
                wrong += 1
                upsets.append(g)
        report[rnd] = {
            "correct": correct, "wrong": wrong,
            "total": correct + wrong,
            "accuracy": correct / (correct + wrong) * 100 if (correct + wrong) > 0 else 0,
            "upsets": upsets
        }
    return report

# ─── 10. HTML Generation ─────────────────────────────────────────────────

def generate_updated_html(r64_locked, predicted_results, champion, advancement,
                          champion_counts, team_cache_by_name, n_sims, feat_imp,
                          cv_accuracy, accuracy_report, actual_r32_results):
    """Generate HTML with actual R64 results and predicted R32+ results."""

    # Figure out which R32 games are actual vs predicted
    r32_actual_winners = set()
    if actual_r32_results:
        for g in actual_r32_results:
            r32_actual_winners.add(g["winner"])

    # Build rounds data
    rounds_data = {}

    # R64: all actual
    if r64_locked:
        games = []
        for winner, loser, wp in r64_locked:
            games.append({
                'winner': winner['name'], 'winner_seed': winner['seed'],
                'loser': loser['name'], 'loser_seed': loser['seed'],
                'prob': None, 'actual': True,
            })
        rounds_data['Round of 64'] = games

    # R32 through Championship: predicted (or actual if already played)
    for rname in ['Round of 32', 'Sweet 16', 'Elite Eight', 'Final Four', 'Championship']:
        if rname in predicted_results:
            games = []
            for winner, loser, wp in predicted_results[rname]:
                is_actual = (rname == 'Round of 32' and winner['name'] in r32_actual_winners)
                games.append({
                    'winner': winner['name'], 'winner_seed': winner['seed'],
                    'loser': loser['name'], 'loser_seed': loser['seed'],
                    'prob': round(wp * 100, 1) if wp is not None and not is_actual else None,
                    'actual': is_actual,
                })
            rounds_data[rname] = games

    # Championship and Final Four odds
    champ_sorted = sorted(champion_counts.items(), key=lambda x: -x[1])[:15]
    champ_odds = [{'team': t, 'seed': team_cache_by_name[t]['seed'],
                   'pct': round(c / n_sims * 100, 1)} for t, c in champ_sorted
                  if t in team_cache_by_name]

    f4 = {}
    for name, counts in advancement.items():
        f4_count = counts.get(4, 0)
        if f4_count > 0:
            f4[name] = f4_count / n_sims * 100
    f4_sorted = sorted(f4.items(), key=lambda x: -x[1])[:15]
    f4_odds = [{'team': t, 'seed': team_cache_by_name[t]['seed'],
                'pct': round(p, 1)} for t, p in f4_sorted
               if t in team_cache_by_name]

    feat_data = [{'name': n.replace('_DIFF', ''), 'importance': round(v * 100, 2)}
                 for n, v in feat_imp[:15]]

    # Accuracy summary
    acc_html = ""
    if accuracy_report:
        for rnd, report in accuracy_report.items():
            acc_html += f"""
    <div class="accuracy-row">
      <span class="acc-round">{rnd}</span>
      <span class="acc-score">{report['correct']}/{report['total']}</span>
      <span class="acc-pct">{report['accuracy']:.1f}%</span>
    </div>"""

    # Count completed rounds
    completed_rounds = ["Round of 64"]
    if actual_r32_results and len(actual_r32_results) >= 16:
        completed_rounds.append("Round of 32")

    completed_str = ", ".join(completed_rounds)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>March Madness 2026 - Updated Bracket (Round of 32)</title>
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
    font-size: 3rem; font-weight: 900;
    background: linear-gradient(135deg, var(--accent), var(--accent2));
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin-bottom: 0.5rem;
  }}

  .hero .subtitle {{ font-size: 1.1rem; color: var(--text-dim); margin-bottom: 1.5rem; }}

  .hero .model-badge {{
    display: inline-block; background: var(--card); border: 1px solid var(--card-border);
    border-radius: 2rem; padding: 0.5rem 1.5rem; font-size: 0.85rem; color: var(--accent3);
  }}

  .container {{ max-width: 1200px; margin: 0 auto; padding: 2rem; }}

  .update-banner {{
    background: var(--actual-bg); border: 1px solid var(--actual-border);
    border-radius: 0.75rem; padding: 1rem 1.5rem; margin: 1.5rem 0;
    font-size: 0.9rem; display: flex; align-items: center; gap: 0.75rem;
  }}
  .update-banner .icon {{ font-size: 1.2rem; }}

  .section-title {{
    font-size: 1.5rem; font-weight: 800; margin: 2.5rem 0 1rem;
    padding-bottom: 0.5rem; border-bottom: 2px solid var(--accent); display: inline-block;
  }}

  .champion-card {{
    background: linear-gradient(135deg, #1a0f00 0%, #2a1800 100%);
    border: 2px solid var(--accent); border-radius: 1rem;
    padding: 2rem; text-align: center; margin: 2rem 0;
    position: relative; overflow: hidden;
  }}
  .champion-card::before {{
    content: ''; position: absolute; top: -50%; left: -50%; width: 200%; height: 200%;
    background: radial-gradient(circle, rgba(249,115,22,0.1) 0%, transparent 60%);
  }}
  .champion-card .label {{
    font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.15em;
    color: var(--accent); margin-bottom: 0.5rem;
  }}
  .champion-card .team {{ font-size: 2.5rem; font-weight: 900; color: var(--accent); }}
  .champion-card .seed {{ font-size: 1rem; color: var(--text-dim); margin-top: 0.25rem; }}

  .odds-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 2rem; margin: 1rem 0; }}
  @media (max-width: 768px) {{ .odds-grid {{ grid-template-columns: 1fr; }} .hero h1 {{ font-size: 2rem; }} }}

  .odds-card {{
    background: var(--card); border: 1px solid var(--card-border);
    border-radius: 0.75rem; padding: 1.5rem;
  }}
  .odds-card h3 {{ font-size: 1.1rem; font-weight: 700; margin-bottom: 1rem; color: var(--accent2); }}

  .odds-row {{ display: flex; align-items: center; padding: 0.35rem 0; border-bottom: 1px solid rgba(255,255,255,0.05); }}
  .odds-row:last-child {{ border-bottom: none; }}
  .odds-seed {{ width: 2rem; font-size: 0.75rem; color: var(--text-dim); text-align: center; }}
  .odds-team {{ flex: 1; font-weight: 600; font-size: 0.9rem; }}
  .odds-pct {{ width: 3.5rem; text-align: right; font-weight: 700; font-size: 0.9rem; }}
  .odds-bar-wrap {{ width: 120px; height: 8px; background: rgba(255,255,255,0.05); border-radius: 4px; margin-left: 0.75rem; overflow: hidden; }}
  .odds-bar {{ height: 100%; border-radius: 4px; }}
  .odds-bar.champ {{ background: linear-gradient(90deg, var(--accent), #fb923c); }}
  .odds-bar.f4 {{ background: linear-gradient(90deg, var(--accent2), #60a5fa); }}

  .accuracy-card {{
    background: var(--card); border: 1px solid var(--card-border);
    border-radius: 0.75rem; padding: 1.5rem; margin: 1.5rem 0;
  }}
  .accuracy-card h3 {{ font-size: 1.1rem; font-weight: 700; margin-bottom: 1rem; color: var(--accent2); }}
  .accuracy-row {{
    display: flex; align-items: center; padding: 0.5rem 0;
    border-bottom: 1px solid rgba(255,255,255,0.05);
  }}
  .accuracy-row:last-child {{ border-bottom: none; }}
  .acc-round {{ flex: 1; font-weight: 600; }}
  .acc-score {{ width: 60px; text-align: center; color: var(--text-dim); }}
  .acc-pct {{ width: 60px; text-align: right; font-weight: 700; color: var(--accent3); }}

  .bracket-round {{
    background: var(--card); border: 1px solid var(--card-border);
    border-radius: 0.75rem; padding: 1.5rem; margin: 1.5rem 0;
  }}
  .bracket-round.actual-round {{
    border-color: var(--actual-border);
    background: linear-gradient(135deg, var(--card) 0%, rgba(16, 185, 129, 0.04) 100%);
  }}
  .bracket-round h3 {{ font-size: 1.1rem; font-weight: 700; margin-bottom: 1rem; color: var(--accent2); }}
  .bracket-round h3 .badge {{
    display: inline-block; font-size: 0.7rem; font-weight: 600;
    padding: 0.15rem 0.5rem; border-radius: 1rem; margin-left: 0.5rem; vertical-align: middle;
  }}
  .badge-actual {{ background: rgba(16, 185, 129, 0.15); color: var(--accent3); }}
  .badge-predicted {{ background: rgba(59, 130, 246, 0.15); color: var(--accent2); }}

  .matchup {{
    display: flex; align-items: center; padding: 0.4rem 0.75rem;
    margin: 0.25rem 0; border-radius: 0.5rem; background: rgba(255,255,255,0.02);
  }}
  .matchup:hover {{ background: rgba(255,255,255,0.05); }}
  .matchup .winner {{ color: var(--winner); font-weight: 700; }}
  .matchup .loser {{ color: var(--text-dim); }}
  .matchup.actual-game {{
    background: var(--actual-bg); border-left: 3px solid var(--accent3);
  }}
  .matchup .m-seed {{ width: 1.8rem; font-size: 0.75rem; text-align: center; }}
  .matchup .m-team {{ width: 180px; font-size: 0.85rem; }}
  .matchup .m-vs {{ width: 40px; text-align: center; font-size: 0.7rem; color: var(--text-dim); }}
  .matchup .m-prob {{
    font-size: 0.8rem; font-weight: 600; margin-left: auto;
    padding: 0.15rem 0.5rem; border-radius: 1rem; font-variant-numeric: tabular-nums;
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
    flex: 1; height: 12px; background: rgba(255,255,255,0.05);
    border-radius: 6px; overflow: hidden; margin: 0 1rem;
  }}
  .feat-bar {{ height: 100%; background: linear-gradient(90deg, var(--accent2), var(--accent3)); border-radius: 6px; }}
  .feat-val {{ width: 50px; text-align: right; font-size: 0.8rem; color: var(--text-dim); }}

  .methodology {{
    background: var(--card); border: 1px solid var(--card-border);
    border-radius: 0.75rem; padding: 2rem; margin: 2rem 0;
    font-size: 0.9rem; line-height: 1.8;
  }}
  .methodology h3 {{ color: var(--accent2); margin-bottom: 1rem; }}
  .methodology ul {{ padding-left: 1.5rem; margin: 0.5rem 0; }}
  .methodology li {{ margin: 0.3rem 0; }}

  .footer {{
    text-align: center; padding: 2rem; color: var(--text-dim); font-size: 0.8rem;
    border-top: 1px solid var(--card-border); margin-top: 3rem;
  }}
  .footer a {{ color: var(--accent2); text-decoration: none; }}
</style>
</head>
<body>

<div class="hero">
  <h1>March Madness 2026</h1>
  <p class="subtitle">Updated Predictions: Round of 32</p>
  <span class="model-badge">Stacked Ensemble &middot; {cv_accuracy:.1f}% accuracy &middot; {n_sims:,} simulations</span>
</div>

<div class="container">

  <div class="update-banner">
    <span class="icon">&#9989;</span>
    <div>
      <strong>Bracket updated with real results.</strong>
      Completed: {completed_str}. Remaining rounds are ML predictions recalculated from actual outcomes.
    </div>
  </div>
"""

    # Accuracy card
    if acc_html:
        html += f"""
  <div class="accuracy-card">
    <h3>Model Accuracy vs Actual Results</h3>
    {acc_html}
  </div>
"""

    html += f"""
  <div class="champion-card">
    <div class="label">Updated Predicted Champion</div>
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

    # Render each round
    for rname in ['Round of 64', 'Round of 32', 'Sweet 16',
                  'Elite Eight', 'Final Four', 'Championship']:
        if rname not in rounds_data:
            continue
        games = rounds_data[rname]
        all_actual = all(g['actual'] for g in games)
        round_class = "bracket-round actual-round" if all_actual else "bracket-round"
        badge = '<span class="badge badge-actual">ACTUAL</span>' if all_actual else '<span class="badge badge-predicted">PREDICTED</span>'

        html += f'  <div class="{round_class}"><h3>{rname} {badge}</h3>\n'
        for g in games:
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
    Updated with actual Round of 64 results. Built with Python, scikit-learn, and too much coffee.<br>
    <a href="https://jasonhorne.org">jasonhorne.org</a> &middot; March 2026
  </div>

</div>
</body>
</html>"""

    return html

# ─── MAIN ─────────────────────────────────────────────────────────────────

print("=" * 60)
print("  MARCH MADNESS 2026 - ROUND OF 32 PREDICTIONS")
print("  (Updated with actual Round of 64 results)")
print("=" * 60)

# Step 1: Train model (same as before)
print("\n[1/7] Loading data...")
kb, resume, cbb = load_team_data()
games = load_matchups()

print("\n[2/7] Building training features...")
train_df = build_training_data(games, kb, resume, cbb)

print("\n[3/7] Training model...")
model, feature_cols, feat_imp, cv_accuracy = train_model(train_df)

print("\n[4/7] Loading 2026 bracket...")
play_in_matchups, r64_matchups = get_2026_bracket()

# Build team cache
team_cache = build_team_cache(kb, resume, cbb, play_in_matchups, r64_matchups)

# Step 2: Fetch actual results from ESPN
print("\n[5/7] Fetching actual results from ESPN...")
bracket_team_names = set(team_cache.keys())
actual_results = fetch_espn_scores(bracket_team_names)

for rnd, games_list in actual_results.items():
    print(f"  {rnd}: {len(games_list)} games")

if "Round of 64" not in actual_results or len(actual_results["Round of 64"]) < 32:
    r64_count = len(actual_results.get("Round of 64", []))
    print(f"\n  WARNING: Only {r64_count}/32 Round of 64 games found.")
    print("  Results may be incomplete. The model will use predictions for missing games.")

# Step 3: Lock R64 actuals and build R32 bracket
print("\n[6/7] Locking actual results and re-predicting...")
r32_bracket, r64_locked = advance_with_actuals(actual_results, team_cache, r64_matchups)
print(f"  R32 bracket: {len(r32_bracket)} teams")

# Get any actual R32 results (games already played today)
actual_r32 = actual_results.get("Round of 32", [])
if actual_r32:
    print(f"  Found {len(actual_r32)} completed Round of 32 games")

# Predict R32 through Championship
predicted_results, champion = predict_from_bracket(model, feature_cols, r32_bracket, 32)

# Compare with original predictions
original_picks_path = f"{OUT_DIR}/bracket_picks.json"
accuracy_report = compare_predictions(actual_results, original_picks_path)
if accuracy_report:
    for rnd, report in accuracy_report.items():
        print(f"  {rnd}: {report['correct']}/{report['total']} correct ({report['accuracy']:.1f}%)")
        for g in report['upsets']:
            print(f"    MISSED: ({g['winner_seed']}) {g['winner']} beat our pick")

# Step 4: Monte Carlo simulation from R32
print("\n[7/7] Simulating tournament from Round of 32...")
N_SIMS = 10000
advancement, champion_counts = simulate_from_r32(
    model, feature_cols, r32_bracket, actual_r32, n_sims=N_SIMS
)

# Print updated bracket
print(f"\n{'=' * 60}")
print("  UPDATED BRACKET")
print(f"{'=' * 60}")

print("\n  Round of 64 [ACTUAL]:")
for winner, loser, wp in r64_locked:
    print(f"    ({winner['seed']:>2}) {winner['name']:<22} over ({loser['seed']:>2}) {loser['name']}")

for rname in ['Round of 32', 'Sweet 16', 'Elite Eight', 'Final Four', 'Championship']:
    if rname in predicted_results:
        print(f"\n  {rname} [PREDICTED]:")
        for winner, loser, wp in predicted_results[rname]:
            print(f"    ({winner['seed']:>2}) {winner['name']:<22} over "
                  f"({loser['seed']:>2}) {loser['name']:<22} ({wp:.1%})")

print(f"\n  {'=' * 55}")
print(f"  UPDATED PREDICTED CHAMPION: ({champion['seed']}) {champion['name']}")
champ_pct = champion_counts.get(champion['name'], 0) / N_SIMS * 100
print(f"  Championship probability: {champ_pct:.1f}%")
print(f"  {'=' * 55}")

# Updated championship odds
print("\n  UPDATED CHAMPIONSHIP ODDS:")
champ_sorted = sorted(champion_counts.items(), key=lambda x: -x[1])
for team, count in champ_sorted[:15]:
    seed = team_cache[team]['seed']
    pct = count / N_SIMS * 100
    bar = "\u2588" * int(pct * 1.5)
    print(f"    ({seed:>2}) {team:<25} {pct:5.1f}%  {bar}")

# Generate HTML
html = generate_updated_html(
    r64_locked, predicted_results, champion, advancement,
    champion_counts, team_cache, N_SIMS, feat_imp, cv_accuracy,
    accuracy_report, actual_r32
)

html_path = f"{OUT_DIR}/march_madness_2026_r32.html"
with open(html_path, 'w') as f:
    f.write(html)
print(f"\n  HTML saved to: {html_path}")

# Save updated picks JSON
all_picks = {}
# R64 actuals
all_picks["Round of 64"] = [
    {"winner": w['name'], "winner_seed": w['seed'],
     "loser": l['name'], "loser_seed": l['seed'],
     "actual": True}
    for w, l, _ in r64_locked
]
# Predicted rounds
for rname in ['Round of 32', 'Sweet 16', 'Elite Eight', 'Final Four', 'Championship']:
    if rname in predicted_results:
        all_picks[rname] = [
            {"winner": w['name'], "winner_seed": w['seed'],
             "loser": l['name'], "loser_seed": l['seed'],
             "prob": round(wp * 100, 1) if wp else None}
            for w, l, wp in predicted_results[rname]
        ]

picks_path = f"{OUT_DIR}/bracket_picks_r32.json"
with open(picks_path, 'w') as f:
    json.dump(all_picks, f, indent=2)
print(f"  Updated picks saved to: {picks_path}")

# Save results JSON for blog
results_json = {
    'champion': {'name': champion['name'], 'seed': champion['seed']},
    'championship_odds': [{'team': t, 'seed': team_cache[t]['seed'],
                           'pct': round(c / N_SIMS * 100, 1)}
                          for t, c in champ_sorted[:25] if t in team_cache],
    'cv_accuracy': round(cv_accuracy, 1),
    'r64_accuracy': accuracy_report.get('Round of 64', {}).get('accuracy', None) if accuracy_report else None,
    'r64_correct': accuracy_report.get('Round of 64', {}).get('correct', None) if accuracy_report else None,
    'r64_total': accuracy_report.get('Round of 64', {}).get('total', None) if accuracy_report else None,
    'n_simulations': N_SIMS,
    'updated_from': 'Round of 32',
}
results_path = f"{OUT_DIR}/results_r32.json"
with open(results_path, 'w') as f:
    json.dump(results_json, f, indent=2)
print(f"  Results JSON saved to: {results_path}")

# Display inline
from IPython.display import HTML, display
display(HTML(html))

print("\nDone! All outputs saved to Google Drive.")
