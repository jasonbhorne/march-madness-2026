"""
March Madness 2026 Bracket Predictor
====================================
Uses historical tournament data + KenPom/Barttorvik team metrics to train an ML model,
then predicts P(win) for every possible 2026 tournament matchup.

Approach:
1. Build team feature profiles from KenPom Barttorvik (103 columns, 2008-2026)
2. Supplement with Resume data (ELO, NET, quad records)
3. Pair historical tournament matchups with team features (as deltas)
4. Train a stacked ensemble (logistic regression + gradient boosting + random forest)
5. Simulate the 2026 bracket using predicted win probabilities

Blog: jasonhorne.org
"""

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
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = "/Users/hornej/Documents/Research/march-madness-2026/data"
OUT_DIR = "/Users/hornej/Documents/Research/march-madness-2026"

# ─── 1. Features to extract ─────────────────────────────────────────────

# Core efficiency metrics (KenPom Barttorvik)
KB_FEATURES = [
    'KADJ O',       # KenPom adjusted offensive efficiency
    'KADJ D',       # KenPom adjusted defensive efficiency
    'KADJ EM',      # KenPom adjusted efficiency margin
    'BADJ EM',      # Barttorvik adjusted efficiency margin
    'BADJ O',       # Barttorvik adjusted offense
    'BADJ D',       # Barttorvik adjusted defense
    'BARTHAG',      # Barttorvik power rating (0-1)
    'KADJ T',       # KenPom adjusted tempo
    'WIN%',         # Win percentage
    # Four factors
    'EFG%',         # Effective FG%
    'EFG%D',        # Effective FG% defense
    'FTR',          # Free throw rate
    'FTRD',         # Free throw rate defense
    'TOV%',         # Turnover %
    'TOV%D',        # Opponent turnover %
    'OREB%',        # Offensive rebound %
    'DREB%',        # Defensive rebound %
    # Shooting
    '2PT%',         # 2-point %
    '2PT%D',        # 2-point defense %
    '3PT%',         # 3-point %
    '3PT%D',        # 3-point defense %
    'FT%',          # Free throw %
    # Shot selection
    '2PTR',         # 2-point attempt rate
    '3PTR',         # 3-point attempt rate
    # Advanced
    'BLK%',         # Block %
    'AST%',         # Assist %
    'OP AST%',      # Opponent assist %
    # Points per possession
    'PPPO',         # Points per possession offense
    'PPPD',         # Points per possession defense
    # Schedule / experience
    'ELITE SOS',    # Elite strength of schedule
    'WAB',          # Wins above bubble
    'EXP',          # Experience
    'TALENT',       # Talent rating
    'AVG HGT',      # Average height
    'EFF HGT',      # Effective height
]

# Resume features
RESUME_FEATURES = [
    'ELO',          # ELO rating
    'NET RPI',      # NET/RPI ranking
    'Q1 W',         # Quad 1 wins
    'Q2 W',         # Quad 2 wins
    'Q1 PLUS Q2 W', # Q1+Q2 wins combined
    'Q3 Q4 L',      # Quad 3/4 losses (bad losses)
    'PLUS 500',     # Games above .500
    'R SCORE',      # Resume score
]

# Engineered features (computed from raw)
def engineer_features(row):
    """Create derived features from raw stats."""
    feats = {}
    # Net efficiency margin (the most predictive single metric)
    feats['NET_EFF'] = row.get('KADJ O', 0) - row.get('KADJ D', 0)
    # Offensive vs defensive balance
    kadj_o = row.get('KADJ O', 0)
    kadj_d = row.get('KADJ D', 0)
    feats['OFF_DEF_RATIO'] = kadj_o / max(kadj_d, 1)
    # Four factors composite (Dean Oliver's formula, weighted)
    feats['FOUR_FACTORS_O'] = (
        0.4 * row.get('EFG%', 0) +
        0.25 * (100 - row.get('TOV%', 0)) +
        0.20 * row.get('OREB%', 0) +
        0.15 * row.get('FTR', 0)
    )
    feats['FOUR_FACTORS_D'] = (
        0.4 * (100 - row.get('EFG%D', 50)) +
        0.25 * row.get('TOV%D', 0) +
        0.20 * (100 - row.get('OP OREB%', 50)) +
        0.15 * (100 - row.get('FTRD', 0))
    )
    # Shooting versatility (can score from anywhere)
    feats['SHOOT_VERSATILITY'] = row.get('2PT%', 0) * 0.5 + row.get('3PT%', 0) * 0.5
    # Defensive pressure
    feats['DEF_PRESSURE'] = row.get('BLK%', 0) + row.get('TOV%D', 0)
    # Ball security
    feats['BALL_SECURITY'] = 100 - row.get('TOV%', 0)
    # Rebounding margin
    feats['REB_MARGIN'] = row.get('OREB%', 0) + row.get('DREB%', 0) - 100
    # Quality wins ratio (Q1+Q2 wins relative to total wins)
    q12 = row.get('Q1 PLUS Q2 W', 0)
    total_w = row.get('W', 1)
    feats['QUALITY_WIN_PCT'] = q12 / max(total_w, 1)
    # Bad loss penalty
    feats['BAD_LOSS_RATE'] = row.get('Q3 Q4 L', 0)
    return feats

ENGINEERED_NAMES = [
    'NET_EFF', 'OFF_DEF_RATIO', 'FOUR_FACTORS_O', 'FOUR_FACTORS_D',
    'SHOOT_VERSATILITY', 'DEF_PRESSURE', 'BALL_SECURITY', 'REB_MARGIN',
    'QUALITY_WIN_PCT', 'BAD_LOSS_RATE'
]

# ─── 2. Load data ───────────────────────────────────────────────────────

def load_team_data():
    """Load KenPom Barttorvik + Resume data, merge into one team profile per year."""
    kb = pd.read_csv(f"{DATA_DIR}/KenPom Barttorvik.csv")
    print(f"KenPom Barttorvik: {kb.shape[0]} team-seasons, years {kb['YEAR'].min()}-{kb['YEAR'].max()}")

    resume = pd.read_csv(f"{DATA_DIR}/Resumes.csv")
    print(f"Resumes: {resume.shape[0]} team-seasons")

    # Also load the cbb files for teams NOT in KenPom Barttorvik (covers more teams)
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
    """Parse tournament matchups into game-level rows with winner/loser."""
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
                # Winner advanced further (lower ROUND value) or won by score
                if t1['ROUND'] < t2['ROUND']:
                    winner, loser = t1, t2
                elif t2['ROUND'] < t1['ROUND']:
                    winner, loser = t2, t1
                elif t1['SCORE'] > t2['SCORE']:
                    winner, loser = t1, t2
                else:
                    winner, loser = t2, t1

                games.append({
                    'YEAR': year,
                    'ROUND': rnd,
                    'WINNER': winner['TEAM'],
                    'WINNER_SEED': winner['SEED'],
                    'WINNER_NO': winner['TEAM NO'],
                    'LOSER': loser['TEAM'],
                    'LOSER_SEED': loser['SEED'],
                    'LOSER_NO': loser['TEAM NO'],
                    'WIN_SCORE': winner['SCORE'],
                    'LOSE_SCORE': loser['SCORE'],
                    'MARGIN': winner['SCORE'] - loser['SCORE'],
                })

    games_df = pd.DataFrame(games)
    print(f"Parsed {len(games_df)} tournament games across {games_df['YEAR'].nunique()} years")
    return games_df

# ─── 3. Feature building ────────────────────────────────────────────────

def get_team_features(team_name, team_no, year, kb, resume, cbb):
    """Get full feature vector for a team-year from all data sources."""
    feats = {}

    # Match in KenPom Barttorvik by TEAM NO and YEAR (most reliable)
    kb_match = kb[(kb['YEAR'] == year) & (kb['TEAM NO'] == team_no)]
    if len(kb_match) == 0:
        # Fallback: match by name
        kb_match = kb[(kb['YEAR'] == year) & (kb['TEAM'] == team_name)]

    if len(kb_match) > 0:
        row = kb_match.iloc[0]
        for f in KB_FEATURES:
            if f in row.index:
                feats[f] = row[f]
        feats['W'] = row.get('W', 0)
        feats['OP OREB%'] = row.get('OP OREB%', 0)
        # Add engineered features
        feats.update(engineer_features({**feats, **row.to_dict()}))
    else:
        # Fallback to cbb data
        cbb_match = cbb[(cbb['YEAR'] == year) & (cbb['TEAM'] == team_name)]
        if len(cbb_match) > 0:
            row = cbb_match.iloc[0]
            # Map cbb columns to our feature names
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

    # Add resume features
    res_match = resume[(resume['YEAR'] == year) & (resume['TEAM NO'] == team_no)]
    if len(res_match) == 0:
        res_match = resume[(resume['YEAR'] == year) & (resume['TEAM'] == team_name)]
    if len(res_match) > 0:
        row = res_match.iloc[0]
        for f in RESUME_FEATURES:
            if f in row.index:
                feats[f] = row[f]
        # Re-compute quality features with resume data
        if 'Q1 PLUS Q2 W' in feats:
            feats['QUALITY_WIN_PCT'] = feats['Q1 PLUS Q2 W'] / max(feats.get('W', 1), 1)
        if 'Q3 Q4 L' in feats:
            feats['BAD_LOSS_RATE'] = feats['Q3 Q4 L']

    return feats if feats else None

ALL_FEATURE_NAMES = KB_FEATURES + RESUME_FEATURES + ENGINEERED_NAMES

def build_training_data(games, kb, resume, cbb):
    """For each tournament game, compute feature deltas between winner and loser."""
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

        # Seed-based features
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

# ─── 4. Train model ─────────────────────────────────────────────────────

def train_model(train_df):
    """Train a stacked ensemble to predict P(team with features wins)."""
    feature_cols = ['SEED_DIFF', 'SEED_SUM', 'SEED_PRODUCT']
    feature_cols += [f'{f}_DIFF' for f in ALL_FEATURE_NAMES if f'{f}_DIFF' in train_df.columns]

    # Drop features with too many NaNs
    nan_pcts = train_df[feature_cols].isna().mean()
    good_cols = [c for c in feature_cols if nan_pcts[c] < 0.3]
    dropped_cols = set(feature_cols) - set(good_cols)
    if dropped_cols:
        print(f"  Dropped {len(dropped_cols)} features with >30% NaN")
    feature_cols = good_cols

    # Fill remaining NaNs with 0 (neutral difference)
    train_clean = train_df[feature_cols].fillna(0)

    X = train_clean.values
    # Create balanced dataset: both orientations
    X_flip = -X.copy()
    # Don't flip SEED_SUM and SEED_PRODUCT (they're symmetric)
    sum_idx = feature_cols.index('SEED_SUM') if 'SEED_SUM' in feature_cols else None
    prod_idx = feature_cols.index('SEED_PRODUCT') if 'SEED_PRODUCT' in feature_cols else None
    if sum_idx is not None:
        X_flip[:, sum_idx] = X[:, sum_idx]
    if prod_idx is not None:
        X_flip[:, prod_idx] = X[:, prod_idx]

    X_full = np.vstack([X, X_flip])
    y_full = np.concatenate([np.ones(len(X)), np.zeros(len(X))])

    # Shuffle
    rng = np.random.RandomState(42)
    idx = rng.permutation(len(X_full))
    X_full, y_full = X_full[idx], y_full[idx]

    print(f"  Training on {len(X_full)} samples, {len(feature_cols)} features")

    # Stacked ensemble
    base_estimators = [
        ('lr', Pipeline([
            ('scaler', StandardScaler()),
            ('lr', LogisticRegression(C=0.5, max_iter=2000))
        ])),
        ('gb', GradientBoostingClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, min_samples_leaf=10, random_state=42
        )),
        ('rf', RandomForestClassifier(
            n_estimators=300, max_depth=6, min_samples_leaf=10,
            random_state=42
        )),
        ('hgb', HistGradientBoostingClassifier(
            max_iter=300, max_depth=5, learning_rate=0.05,
            min_samples_leaf=20, random_state=42
        )),
    ]

    model = StackingClassifier(
        estimators=base_estimators,
        final_estimator=LogisticRegression(C=1.0, max_iter=1000),
        cv=3,
        stack_method='predict_proba',
        passthrough=False,
        n_jobs=2  # was -1, which spawns too many processes and kills the machine
    )

    # Quick holdout check instead of full CV (stacking already does internal CV)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_full, y_full, test_size=0.2, random_state=42, stratify=y_full
    )
    model.fit(X_tr, y_tr)
    holdout_acc = model.score(X_te, y_te)
    holdout_proba = model.predict_proba(X_te)[:, 1]
    holdout_brier = brier_score_loss(y_te, holdout_proba)
    print(f"  Holdout Accuracy: {holdout_acc:.4f}")
    print(f"  Holdout Brier Score: {holdout_brier:.4f}")

    # Refit on all data for final predictions
    model.fit(X_full, y_full)

    # Feature importance from gradient boosting
    gb_model = model.estimators_[1]  # The GB estimator
    importances = gb_model.feature_importances_
    feat_imp = sorted(zip(feature_cols, importances), key=lambda x: -x[1])

    print(f"\n  Top 20 Feature Importances (Gradient Boosting):")
    print(f"  {'Feature':<30} {'Importance':>10}")
    print(f"  {'-'*40}")
    for name, imp in feat_imp[:20]:
        bar = "█" * int(imp * 200)
        print(f"  {name:<30} {imp:>10.4f} {bar}")

    return model, feature_cols, feat_imp, holdout_acc * 100

# ─── 5. Predict matchup ─────────────────────────────────────────────────

def predict_matchup(model, feature_cols, t1_feats, t2_feats, seed1, seed2):
    """Predict P(team1 wins)."""
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
    prob = model.predict_proba(X)[0][1]
    return prob

# ─── 6. Bracket extraction and simulation ───────────────────────────────

def get_2026_bracket():
    """Extract the 2026 bracket, handling play-in games correctly."""
    df = pd.read_csv(f"{DATA_DIR}/Tournament Matchups.csv")
    bracket = df[df['YEAR'] == 2026].sort_values('BY YEAR NO', ascending=False)

    # First, handle play-in games (CURRENT ROUND == 16 means it's a play-in)
    # Actually, looking at the data: CURRENT ROUND 16 has 16 entries = 8 play-in games
    # CURRENT ROUND 64 has 72 entries = 36 R64 games (but some are play-in winners)
    # The play-in teams appear twice in R64
    # We need to: identify play-in matchups, then build R64 with play-in placeholders

    play_in = bracket[bracket['CURRENT ROUND'] == 16].sort_values('BY YEAR NO', ascending=False)
    r64 = bracket[bracket['CURRENT ROUND'] == 64].sort_values('BY YEAR NO', ascending=False)

    # Find teams that appear twice in R64 (they're in play-in games)
    r64_teams = r64['TEAM'].value_counts()
    play_in_team_names = set(r64_teams[r64_teams > 1].index)

    # Build play-in matchups
    play_in_matchups = []
    pi_list = play_in.to_dict('records')
    for i in range(0, len(pi_list) - 1, 2):
        play_in_matchups.append((pi_list[i], pi_list[i + 1]))

    # Build R64 matchups, deduplicating play-in teams (keep first occurrence)
    seen_play_in = set()
    r64_deduped = []
    for _, row in r64.iterrows():
        if row['TEAM'] in play_in_team_names:
            if row['TEAM'] not in seen_play_in:
                seen_play_in.add(row['TEAM'])
                r64_deduped.append(row.to_dict())
        else:
            r64_deduped.append(row.to_dict())

    # Pair into matchups
    r64_matchups = []
    for i in range(0, len(r64_deduped) - 1, 2):
        r64_matchups.append((r64_deduped[i], r64_deduped[i + 1]))

    print(f"  Play-in games: {len(play_in_matchups)}")
    print(f"  Round of 64 matchups: {len(r64_matchups)}")

    return play_in_matchups, r64_matchups

def simulate_bracket(model, feature_cols, kb, resume, cbb, play_in_matchups, r64_matchups, n_sims=10000):
    """Simulate the full tournament bracket n_sims times.
    Pre-computes all pairwise probabilities for speed."""
    # Pre-compute all team features
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

    # Pre-compute pairwise win probabilities for all team pairs
    print("  Pre-computing pairwise probabilities...")
    team_names = list(team_cache.keys())
    prob_cache = {}
    pairs_to_compute = []
    pair_keys = []

    for i, n1 in enumerate(team_names):
        for n2 in team_names[i+1:]:
            pairs_to_compute.append((n1, n2))

    # Batch predict: build feature matrix for all pairs
    if pairs_to_compute:
        X_batch = []
        for n1, n2 in pairs_to_compute:
            t1, t2 = team_cache[n1], team_cache[n2]
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

        X_batch = np.array(X_batch)
        probs = model.predict_proba(X_batch)[:, 1]

        for idx, (n1, n2) in enumerate(pairs_to_compute):
            prob_cache[(n1, n2)] = probs[idx]
            prob_cache[(n2, n1)] = 1.0 - probs[idx]

    print(f"  Cached {len(prob_cache)} pairwise probabilities")

    def get_prob_cached(name1, name2):
        if name1 == name2:
            return 0.5
        return prob_cache.get((name1, name2), 0.5)

    all_team_names_set = set(team_names)
    advancement = {name: {} for name in team_names}
    champion_counts = {}

    # Pre-generate all random numbers needed
    # Max games per sim: 8 play-in + 32 + 16 + 8 + 4 + 2 + 1 = 71
    rng = np.random.RandomState(42)
    all_randoms = rng.random((n_sims, 75))

    print(f"  Running {n_sims:,} simulations...")
    for sim in range(n_sims):
        rand_idx = 0
        randoms = all_randoms[sim]

        # Resolve play-in games
        play_in_winners = {}
        for t1_data, t2_data in play_in_matchups:
            n1, n2 = t1_data['TEAM'], t2_data['TEAM']
            prob = get_prob_cached(n1, n2)
            winner_name = n1 if randoms[rand_idx] < prob else n2
            rand_idx += 1
            play_in_winners[n1] = winner_name
            play_in_winners[n2] = winner_name

        # Build R64 with resolved play-ins
        bracket_names = []
        for t1_data, t2_data in r64_matchups:
            n1 = play_in_winners.get(t1_data['TEAM'], t1_data['TEAM'])
            n2 = play_in_winners.get(t2_data['TEAM'], t2_data['TEAM'])
            bracket_names.append(n1)
            bracket_names.append(n2)

        # Simulate rounds
        current = list(bracket_names)
        round_size = 64
        while len(current) > 1:
            next_round = []
            rnd_key = round_size // 2
            for i in range(0, len(current) - 1, 2):
                n1, n2 = current[i], current[i + 1]
                prob = get_prob_cached(n1, n2)
                winner = n1 if randoms[rand_idx] < prob else n2
                rand_idx += 1
                next_round.append(winner)
                advancement[winner][rnd_key] = advancement[winner].get(rnd_key, 0) + 1

            current = next_round
            round_size //= 2

        champ = current[0]
        champion_counts[champ] = champion_counts.get(champ, 0) + 1

    return advancement, champion_counts, team_cache

def _get_prob(model, feature_cols, t1, t2):
    """Get P(t1 wins t2) with fallback."""
    if t1['feats'] is not None and t2['feats'] is not None:
        return predict_matchup(
            model, feature_cols,
            t1['feats'], t2['feats'],
            t1['seed'], t2['seed']
        )
    else:
        # Seed-based fallback
        prob = 0.5 + (t2['seed'] - t1['seed']) * 0.03
        return max(0.05, min(0.95, prob))

def run_deterministic_bracket(model, feature_cols, team_cache, play_in_matchups, r64_matchups):
    """Run a single deterministic bracket, always picking the higher-probability team."""
    round_names = {64: 'Round of 64', 32: 'Round of 32', 16: 'Sweet 16',
                   8: 'Elite Eight', 4: 'Final Four', 2: 'Championship'}

    results = {}  # round_name -> list of (winner, loser, prob)

    # Resolve play-ins
    play_in_winners = {}
    pi_results = []
    for t1_data, t2_data in play_in_matchups:
        t1 = team_cache[t1_data['TEAM']]
        t2 = team_cache[t2_data['TEAM']]
        prob = _get_prob(model, feature_cols, t1, t2)
        if prob >= 0.5:
            winner, loser, wp = t1, t2, prob
        else:
            winner, loser, wp = t2, t1, 1 - prob
        play_in_winners[t1['name']] = winner
        play_in_winners[t2['name']] = winner
        pi_results.append((winner, loser, wp))
    results['Play-In'] = pi_results

    # Build R64
    bracket = []
    for t1_data, t2_data in r64_matchups:
        t1 = team_cache[t1_data['TEAM']]
        t2 = team_cache[t2_data['TEAM']]
        if t1['name'] in play_in_winners:
            t1 = play_in_winners[t1['name']]
        if t2['name'] in play_in_winners:
            t2 = play_in_winners[t2['name']]
        bracket.append(t1)
        bracket.append(t2)

    current = list(bracket)
    round_size = 64

    while len(current) > 1:
        rname = round_names.get(round_size, f"Round of {round_size}")
        rnd_results = []
        next_round = []
        for i in range(0, len(current) - 1, 2):
            t1, t2 = current[i], current[i + 1]
            prob = _get_prob(model, feature_cols, t1, t2)
            if prob >= 0.5:
                winner, loser, wp = t1, t2, prob
            else:
                winner, loser, wp = t2, t1, 1 - prob
            rnd_results.append((winner, loser, wp))
            next_round.append(winner)

        results[rname] = rnd_results
        current = next_round
        round_size //= 2

    champion = current[0]
    return results, champion

# ─── 7. Print results ───────────────────────────────────────────────────

def print_results(advancement, champion_counts, team_cache, n_sims, det_results, champion):
    """Print full results."""
    print("\n" + "=" * 80)
    print(f"  2026 NCAA TOURNAMENT PREDICTIONS")
    print(f"  Model: Stacked Ensemble (LR + GB + RF + HGB)")
    print(f"  Simulations: {n_sims:,}")
    print("=" * 80)

    # Championship odds
    print("\n  CHAMPIONSHIP ODDS:")
    print("  " + "-" * 55)
    champ_sorted = sorted(champion_counts.items(), key=lambda x: -x[1])
    for team, count in champ_sorted[:25]:
        seed = team_cache[team]['seed']
        pct = count / n_sims * 100
        bar = "█" * int(pct * 1.5)
        print(f"    ({seed:>2}) {team:<25} {pct:5.1f}%  {bar}")

    # Final Four odds
    print("\n  FINAL FOUR ODDS:")
    print("  " + "-" * 55)
    f4 = {}
    for name, counts in advancement.items():
        f4_count = counts.get(4, 0)
        if f4_count > 0:
            f4[name] = f4_count / n_sims * 100
    f4_sorted = sorted(f4.items(), key=lambda x: -x[1])[:20]
    for team, pct in f4_sorted:
        seed = team_cache[team]['seed']
        bar = "█" * int(pct / 2)
        print(f"    ({seed:>2}) {team:<25} {pct:5.1f}%  {bar}")

    # Deterministic bracket
    print("\n  PREDICTED BRACKET (most likely outcome):")
    print("  " + "-" * 65)
    for rname in ['Play-In', 'Round of 64', 'Round of 32', 'Sweet 16',
                  'Elite Eight', 'Final Four', 'Championship']:
        if rname in det_results:
            print(f"\n  {rname}:")
            for winner, loser, wp in det_results[rname]:
                print(f"    ({winner['seed']:>2}) {winner['name']:<22} over "
                      f"({loser['seed']:>2}) {loser['name']:<22} ({wp:.1%})")

    print(f"\n  {'='*65}")
    print(f"  PREDICTED CHAMPION: ({champion['seed']}) {champion['name']}")
    print(f"  {'='*65}")

    return champ_sorted

# ─── 8. Silver baseline for comparison ──────────────────────────────────

def silver_wpct(pwr1, pwr2):
    tscore = (pwr1 - pwr2) / 11.0
    return norm.cdf(tscore)

# ─── 9. Generate HTML bracket ───────────────────────────────────────────

def generate_html_bracket(det_results, champion, advancement, champion_counts,
                          team_cache, n_sims, feat_imp, cv_accuracy):
    """Generate a beautiful HTML bracket visualization."""

    # Build round data for JS
    rounds_data = {}
    for rname in ['Round of 64', 'Round of 32', 'Sweet 16',
                  'Elite Eight', 'Final Four', 'Championship']:
        if rname in det_results:
            games = []
            for winner, loser, wp in det_results[rname]:
                games.append({
                    'winner': winner['name'], 'winner_seed': winner['seed'],
                    'loser': loser['name'], 'loser_seed': loser['seed'],
                    'prob': round(wp * 100, 1)
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

    # Feature importance (top 15)
    feat_data = [{'name': n.replace('_DIFF', ''), 'importance': round(v * 100, 2)}
                 for n, v in feat_imp[:15]]

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>March Madness 2026 - ML Bracket Predictions</title>
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

  /* Champion callout */
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
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
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

  /* Odds tables */
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

  .odds-seed {{
    width: 2rem;
    font-size: 0.75rem;
    color: var(--text-dim);
    text-align: center;
  }}

  .odds-team {{
    flex: 1;
    font-weight: 600;
    font-size: 0.9rem;
  }}

  .odds-pct {{
    width: 3.5rem;
    text-align: right;
    font-weight: 700;
    font-size: 0.9rem;
  }}

  .odds-bar-wrap {{
    width: 120px;
    height: 8px;
    background: rgba(255,255,255,0.05);
    border-radius: 4px;
    margin-left: 0.75rem;
    overflow: hidden;
  }}

  .odds-bar {{
    height: 100%;
    border-radius: 4px;
    transition: width 0.3s;
  }}

  .odds-bar.champ {{ background: linear-gradient(90deg, var(--accent), #fb923c); }}
  .odds-bar.f4 {{ background: linear-gradient(90deg, var(--accent2), #60a5fa); }}

  /* Bracket rounds */
  .bracket-round {{
    background: var(--card);
    border: 1px solid var(--card-border);
    border-radius: 0.75rem;
    padding: 1.5rem;
    margin: 1.5rem 0;
  }}

  .bracket-round h3 {{
    font-size: 1.1rem;
    font-weight: 700;
    margin-bottom: 1rem;
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

  .matchup .winner {{
    color: var(--winner);
    font-weight: 700;
  }}

  .matchup .loser {{
    color: var(--text-dim);
  }}

  .matchup .m-seed {{
    width: 1.8rem;
    font-size: 0.75rem;
    text-align: center;
  }}

  .matchup .m-team {{
    width: 180px;
    font-size: 0.85rem;
  }}

  .matchup .m-vs {{
    width: 40px;
    text-align: center;
    font-size: 0.7rem;
    color: var(--text-dim);
  }}

  .matchup .m-prob {{
    font-size: 0.8rem;
    font-weight: 600;
    margin-left: auto;
    padding: 0.15rem 0.5rem;
    border-radius: 1rem;
    font-variant-numeric: tabular-nums;
  }}

  .prob-high {{ background: rgba(16,185,129,0.15); color: var(--prob-high); }}
  .prob-mid {{ background: rgba(245,158,11,0.15); color: var(--prob-mid); }}
  .prob-low {{ background: rgba(239,68,68,0.15); color: var(--prob-low); }}

  /* Feature importance */
  .feat-row {{
    display: flex;
    align-items: center;
    padding: 0.4rem 0;
  }}

  .feat-name {{
    width: 200px;
    font-size: 0.85rem;
    font-weight: 500;
  }}

  .feat-bar-wrap {{
    flex: 1;
    height: 12px;
    background: rgba(255,255,255,0.05);
    border-radius: 6px;
    overflow: hidden;
    margin: 0 1rem;
  }}

  .feat-bar {{
    height: 100%;
    background: linear-gradient(90deg, var(--accent2), var(--accent3));
    border-radius: 6px;
  }}

  .feat-val {{
    width: 50px;
    text-align: right;
    font-size: 0.8rem;
    color: var(--text-dim);
  }}

  .methodology {{
    background: var(--card);
    border: 1px solid var(--card-border);
    border-radius: 0.75rem;
    padding: 2rem;
    margin: 2rem 0;
    font-size: 0.9rem;
    line-height: 1.8;
  }}

  .methodology h3 {{
    color: var(--accent2);
    margin-bottom: 1rem;
  }}

  .methodology ul {{
    padding-left: 1.5rem;
    margin: 0.5rem 0;
  }}

  .methodology li {{
    margin: 0.3rem 0;
  }}

  .footer {{
    text-align: center;
    padding: 2rem;
    color: var(--text-dim);
    font-size: 0.8rem;
    border-top: 1px solid var(--card-border);
    margin-top: 3rem;
  }}

  .footer a {{ color: var(--accent2); text-decoration: none; }}
</style>
</head>
<body>

<div class="hero">
  <h1>March Madness 2026</h1>
  <p class="subtitle">Machine Learning Bracket Predictions</p>
  <span class="model-badge">Stacked Ensemble &middot; {cv_accuracy:.1f}% accuracy &middot; {n_sims:,} simulations</span>
</div>

<div class="container">

  <div class="champion-card">
    <div class="label">Predicted Champion</div>
    <div class="team">{champion['name']}</div>
    <div class="seed">#{champion['seed']} Seed &middot; {round(champion_counts.get(champion['name'], 0) / n_sims * 100, 1)}% probability</div>
  </div>

  <div class="odds-grid">
    <div class="odds-card">
      <h3>Championship Odds</h3>
      {''.join(f"""<div class="odds-row">
        <span class="odds-seed">{o['seed']}</span>
        <span class="odds-team">{o['team']}</span>
        <span class="odds-pct">{o['pct']}%</span>
        <div class="odds-bar-wrap"><div class="odds-bar champ" style="width:{min(o['pct'] / champ_odds[0]['pct'] * 100, 100)}%"></div></div>
      </div>""" for o in champ_odds)}
    </div>
    <div class="odds-card">
      <h3>Final Four Odds</h3>
      {''.join(f"""<div class="odds-row">
        <span class="odds-seed">{o['seed']}</span>
        <span class="odds-team">{o['team']}</span>
        <span class="odds-pct">{o['pct']}%</span>
        <div class="odds-bar-wrap"><div class="odds-bar f4" style="width:{min(o['pct'] / f4_odds[0]['pct'] * 100, 100)}%"></div></div>
      </div>""" for o in f4_odds)}
    </div>
  </div>

  <div class="section-title">Predicted Bracket</div>
"""

    # Add each round
    for rname in ['Round of 64', 'Round of 32', 'Sweet 16',
                  'Elite Eight', 'Final Four', 'Championship']:
        if rname not in rounds_data:
            continue
        html += f'  <div class="bracket-round"><h3>{rname}</h3>\n'
        for g in rounds_data[rname]:
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

    html += """  </div>

  <div class="section-title">Methodology</div>
  <div class="methodology">
    <h3>How This Works</h3>
    <p>This model uses a stacked machine learning ensemble trained on 18 years of NCAA tournament data (2008-2025). For each historical tournament game, it computes the difference in 45+ team metrics between the two opponents and learns which statistical advantages best predict tournament wins.</p>
    <ul>
      <li><strong>Data sources:</strong> KenPom efficiency ratings, Barttorvik metrics, team resumes (ELO, NET, quad records), and season statistics</li>
      <li><strong>Features:</strong> Adjusted offensive/defensive efficiency, four factors (EFG%, TOV%, OREB%, FTR), shooting splits, experience, talent, height, strength of schedule, and 10 engineered composite metrics</li>
      <li><strong>Model:</strong> Stacking classifier combining Logistic Regression, Gradient Boosting, Random Forest, and Histogram Gradient Boosting, with a meta-learner to weight their predictions</li>
      <li><strong>Bracket:</strong> The deterministic bracket always picks the team with higher win probability. The simulation runs 10,000 Monte Carlo iterations, sampling from predicted probabilities each time</li>
      <li><strong>Baseline:</strong> The Nate Silver method (point spread / 11, normal CDF) is used as a sanity check against the ML model</li>
    </ul>
    <p>The model was inspired by the Kaggle March Machine Learning Mania competitions and the Silver/538 methodology for tournament prediction.</p>
  </div>

  <div class="footer">
    Built with Python, scikit-learn, and too much coffee.<br>
    <a href="https://jasonhorne.org">jasonhorne.org</a> &middot; March 2026
  </div>

</div>
</body>
</html>"""

    return html

# ─── Main ────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("=" * 60)
    print("  MARCH MADNESS 2026 ML BRACKET PREDICTOR")
    print("=" * 60)

    print("\n[1/6] Loading data...")
    kb, resume, cbb = load_team_data()
    games = load_matchups()

    print("\n[2/6] Building training features...")
    train_df = build_training_data(games, kb, resume, cbb)

    print("\n[3/6] Training model...")
    model, feature_cols, feat_imp, cv_accuracy = train_model(train_df)

    print("\n[4/6] Loading 2026 bracket...")
    play_in_matchups, r64_matchups = get_2026_bracket()

    print("\n[5/6] Simulating tournament (10,000x)...")
    N_SIMS = 10000
    advancement, champion_counts, team_cache = simulate_bracket(
        model, feature_cols, kb, resume, cbb, play_in_matchups, r64_matchups, n_sims=N_SIMS
    )

    # Deterministic bracket
    det_results, champion = run_deterministic_bracket(
        model, feature_cols, team_cache, play_in_matchups, r64_matchups
    )

    # cv_accuracy already set by train_model()

    print("\n[6/6] Generating outputs...")

    # Print results
    champ_sorted = print_results(
        advancement, champion_counts, team_cache, N_SIMS, det_results, champion
    )

    cv_acc = cv_accuracy

    # Generate HTML
    html = generate_html_bracket(
        det_results, champion, advancement, champion_counts,
        team_cache, N_SIMS, feat_imp, cv_acc
    )

    html_path = f"{OUT_DIR}/march_madness_2026.html"
    with open(html_path, 'w') as f:
        f.write(html)
    print(f"\n  HTML bracket saved to: {html_path}")

    # Save JSON results for blog
    results_json = {
        'champion': {'name': champion['name'], 'seed': champion['seed']},
        'championship_odds': [{'team': t, 'seed': team_cache[t]['seed'],
                               'pct': round(c / N_SIMS * 100, 1)}
                              for t, c in champ_sorted[:25]],
        'cv_accuracy': round(cv_acc, 1),
        'n_features': len(feature_cols),
        'n_training_games': len(train_df),
        'n_simulations': N_SIMS,
        'feature_importance': [{'name': n, 'importance': round(v, 4)}
                               for n, v in feat_imp[:20]],
    }
    json_path = f"{OUT_DIR}/results.json"
    with open(json_path, 'w') as f:
        json.dump(results_json, f, indent=2)
    print(f"  Results JSON saved to: {json_path}")

    print("\nDone!")
