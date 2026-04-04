# March Madness 2026 ML Bracket Predictor

A machine learning bracket predictor for the 2026 NCAA Men's Basketball Tournament. Stacked ensemble trained on 18 years of KenPom/Barttorvik data, inspired by [Jared Cross's 1st-place Kaggle solution](https://www.kaggle.com/competitions/march-machine-learning-mania-2024).

## Results

**Overall accuracy: 66.7%** across 60 games (through Elite Eight).

| Round | Correct | Total | Accuracy |
|-------|---------|-------|----------|
| Round of 64 | 22 | 32 | 68.8% |
| Round of 32 | 11 | 16 | 68.8% |
| Sweet 16 | 5 | 8 | 62.5% |
| Elite Eight | 2 | 4 | 50.0% |

**Final Four picks:** Duke, Houston, Arizona, Michigan (2 of 4 correct: Arizona, Michigan)

**Championship pick:** Duke (eliminated in Elite Eight by UConn, 73-72)

## Methodology

### Data Sources (66 files, 2008-2026)

- **KenPom Barttorvik** -- adjusted efficiency ratings, tempo, four factors, shooting splits, experience, talent, height
- **Resumes** -- ELO, NET/RPI rankings, quad records (Q1-Q4), wins above bubble
- **CBB stats** (cbb13-cbb26) -- fallback team stats for broader coverage
- **Tournament Matchups** -- historical game results for training labels

### Feature Engineering

The model computes **pairwise feature deltas** between opponents. For each matchup, Team A's stats minus Team B's stats become the input features. This yields ~50 features per game:

- 33 KenPom/Barttorvik metrics (efficiency, four factors, shooting, rebounding, experience)
- 8 resume metrics (ELO, NET, quad records)
- 10 engineered features:
  - Net efficiency margin, offensive/defensive balance ratio
  - Dean Oliver's Four Factors composite (offense and defense)
  - Shooting versatility, defensive pressure, ball security
  - Rebounding margin, quality win percentage, bad loss rate
- 3 seed-based features (difference, sum, product)

Training data is doubled by flipping perspectives (Team A vs B becomes B vs A with inverted label), which eliminates ordering bias and improves calibration.

### Model Architecture

**Stacked ensemble** with logistic regression meta-learner:

| Layer | Model | Key Hyperparameters |
|-------|-------|-------------------|
| Base | Logistic Regression | C=0.5, StandardScaler |
| Base | Gradient Boosting | 300 trees, depth 4, lr 0.05, subsample 0.8 |
| Base | Random Forest | 300 trees, depth 6, min leaf 10 |
| Base | Hist Gradient Boosting | 300 iterations, depth 5, lr 0.05 |
| Meta | Logistic Regression | C=1.0, 3-fold CV stacking |

Base models generate probability predictions via 3-fold cross-validation. The meta-learner combines these into final win probabilities.

### Bracket Simulation

The full 68-team bracket is simulated round by round. In each game, the model predicts P(win) and the higher-probability team advances. All win probabilities are recorded in `bracket_picks.json`.

## Repository Structure

```
predict_bracket.py          # Main prediction pipeline (features, training, simulation)
predict_bracket_colab.py    # Colab-ready version with inline data loading
predict_r32_colab.py        # Round of 32 re-prediction with locked R64 results
bracket_picks.json          # Full bracket predictions with probabilities
tournament_state.json       # Actual results tracker with accuracy stats
fetch_scores.py             # ESPN API score fetcher
update_results.py           # Update tournament_state with actual outcomes
rerun.py                    # Re-run predictions from any round forward
tracker.html                # Visual bracket tracker
data/                       # 66 CSV files (KenPom, Barttorvik, resumes, matchups)
```

## Quick Start

### Run locally

```bash
python predict_bracket.py
```

Outputs `bracket_picks.json` with every predicted matchup and win probability.

### Run on Google Colab

Open `march_madness_2026_colab.ipynb` or `predict_bracket_colab.py` in Colab. Data files load from the `data/` directory.

### Re-run from a later round

```bash
python rerun.py
```

Locks in actual results and re-predicts remaining games.

## What Worked, What Didn't

**Hits:** The model correctly picked Arizona (1) and Michigan (1) to the Final Four, nailed UConn over Michigan State in the Sweet 16 at 62.1% confidence, and maintained ~69% accuracy through the first two rounds.

**Misses:** Duke was the model's champion pick at 57% confidence but fell to UConn by one point in the Elite Eight. Houston (2-seed, picked to the Final Four) lost to Illinois in the Sweet 16. The model struggled with Iowa's 9-seed Cinderella run and Tennessee's upset of Iowa State.

**Takeaway:** Efficiency-based models are strong in early rounds where talent gaps are wide. As the field narrows to elite teams, single-game variance (shooting luck, foul trouble, one-point finishes) dominates, and no model can reliably predict coin-flip games.

## License

MIT

## Author

[Jason Horne](https://jasonhorne.org) | [@jasonbhorne](https://github.com/jasonbhorne)
