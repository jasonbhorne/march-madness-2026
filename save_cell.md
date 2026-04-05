## 12. Save outputs to Drive

```python
OUT_DIR = '/content/drive/MyDrive/March Madness 2026'
os.makedirs(OUT_DIR, exist_ok=True)

# Save prediction JSON
prediction = {
    'game': 'NCAA Championship 2026',
    'date': '2026-04-06',
    'location': 'Lucas Oil Stadium, Indianapolis',
    'teams': {'uconn_seed': 2, 'michigan_seed': 1},
    'models': {
        'full_ensemble':         {'p_uconn': float(p_uconn_full), 'p_michigan': float(1-p_uconn_full)},
        'late_round_specialist': {'p_uconn': float(p_uconn_late), 'p_michigan': float(1-p_uconn_late)},
        'blended':               {'p_uconn': float(p_uconn),      'p_michigan': float(1-p_uconn)},
    },
    'predicted_winner': winner,
    'monte_carlo': {
        'n_sims': int(N_SIMS),
        'uconn_win_share':    float(uconn_wins/N_SIMS),
        'michigan_win_share': float(mich_wins/N_SIMS),
        'p_uconn_mean':       float(probs_blend.mean()),
        'p_uconn_std':        float(probs_blend.std()),
        'p_uconn_p5':         float(p5),
        'p_uconn_p50':        float(p50),
        'p_uconn_p95':        float(p95),
    },
    'score_projection': {
        'tempo': float(tempo),
        'uconn': float(uc),
        'michigan': float(mi),
        'margin': float(abs(uc-mi)),
        'projected_winner': 'UConn' if uc > mi else 'Michigan',
    },
    'model_accuracy_through_elite_eight': STATE['model_accuracy'],
}
pred_path = f'{OUT_DIR}/championship_prediction.json'
with open(pred_path, 'w') as f:
    json.dump(prediction, f, indent=2)
print(f'Saved: {pred_path}')

# Save histogram PNG
fig, ax = plt.subplots(figsize=(10, 4))
ax.hist(probs_blend*100, bins=50, color='#f97316', edgecolor='#0a0e17', alpha=0.85)
ax.axvline(50, color='#64748b', linestyle='--', linewidth=1, label='Coin flip (50%)')
ax.axvline(probs_blend.mean()*100, color='#3b82f6', linewidth=2, label=f'Mean P(UConn) = {probs_blend.mean()*100:.1f}%')
ax.set_xlabel('P(UConn wins) — %')
ax.set_ylabel('Simulation count')
ax.set_title('Championship Predictor — Monte Carlo Distribution (UConn vs Michigan)')
ax.legend()
ax.set_xlim(0, 100)
plt.tight_layout()
hist_path = f'{OUT_DIR}/championship_mc_histogram.png'
plt.savefig(hist_path, dpi=150, bbox_inches='tight')
plt.show()
print(f'Saved: {hist_path}')

# Save tournament recap as markdown
recap_lines = []
recap_lines.append('# March Madness 2026 — Championship Predictor Results\n\n')
recap_lines.append('## Model accuracy through Elite Eight\n')
for r, d in STATE['model_accuracy'].items():
    if r == 'cumulative': continue
    recap_lines.append(f"- {r.replace('_',' ').title()}: {d['correct']}/{d['total']} ({d['pct']}%)\n")
c = STATE['model_accuracy']['cumulative']
recap_lines.append(f"- **Cumulative: {c['correct']}/{c['total']} ({c['pct']}%)**\n\n")

recap_lines.append('## Final Four picks\n')
recap_lines.append(f"- Model picks: {STATE['model_final_four_picks']}\n")
recap_lines.append(f"- Correct: {STATE['model_final_four_correct']}/4 (Arizona, Michigan)\n")
recap_lines.append(f"- Original champion pick: {STATE['model_championship_pick']} (eliminated Elite Eight)\n\n")

recap_lines.append('## Championship prediction (UConn #2 vs Michigan #1)\n')
recap_lines.append(f'- Full ensemble: UConn {p_uconn_full*100:.1f}% | Michigan {(1-p_uconn_full)*100:.1f}%\n')
recap_lines.append(f'- Late-round specialist: UConn {p_uconn_late*100:.1f}% | Michigan {(1-p_uconn_late)*100:.1f}%\n')
recap_lines.append(f'- **Blended: UConn {p_uconn*100:.1f}% | Michigan {(1-p_uconn)*100:.1f}%**\n')
recap_lines.append(f'- Predicted winner: **{winner}**\n\n')

recap_lines.append(f'## Monte Carlo ({N_SIMS:,} sims)\n')
recap_lines.append(f'- UConn win share: {uconn_wins/N_SIMS*100:.1f}%\n')
recap_lines.append(f'- Michigan win share: {mich_wins/N_SIMS*100:.1f}%\n')
recap_lines.append(f'- P(UConn) 5th/50th/95th: {p5*100:.1f}% / {p50*100:.1f}% / {p95*100:.1f}%\n\n')

recap_lines.append('## Score projection\n')
recap_lines.append(f'- Projected tempo: {tempo:.1f} possessions\n')
recap_lines.append(f'- UConn {uc:.1f} — Michigan {mi:.1f}\n')
recap_lines.append(f'- {"UConn" if uc>mi else "Michigan"} by {abs(uc-mi):.1f}\n')

recap_path = f'{OUT_DIR}/championship_recap.md'
with open(recap_path, 'w') as f:
    f.writelines(recap_lines)
print(f'Saved: {recap_path}')

print('\nAll files saved to:', OUT_DIR)
```
