"""Figure 2: gain of the model chosen on the validation year over the local model, with block-bootstrap 95% intervals."""
from __future__ import annotations

import pandas as pd

NAMES = {'IN': 'India', 'BR': 'Brazil', 'HK': 'Hong Kong', 'CA': 'Canada'}


def figure2(bootstrap_csv: str, table6_csv: str, path: str):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    B = pd.read_csv(bootstrap_csv); B = B[B['model'] == 'chosen on validation']
    B = B[((B['period'] == '2023-25') & (B['blocks'] == '3-month blocks')) | ((B['period'] == '2026') & (B['blocks'] == '1-month blocks'))]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for j, m in enumerate(NAMES):
        for k, (per, alpha) in enumerate((('2023-25', 1.0), ('2026', 0.45))):
            r = B[(B['market'] == m) & (B['period'] == per)]
            if not len(r): continue
            r = r.iloc[0]; x = j + (-0.12 if k == 0 else 0.12)
            ax.plot([x, x], [r['95% low'], r['95% high']], color='C0', alpha=alpha, lw=2)
            ax.plot(x, r['gain'], 'o', color='C0', alpha=alpha, ms=7, label=per if j == 0 else None)
            ax.annotate(f"{r['gain']:+.1f}", (x, r['gain']), xytext=(6 if k else -6, 0), textcoords='offset points', ha='left' if k else 'right', va='center', fontsize=9)
    ax.axhline(0, color='0.4', lw=0.8); ax.set_xticks(range(len(NAMES))); ax.set_xticklabels(NAMES.values())
    ax.set_ylabel('gain over the local model (points of skill)'); ax.legend(frameon=False, loc='upper right')
    ax.set_title('The model chosen on validation beats the local model in every market in 2023-25', fontsize=11)
    for s in ('top', 'right'): ax.spines[s].set_visible(False)
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)
    return path
