import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(context='talk', style='ticks',
        color_codes=True, rc={'legend.frameon': False})

test = pd.read_csv('trained_model_icsd_zintl/predicted_test.csv')
icsd = pd.read_csv('/projects/rlmolecule/shubham/file_transfer/decorations/relaxed/icsd/icsd_energies.csv')

test_nrel = test[test.id.isin(icsd.id)]
test_hypo = test[~test.id.isin(icsd.id)]

fig = plt.figure(figsize=(4,4))
ax = fig.add_subplot(111, aspect='equal')
ax.plot(test_nrel.energyperatom, test_nrel.predicted_energyperatom, '.', ms=2, c='b', label='ICSD')
ax.plot(test_hypo.energyperatom, test_hypo.predicted_energyperatom, '.', ms=2, c='r', label='Hypothetical')
ax.plot([-9, -1], [-9, -1], '--', color='.8', zorder=0)
ax.set_xticks([-1, -3, -5, -7, -9])
ax.set_yticks([-1, -3, -5, -7, -9])
ax.legend(loc='upper left', frameon=False, prop={'size':10}, markerscale=6)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.tick_params(direction='in', length=5)
#plt.ylabel('Energy per atom,\npredicted (eV/atom)', fontsize=12)
#plt.xlabel('Energy per atom,\nDFT (eV/atom)', fontsize=12)
plt.ylabel('Predicted Total Energy (eV/atom)', fontsize=14)
plt.xlabel('DFT Total Energy (eV/atom)', fontsize=14)
plt.tight_layout()

ax.text(1, 0.025, f'Test MAE: {(test.energyperatom - test.predicted_energyperatom.squeeze()).abs().mean():.3f} eV/atom',
        ha='right', va='bottom', transform=ax.transAxes, fontsize=14)

sns.despine(trim=False)
plt.savefig('test_parity.pdf')
