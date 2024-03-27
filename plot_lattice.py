# This code is part of hubbard.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
This code simply plots the lattice in a nice way, highlighting the 1d mapping
"""

import matplotlib.pyplot as plt
import numpy as np

coordinates = np.array([[ii, jj] for jj in range(4) for ii in range(4)])

vrishon_coordinates = np.array([[ii, jj+0.5] for jj in range(3) for ii in range(4) ])
hrishon_coordinates = np.array([[ii+0.5, jj] for jj in range(4) for ii in range(3) ])

fig, ax = plt.subplots(figsize=(8, 8))

ax.hlines(range(4), xmin=np.repeat(0, 4), xmax=np.repeat(3, 4), color='black', zorder=-1)
ax.vlines(range(4), ymin=np.repeat(0, 4), ymax=np.repeat(3, 4), color='black', zorder=-1)
ax.scatter(coordinates[:, 0], coordinates[:, 1], marker='o', edgecolors='black', s=500, c='navy', label='Site')
ax.scatter(vrishon_coordinates[:, 0], vrishon_coordinates[:, 1], marker='^', edgecolors='black', s=200, c='red', label='Rishon')
ax.scatter(hrishon_coordinates[:, 0], hrishon_coordinates[:, 1], marker='^', edgecolors='black', s=200, c='red')
ax.set_xticks(range(4))
ax.set_yticks(range(4))


qubit_initial_ordering = np.vstack((vrishon_coordinates, hrishon_coordinates, coordinates))
ordering = [24, 12, 25, 1, 13, 26, 14, 27, 3, 31, 17, 2, 30, 16, 5, 29,
            15, 0, 28, 4, 32, 18, 9, 33, 19, 6, 34, 20, 7, 35, 11, 39,
            23, 10, 38, 22, 37, 21, 8, 36]
qubit_final_ordering = qubit_initial_ordering[ordering, :]
ax.plot(qubit_final_ordering[:, 0], qubit_final_ordering[:, 1], lw=5, color='forestgreen', alpha=0.5, label='1d mapping')

ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
          ncol=3, fancybox=True, shadow=True, fontsize=14)
plt.tight_layout()
plt.savefig('images/1d_mapping.pdf')

plt.show()