import math

x = list(range(-5, 5))
y = [math.exp(-(i**2)/5) for i in x]
den = sum(y)
for i in range(len(y)):
    y[i] = y[i] / den
d = [0 for _ in range(7)]
for i, val in enumerate(y):
    # d[(i+2)%7] += y[i]
    d[(i+3)%7] += y[i]

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({'font.size': 16})
# Fixing random state for reproducibility
np.random.seed(19680801)

# Compute pie slices
N = 7
theta = np.linspace(0.0, 2 * np.pi, N, endpoint=False) + 0.44
# theta = np.linspace(0.0, 2 * np.pi, N, endpoint=False)
# radii = 10 * np.random.rand(N)
radii = d
# width = np.pi / 4 * np.random.rand(N)
width = [0.3 for _ in range(N)]
# colors = plt.cm.viridis(radii / 10.)
colors = ['tab:blue' for _ in range(7)]

ax = plt.subplot(projection='polar')
# ax = plt.subplot(projection='radar')
# ax.bar(theta, radii, width=width, bottom=0.0, color=colors)
ax.bar(theta, radii, width=width, bottom=0.1)
spoke_labels = ['Mo', 'Tu', 'We', 'Th', 'Fr', 'Sa', 'Su']
# ax.set_varlabels(spoke_labels)
# ax.set_label(spoke_labels)
ax.set_xticks(np.pi/180. * np.linspace(180,  -180, 7, endpoint=False), spoke_labels)
ax.set_thetalim(-np.pi, np.pi)
ax.get_yaxis().set_visible(False)
plt.tight_layout()

# plt.show()
plt.savefig('spectra2.pdf')
