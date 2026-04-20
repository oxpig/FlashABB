"""Combined spectra plots in a 2x2 grid with consistent styling."""

import math
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

plt.rcParams.update({'font.size': 14})

# Color scheme from speed_comparison.py
bar_color = 'lightblue'
edge_color = 'black'
edge_width = 0.5

# Compute base data
days = ['Mo', 'Tu', 'We', 'Th', 'Fr', 'Sa', 'Su']
x = list(range(-5, 5))
y = [math.exp((-i**2)/5) for i in x]
den = sum(y)
for i in range(len(y)):
    y[i] = y[i] / den

# Create figure with GridSpec for better control
fig = plt.figure(figsize=(20, 5), constrained_layout=True)
gs = gridspec.GridSpec(1, 4, figure=fig)

# ========== Plot 1: Linear spectra 1 ==========
ax1 = fig.add_subplot(gs[0, 0])
labels1 = [days[(i+2)%7] for i in x]
ax1.bar(x, y, color=bar_color, edgecolor=edge_color, linewidth=edge_width)
ax1.set_xticks(x)
ax1.set_xticklabels(labels1)
ax1.set_title('(a) Absolute', pad=10)

# ========== Plot 2: Rotary spectra (polar) ==========
# Compute rotary data
d_rotary = [0 for _ in range(7)]
for i, val in enumerate(y):
    d_rotary[(i+3)%7] += y[i]

ax2 = fig.add_subplot(gs[0, 1], projection='polar')
N = 7
theta = np.linspace(0.0, 2 * np.pi, N, endpoint=False) + 0.44
width = [0.3 for _ in range(N)]
ax2.bar(theta, d_rotary, width=width, bottom=0.1,
        color=bar_color, edgecolor=edge_color, linewidth=edge_width)
spoke_labels = ['Mo', 'Tu', 'We', 'Th', 'Fr', 'Sa', 'Su']
ax2.set_xticks(np.pi/180. * np.linspace(180, -180, 7, endpoint=False))
ax2.set_xticklabels(spoke_labels)
ax2.set_thetalim(-np.pi, np.pi)
ax2.get_yaxis().set_visible(False)
ax2.set_title('(b) Modulus 7', pad=10)

# ========== Plot 3: Linear spectra 2 ==========
ax3 = fig.add_subplot(gs[0, 2])
# Compute aggregated data
d_linear = [0 for _ in range(7)]
for i, val in enumerate(y):
    d_linear[(i-2)%7] += y[i]

labels3 = [-3, -2, -1, 0, 1, 2, 3]
ax3.bar(labels3, d_linear, color=bar_color, edgecolor=edge_color, linewidth=edge_width)
ax3.set_xticks(labels3)
ax3.set_title('(c) Relative', pad=10)

# ========== Plot 4: Linear spectra 3 ==========
ax4 = fig.add_subplot(gs[0, 3])
# Remap aggregated data back to position space
y_remapped = [0 for _ in y]
for i in range(len(d_linear)):
    y_remapped[i+2] = d_linear[i]

labels4 = [days[(i+2)%7] for i in x]
ax4.bar(x, y_remapped, color=bar_color, edgecolor=edge_color, linewidth=edge_width)
ax4.set_xticks(x)
ax4.set_xticklabels(labels4)
ax4.set_title('(d) Reconstructed', pad=10)

fig.align_titles()
plt.savefig('combined_spectra.pdf', dpi=150, bbox_inches='tight')
plt.savefig('combined_spectra.png', dpi=150, bbox_inches='tight')
print("Saved combined_spectra.pdf and combined_spectra.png")
plt.show()
