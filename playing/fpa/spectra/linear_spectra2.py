import math

from matplotlib import pyplot as plt

plt.rcParams.update({'font.size': 16})

days = ['Mo', 'Tu', 'We', 'Th', 'Fr', 'Sa', 'Su']
x = list(range(-5, 5))
y = [math.exp((-i**2)/5) for i in x]
den = sum(y)
for i in range(len(y)):
    y[i] = y[i] / den
d = [0 for _ in range(7)]
for i, val in enumerate(y):
    d[(i-2)%7] += y[i]

labels = [-3, -2, -1, 0, 1, 2, 3]
plt.bar(labels, d)
plt.tight_layout()
# plt.xticks(range(7), labels)
# plt.show()
plt.savefig('spectra3.pdf')
