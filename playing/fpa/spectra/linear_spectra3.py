import math

from matplotlib import pyplot as plt

plt.rcParams.update({'font.size': 16})

days = ['Mo', 'Tu', 'We', 'Th', 'Fr', 'Sa', 'Su']
x = list(range(-5, 5))
y = [math.exp((-i**2)/5) for i in x]
den = sum(y)
for i in range(len(y)):
    y[i] = y[i] / den
labels = [days[(i+2)%7] for i in x]
d = [0 for _ in range(7)]
for i, val in enumerate(y):
    d[(i-2)%7] += y[i]

y = [0 for _ in y]
for i in range(len(d)):
    y[i+2] = d[i]


plt.bar(x, y)
plt.xticks(x, labels)
plt.tight_layout()
# plt.show()
plt.savefig('spectra4.pdf')
