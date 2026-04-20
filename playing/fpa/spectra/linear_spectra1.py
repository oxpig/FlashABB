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

plt.bar(x, y)
plt.xticks(x, labels)
plt.tight_layout()
plt.savefig('spectra1.pdf')
