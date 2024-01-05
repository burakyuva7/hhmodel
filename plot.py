"""Script for plotting g_ACH,min vs. time for Question 2."""

import matplotlib.pyplot as plt

plt.grid()
plt.plot([1.0, 1.5, 2.0, 3.0], [0.072, 0.051, 0.041, 0.031], marker='o')
plt.title(r'$g_{ACH, min}$ vs. Pulse Duration')
plt.ylabel(r'$g_{ACH, min}$')
plt.xlabel('Pulse Duration (ms)')
plt.savefig("Q2-gACH-vs-ps.png")
plt.clf()