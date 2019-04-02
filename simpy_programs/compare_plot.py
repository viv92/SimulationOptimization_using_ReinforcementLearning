import numpy as np
import matplotlib.pyplot as plt


N = 5
totalTimes = (56.10, 75.47, 113.12, 75.52, 46.20)

fig, ax = plt.subplots()

ind = np.arange(N)*2    # the x locations for the groups
width = 0.35         # the width of the bars
p1 = ax.bar(ind, totalTimes, width, color='r', bottom=0)


prodRates = (178.18, 132.46, 88.37, 132.37, 215.30)
p2 = ax.bar(ind + width, prodRates, width,
            color='b', bottom=0)

ax.set_title('Comparison for different policies')
ax.set_xticks(ind + width / 2)
ax.set_xticklabels(('Alternating', 'TrucksInQueue', 'CapacityInQue', 'ExpLoadTime', 'LearntPolicy(RL)'))
ax.set_axisbelow(True)
ax.yaxis.grid(color='gray', linestyle='dashed')
ax.legend((p1[0], p2[0]), ('Total time', 'Production rate'))
#ax.yaxis.set_units(inch)
ax.autoscale_view()

plt.show()
