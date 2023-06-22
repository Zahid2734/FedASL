import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from numpy import linspace
import statistics as st
from Code.utils.file_handler import save_file, open_file

plt.rcParams["font.family"] = "Palatino linotype"
plt.rcParams.update({'font.size': 40,'font.weight':'bold','pdf.fonttype':42})

x = np.linspace(0, 3.15, 51)
y = np.sin(x)
bla= np.sin(x)

for i in range(len(bla)):
    if (i % 2) == 1:
        bla[i]= bla[i]+.1
    else:
        bla[i]= bla[i]-.1

plt.figure(figsize=(13, 8))
plt.grid(zorder=1, color='#999999', linewidth=2.0, linestyle='--', alpha=0.5)

plt.plot(bla*5, '-o', c='blue', linewidth=5.0, zorder=5, marker='D',markevery=1,markersize=0, mfc='yellow', mec='black', mew=1, label='Actual')
plt.plot(y*5, '--', c='red', linewidth=5.0, zorder=5, marker='D',markevery=1,markersize=0, mfc='yellow', mec='black', mew=1,label='Reference')


my_xticks = []
x = np.array([0,10,20,30,40,50])
plt.xticks(x, my_xticks)
plt.xlabel("Time", weight='bold')
plt.ylabel('PFC Input Current', weight='bold')
y = np.array([1,2,3,4,5,6])
plt.yticks(y, my_xticks)
plt.ylim(0, 6)
plt.legend(bbox_to_anchor=(.25,0,.9,.3), loc="lower left",frameon=False,
                mode="expand", ncol=1)
plt.show
plt.savefig('../../Figures/sensors/sine_wave_2.pdf', bbox_inches='tight')