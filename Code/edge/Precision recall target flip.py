import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from numpy import linspace
import statistics as st
from Code.utils.file_handler import save_file, open_file


plt.rcParams["font.family"] = "Palatino linotype"
plt.rcParams.update({'font.size': 40,'font.weight':'bold','pdf.fonttype':42})

precision= [48,49,48,47,42,37]
recall=[61,62,61,63,69,78]
acuracy=[63,63,63,62,62,62]
plt.figure(figsize=(10, 8))
plt.grid(zorder=1, color='#999999', linewidth=2.0, linestyle='--', alpha=0.5)

plt.plot(acuracy, '-o', c='red', linewidth=10.0, zorder=5, marker='o',markevery=1,markersize=25, mfc='white', mec='red', mew=1,label='Overall accuracy')
plt.plot(recall, '-o', c='blue', linewidth=10.0, zorder=5, marker='D',markevery=1,markersize=25, mfc='white', mec='blue', mew=1,label='Class 5 accuracy')
plt.plot(precision, '-o', c='darkorange', linewidth=10.0, zorder=5, marker='^',markevery=1,markersize=25, mfc='white', mec='darkorange', mew=1,label='Class 5 precision')

plt.legend(loc=4)
my_xticks = ['5','10','15','20','25','30']
x = np.array([0,1,2,3,4,5])
plt.xticks(x, my_xticks)
plt.xlabel("Coordinating Attacker (%)", weight='bold')
plt.ylabel('Percentage (%)', weight='bold')
plt.yticks(np.arange(30, 91, 15))
plt.ylim(30, 90)
plt.legend(bbox_to_anchor=(0,0.6,.75,.3), loc="lower left",fontsize=32,
                mode="expand", ncol=1)
plt.show
plt.savefig('../../Figures/edge/power point/precision_recall.png', bbox_inches='tight')