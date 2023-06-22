import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from numpy import linspace
import statistics as st
from Code.utils.file_handler import save_file, open_file


plt.rcParams["font.family"] = "Palatino linotype"
plt.rcParams.update({'font.size': 40,'font.weight':'bold','pdf.fonttype':42})

acuracy=[63,63,63,62,62,62]
class_5=[61,62,61,62,69,78]
class_2=[51,45, 42, 44, 49, 38]
class_3=[33,42, 42, 36, 35, 35]
plt.figure(figsize=(10, 8))
plt.grid(zorder=1, color='#999999', linewidth=2.0, linestyle='--', alpha=0.5)


plt.plot(acuracy, '-o', c='red', linewidth=10.0, zorder=5, marker='o',markevery=1,markersize=25, mfc='white', mec='black', mew=1,label='Overall acuuracy')
plt.plot(class_5, '-o', c='blue', linewidth=10.0, zorder=5, marker='D',markevery=1,markersize=25, mfc='yellow', mec='black', mew=1,label='Class 5')
plt.plot(class_2, '-o', c='green', linewidth=10.0, zorder=5, marker='>',markevery=1,markersize=25, mfc='white', mec='black', mew=1,label='Class 2')
plt.plot(class_3, '-o', c='orange', linewidth=10.0, zorder=5, marker='^',markevery=1,markersize=25, mfc='white', mec='black', mew=1,label='Class 3')

plt.legend(loc=4)
my_xticks = ['5','10','15','20','25','30']
x = np.array([0,1,2,3,4,5])
plt.xticks(x, my_xticks)
plt.xlabel("Percentage of coordinating attacker", weight='bold')
plt.ylabel('Accuracy(%)', weight='bold')
plt.yticks(np.arange(30, 91, 15))
plt.ylim(30, 90)
plt.legend(bbox_to_anchor=(0,.5,.35,.3), loc="lower left",fontsize=30,
                mode="expand", ncol=1)
plt.show
plt.savefig('../../Figures/edge/target_flip.pdf', bbox_inches='tight')