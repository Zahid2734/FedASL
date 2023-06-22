import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from numpy import linspace
from Code.utils.file_handler import save_file, open_file

plt.rcParams["font.family"] = "Palatino linotype"
plt.rcParams.update({'font.size': 40,'font.weight':'bold','pdf.fonttype':42})

data1= '../../Data/Femnist/clean/FedSTD_clean_Femnist_150.pkl'
data1= open_file(data1)

data2= '../../Data/Femnist/clean/FedAVG_clean_Femnist_150.pkl '
data2= open_file(data2)




plt.figure(figsize=(20, 12))
plt.grid(zorder=1, color='#999999', linestyle='--', alpha=.04)
plt.plot(data1[0], '-o', c='royalblue', linewidth=3.0, zorder=3, markersize=1, mfc='w', mec='royalblue', mew=1,label='FedSTD')
plt.plot(data2[0], '-o', c='red', linewidth=3.0, zorder=3, markersize=1, mfc='w', mec='red', mew=1,label='FedAVG')


plt.legend(loc=4)
plt.xticks(np.arange(0.0, 151, 25))
plt.xlabel("Rounds of Training", weight='bold')
plt.ylabel('Accuracy(%)', weight='bold')
plt.legend(bbox_to_anchor=(0,1.02,1,0.2), loc="lower left",
                mode="expand", ncol=2)
plt.show
plt.savefig('../../Figures/femnist/Femnist_clean.pdf', bbox_inches='tight')