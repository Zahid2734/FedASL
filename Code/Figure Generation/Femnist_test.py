
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from numpy import linspace
from Code.utils.file_handler import save_file, open_file

plt.rcParams["font.family"] = "palatino linotype"
plt.rcParams.update({'font.size': 40,'font.weight':'bold','pdf.fonttype':42})

data1= '../../Data/Femnist/flipping/FedSTD_0.3_1_500_200_0.05_0.03_flip_Femnist_test1.pkl'
FedSTD_1= open_file(data1)

data2= '../../Data/Femnist/flipping/FedSTD_0.3_1_500_200_0.9_0.2_flip_Femnist.pkl'
FedSTD_2= open_file(data2)



plt.figure(figsize=(20, 12))
plt.grid(zorder=1, color='#999999', linestyle='--', alpha=.04)

plt.plot(FedSTD_1[0], '-o', c='royalblue', linewidth=3.0, zorder=3, markersize=1, mfc='w', mec='royalblue', mew=1,label='FedSTD')
plt.plot(FedSTD_2[0], '-o', c='red', linewidth=3.0, zorder=3, markersize=1, mfc='w', mec='red', mew=1,label='FedSTD_pre')
plt.legend(loc=4)
plt.xticks(np.arange(0.0, 201, 50))
plt.xlabel("Rounds of Training", weight='bold')
plt.ylabel('Accuracy(%)', weight='bold')
# plt.legend(bbox_to_anchor=(.1,0,.9,.3), loc="lower left",
#                 mode="expand", ncol=3)

plt.show
# plt.savefig('../../Figures/femnist/femnist_flipping.pdf', bbox_inches='tight')
