import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from numpy import linspace
from Code.utils.file_handler import save_file, open_file

plt.rcParams["font.family"] = "palatino linotype"
plt.rcParams.update({'font.size': 40,'font.weight':'bold','pdf.fonttype':42})

data1= '../../Data/Femnist/flipping/FedAVG_0.3_1_3000_250_flip_Femnist.pkl'
FedAVG_3= open_file(data1)

data2= '../../Data/Femnist/flipping/FedSTD_0.3_1_3000_250_0.7_0.2_flip_Femnist.pkl'
FedSTD_3= open_file(data2)

data3= '../../Data/Femnist/Shuffling/FedAVG_0.3_1_3000_250_shuffle_Femnist.pkl'
FedAVG_1= open_file(data3)

data4= '../../Data/Femnist/Shuffling/FedSTD_0.3_1_3000_250_0.7_0.2_shuffle_Femnist.pkl'
FedSTD_1= open_file(data4)

data5= '../../Data/Femnist/Noisy/FedAVG_0.3_1_3000_250_noise_Femnist.pkl'
FedAVG_4= open_file(data5)

data6= '../../Data/Femnist/Noisy/FedSTD_0.3_1_3000_250_0.7_0.2_noise_Femnist.pkl'
FedSTD_4= open_file(data6)

data7= '../../Data/Femnist/Shuffling/median_0.3_1_3000_250_shuffle_Femnist.pkl'
median_shuffle= open_file(data7)

data8= '../../Data/Femnist/Shuffling/trimmed_mean_0.3_1_3000_250_shuffle_Femnist.pkl'
TM_shuffle= open_file(data8)

data9= '../../Data/Femnist/flipping/median_0.3_1_3000_250_flip_Femnist.pkl'
median_flip= open_file(data9)

data10= '../../Data/Femnist/flipping/trimmed_mean_0.3_1_3000_250_flip_Femnist.pkl'
TM_flip= open_file(data10)

data11= '../../Data/Femnist/Noisy/median_0.3_1_3000_250_noise_Femnist.pkl'
median_noise= open_file(data11)

data12= '../../Data/Femnist/Noisy/trimmed_mean_0.3_1_3000_250_noise_Femnist.pkl'
TM_noise= open_file(data12)


plt.figure(figsize=(50,12))
plt.grid(zorder=1, color='#999999', linestyle='--', alpha=.04)
plt.subplot(1, 3, 1)
plt.plot(FedSTD_1[0], '-o', c='red', linewidth=3.0, zorder=5, marker='o',markevery=25,markersize=15, mfc='red', mec='red', mew=1, label='FedSTD')
plt.plot(median_shuffle[0], '-o', c='green', linewidth=3.0, zorder=5, marker='*',markevery=35,markersize=15, mfc='green', mec='green', mew=1, label='Median')
plt.plot(TM_shuffle[0], '-o', c='brown', linewidth=3.0, zorder=5, marker='>',markevery=40,markersize=15, mfc='brown', mec='brown', mew=1,label='Trimmed_mean')
plt.plot(FedAVG_1[0], '-o', c='blue', linewidth=3.0, zorder=5, marker='D',markevery=30,markersize=15, mfc='blue', mec='blue', mew=1,label='FedAVG')
plt.legend(loc=4)
plt.xticks(np.arange(0.0, 251, 50))
plt.title('Label Shuffling')
plt.xlabel("Rounds of Training", weight='bold')
plt.ylabel('Accuracy(%)', weight='bold')
# plt.legend(bbox_to_anchor=(.1,0,.9,.3), loc="lower left",
#                 mode="expand", ncol=3)

plt.subplot(1, 3, 2)
plt.plot(FedSTD_3[0], '-o', c='red', linewidth=3.0, zorder=5, marker='o',markevery=25,markersize=15, mfc='red', mec='red', mew=1, label='FedSTD')
plt.plot(median_flip[0], '-o', c='green', linewidth=3.0, zorder=5, marker='*',markevery=35,markersize=15, mfc='green', mec='green', mew=1, label='Median')
plt.plot(TM_flip[0], '-o', c='brown', linewidth=3.0, zorder=5, marker='>',markevery=40,markersize=15, mfc='brown', mec='brown', mew=1,label='Trimmed_mean')
plt.plot(FedAVG_3[0], '-o', c='blue', linewidth=3.0, zorder=5, marker='D',markevery=30,markersize=15, mfc='blue', mec='blue', mew=1,label='FedAVG')

plt.legend(loc=4)
plt.xticks(np.arange(0.0, 251, 50))
plt.title('Label Flipping')
plt.xlabel("Rounds of Training", weight='bold')
plt.ylabel('Accuracy(%)', weight='bold')
# plt.legend(bbox_to_anchor=(.1,0,.9,.3), loc="lower left",
#                 mode="expand", ncol=3)

plt.subplot(1, 3, 3)
plt.plot(FedSTD_4[0], '-o', c='red', linewidth=3.0, zorder=5, marker='o',markevery=25,markersize=15, mfc='red', mec='red', mew=1, label='FedSTD')
plt.plot(median_noise[0], '-o', c='green', linewidth=3.0, zorder=5, marker='*',markevery=35,markersize=15, mfc='green', mec='green', mew=1, label='Median')
plt.plot(TM_noise[0], '-o', c='brown', linewidth=3.0, zorder=5, marker='>',markevery=40,markersize=15, mfc='brown', mec='brown', mew=1,label='Trimmed_mean')
plt.plot(FedAVG_4[0], '-o', c='blue', linewidth=3.0, zorder=5, marker='D',markevery=30,markersize=15, mfc='blue', mec='blue', mew=1,label='FedAVG')

plt.legend(loc=4)
plt.xticks(np.arange(0.0, 251, 50))
plt.title('Noisy Data')
plt.xlabel("Rounds of Training", weight='bold')
plt.ylabel('Accuracy(%)', weight='bold')
# plt.legend(bbox_to_anchor=(.1,0,.9,.3), loc="lower left",
#                 mode="expand", ncol=3)

plt.show
plt.savefig('../../Figures/Femnist/Femnist_all_cases_4_algo.pdf', bbox_inches='tight')