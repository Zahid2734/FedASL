
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from numpy import linspace
from Code.utils.file_handler import save_file, open_file

plt.rcParams["font.family"] = "palatino linotype"
plt.rcParams.update({'font.size': 40,'font.weight':'bold','pdf.fonttype':42})

data1= '../../Data/Femnist/Shuffling/FedAVG_0.3_1_3000_250_shuffle_Femnist.pkl'
FedAVG_3= open_file(data1)

data2= '../../Data/Femnist/Shuffling/FedSTD_0.3_1_3000_250_0.7_0.2_shuffle_Femnist.pkl'
FedSTD_3= open_file(data2)

data3= '../../Data/Femnist/Shuffling/FedAVG_0.1_1_3000_250_shuffle_Femnist.pkl'
FedAVG_1= open_file(data3)

data4='../../Data/Femnist/Shuffling/FedSTD_0.1_1_3000_250_1_0.2_shuffle_Femnist.pkl'
FedSTD_1= open_file(data4)

data5= '../../Data/Femnist/Shuffling/FedAVG_0.4_1_3000_250_shuffle_Femnist.pkl'
FedAVG_4= open_file(data5)

data6= '../../Data/Femnist/Shuffling/FedSTD_0.4_1_3000_250_0.5_0.2_shuffle_Femnist.pkl'
FedSTD_4= open_file(data6)

data7= '../../Data/Femnist/Shuffling/median_0.3_1_3000_250_shuffle_Femnist.pkl'
FedPA_4_2= open_file(data7)

data8= '../../Data/Femnist/Shuffling/trimmed_mean_0.3_1_3000_250_shuffle_Femnist.pkl'
FedPA_4= open_file(data8)

data9= '../../Data/Femnist/Shuffling/FedSTD_loss_0.3_1_3000_250_0.7_0.2_shuffle_Femnist.pkl'
FedSTD_loss_3= open_file(data9)

data10= '../../Data/Femnist/Shuffling/FedSTD_loss_0.4_1_3000_250_0.7_0.2_shuffle_Femnist.pkl'
FedSTD_loss_4= open_file(data10)

plt.figure(figsize=(40, 12))
plt.grid(zorder=1, color='#999999', linestyle='--', alpha=.04)
plt.subplot(1, 3, 1)
plt.plot(FedSTD_1[0], '-o', c='royalblue', linewidth=3.0, zorder=3, markersize=1, mfc='w', mec='royalblue', mew=1,label='FedSTD')
plt.plot(FedAVG_1[0], '-o', c='red', linewidth=3.0, zorder=3, markersize=1, mfc='w', mec='red', mew=1,label='FedAVG')
plt.legend(loc=4)
plt.xticks(np.arange(0.0, 251, 50))
plt.xlabel("Rounds of Training", weight='bold')
plt.ylabel('Accuracy(%)', weight='bold')
# plt.legend(bbox_to_anchor=(.1,0,.9,.3), loc="lower left",
#                 mode="expand", ncol=3)

plt.subplot(1, 3, 2)
plt.plot(FedSTD_3[0], '-o', c='royalblue', linewidth=3.0, zorder=3, markersize=1, mfc='w', mec='royalblue', mew=1,label='FedSTD')
plt.plot(FedSTD_loss_3[0], '-o', c='orange', linewidth=3.0, zorder=3, markersize=1, mfc='w', mec='blue', mew=1,label='FedSTD_loss')
plt.plot(FedAVG_3[0], '-o', c='red', linewidth=3.0, zorder=3, markersize=1, mfc='w', mec='red', mew=1,label='FedAVG')
plt.plot(FedPA_4[0], '-o', c='green', linewidth=3.0, zorder=3, markersize=1, mfc='w', mec='green', mew=1,label='median')
plt.plot(FedPA_4_2[0], '-o', c='brown', linewidth=3.0, zorder=3, markersize=1, mfc='w', mec='brown', mew=1,label='trimmed mean')

plt.legend(loc=4)
plt.xticks(np.arange(0.0, 251, 50))
plt.xlabel("Rounds of Training", weight='bold')
plt.ylabel('Accuracy(%)', weight='bold')
# plt.legend(bbox_to_anchor=(.1,0,.9,.3), loc="lower left",
#                 mode="expand", ncol=3)

plt.subplot(1, 3, 3)
plt.plot(FedSTD_4[0], '-o', c='royalblue', linewidth=3.0, zorder=3, markersize=1, mfc='w', mec='royalblue', mew=1,label='FedSTD')
plt.plot(FedSTD_loss_4[0], '-o', c='orange', linewidth=3.0, zorder=3, markersize=1, mfc='w', mec='blue', mew=1,label='FedSTD_loss')
plt.plot(FedAVG_4[0], '-o', c='red', linewidth=3.0, zorder=3, markersize=1, mfc='w', mec='red', mew=1,label='FedAVG')


plt.legend(loc=4)
plt.xticks(np.arange(0.0, 251, 50))
plt.xlabel("Rounds of Training", weight='bold')
plt.ylabel('Accuracy(%)', weight='bold')
# plt.legend(bbox_to_anchor=(.1,0,.9,.3), loc="lower left",
#                 mode="expand", ncol=3)

plt.show
plt.savefig('../../Figures/femnist/femnist_shuffling_3000.pdf', bbox_inches='tight')




