
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from numpy import linspace
from Code.utils.file_handler import save_file, open_file

plt.rcParams["font.family"] = "palatino linotype"
plt.rcParams.update({'font.size': 40,'font.weight':'bold','pdf.fonttype':42})

data1= '../../Data/mnist/clean/FedAVG_0.3_100_200_clean_mnist.pkl'
FedAVG= open_file(data1)

data2= '../../Data/mnist/clean/FedSTD_0.3_100_200_0.2_0.05_clean_mnist.pkl'
FedSTD= open_file(data2)

data3= '../../Data/mnist/clean/FedSTD_loss_0.3_100_200_0.2_0.05_clean_mnist.pkl'
FedSTD_loss= open_file(data3)

data4='../../Data/mnist/clean/median_0.3_100_200_clean_mnist.pkl'
median= open_file(data4)

data5= '../../Data/mnist/clean/trimmed_mean_0.3_100_200_clean_mnist.pkl'
TM= open_file(data5)



plt.figure(figsize=(10, 6))
plt.grid(zorder=1, color='#999999', linestyle='--', alpha=.04)

plt.plot(FedSTD[0], '-o', c='royalblue', linewidth=3.0, zorder=3, markersize=1, mfc='w', mec='royalblue', mew=1,label='FedSTD')
plt.plot(FedSTD_loss[0], '-o', c='red', linewidth=3.0, zorder=3, markersize=1, mfc='w', mec='red', mew=1,label='FedSTD_loss')
plt.plot(FedAVG[0], '-o', c='green', linewidth=3.0, zorder=3, markersize=1, mfc='w', mec='green', mew=1,label='FedAVG')
plt.plot(median[0], '-o', c='brown', linewidth=3.0, zorder=3, markersize=1, mfc='w', mec='brown', mew=1,label='Median')
plt.plot(TM[0], '-o', c='orange', linewidth=3.0, zorder=3, markersize=1, mfc='w', mec='brown', mew=1,label='Trimmed mean')

plt.legend(loc=4)
plt.xticks(np.arange(0.0, 201, 50))
plt.xlabel("Rounds of Training", weight='bold')
plt.ylabel('Accuracy(%)', weight='bold')
plt.legend(bbox_to_anchor=(-.1,1.01,1.1,.1), loc="lower left",
                 mode="expand", ncol=2)

plt.show
# plt.savefig('../../Figures/mnist/mnist_flipping.pdf', bbox_inches='tight')




