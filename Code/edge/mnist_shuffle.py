import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from numpy import linspace
from Code.utils.file_handler import save_file, open_file

plt.rcParams["font.family"] = "palatino linotype"
plt.rcParams.update({'font.size': 40,'font.weight':'bold','pdf.fonttype':42})



data5= '../../Data/mnist/shuffling/table/FedAVG_0.3_1_0.3_100_200_shuffle_mnist.pkl'
FedAVG= open_file(data5)

data6= '../../Data/mnist/shuffling/table/FedSTD_0.3_1_0.3_100_200_0.9_0.2_shuffle_mnist.pkl'
FedSTD= open_file(data6)

data7= '../../Data/mnist/shuffling/table/FedSTD_loss_0.3_1_0.3_100_200_1_0.2_shuffle_mnist.pkl'
FedSTD_loss= open_file(data7)

data8= '../../Data/mnist/shuffling/table/median_0.3_1_0.3_100_200_shuffle_mnist.pkl'
median= open_file(data8)

data9= '../../Data/mnist/shuffling/table/trimmed_mean_0.3_1_0.3_100_200_shuffle_mnist.pkl'
TM= open_file(data9)



plt.figure(figsize=(10,8))
plt.grid(zorder=1, color='#999999', linewidth=2.0, linestyle='--', alpha=0.5)

plt.plot(np.array(FedAVG[0])*100, '--', c='red', linewidth=3.0, zorder=4,  mfc='red', mec='red', mew=1, label='FedAVG')
# plt.plot(np.array(FedSTD[0])*100, '-', c='green', linewidth=3.0, zorder=5,  mfc='green', mec='green', mew=1, label='FedSTD')
plt.plot(np.array(FedSTD_loss[0])*100, '-', c='blue', linewidth=3.0, zorder=6,  mfc='blue', mec='blue', mew=1,label='FedASL')
plt.plot(np.array(TM[0])*100, '-.', c='green', linewidth=3.0, zorder=3,  mfc='green', mec='green', mew=1,label='Trimmed Mean')
plt.plot(np.array(median[0])*100, ':', c='magenta', linewidth=3.0, zorder=2,  mfc='magenta', mec='magenta', mew=1,label='Median')

plt.xticks(np.arange(0.0, 201, 50))
plt.xlabel("Rounds of Training", weight='bold')
plt.ylabel('Accuracy (%)', weight='bold')
plt.yticks(np.arange(30, 91, 15))
plt.ylim(30, 90)
plt.legend(loc=4,  handlelength=1,fontsize=32,
               handletextpad=0.2, labelspacing=0.2)
plt.tight_layout(pad=0.0, h_pad=0.0, w_pad=0.0)



plt.show
plt.savefig('../../Figures/edge/power point/mnist_shuffle.png', bbox_inches='tight')
