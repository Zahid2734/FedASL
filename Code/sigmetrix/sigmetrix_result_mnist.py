import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from numpy import linspace
from Code.utils.file_handler import save_file, open_file

plt.rcParams["font.family"] = "palatino linotype"
plt.rcParams.update({'font.size': 40,'font.weight':'bold','pdf.fonttype':42})



data5= '../../Data/mnist/noisy/FedAVG_0.3_1_0.3_100_200_noise_mnist.pkl'
FedAVG_mnist= open_file(data5)

data6= '../../Data/mnist/noisy/FedAVG_plus_FedPA_0.3_1_0.3_100_200_noise_mnist.pkl'
FedAVG_FedPA_mnist= open_file(data6)

data7= '../../Data/mnist/noisy/trimmed_mean_plus_FedPA_0.3_1_0.3_100_200_noise_mnist.pkl'
TM_FedPA_mnist= open_file(data7)

data8= '../../Data/mnist/noisy/trimmed_mean_0.3_1_0.3_100_200_noise_mnist.pkl'
TM_mnist= open_file(data8)



plt.figure(figsize=(13.33,6))
plt.grid(zorder=1, color='#999999', linewidth=2.0, linestyle='--', alpha=0.5)


plt.plot(np.array(FedAVG_mnist[0])*100, '--', c='red', linewidth=3.0, zorder=1,  mfc='red', mec='red', mew=1, label='FedAVG')
plt.plot(np.array(TM_mnist[0])*100, '-.', c='green', linewidth=3.0, zorder=2,  mfc='green', mec='green', mew=1, label='TM')
plt.plot(np.array(FedAVG_FedPA_mnist[0])*100, '-', c='brown', linewidth=3.0, zorder=5,  mfc='brown', mec='brown', mew=1,label='FedAVG+FedSRC')
plt.plot(np.array(TM_FedPA_mnist[0])*100, '-', c='blue', linewidth=3.0, zorder=5,  mfc='blue', mec='blue', mew=1,label='TM+FedSRC')

plt.xticks(np.arange(0.0, 201, 50))
plt.xlabel("Rounds of Training", weight='bold')
plt.ylabel('Accuracy (%)', weight='bold')
plt.yticks(np.arange(0, 90, 20))
plt.ylim(0, 90)
plt.legend(loc=4,  handlelength=1,
               handletextpad=0.2, labelspacing=0.2)
plt.tight_layout(pad=0.0, h_pad=0.0, w_pad=0.0)



plt.show
plt.savefig('../../Figures/sigmetrix/Sigmetrix_result_mnist.pdf', bbox_inches='tight')
