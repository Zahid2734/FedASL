import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from numpy import linspace
from Code.utils.file_handler import save_file, open_file

plt.rcParams["font.family"] = "palatino linotype"
plt.rcParams.update({'font.size': 40,'font.weight':'bold','pdf.fonttype':42})

data1= '../../Data/cifar10/noisy/FedAVG_0.3_1_0.3_100_200_noise_cifar10.pkl'
FedAVG_cifar10= open_file(data1)

data2= '../../Data/cifar10/noisy/FedAVG_plus_FedPA_0.3_1_0.3_100_200_noise_cifar10.pkl'
FedAVG_FedPA_cifar10= open_file(data2)

data3= '../../Data/cifar10/noisy/trimmed_mean_plus_FedPA_0.3_1_0.3_100_200_noise_cifar10.pkl'
TM_FedPA_cifar10= open_file(data3)

data4= '../../Data/cifar10/noisy/trimmed_mean_0.3_1_0.3_100_200_noise_cifar10.pkl'
TM_cifar10= open_file(data4)



plt.figure(figsize=(13.33,6))
plt.grid(zorder=1, color='#999999', linewidth=2.0, linestyle='--', alpha=0.5)
plt.plot(np.array(FedAVG_cifar10[0])*100, '--', c='red', linewidth=3.0, zorder=5, mfc='red', mec='red', mew=1, label='FedAVG')
plt.plot(np.array(TM_cifar10[0])*100, '-.', c='green', linewidth=3.0, zorder=5, mfc='green', mec='green', mew=1, label='TM')
plt.plot(np.array(FedAVG_FedPA_cifar10[0])*100, '-', c='orange', linewidth=3.0, zorder=5, mfc='orange', mec='orange', mew=1,label='FedAVG+FedSRC')
plt.plot(np.array(TM_FedPA_cifar10[0])*100, '-', c='blue', linewidth=3.0, zorder=5, mfc='blue', mec='blue', mew=1,label='TM+FedSRC')

plt.xticks(np.arange(0.0, 201, 50))

plt.xlabel("Rounds of Training", weight='bold')
plt.ylabel('Accuracy (%)', weight='bold')
plt.yticks(np.arange(0, 70, 15))
plt.ylim(0, 70)
plt.legend(loc=4,  handlelength=1,
               handletextpad=0.2, labelspacing=0.2)
plt.tight_layout(pad=0.0, h_pad=0.0, w_pad=0.0)


plt.show
plt.savefig('../../Figures/sigmetrix/Sigmetrix_result_cifar10.pdf', bbox_inches='tight')
# ax1.plot(idx/1000,psd[:,1],'-',dashes=[10,5],dash_capstyle='butt',c='red',linewidth=3.0,zorder=3,markersize=15,mfc='w',mec='red',mew=1,label='410W')