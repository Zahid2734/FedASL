import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from numpy import linspace
from Code.utils.file_handler import save_file, open_file

plt.rcParams["font.family"] = "palatino linotype"
plt.rcParams.update({'font.size': 200,'font.weight':'bold','pdf.fonttype':42})

data1= '../../Data/cifar10/noisy/FedAVG_0.3_1_0.3_100_200_noise_cifar10.pkl'
FedAVG_cifar10= open_file(data1)

data2= '../../Data/cifar10/noisy/FedAVG_plus_FedPA_0.3_1_0.3_100_200_noise_cifar10.pkl'
FedAVG_FedPA_cifar10= open_file(data2)

data3= '../../Data/cifar10/noisy/trimmed_mean_plus_FedPA_0.3_1_0.3_100_200_noise_cifar10.pkl'
TM_FedPA_cifar10= open_file(data3)

data4= '../../Data/cifar10/noisy/trimmed_mean_0.3_1_0.3_100_200_noise_cifar10.pkl'
TM_cifar10= open_file(data4)


data5= '../../Data/mnist/noisy/FedAVG_0.3_1_0.3_100_200_noise_mnist.pkl'
FedAVG_mnist= open_file(data5)

data6= '../../Data/mnist/noisy/FedAVG_plus_FedPA_0.3_1_0.3_100_200_noise_mnist.pkl'
FedAVG_FedPA_mnist= open_file(data6)

data7= '../../Data/mnist/noisy/trimmed_mean_plus_FedPA_0.3_1_0.3_100_200_noise_mnist.pkl'
TM_FedPA_mnist= open_file(data7)

data8= '../../Data/mnist/noisy/trimmed_mean_0.3_1_0.3_100_200_noise_mnist.pkl'
TM_mnist= open_file(data8)


data9= '../../Data/Femnist/Noisy/FedAVG_0.3_1_3000_250_noise_Femnist.pkl'
FedAVG_femnit= open_file(data9)

data10= '../../Data/Femnist/Noisy/FedAVG_plus_FedPA_0.3_1_3000_250_noise_Femnist_2.pkl'
FedAVG_FedPA_femnist= open_file(data10)

data11= '../../Data/Femnist/Noisy/trimmed_mean_plus_FedPA_0.3_1_3000_250_noise_Femnist_2.pkl'
TM_FedPA_femnist= open_file(data11)

data12= '../../Data/Femnist/Noisy/trimmed_mean_0.3_1_3000_250_noise_Femnist.pkl'
TM_femnist= open_file(data12)

plt.figure(figsize=(130,40))
plt.grid(zorder=1, color='#999999', linestyle='--', alpha=.04)
plt.subplot(1, 3, 1)
plt.plot(np.array(FedAVG_cifar10[0])*100, '--', c='red', linewidth=35.0, zorder=5, mfc='red', mec='red', mew=1, label='FedAVG')
plt.plot(np.array(TM_cifar10[0])*100, '-.', c='green', linewidth=35.0, zorder=5, mfc='green', mec='green', mew=1, label='TM')
plt.plot(np.array(FedAVG_FedPA_cifar10[0])*100, '-', c='orange', linewidth=35.0, zorder=5, mfc='orange', mec='orange', mew=1,label='FedAVG+FedSRC')
plt.plot(np.array(TM_FedPA_cifar10[0])*100, '-', c='blue', linewidth=35.0, zorder=5, mfc='blue', mec='blue', mew=1,label='TM+FedSRC')
plt.legend(loc=4)
plt.xticks(np.arange(0.0, 201, 50))
plt.title('CIFAR10',weight='bold')
plt.xlabel("Rounds of Training", weight='bold')
plt.ylabel('Accuracy(%)', weight='bold')
plt.yticks(np.arange(0, 70, 15))
plt.ylim(0, 70)
plt.legend(loc=4,  handlelength=1,bbox_to_anchor=(.3,.05,.7,0.2),
               handletextpad=0.2, labelspacing=0.2)
plt.tight_layout(pad=0.0, h_pad=0.0, w_pad=0.0)

plt.subplot(1, 3, 2)
plt.plot(np.array(FedAVG_mnist[0])*100, '--', c='red', linewidth=35.0, zorder=1,  mfc='red', mec='red', mew=1, label='FedAVG')
plt.plot(np.array(TM_mnist[0])*100, '-.', c='green', linewidth=35.0, zorder=2,  mfc='green', mec='green', mew=1, label='TM')
plt.plot(np.array(FedAVG_FedPA_mnist[0])*100, '-',  c='orange', linewidth=35.0, zorder=5, mfc='orange', mec='orange', mew=1,label='FedAVG+FedSRC')
plt.plot(np.array(TM_FedPA_mnist[0])*100, '-', c='blue', linewidth=35.0, zorder=5,  mfc='blue', mec='blue', mew=1,label='TM+FedSRC')

plt.legend(loc=4)
plt.xticks(np.arange(0.0, 201, 50))
plt.title('MNIST',weight='bold')
plt.xlabel("Rounds of Training", weight='bold')
plt.ylabel('Accuracy(%)', weight='bold')
plt.yticks(np.arange(0, 90, 20))
plt.ylim(0, 90)
plt.legend(loc=4,  handlelength=1,bbox_to_anchor=(.3,.05,.7,0.2),
               handletextpad=0.2, labelspacing=0.2)
plt.tight_layout(pad=0.0, h_pad=0.0, w_pad=0.0)

plt.subplot(1, 3, 3)
plt.plot(np.array(FedAVG_femnit[0])*100, '--', c='red', linewidth=35.0, zorder=1, mfc='red', mec='red', mew=1, label='FedAVG')
plt.plot(np.array(TM_femnist[0])*100, '-.', c='green', linewidth=35.0, zorder=2,  mfc='green', mec='green', mew=1, label='TM')
plt.plot(np.array(FedAVG_FedPA_femnist[0])*100, '-', c='orange', linewidth=35.0, zorder=5, mfc='orange', mec='orange', mew=1,label='FedAVG+FedSRC')
plt.plot(np.array(TM_FedPA_femnist[0])*100, '-', c='blue', linewidth=35.0, zorder=5,  mfc='blue', mec='blue', mew=1,label='TM+FedSRC')

plt.legend(loc=4)
plt.xticks(np.arange(0.0, 251, 50))
plt.title('FEMNIST', weight='bold')
plt.xlabel("Rounds of Training", weight='bold')
plt.ylabel('Accuracy(%)', weight='bold')
plt.yticks(np.arange(0, 90, 20))
plt.ylim(0, 90)
plt.legend(loc=4,  handlelength=1,bbox_to_anchor=(.3,.05,.7,0.2),
               handletextpad=0.2, labelspacing=0.2)
plt.tight_layout(pad=0.0, h_pad=0.0, w_pad=0.0)

plt.show
plt.savefig('../../Figures/sigmetrix/Sigmetrix_result.png', bbox_inches='tight')
