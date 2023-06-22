import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from numpy import linspace
from Code.utils.file_handler import save_file, open_file

plt.rcParams["font.family"] = "palatino linotype"
plt.rcParams.update({'font.size': 40,'font.weight':'bold','pdf.fonttype':42})


data6= '../../Data/cifar10/target/target_flipping/FedSTD_loss_0.3_1_0.3_100_200_15_target_flip_cifar10.pkl'
FedSTD_p1= open_file(data6)

data5= '../../Data/cifar10/target/FedSTD_loss_0.3_1_0.3_100_200_0.9_0.2_shuffle_cifar10.pkl'
FedSTD_p3= open_file(data5)

data7= '../../Data/cifar10/target/FedSTD_loss_0.4_1_0.3_100_200_0.9_0.2_shuffle_cifar10.pkl'
FedSTD_p4= open_file(data7)





plt.figure(figsize=(20,13))
plt.grid(zorder=1, color='#999999', linewidth=2.0, linestyle='--', alpha=0.6)

plt.plot(np.array(FedSTD_p3[0])*100, '--', c='red', linewidth=3.0, zorder=4,  mfc='red', mec='red', mew=1, label='FedASL 0.3')
# plt.plot(np.array(FedSTD[0])*100, '-', c='green', linewidth=3.0, zorder=5,  mfc='green', mec='green', mew=1, label='FedSTD')
plt.plot(np.array(FedSTD_p1[0])*100, '-', c='blue', linewidth=3.0, zorder=6,  mfc='blue', mec='blue', mew=1,label='FedASL 0.1')
plt.plot(np.array(FedSTD_p4[0])*100, '-.', c='green', linewidth=3.0, zorder=3,  mfc='green', mec='green', mew=1,label='FedASL 0.4')


plt.xticks(np.arange(0.0, 201, 50))
plt.xlabel("Rounds of Training", weight='bold')
plt.ylabel('Accuracy(%)', weight='bold')
plt.yticks(np.arange(0, 70, 15))
plt.ylim(0, 70)
plt.legend(loc=4,  handlelength=1, ncol=1,
               handletextpad=0.2, labelspacing=0.2)
plt.tight_layout(pad=0.0, h_pad=0.0, w_pad=0.0)



plt.show
# plt.savefig('../../Figures/edge/cifar10_target.pdf', bbox_inches='tight')
