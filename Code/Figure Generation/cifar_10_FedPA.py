import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from numpy import linspace
from Code.utils.file_handler import save_file, open_file

plt.rcParams["font.family"] = "palatino linotype"
plt.rcParams.update({'font.size': 40,'font.weight':'bold','pdf.fonttype':42})

data1= '../../Data/cifar10/flipping/FedAVG_0.3_1_0.3_100_200_flip_cifar10.pkl'
FedAVG_3= open_file(data1)

data2= '../../Data/cifar10/flipping/FedSTD_0.3_1_0.3_100_200_0.9_0.2_flip_cifar10.pkl'
FedSTD_3= open_file(data2)

data3= '../../Data/cifar10/flipping/FedPA_test_.3.pkl'
FedPA_3= open_file(data3)

data8= '../../Data/cifar10/flipping/FedPA_test_.4.pkl'
FedPA_4= open_file(data8)


plt.figure(figsize=(40, 12))
plt.grid(zorder=1, color='#999999', linestyle='--', alpha=.04)

# plt.plot(FedSTD_3[0], '-o', c='royalblue', linewidth=3.0, zorder=3, markersize=1, mfc='w', mec='royalblue', mew=1,label='FedSTD')
plt.plot(FedPA_4[0], '-o', c='red', linewidth=3.0, zorder=3, markersize=1, mfc='w', mec='red', mew=1,label='FedAVG')
plt.plot(FedPA_3[0], '-o', c='green', linewidth=3.0, zorder=3, markersize=1, mfc='w', mec='green', mew=1,label='FedPA')
plt.legend(loc=4)
plt.xticks(np.arange(0.0, 201, 50))
plt.xlabel("Rounds of Training", weight='bold')
plt.ylabel('Accuracy(%)', weight='bold')
# plt.legend(bbox_to_anchor=(.1,0,.9,.3), loc="lower left",
#                 mode="expand", ncol=3)



plt.show
plt.savefig('../../Figures/cifar10/cifar10_FedPA.pdf', bbox_inches='tight')

