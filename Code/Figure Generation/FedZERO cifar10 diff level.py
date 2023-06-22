import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from numpy import linspace
from Code.utils.file_handler import save_file, open_file

plt.rcParams["font.family"] = "palatino linotype"
plt.rcParams.update({'font.size': 40,'font.weight':'bold','pdf.fonttype':42})

data1= '../../Data/cifar10/shuffling/FedAVG_0_10_shuffle_cifar10.pkl'
FedAVG_0= open_file(data1)

data2= '../../Data/cifar10/shuffling/FedZERO_0.1_10_cifar10.pkl'
FedAVG_1= open_file(data2)

data3= '../../Data/cifar10/shuffling/FedZERO_0.2_10_cifar10.pkl'
FedAVG_2= open_file(data3)

data4= '../../Data/cifar10/shuffling/FedZERO_0.3_10_cifar10.pkl'
FedAVG_3= open_file(data4)

data5= '../../Data/cifar10/shuffling/FedZERO_0.4_10_cifar10.pkl'
FedAVG_4= open_file(data5)

data6= '../../Data/cifar10/shuffling/FedZERO_0.5_10_cifar10.pkl'
FedAVG_5= open_file(data6)




plt.figure(figsize=(20, 12))
plt.grid(zorder=1, color='#999999', linestyle='--', alpha=.04)
plt.plot(FedAVG_0[0], '-o', c='royalblue', linewidth=5.0, zorder=5, marker='o',markevery=25,markersize=25, mfc='royalblue', mec='royalblue', mew=1,label='No bad client')
plt.plot(FedAVG_1[0], '-o', c='orange',  linewidth=5.0, zorder=5, marker='*',markevery=25,markersize=25, mfc='orange', mec='orange', mew=1,label='1 bad client')
plt.plot(FedAVG_2[0], '-o', c='green', linewidth=5.0, zorder=5, marker='D',markevery=25,markersize=25, mfc='green', mec='green', mew=1,label='2 bad client')
plt.plot(FedAVG_3[0], '-o', c='black', linewidth=5.0, zorder=5, marker='H',markevery=25,markersize=25, mfc='black', mec='black', mew=1,label='3 bad client')
plt.plot(FedAVG_4[0], '-o', c='brown', linewidth=5.0, zorder=5, marker='v',markevery=25,markersize=25, mfc='brown', mec='brown', mew=1,label='4 bad client')
plt.plot(FedAVG_5[0], '-o', c='red', linewidth=5.0, zorder=5, marker='>',markevery=25,markersize=25, mfc='red', mec='red', mew=1,label='5 bad client')

plt.legend(loc=4)
plt.xticks(np.arange(0.0, 201, 25))
plt.xlabel("Rounds of Training", weight='bold')
plt.ylabel('Accuracy(%)', weight='bold')
plt.legend(bbox_to_anchor=(0,1.02,1,0.2), loc="lower left",
                mode="expand", ncol=3)

plt.show
plt.savefig('../../Figures/cifar10/FedZERO cifar 10 diff level.pdf', bbox_inches='tight')
