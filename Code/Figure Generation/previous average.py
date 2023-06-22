import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from numpy import linspace
from Code.utils.file_handler import save_file, open_file

plt.rcParams["font.family"] = "Palatino linotype"
plt.rcParams.update({'font.size': 40,'font.weight':'bold','pdf.fonttype':42})

data1= '../../Data/cifar10/alpha beta/alpha_1_beta_.2.pkl '
beta1= open_file(data1)

data2= '../../Data/cifar10/previous average/previous_average_2.pkl '
beta2= open_file(data2)

data3= '../../Data/cifar10/previous average/previous_average_3.pkl '
beta3= open_file(data3)

data4= '../../Data/cifar10/previous average/previous_average_4.pkl '
beta4= open_file(data4)

data5= '../../Data/cifar10/previous average/previous_average_5.pkl '
beta5= open_file(data5)

data6= '../../Data/cifar10/alpha beta/fed_avg.pkl '
beta6= open_file(data6)

previous_average= [max(beta1[0]),max(beta2[0]),max(beta3[0]),max(beta4[0]),max(beta5[0])]

plt.figure(figsize=(20, 12))
plt.grid(zorder=1, color='#999999', linestyle='--', alpha=.04)
# plt.plot(beta1[0], '-o', c='royalblue', linewidth=3.0, zorder=3, markersize=1, mfc='w', mec='royalblue', mew=1,label='no average')
# plt.plot(beta2[0], '-o', c='orange', linewidth=3.0, zorder=3, markersize=1, mfc='w', mec='orange', mew=1,label='average 2')
# plt.plot(beta3[0], '-o', c='green', linewidth=3.0, zorder=3, markersize=1, mfc='w', mec='green', mew=1,label='average3')
# plt.plot(beta4[0], '-o', c='black', linewidth=3.0, zorder=3, markersize=1, mfc='w', mec='black', mew=1,label='average 3')
# plt.plot(beta5[0], '-o', c='yellow', linewidth=3.0, zorder=3, markersize=1, mfc='w', mec='yellow', mew=1,label='average 4')
# plt.plot(beta6[0], '-o', c='red', linewidth=3.0, zorder=3, markersize=1, mfc='w', mec='red', mew=1,label='FedAVG')


plt.plot(previous_average, '-o', c='black', linewidth=10.0, zorder=5, marker='D',markevery=1,markersize=35, mfc='red', mec='black', mew=1,label='Previous average')


plt.legend(loc=4)
x = np.array([0,1,2,3,4])
plt.xticks(x, x)
plt.xlabel("Average of previous global model", weight='bold')
plt.ylabel('Accuracy(%)', weight='bold')
plt.legend(bbox_to_anchor=(.5,.8,.45,.3), loc="lower left",
                mode="expand", ncol=2)
plt.show
plt.savefig('../../Figures/cifar10/previous_average_single.pdf', bbox_inches='tight')

