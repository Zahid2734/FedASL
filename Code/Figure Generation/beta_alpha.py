import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from numpy import linspace
from Code.utils.file_handler import save_file, open_file

plt.rcParams["font.family"] = "Palatino linotype"
plt.rcParams.update({'font.size': 40,'font.weight':'bold','pdf.fonttype':42})

data1= '../../Data/cifar10/beta alpha/alpha_.2_beta_.1.pkl '
beta1= open_file(data1)

data2= '../../Data/cifar10/beta alpha/alpha_.3_beta_.15.pkl '
beta2= open_file(data2)

data3= '../../Data/cifar10/beta alpha/alpha_.4_beta_.2.pkl '
beta3= open_file(data3)

data4= '../../Data/cifar10/beta alpha/alpha_.5_beta_.25.pkl '
beta4= open_file(data4)

data5= '../../Data/cifar10/beta alpha/alpha_.6_beta_.3.pkl '
beta5= open_file(data5)

data6= '../../Data/cifar10/beta alpha/alpha_.7_beta_.35.pkl '
beta6= open_file(data6)

data7= '../../Data/cifar10/beta alpha/alpha_.8_beta_.4.pkl '
beta7= open_file(data7)

data8= '../../Data/cifar10/beta alpha/alpha_.9_beta_.45.pkl '
beta8= open_file(data8)

data9= '../../Data/cifar10/beta alpha/alpha_1_beta_.45.pkl '
beta9= open_file(data5)

data10= '../../Data/cifar10/shuffling/FedAVG_0.3_1_0.3_100_150_shuffle_cifar10.pkl'
beta10= open_file(data10)

beta_alpha= [max(beta1[0]),max(beta2[0]),max(beta3[0]),max(beta4[0]),max(beta5[0]),max(beta6[0]),max(beta7[0]),max(beta8[0]),max(beta9[0]),]

plt.figure(figsize=(20, 12))
plt.grid(zorder=1, color='#999999', linestyle='--', alpha=.04)
# plt.plot(beta1[0], '-o', c='royalblue', linewidth=3.0, zorder=3, markersize=1, mfc='w', mec='royalblue', mew=1,label='alpha .2')
# plt.plot(beta2[0], '-o', c='orange', linewidth=3.0, zorder=3, markersize=1, mfc='w', mec='orange', mew=1,label='alpha 0.3')
# plt.plot(beta3[0], '-o', c='green', linewidth=3.0, zorder=3, markersize=1, mfc='w', mec='green', mew=1,label='alpha 0.4')
# plt.plot(beta4[0], '-o', c='black', linewidth=3.0, zorder=3, markersize=1, mfc='w', mec='black', mew=1,label='alpha 0.5')
# plt.plot(beta5[0], '-o', c='yellow', linewidth=3.0, zorder=3, markersize=1, mfc='w', mec='yellow', mew=1,label='alpha .6')
# plt.plot(beta6[0], '-o', c='grey', linewidth=3.0, zorder=3, markersize=1, mfc='w', mec='grey', mew=1,label='alpha .7')
# plt.plot(beta7[0], '-o', c='aqua', linewidth=3.0, zorder=3, markersize=1, mfc='w', mec='aqua', mew=1,label='alpha 0.8')
# plt.plot(beta9[0], '-o', c='maroon', linewidth=3.0, zorder=3, markersize=1, mfc='w', mec='maroon', mew=1,label='alpha 0.9')
# plt.plot(beta10[0], '-o', c='red', linewidth=3.0, zorder=3, markersize=1, mfc='w', mec='red', mew=1,label='FedAVG')


plt.plot(beta_alpha, '-o', c='black', linewidth=10.0, zorder=5, marker='D',markevery=1,markersize=35, mfc='red', mec='black', mew=1,label='Beta alpha')


plt.legend(loc=4)
my_xticks = ['0.2','0.3','0.4','0.5','0.6','0.7','0.8','0.9','1']
x = np.array([0,1,2,3,4,5,6,7,8])
plt.xticks(x, my_xticks)
plt.xlabel("Alpha value changed with half beta value", weight='bold')
plt.ylabel('Accuracy(%)', weight='bold')
plt.legend(bbox_to_anchor=(.6,0.8,.35,.3), loc="lower left",
                mode="expand", ncol=2)
plt.show
plt.savefig('../../Figures/cifar10/beta_alpha_single.pdf', bbox_inches='tight')

