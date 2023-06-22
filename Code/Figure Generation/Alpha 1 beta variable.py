import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from numpy import linspace
from Code.utils.file_handler import save_file, open_file

plt.rcParams["font.family"] = "Palatino linotype"
plt.rcParams.update({'font.size': 40,'font.weight':'bold','pdf.fonttype':42})

data1= '../../Data/cifar10/alpha beta/alpha_1_beta_0.pkl '
data1= open_file(data1)

data2= '../../Data/cifar10/alpha beta/alpha_1_beta_.01_2.pkl '
data2= open_file(data2)

data3= '../../Data/cifar10/alpha beta/alpha_1_beta_.05_2.pkl '
data3= open_file(data3)

data4= '../../Data/cifar10/alpha beta/alpha_1_beta_.1_2.pkl '
data4= open_file(data4)

data5= '../../Data/cifar10/alpha beta/alpha_1_beta_.2_2.pkl '
data5= open_file(data5)

data6= '../../Data/cifar10/alpha beta/alpha_1_beta_.4.pkl '
data6= open_file(data6)

data7= '../../Data/cifar10/alpha beta/alpha_1_beta_.4_2.pkl '
data7= open_file(data7)

data8= '../../Data/cifar10/alpha beta/alpha_1_beta_.6.pkl '
data8= open_file(data8)

data9= '../../Data/cifar10/alpha beta/alpha_1_beta_.6_2.pkl '
data9= open_file(data9)

data10= '../../Data/cifar10/alpha beta/alpha_1_beta_.8_2.pkl '
data10= open_file(data10)

data11= '../../Data/cifar10/alpha beta/alpha_1_beta_1_2.pkl '
data11= open_file(data11)

alpha_beta= [max(data1[0]),max(data2[0]),max(data3[0]),max(data4[0]),max(data5[0]),max(data7[0]),max(data9[0]),max(data10[0]),max(data11[0])]

plt.figure(figsize=(20, 12))
plt.grid(zorder=1, color='#999999', linestyle='--', alpha=.04)
# plt.plot(data1[0], '-o', c='royalblue', linewidth=3.0, zorder=3, markersize=1, mfc='w', mec='royalblue', mew=1,label='percent 0.1')
# plt.plot(data2[0], '-o', c='orange', linewidth=3.0, zorder=3, markersize=1, mfc='w', mec='orange', mew=1,label='percent 0.2')
# plt.plot(data3[0], '-o', c='green', linewidth=3.0, zorder=3, markersize=1, mfc='w', mec='green', mew=1,label='percent 0.3')
# # plt.plot(data4[0], '-o', c='black', linewidth=3.0, zorder=3, markersize=1, mfc='w', mec='black', mew=1,label='percent 0.4')
# plt.plot(data5[0], '-o', c='yellow', linewidth=3.0, zorder=3, markersize=1, mfc='w', mec='yellow', mew=1,label='percent 0.5')
# plt.plot(data11[0], '-o', c='brown', linewidth=3.0, zorder=3, markersize=1, mfc='w', mec='brown', mew=1,label='percent 0.6')
# plt.plot(data7[0], '-o', c='blue', linewidth=3.0, zorder=3, markersize=1, mfc='w', mec='blue', mew=1,label='percent 0.7')
# plt.plot(data10[0], '-o', c='purple', linewidth=3.0, zorder=3, markersize=1, mfc='w', mec='purple', mew=1,label='percent 0.8')
# plt.plot(data9[0], '-o', c='pink', linewidth=3.0, zorder=3, markersize=1, mfc='w', mec='pink', mew=1,label='percent 0.9')

plt.plot(alpha_beta, '-o', c='black', linewidth=10.0, zorder=5, marker='D',markevery=1,markersize=35, mfc='red', mec='black', mew=1,label='Alpha beta')


plt.legend(loc=4)
my_xticks = ['0','0.01','0.05','0.1','0.2','0.4','0.6','0.8','1']
x = np.array([0,1,2,3,4,5,6,7,8])
plt.xticks(x, my_xticks)
plt.xlabel("Beta value in fixed Alpha", weight='bold')
plt.ylabel('Accuracy(%)', weight='bold')
plt.legend(bbox_to_anchor=(.6,0,.35,.3), loc="lower left",
                mode="expand", ncol=2)
plt.show
plt.savefig('../../Figures/cifar10/alpha beta single.pdf', bbox_inches='tight')
