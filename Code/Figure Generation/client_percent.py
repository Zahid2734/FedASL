import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from numpy import linspace
from Code.utils.file_handler import save_file, open_file

plt.rcParams["font.family"] = "Palatino linotype"
plt.rcParams.update({'font.size': 40,'font.weight':'bold','pdf.fonttype':42})

# data1= '../../Data/cifar10/client percent/client_percent.1.pkl '
# data1= open_file(data1)

data2= '../../Data/cifar10/client percent/client_percent.2.pkl '
data2= open_file(data2)

data3= '../../Data/cifar10/client percent/client_percent.3.pkl '
data3= open_file(data3)

data4= '../../Data/cifar10/client percent/client_percent.4.pkl '
data4= open_file(data4)

data5= '../../Data/cifar10/client percent/client_percent.5.pkl '
data5= open_file(data5)

data6= '../../Data/cifar10/client percent/client_percent.6.pkl '
data6= open_file(data6)

data7= '../../Data/cifar10/client percent/client_percent.7.pkl '
data7= open_file(data7)

data8= '../../Data/cifar10/client percent/client_percent.8.pkl '
data8= open_file(data8)

data9= '../../Data/cifar10/client percent/client_percent.9.pkl '
data9= open_file(data9)

client_percent= [max(data2[0]),max(data3[0]),max(data4[0]),max(data5[0]),max(data6[0]),max(data7[0]),max(data8[0]),max(data9[0])]


plt.figure(figsize=(20, 12))
plt.grid(zorder=1, color='#999999', linestyle='--', alpha=.04)
# plt.plot(data1[0], '-o', c='royalblue', linewidth=3.0, zorder=3, markersize=1, mfc='w', mec='royalblue', mew=1,label='percent 0.1')
# plt.plot(data2[0], '-o', c='orange', linewidth=3.0, zorder=3, markersize=1, mfc='w', mec='orange', mew=1,label='percent 0.2')
# plt.plot(data3[0], '-o', c='green', linewidth=3.0, zorder=3, markersize=1, mfc='w', mec='green', mew=1,label='percent 0.3')
# plt.plot(data4[0], '-o', c='black', linewidth=3.0, zorder=3, markersize=1, mfc='w', mec='black', mew=1,label='percent 0.4')
# plt.plot(data5[0], '-o', c='yellow', linewidth=3.0, zorder=3, markersize=1, mfc='w', mec='yellow', mew=1,label='percent 0.5')
# plt.plot(data10[0], '-o', c='brown', linewidth=3.0, zorder=3, markersize=1, mfc='w', mec='brown', mew=1,label='percent 0.6')
# plt.plot(data7[0], '-o', c='blue', linewidth=3.0, zorder=3, markersize=1, mfc='w', mec='blue', mew=1,label='percent 0.7')
# plt.plot(data8[0], '-o', c='purple', linewidth=3.0, zorder=3, markersize=1, mfc='w', mec='purple', mew=1,label='percent 0.8')
# plt.plot(data9[0], '-o', c='pink', linewidth=3.0, zorder=3, markersize=1, mfc='w', mec='pink', mew=1,label='percent 0.9')



plt.plot(client_percent, '-o', c='black', linewidth=10.0, zorder=5, marker='D',markevery=1,markersize=35, mfc='red', mec='black', mew=1,label='CLient percentage')


plt.legend(loc=4)
my_xticks = ['0.2','0.3','0.4','0.5','0.6','0.7','0.8','0.9']
x = np.array([0,1,2,3,4,5,6,7])
plt.xticks(x, my_xticks)
plt.xlabel("Percentage of selected clients for training", weight='bold')
plt.ylabel('Accuracy(%)', weight='bold')
plt.legend(bbox_to_anchor=(.5,0,.45,.3), loc="lower left",
                mode="expand", ncol=2)
plt.show
plt.savefig('../../Figures/cifar10/client_percent_single.pdf', bbox_inches='tight')

