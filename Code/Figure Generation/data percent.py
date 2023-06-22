
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from numpy import linspace
from Code.utils.file_handler import save_file, open_file

plt.rcParams["font.family"] = "Palatino linotype"
plt.rcParams.update({'font.size': 40,'font.weight':'bold','pdf.fonttype':42})

data1= '../../Data/cifar10/data percent/FedSTD_.1.pkl'
fedSTD1= open_file(data1)

data2= '../../Data/cifar10/data percent/FedSTD_.3.pkl'
fedSTD2= open_file(data2)

data3= '../../Data/cifar10/data percent/FedSTD_.5.pkl'
fedSTD3= open_file(data3)

data4= '../../Data/cifar10/data percent/FedSTD_.75.pkl'
fedSTD4= open_file(data4)

data5= '../../Data/cifar10/data percent/FedVG_.1.pkl'
fedAVG1= open_file(data5)

data6= '../../Data/cifar10/data percent/FedVG_.3.pkl'
fedAVG2= open_file(data6)

data7= '../../Data/cifar10/data percent/FedVG_.5.pkl'
fedAVG3= open_file(data7)

data8= '../../Data/cifar10/data percent/FedVG_.75.pkl'
fedAVG4= open_file(data8)

list_std= [max(fedSTD1[0]), max(fedSTD2[0]),max(fedSTD3[0]),max(fedSTD4[0])]
list_avg = [max(fedAVG1[0]),max(fedAVG2[0]),max(fedAVG3[0]),max(fedAVG4[0])]

plt.figure(figsize=(20, 12))
plt.grid(zorder=1, color='#999999', linestyle='--', alpha=.04)
# plt.plot(fedSTD1[0], '-o', c='royalblue', linewidth=3.0, zorder=3, markersize=1, mfc='w', mec='royalblue', mew=1,label='FedSTD_.1')
# plt.plot(fedSTD2[0], '-o', c='orange', linewidth=3.0, zorder=3, markersize=1, mfc='w', mec='orange', mew=1,label='FedSTD_.3')
# plt.plot(fedSTD3[0], '-o', c='green', linewidth=3.0, zorder=3, markersize=1, mfc='w', mec='green', mew=1,label='FedSTD_.5')
# plt.plot(fedSTD4[0], '-o', c='black', linewidth=3.0, zorder=3, markersize=1, mfc='w', mec='black', mew=1,label='FedSTD_.75')
# plt.plot(fedAVG1[0], '-o', c='yellow', linewidth=3.0, zorder=3, markersize=1, mfc='w', mec='yellow', mew=1,label='FedAVG_.1')
# plt.plot(fedAVG2[0], '-o', c='red', linewidth=3.0, zorder=3, markersize=1, mfc='w', mec='red', mew=1,label='FedAVG_.3')
# plt.plot(fedAVG3[0], '-o', c='pink', linewidth=3.0, zorder=3, markersize=1, mfc='w', mec='pink', mew=1,label='FedAVG_.5')
# plt.plot(fedAVG4[0], '-o', c='brown', linewidth=3.0, zorder=3, markersize=1, mfc='w', mec='brown', mew=1,label='FedAVG_.75')


plt.plot(list_std, '-o', c='royalblue', linewidth=10.0, zorder=5, marker='o',markevery=1,markersize=35, mfc='cyan', mec='royalblue', mew=1,label='FedSTD')
plt.plot(list_avg, '-o', c='red', linewidth=10.0, zorder=5, marker='^',markevery=1,markersize=35, mfc='black', mec='red', mew=1,label='FedAVG')


plt.legend(loc=4)
my_xticks = ['0.1','0.3','0.5','0.75']
x = np.array([0,1,2,3])
plt.xticks(x, my_xticks)
plt.xlabel("Bad Data Percent", weight='bold')
plt.ylabel('Accuracy(%)', weight='bold')
plt.legend(bbox_to_anchor=(.1,0,.6,.3), loc="lower left",
                mode="expand", ncol=2)
plt.show
# plt.savefig('../../Figures/cifar10/data percent.pdf', bbox_inches='tight')

