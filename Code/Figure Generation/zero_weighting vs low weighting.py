import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from numpy import linspace
from Code.utils.file_handler import save_file, open_file

plt.rcParams["font.family"] = "Palatino linotype"
plt.rcParams.update({'font.size': 40,'font.weight':'bold','pdf.fonttype':42})

data1= '../../Data/cifar10/shuffling/FedSTD_barplot.pkl'
FedSTD= open_file(data1)

data2= '../../Data/cifar10/shuffling/FedAVG_barplot.pkl'
FedAVG= open_file(data2)

data3= '../../Data/cifar10/shuffling/FedZERO_barplot.pkl'
FedZERO= open_file(data3)

plt.figure(figsize=(20, 12))
plt.grid(zorder=1, color='#999999', linestyle='--', alpha=.04)
plt.plot(FedSTD[0], '-o', c='royalblue', linewidth=6.0, zorder=3, markersize=1, mfc='w', mec='royalblue', mew=1,
         label='FedSTD')
plt.plot(FedAVG[0], '-s',  c='red', linewidth=6.0, zorder=3, markersize=1, mfc='w',
         mec='red', mew=1, label='FedAVG')
plt.plot(FedZERO[0], '-d',  c='k', linewidth=6.0, zorder=3, markersize=1, mfc='w',
         mec='k', mew=1, label='zero')

plt.legend(loc=4)
plt.xticks(np.arange(0.0, 101, 25))
plt.xlabel("Epochs", weight='bold')
plt.ylabel('Accuracy(%)', weight='bold')
plt.show
plt.savefig('../../Figures/mnist/zero_weighting_vs_low_weighting.pdf', bbox_inches='tight')

