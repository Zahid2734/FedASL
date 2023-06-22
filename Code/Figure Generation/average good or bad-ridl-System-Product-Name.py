import pickle
import tkinter
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
from matplotlib import cm
import matplotlib.font_manager
from numpy import linspace
from Code.utils.file_handler import save_file, open_file


def average_good_bad(client_acc, bad_client):
    good_average = []
    bad_average = []
    for i in range(len(client_acc)):
        good = 0
        bad = 0
        x = 0
        for j in range(len(client_acc[0])):
            if bad_client[x] == j:
                bad += client_acc[i][j]
                if x < len(bad_client) - 1:
                    x += 1
            else:
                good += client_acc[i][j]
        avg_good = good / (len(client_acc[0]) - len(bad_client))
        avg_bad = bad / len(bad_client)
        good_average.append(avg_good)
        bad_average.append(avg_bad)
    return good_average, bad_average


plt.rcParams["font.family"] = "palatino linotype"
plt.rcParams.update({'font.size': 40,'font.weight':'bold','pdf.fonttype':42})

data1= '../../Data/cifar10/shuffling/FedAVG 0.4 for barplot.pkl'
FedAVG= open_file(data1)

global_acc= FedAVG[0]
client_acc= FedAVG[1]
bad_client= FedAVG[3]

good_average, bad_average=average_good_bad(client_acc, bad_client)

plt.figure(figsize=(20, 12))
plt.grid(zorder=1, color='#999999', linestyle='--', alpha=.04)
plt.plot(good_average, '-o', c='royalblue', linewidth=5.0, zorder=5, marker='o',markevery=25,markersize=25, mfc='royalblue', mec='royalblue', mew=1,label='Average good clients training accuracy')
plt.plot(bad_average, '-o', c='orange',  linewidth=5.0, zorder=5, marker='*',markevery=25,markersize=25, mfc='orange', mec='orange', mew=1,label='Average bad clients training accuracy')
plt.plot(global_acc, '-o', c='red', linewidth=5.0, zorder=5, marker='>',markevery=25,markersize=25, mfc='red', mec='red', mew=1,label='Global accuracy')

plt.legend(loc=4)
plt.xticks(np.arange(0.0, 201, 25))
plt.xlabel("Rounds of Training", weight='bold')
plt.ylabel('Accuracy(%)', weight='bold')
plt.legend(bbox_to_anchor=(0,1.02,1,0.2), loc="lower left",
                mode="expand", ncol=1)

plt.show
# plt.savefig('../../Figures/cifar10/Average good and bad.pdf', bbox_inches='tight')
