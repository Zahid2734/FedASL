import pickle
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')

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
plt.rcParams["font.family"] = "palatino linotype"
plt.rcParams.update({'font.size': 40,'font.weight':'bold','pdf.fonttype':42})


# average good and bad
data12= '../../Data/cifar10/shuffling/FedAVG 0.4 for barplot.pkl'
FedAVG= open_file(data12)

global_acc= FedAVG[0]
client_acc= FedAVG[1]
bad_client= FedAVG[3]

good_average, bad_average=average_good_bad(client_acc, bad_client)
good_average=np.array(good_average)
bad_average=np.array(bad_average)
global_acc=np.array(global_acc)

plt.figure(figsize=(13,10))
plt.grid(zorder=1, color='#999999', linewidth=2.0, linestyle='--', alpha=0.5)
plt.plot(good_average*100, '-', c='red', linewidth=3.0, zorder=5,  mfc='red', mec='red', mew=1, label='Avg good')
plt.plot(global_acc*100, '-', c='blue', linewidth=3.0, zorder=1,  mfc='blue', mec='blue', mew=1,label='Global accuracy')
plt.plot(bad_average*100, '-', c='green', linewidth=3.0, zorder=2,  mfc='green', mec='green', mew=1, label='Avg bad')


plt.xticks(np.arange(0.0, 201, 50))

plt.xlabel("Rounds of Training", weight='bold')
plt.ylabel('Accuracy (%)', weight='bold')
plt.yticks(np.arange(0, 80, 15))
plt.ylim(0, 75)
plt.legend(bbox_to_anchor=(.5,.55,.5,0.2), loc="lower left",
                mode="expand", ncol=1,handlelength=1,
               handletextpad=0.2, labelspacing=0.2)
plt.tight_layout(pad=0.0, h_pad=0.0, w_pad=0.0)
plt.show
plt.savefig('../../Figures/sigmetrix/Sigmetrix_intro_avg_good_bad.png', bbox_inches='tight')
