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

data1= '../../Data/cifar10/shuffling/FedAVG_0_10_shuffle_cifar10.pkl'
FedAVG_0= open_file(data1)

data2= '../../Data/cifar10/shuffling/FedAVG_0.1_10_shuffle_cifar10.pkl'
FedAVG_1= open_file(data2)

data3= f'../../Data/cifar10/shuffling/FedAVG_0.2_10_shuffle_cifar10.pkl'
FedAVG_2= open_file(data3)

data4= '../../Data/cifar10/shuffling/FedAVG_0.3_10_shuffle_cifar10_1.pkl'
FedAVG_3= open_file(data4)

data5= '../../Data/cifar10/shuffling/FedAVG_0.4_10_shuffle_cifar10.pkl'
FedAVG_4= open_file(data5)

data6= '../../Data/cifar10/shuffling/FedAVG_0.5_10_shuffle_cifar10.pkl'
FedAVG_5= open_file(data6)

# Fed zero
data7= '../../Data/cifar10/shuffling/FedZERO_0.1_10_cifar10.pkl'
Fed_z_1= open_file(data7)

data8= '../../Data/cifar10/shuffling/FedZERO_0.2_10_cifar10.pkl'
Fed_z_2= open_file(data8)

data9= '../../Data/cifar10/shuffling/FedZERO_0.3_10_cifar10.pkl'
Fed_z_3= open_file(data9)

data10= '../../Data/cifar10/shuffling/FedZERO_0.4_10_cifar10.pkl'
Fed_z_4= open_file(data10)

data11= '../../Data/cifar10/shuffling/FedZERO_0.5_10_cifar10.pkl'
Fed_z_5= open_file(data11)

# average good and bad
data12= '../../Data/cifar10/shuffling/FedAVG 0.4 for barplot.pkl'
FedAVG= open_file(data12)

global_acc= FedAVG[0]
client_acc= FedAVG[1]
bad_client= FedAVG[3]

good_average, bad_average=average_good_bad(client_acc, bad_client)

plt.figure(figsize=(50,12))
plt.grid(zorder=1, color='#999999', linestyle='--', alpha=.04)
plt.subplot(1, 3, 1)
plt.plot(FedAVG_0[0], '-o', c='red', linewidth=3.0, zorder=5, marker='o',markevery=25,markersize=15, mfc='red', mec='red', mew=1, label='All good')
plt.plot(FedAVG_1[0], '-o', c='green', linewidth=3.0, zorder=5, marker='*',markevery=35,markersize=15, mfc='green', mec='green', mew=1, label='1 bad')
plt.plot(FedAVG_2[0], '-o', c='brown', linewidth=3.0, zorder=5, marker='>',markevery=40,markersize=15, mfc='brown', mec='brown', mew=1,label='2 bad')
plt.plot(FedAVG_3[0], '-o', c='blue', linewidth=3.0, zorder=5, marker='D',markevery=30,markersize=15, mfc='blue', mec='blue', mew=1,label='3 bad')
plt.plot(FedAVG_4[0], '-o', c='orange', linewidth=3.0, zorder=5, marker='>',markevery=40,markersize=15, mfc='orange', mec='orange', mew=1,label='4 bad')
plt.plot(FedAVG_5[0], '-o', c='purple', linewidth=3.0, zorder=5, marker='D',markevery=30,markersize=15, mfc='purple', mec='purple', mew=1,label='5 bad')

plt.legend(loc=4)
plt.xticks(np.arange(0.0, 201, 50))

plt.xlabel("Rounds of Training", weight='bold')
plt.ylabel('Accuracy(%)', weight='bold')
plt.legend(bbox_to_anchor=(-.05,1,1.1,0.2), loc="center",
                mode="expand", ncol=3)

plt.subplot(1, 3, 2)
plt.plot(FedAVG_0[0], '-o', c='red', linewidth=3.0, zorder=5, marker='o',markevery=25,markersize=15, mfc='red', mec='red', mew=1, label='All good')
plt.plot(Fed_z_1[0], '-o', c='green', linewidth=3.0, zorder=5, marker='*',markevery=35,markersize=15, mfc='green', mec='green', mew=1, label='1 bad')
plt.plot(Fed_z_2[0], '-o', c='brown', linewidth=3.0, zorder=5, marker='>',markevery=40,markersize=15, mfc='brown', mec='brown', mew=1,label='2 bad')
plt.plot(Fed_z_3[0], '-o', c='blue', linewidth=3.0, zorder=5, marker='D',markevery=30,markersize=15, mfc='blue', mec='blue', mew=1,label='3 bad')
plt.plot(Fed_z_4[0], '-o', c='orange', linewidth=3.0, zorder=5, marker='>',markevery=40,markersize=15, mfc='orange', mec='orange', mew=1,label='4 bad')
plt.plot(Fed_z_5[0], '-o', c='purple', linewidth=3.0, zorder=5, marker='D',markevery=30,markersize=15, mfc='purple', mec='purple', mew=1,label='5 bad')

plt.legend(loc=4)

plt.xlabel("Rounds of Training", weight='bold')
plt.ylabel('Accuracy(%)', weight='bold')
plt.legend(bbox_to_anchor=(-.05,1,1.1,0.2), loc="center",
                mode="expand", ncol=3)

plt.subplot(1, 3, 3)
plt.plot(good_average, '-o', c='red', linewidth=3.0, zorder=5, marker='o',markevery=25,markersize=15, mfc='red', mec='red', mew=1, label='Avg good')
plt.plot(bad_average, '-o', c='green', linewidth=3.0, zorder=5, marker='*',markevery=35,markersize=15, mfc='green', mec='green', mew=1, label='Avg bad')
plt.plot(global_acc, '-o', c='blue', linewidth=3.0, zorder=5, marker='D',markevery=30,markersize=15, mfc='blue', mec='blue', mew=1,label='Global accuracy')

plt.legend(loc=4)
plt.xticks(np.arange(0.0, 201, 50))

plt.xlabel("Rounds of Training", weight='bold')
plt.ylabel('Accuracy(%)', weight='bold')
plt.legend(bbox_to_anchor=(-.05,1,1.1,0.2), loc="center",
                mode="expand", ncol=2)

plt.show
# plt.savefig('../../Figures/intro_Sigmetrix.pdf', bbox_inches='tight')
