import pickle
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')

from Code.utils.file_handler import save_file, open_file


plt.rcParams["font.family"] = "palatino linotype"
plt.rcParams["font.family"] = "palatino linotype"
plt.rcParams.update({'font.size': 40,'font.weight':'bold','pdf.fonttype':42})


# Fed zero
data1= '../../Data/cifar10/shuffling/FedAVG_0_10_shuffle_cifar10.pkl'
FedAVG_0= open_file(data1)

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



plt.figure(figsize=(10,6))
plt.grid(zorder=1, color='#999999', linewidth=2.0, linestyle='--', alpha=0.5)

plt.plot(np.array(FedAVG_0[0])*100, '-', c='red', linewidth=3.0, zorder=6, mfc='red', mec='red', mew=1, label='All good')
plt.plot(np.array(Fed_z_1[0])*100, '-', c='green', linewidth=3.0, zorder=5, mfc='green', mec='green', mew=1, label='1 bad')
plt.plot(np.array(Fed_z_2[0])*100, '-', c='brown', linewidth=3.0, zorder=4,  mfc='brown', mec='brown', mew=1,label='2 bad')
plt.plot(np.array(Fed_z_3[0])*100, '-', c='blue', linewidth=3.0, zorder=3,  mfc='blue', mec='blue', mew=1,label='3 bad')
plt.plot(np.array(Fed_z_4[0])*100, '-', c='orange', linewidth=3.0, zorder=2,  mfc='orange', mec='orange', mew=1,label='4 bad')
plt.plot(np.array(Fed_z_5[0])*100, '-', c='purple', linewidth=3.0, zorder=1,  mfc='purple', mec='purple', mew=1,label='5 bad')

plt.legend(loc=4)
plt.xticks(np.arange(0.0, 201, 50))

plt.xlabel("Rounds of Training", weight='bold')
plt.ylabel('Accuracy (%)', weight='bold')
plt.yticks(np.arange(0, 80, 15))
plt.ylim(0, 75)
plt.legend(bbox_to_anchor=(0.1,-.030,.8,0.2), loc="lower left", fontsize=30,
                mode="expand", ncol=3,handlelength=1,
               handletextpad=0.2, labelspacing=0.2)
plt.tight_layout(pad=0.0, h_pad=0.0, w_pad=0.0)

plt.show
plt.savefig('../../Figures/sigmetrix/Sigmetrix_intro_fedzero_small.pdf', bbox_inches='tight')
