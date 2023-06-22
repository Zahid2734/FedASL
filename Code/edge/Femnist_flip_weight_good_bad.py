import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from numpy import linspace
from Code.utils.file_handler import save_file, open_file

plt.rcParams["font.family"] = "palatino linotype"
plt.rcParams.update({'font.size': 40,'font.weight':'bold','pdf.fonttype':42})

data7= '../../Data/Femnist/flipping/table/FedSTD_loss_0.3_1_3000_250_0.1_0.01_flip_Femnist.pkl'
FedSTD_loss= open_file(data7)

def avg_good_bad(bla,sla,nla):
    good_count=0
    good_sum=0
    bad_count=0
    bad_sum=0
    for i in range(len(bla)):
        if sla[i] in nla:

            bad_count += 1
            bad_sum += bla[i]
        else:
            good_count += 1
            good_sum += bla[i]

    good_avg= good_sum/good_count
    bad_avg= bad_sum/bad_count
    return good_avg,bad_avg

def weight_good_bad(data):
    good_val=[]
    bad_val=[]
    nla = data[3]
    for i in range(len(data[0])):
        bla = data[2][i]
        sla = data[4][i]
        good_avg, bad_avg= avg_good_bad(bla,sla,nla)
        good_val.append(good_avg)
        bad_val.append(bad_avg)
    return good_val,bad_val

good_weight, bad_weight= weight_good_bad(FedSTD_loss)

plt.figure(figsize=(10,8))
plt.grid(zorder=1, color='#999999', linestyle='--', alpha=0.4)
plt.plot(good_weight, '-', c='blue', linewidth=3.0, zorder=6,  mfc='blue', mec='blue', mew=1,label='Avg_Good')
plt.plot(bad_weight, '--', c='red', linewidth=3.0, zorder=4,  mfc='red', mec='red', mew=1, label='Avg_Bad')

plt.xticks(np.arange(0.0, 251, 50))
plt.xlabel("Rounds of Training", weight='bold')
plt.ylabel('Weight Coefficient', weight='bold')
plt.yticks(np.arange(0, .006, .001))
plt.ylim(0, .006)
plt.legend(bbox_to_anchor=(0.13,.8,.9,.3),loc=4,  handlelength=1, ncol=2,
               handletextpad=0.2, labelspacing=0.2)
plt.tight_layout(pad=0.0, h_pad=0.0, w_pad=0.0)



plt.show
plt.savefig('../../Figures/edge/femnist_shuffle_weight_good_bad.pdf', bbox_inches='tight')
