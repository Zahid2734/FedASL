import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from numpy import linspace
from Code.utils.file_handler import save_file, open_file

plt.rcParams["font.family"] = "palatino linotype"
plt.rcParams.update({'font.size': 40,'font.weight':'bold','pdf.fonttype':42})




data9= '../../Data/Femnist/Noisy/FedAVG_0.3_1_3000_250_noise_Femnist.pkl'
FedAVG_femnit= open_file(data9)

data10= '../../Data/Femnist/Noisy/FedAVG_plus_FedPA_0.3_1_3000_250_noise_Femnist_2.pkl'
FedAVG_FedPA_femnist= open_file(data10)

data11= '../../Data/Femnist/Noisy/trimmed_mean_plus_FedPA_0.3_1_3000_250_noise_Femnist_2.pkl'
TM_FedPA_femnist= open_file(data11)

data12= '../../Data/Femnist/Noisy/trimmed_mean_0.3_1_3000_250_noise_Femnist.pkl'
TM_femnist= open_file(data12)


plt.figure(figsize=(13.33,6))
plt.grid(zorder=1, color='#999999', linewidth=2.0, linestyle='--', alpha=0.5)
plt.plot(np.array(FedAVG_femnit[0])*100, '--', c='red', linewidth=3.0, zorder=1, mfc='red', mec='red', mew=1, label='FedAVG')
plt.plot(np.array(TM_femnist[0])*100, '-.', c='green', linewidth=3.0, zorder=2,  mfc='green', mec='green', mew=1, label='TM')
plt.plot(np.array(FedAVG_FedPA_femnist[0])*100, '-', c='brown', linewidth=3.0, zorder=5,  mfc='brown', mec='brown', mew=1,label='FedAVG+FedSRC')
plt.plot(np.array(TM_FedPA_femnist[0])*100, '-', c='blue', linewidth=3.0, zorder=5,  mfc='blue', mec='blue', mew=1,label='TM+FedSRC')

plt.xticks(np.arange(0.0, 251, 50))
plt.xlabel("Rounds of Training", weight='bold')
plt.ylabel('Accuracy (%)', weight='bold')
plt.yticks(np.arange(0, 90, 20))
plt.ylim(0, 90)
plt.legend(loc=4,  handlelength=1,
               handletextpad=0.2, labelspacing=0.2)
plt.tight_layout(pad=0.0, h_pad=0.0, w_pad=0.0)
plt.show
plt.savefig('../../Figures/sigmetrix/Sigmetrix_result_femnist.pdf', bbox_inches='tight')
