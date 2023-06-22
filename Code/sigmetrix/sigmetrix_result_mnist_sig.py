import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from numpy import linspace
from Code.utils.file_handler import save_file, open_file

plt.rcParams["font.family"] = "palatino linotype"
plt.rcParams.update({'font.size': 40,'font.weight':'bold','pdf.fonttype':42})



data5= '../../Data/mnist/sigmetrix/FedAVG_noise_mnist_1.pkl'
FedAVG_mnist= open_file(data5)

data6= '../../Data/mnist/sigmetrix/FedAVG_plus_FedPA_batch_test.pkl'
FedAVG_FedPA_mnist= open_file(data6)

data7= '../../Data/mnist/sigmetrix/TM_plus_FedPA_batch_test.pkl'
TM_FedPA_mnist= open_file(data7)

data8= '../../Data/mnist/sigmetrix/TM_mnist_noise.pkl'
TM_mnist= open_file(data8)




plt.figure(figsize=(13.33,8.03))
plt.grid(zorder=1, color='#999999', linestyle='--', alpha=0.4)


plt.plot(np.array(FedAVG_mnist[0])*100, '--', c='red', linewidth=3.0, zorder=1,  mfc='red', mec='red', mew=1, label='FedAVG')
plt.plot(np.array(TM_mnist[0])*100, '-.', c='green', linewidth=3.0, zorder=2,  mfc='green', mec='green', mew=1, label='Trimmed mean')
plt.plot(np.array(FedAVG_FedPA_mnist[0])*100, '-', c='brown', linewidth=3.0, zorder=5,  mfc='brown', mec='brown', mew=1,label='FedAVG+FedSRC')
plt.plot(np.array(TM_FedPA_mnist[0])*100, '-', c='blue', linewidth=3.0, zorder=5,  mfc='blue', mec='blue', mew=1,label='Trimmed mean+FedSRC')

plt.xticks(np.arange(0.0, 201, 50))
plt.xlabel("Rounds of Training", weight='bold')
plt.ylabel('Accuracy(%)', weight='bold')
plt.yticks(np.arange(0, 90, 20))
plt.ylim(0, 90)
plt.legend(loc=4,  handlelength=1,
               handletextpad=0.2, labelspacing=0.2)
plt.tight_layout(pad=0.0, h_pad=0.0, w_pad=0.0)



plt.show
plt.savefig('../../Figures/sigmetrix/Sigmetrix_result_mnist_sig.pdf', bbox_inches='tight')
