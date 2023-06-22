import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from numpy import linspace

plt.rcParams["font.family"] = "palatino linotype"
plt.rcParams.update({'font.size': 35,'font.weight':'bold','pdf.fonttype':42})


bar_width= .20
FedSTD=[1.00,1.00,1.00]
FedAVG=[3.10/3.14*1.00, 3.10/3.14*1.00,3.10/3.14*1.00]
median =[31.04/3.14*1.00, 51.77/3.14*1.00,65.1/3.12*1.00]
TM =[16.77/3.14*1.00, 27/3.14*1.00,34/3.12*1.00]

labels = ['1-VGG','2-VGG' , '3-VGG']
x = np.arange(len(labels))

r1 = np.arange(len(FedSTD))
r2 = [x + bar_width for x in r1]
r3 = [x + bar_width for x in r2]
r4 = [x + bar_width for x in r3]
bla= [x + bar_width/2 for x in r2]

    # Make the plot


fig,ax = plt.subplots(figsize =(10, 8))
plt.grid(zorder=1, color='#999999', linewidth=2.0, linestyle='--', alpha=0.5)

plt.bar(r1, FedSTD, color='#005CAB', width=bar_width, edgecolor='black', label='FedASL', zorder=5)
plt.bar(r2, FedAVG, color='#E31B23', width=bar_width, edgecolor='black', label='FedAVG', zorder=5)
plt.bar(r3, TM, color='darkorange', width=bar_width, edgecolor='black', label='Trimmed Mean', zorder=5)
plt.bar(r4, median, color='darkslategrey', width=bar_width, edgecolor='black', label='Median', zorder=5)

# Add some text for labels, title and custom x-axis tick labels, etc.
plt.ylabel('Normalized \n Computation Cost', fontweight ='bold')
plt.xlabel('Different Model', fontweight ='bold')
ax.set_xticks(bla)
ax.set_xticklabels(labels,fontsize=35)
plt.yticks(np.arange(0, 24.01, 3.00))
plt.ylim(0, 24)
plt.legend(bbox_to_anchor=(0.01,0.65,.5,.3),loc='upper left', ncol=1, fontsize=31, borderaxespad=0.02, handlelength=0.7,
               handletextpad=0.2, labelspacing=0.2, columnspacing=0.6)
plt.tight_layout(pad=0.0, h_pad=0.0, w_pad=0.0)


# def autolabel(rects):
#     """Attach a text label above each bar in *rects*, displaying its height."""
#     for rect in rects:
#         height = rect.get_height()
#         ax.annotate('{}'.format(height),
#                     xy=(rect.get_x() + rect.get_width() / 2, height),
#                     xytext=(0, 2),  # 3 points vertical offset
#                     textcoords="offset points", fontsize=25,color ='black',
#                     ha='center', va='bottom')
#
#
# autolabel(rects1)
# autolabel(rects2)

fig.tight_layout()

plt.show()
plt.savefig('../../Figures/edge/power point/computation_diff_cifar10.png', bbox_inches='tight')