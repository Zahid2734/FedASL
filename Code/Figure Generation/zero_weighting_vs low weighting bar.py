import numpy as np
IT= [0.05364749, 0.05364749, 0.05364749, 0.05364749, 0.05364749,
       0.01257071, 0.05364749, 0.05364749, 0.05364749, 0.01264562]

ECE= [0.04167 for i in range(10)]
CSE= [0.0588, 0.0588,0.0588,0.0588,0.0588,0,0.0588,0.0588,0.0588,0]

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Palatino linotype"
plt.rcParams.update({'font.size': 40,'font.weight':'bold','pdf.fonttype':42})
# set width of bar
barWidth = .25
fig = plt.subplots(figsize =(20, 12))


# Set position of bar on X axis
br1 = np.arange(len(IT))
br2 = [x + barWidth for x in br1]
br3 = [x + barWidth for x in br2]

# Make the plot
plt.bar(br1, IT, color ='r', width = barWidth,
		edgecolor ='grey', label ='FedSTD')
plt.bar(br2, ECE, color ='g', width = barWidth,
		edgecolor ='grey', label ='FedAVG')
plt.bar(br3, CSE, color ='b', width = barWidth,
		edgecolor ='grey', label ='Zero_weighting')

# Adding Xticks
plt.xlabel('Client ID', fontweight ='bold')
plt.ylabel('Avarage Weight per client', fontweight ='bold')


plt.legend(bbox_to_anchor=(0,1.02,1,1.02), loc="lower left",
                mode="expand", ncol=3)
plt.show()
# plt.savefig('../../Figures/mnist/zero_weighting_vs_low_weighting_bar.pdf', bbox_inches='tight')
