import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from numpy import linspace
import statistics as st
from Code.utils.file_handler import save_file, open_file

plt.rcParams["font.family"] = "Palatino linotype"
plt.rcParams.update({'font.size': 40,'font.weight':'bold','pdf.fonttype':42})

data1= '../../Data/cifar10/alpha beta/new/beta.01.pkl '
beta1= open_file(data1)

data2= '../../Data/cifar10/alpha beta/new/beta.1.pkl '
beta2= open_file(data2)

data3= '../../Data/cifar10/alpha beta/new/beta.2.pkl '
beta3= open_file(data3)

data4='../../Data/cifar10/alpha beta/new/beta.3.pkl '
beta4= open_file(data4)

data5= '../../Data/cifar10/alpha beta/new/beta.4.pkl '
beta5= open_file(data5)

data6= '../../Data/cifar10/alpha beta/new/beta.5.pkl'
beta6= open_file(data6)

data7= '../../Data/cifar10/alpha beta/new/beta.6.pkl '
beta7= open_file(data7)

data8= '../../Data/cifar10/alpha beta/new/beta.7.pkl '
beta8= open_file(data8)

data9= '../../Data/cifar10/alpha beta/new/beta.8.pkl '
beta9= open_file(data9)

data10= '../../Data/cifar10/alpha beta/new/beta.9.pkl '
beta10= open_file(data10)

data11= '../../Data/cifar10/alpha beta/new/beta1.pkl '
beta11= open_file(data11)

#flip
data12= '../../Data/cifar10/alpha beta/new/beta.01_flip.pkl'
beta12= open_file(data12)

data13= '../../Data/cifar10/alpha beta/new/beta0.1_flip.pkl'
beta13= open_file(data13)

data14= '../../Data/cifar10/alpha beta/new/beta0.2_flip.pkl'
beta14= open_file(data14)

data15= '../../Data/cifar10/alpha beta/new/beta0.3_flip.pkl'
beta15= open_file(data15)

data16= '../../Data/cifar10/alpha beta/new/beta0.4_flip.pkl'
beta16= open_file(data16)

data17= '../../Data/cifar10/alpha beta/new/beta0.5_flip.pkl'
beta17= open_file(data17)

data18= '../../Data/cifar10/alpha beta/new/beta0.6_flip.pkl'
beta18= open_file(data18)

data19= '../../Data/cifar10/alpha beta/new/beta0.7_flip.pkl'
beta19= open_file(data19)

data20= '../../Data/cifar10/alpha beta/new/beta0.8_flip.pkl'
beta20= open_file(data20)

data21= '../../Data/cifar10/alpha beta/new/beta0.9_flip.pkl'
beta21= open_file(data21)

data22= '../../Data/cifar10/alpha beta/new/beta1_flip.pkl'
beta22= open_file(data22)

#noise
data23= '../../Data/cifar10/alpha beta/new/beta0.01_noise.pkl'
beta23= open_file(data23)

data24= '../../Data/cifar10/alpha beta/new/beta0.1_noise.pkl'
beta24= open_file(data24)

data25= '../../Data/cifar10/alpha beta/new/beta0.2_noise.pkl'
beta25= open_file(data25)

data26= '../../Data/cifar10/alpha beta/new/beta0.3_noise.pkl'
beta26= open_file(data26)

data27= '../../Data/cifar10/alpha beta/new/beta0.4_noise.pkl'
beta27= open_file(data27)

data28= '../../Data/cifar10/alpha beta/new/beta0.5_noise.pkl'
beta28= open_file(data28)

data29= '../../Data/cifar10/alpha beta/new/beta0.6_noise.pkl'
beta29= open_file(data29)

data30= '../../Data/cifar10/alpha beta/new/beta0.7_noise.pkl'
beta30= open_file(data30)

data31= '../../Data/cifar10/alpha beta/new/beta0.8_noise.pkl'
beta31= open_file(data31)

data32= '../../Data/cifar10/alpha beta/new/beta0.9_noise.pkl'
beta32= open_file(data32)

data33= '../../Data/cifar10/alpha beta/new/beta1_noise.pkl'
beta33= open_file(data33)


beta_alpha_shuffle= [st.median(beta1[0][125:150])*100,st.median(beta2[0][125:150])*100,st.median(beta3[0][125:150])*100,st.median(beta4[0][125:150])*100,st.median(beta5[0][125:150])*100,
             st.median(beta6[0][125:150])*100,st.median(beta7[0][125:150])*100,st.median(beta8[0][125:150])*100,st.median(beta9[0][125:150])*100,st.median(beta10[0][125:150])*100,
             st.median(beta11[0][125:150])*100]

beta_alpha_flip=[st.median(beta12[0][125:150])*100,st.median(beta13[0][125:150])*100,st.median(beta14[0][125:150])*100,st.median(beta15[0][125:150])*100,
             st.median(beta16[0][125:150])*100,st.median(beta17[0][125:150])*100,st.median(beta18[0][125:150])*100,st.median(beta19[0][125:150])*100,st.median(beta20[0][125:150])*100,
             st.median(beta21[0][125:150])*100,st.median(beta22[0][125:150])*100]

beta_alpha_noise=[st.median(beta23[0][125:150])*100,st.median(beta24[0][125:150])*100,st.median(beta25[0][125:150])*100,st.median(beta26[0][125:150])*100,
             st.median(beta27[0][125:150])*100,st.median(beta28[0][125:150])*100,st.median(beta29[0][125:150])*100,st.median(beta30[0][125:150])*100,st.median(beta31[0][125:150])*100,
             st.median(beta32[0][125:150])*100,st.median(beta33[0][125:150])*100]

plt.figure(figsize=(10, 8))
plt.grid(zorder=1, color='#999999', linewidth=2.0, linestyle='--', alpha=0.5)


plt.plot(beta_alpha_shuffle, '-o', c='red', linewidth=10.0, zorder=6, marker='D',markevery=1,markersize=25, mfc='white', mec='red', mew=1,label='FedASL Shuffle')
plt.plot(beta_alpha_flip, '-o', c='blue', linewidth=10.0, zorder=5, marker='^',markevery=1,markersize=25, mfc='white', mec='blue', mew=1,label='FedASL Flip')
plt.plot(beta_alpha_noise, '-o', c='darkorange', linewidth=10.0, zorder=5, marker='o',markevery=1,markersize=25, mfc='white', mec='darkorange', mew=1,label='FedASL Noise')

plt.legend(loc=4)
my_xticks = ['0.01','0.2','0.4','0.6','0.8','1']
x = np.array([0,2,4,6,8,10])
plt.xticks(x, my_xticks)
plt.yticks(np.arange(51, 65, 3.00))
plt.ylim(54, 63)
plt.xlabel(r"$\beta$ Value ($\alpha$=1) ", weight='bold')
plt.ylabel('Accuracy (%)', weight='bold')
plt.legend(bbox_to_anchor=(0,0,.7,.3), loc="lower left",fontsize=32,
                mode="expand", ncol=1)
plt.show
plt.savefig('../../Figures/edge/power point/beta_effect.png', bbox_inches='tight')

