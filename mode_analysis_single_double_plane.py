"""
Author:     Wes Johnson 
Date:       June 14th 2023
Purpose:    This script investigates the single-double plane 
            transition of the ion crystal in a penning trap 
            as a function of the rotating wall frequency. 
How to use: python mode_analysis_single_double_plane.py
""" 
import mode_analysis as ma
import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as constants
#import multiprocessing as mp
# parameters
N = 61
ionmass = 9.012182
omega_z = 2*np.pi*1.58e6
B = 4.4588
q = constants.e
m = ionmass*constants.atomic_mass
wcyc = q*B/m
wdel = np.sqrt(wcyc**2 - 2*omega_z**2)
wmag = 1/2 * ( wcyc - wdel )
wrot_crit = 1/2*( wcyc - np.sqrt(wcyc**2 - 4*omega_z**2*(.665/np.sqrt(N) +1/2)) )
Vwall = 1
XR = 3.082
num_ma = 5
ma_list = [None]*num_ma
frot_list = np.linspace(wmag*1.01,wrot_crit*0.99,num_ma)/2/np.pi/1000

ma_crit = ma.ModeAnalysis(ionmass=ionmass  
                            ,omega_z=omega_z
                            ,frot=wrot_crit/2/np.pi/1000
                            ,B=B
                            ,N=N
                            ,XR=XR
                            ,Vwall=Vwall
                        )
ma_crit.run()

#plot results
fontsize_label = 20
fontsize_ticks = 16
fontsize_title = 20
fontsize_legend = 16
cmap = plt.cm.get_cmap('seismic')
fig,axs = plt.subplots(nrows=1,ncols=1,figsize=(8,8))
ax = axs
modes = np.hstack([ma_crit.planarEvalsE[:N],ma_crit.axialEvalsE])/2/np.pi/1e6
mode_numbers = np.arange(1,modes.shape[0]+1)
ax.scatter(mode_numbers, modes, s=50, linewidths=1, edgecolors='k', cmap=cmap, c=mode_numbers)
ax.scatter(x=0,y=-100,s=75,linewidths=2,edgecolors='k',c=cmap(0),label='ExB modes')
ax.scatter(x=0,y=-100,s=75,linewidths=2,edgecolors='k',c=cmap(2*N),label='Axial modes')
ax.set_xlabel('Mode Number',fontsize=fontsize_label)
ax.set_ylabel('Mode Frequency (MHz)',fontsize=fontsize_label)
ax.tick_params(axis='both', which='both', labelsize=fontsize_ticks)
ax.set_title('Mode Frequencies for critcal $\omega_r = 2\pi\\times$ %1.0f kHz, N = %d'%(wrot_crit/2/np.pi/1000,N),fontsize=fontsize_title)
ax.set_ylim([-0.02,1.6])
ax.legend(loc='upper left',fontsize=fontsize_legend)
plt.tight_layout()
plt.show()

# run mode analysis
for i in range(num_ma):
    ma_list[i] = ma.ModeAnalysis(ionmass=ionmass
                                    ,omega_z=omega_z
                                    ,frot=frot_list[i]
                                    ,B=B
                                    ,N=N
                                    ,XR=XR
                                    ,Vwall=Vwall
                                )
    beta = ma_list[i].beta
    beta_crit = .665/np.sqrt(ma_list[i].Nion)
    print('beta < beta_crit:'+str(beta < beta_crit) + ', %.3f < %.3f'%(beta,beta_crit))
    print('frot: ',frot_list[i])
    ma_list[i].run()

# plot results
fig,axs = plt.subplots(nrows=1,ncols=1,figsize=(8.5,11))
ax = axs
for i in range(num_ma):
    modes = np.hstack([ma_list[i].planarEvalsE[:N],ma_list[i].axialEvalsE])/2/np.pi/1e6
    mode_numbers = np.arange(1,modes.shape[0]+1)
    ax.scatter(mode_numbers,modes)
    ax.set_xlabel('Mode Number')
    ax.set_ylabel('Mode Frequency (Hz)')
plt.tight_layout()
plt.show()
