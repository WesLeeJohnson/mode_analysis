"""
The purpose of this script is to compare the results of the new mode analysis
"""

import mode_analysis as ma_new 
import mode_analysis_code_original as ma_old
import matplotlib.pyplot as plt
import numpy as np

# parameters 
ionmass = 9.012182
omega_z = 2*np.pi*1.58e6
frot_kHz = 180
B = 4.4588 
Vwall = 1
N = 19
XR = 3.082
ma_new_instance = ma_new.ModeAnalysis(ionmass=ionmass
                                        ,omega_z=omega_z
                                        ,frot=frot_kHz
                                        ,B=B
                                        ,N=N
                                        ,XR=XR
                                        ,Vwall=Vwall
                                    )
ma_old_instance = ma_old.ModeAnalysis(ionmass=ionmass
                                        ,omega_z=omega_z
                                        ,frot=frot_kHz
                                        ,B=B
                                        ,N=N
                                        ,XR=XR
                                        ,Vwall=Vwall
                                    )

ma_new_instance.run()
ma_old_instance.run()

nrows,ncols = 3,3
half = int(nrows*ncols/2+1/2)
fig,axs = plt.subplots(nrows=nrows,ncols=ncols,figsize=(10,10))
modes = np.hstack((np.arange(0,half),np.arange(N,N+half)))
mode = 0
for row in range(nrows):
    for col in range(ncols):
        ax = axs[row,col]
        ma_new_instance.show_crystal_planar_mode(mode=modes[mode],ax=ax)
        mode += 1
plt.tight_layout()
plt.tight_layout()
plt.show()

nrows,ncols = 4,4
fig,axs = plt.subplots(nrows=nrows,ncols=ncols,figsize=(10,10))
mode = 18
for row in range(nrows):
    for col in range(ncols):
        ax = axs[row,col]
        ma_new_instance.show_crystal_axial_mode(mode=mode,ax=ax)
        mode -= 1
plt.tight_layout()
plt.show()

ma_new_instance.show_cyc_freqs()
ma_new_instance.show_ExB_freqs()
ma_new_instance.show_axial_freqs()
plt.show()

print(ma_new_instance.axialEvects.shape)
fig,axs = plt.subplots(nrows=1,ncols=2,figsize=(10,5))
ax1 = axs[0]
ma_new_instance.show_crystal(ax = ax1,label='new')
x = ma_old_instance.uE[:ma_old_instance.Nion]
y = ma_old_instance.uE[ma_old_instance.Nion:]
ma_new_instance.show_crystal(pos_vect=ma_old_instance.uE,ax=ax1,label='old',color='red')
ax2 = axs[1]
pln_modes_freqs = ma_new_instance.planarEvalsE
axl_modes_freqs = ma_new_instance.axialEvals
all_modes_freqs = np.sort(np.array([*pln_modes_freqs,*axl_modes_freqs]))
ax2.scatter(np.arange(len(all_modes_freqs)),all_modes_freqs,color='blue',label='new')
pln_modes_freqs = ma_old_instance.planarEvalsE
axl_modes_freqs = ma_old_instance.axialEvals
all_modes_freqs = np.sort(np.array([*pln_modes_freqs,*axl_modes_freqs]))
ax2.scatter(np.arange(len(all_modes_freqs)),all_modes_freqs,color='red',label='old')
ax2.set_xlabel('Mode number')
ax2.set_ylabel('Frequency (MHz)')
ax2.set_title('Mode frequencies')
ax2.legend()
plt.show()
