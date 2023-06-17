"""
Author:     Wes Johnson 
Date:       June 16th 2023
Purpose:    This code will plot the structure of the crystal in 3D. 
            The structure will change as the rotation frequency is increased.
How to run: python test_mode_analysis_3D.py
"""
import numpy as np
import matplotlib.pyplot as plt
import mode_analysis_3D as ma3D
from scipy import constants as const
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

#plotting parameters
font_size_ticks = 14
font_size_labels = 18
font_size_title = 22
font_size_legend = 28
font_size_annotation = 18
point_size = 10

#crystal parameters
ionmass = 9.012182 #Be
charge = const.elementary_charge
mass = ionmass*const.atomic_mass
B = 4.4566 #T
wcyc = charge*B/mass
vwall = 5 #V 
omega_z = 2*np.pi*1.58e6 #rad/s
N = 61 #number of ions
frot_min = 200 #kHz
frot_max = 800 #kHz
frots = np.linspace(frot_min,frot_max,5,endpoint=True)
ma_list = []

#run the 3D mode analysis
for frot in frots:
    #run the 3D mode analysis
    ma3D_instance = ma3D.ModeAnalysis(ionmass=ionmass
                                        ,omega_z=omega_z
                                        ,frot=frot
                                        ,B=B
                                        ,N=N
                                        ,Vwall=vwall
                                        ,precision_solving=True
                                        ,method='bfgs'
                                        )
    ma3D_instance.run_3D()
    ma_list.append(ma3D_instance)

#make a 3D plot of the results with 5 subplots
fig = plt.figure(figsize=(16,5))
axs = [fig.add_subplot(1,5,i+1,projection='3d') for i in range(5)]
for i in range(5):
    pos_vect = ma_list[i].uE_3D*1e6
    ma_list[i].show_crystal_3D(ax=axs[i],pos_vect = pos_vect)
    if i == 0:
        xlims = axs[i].get_xlim()
        ylims = axs[i].get_ylim()
        zlims = axs[i].get_zlim()
    else:
        axs[i].set_xlim(xlims)
        axs[i].set_ylim(ylims)
        axs[i].set_zlim(zlims)
    axs[i].set_title(str(frots[i])+' kHz',fontsize=font_size_title)
    axs[i].set_xlabel('x ($\mu$m)',fontsize=font_size_labels)
    axs[i].set_ylabel('y ($\mu$m)',fontsize=font_size_labels)
    axs[i].tick_params(axis='both', which='major', labelsize=font_size_ticks)
    axs[i].legend().remove()
plt.tight_layout()
plt.show();exit()