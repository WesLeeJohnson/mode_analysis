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
font_size_ticks = 16
font_size_labels = 20
font_size_title = 24
font_size_legend = 20 
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
N = 5 #number of ions
frot_min = 180 #kHz
frot_max = 220 #kHz
frots = np.linspace(frot_min,frot_max,5,endpoint=True)
ma_list = []
print('frots = ',frots)

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
fig = plt.figure(figsize=(10,10))
axs = [fig.add_subplot(1,5,i+1,projection='3d') for i in range(5)]
plt.show();exit()