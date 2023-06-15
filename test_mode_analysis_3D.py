"""
Author:     Wes Johnson 
Date:       June 15th 2023
Purpose:    This code tests the mode_analysis_3D script. 
How to run: python test_mode_analysis_3D.py
"""
import numpy as np
import matplotlib.pyplot as plt
import mode_analysis_3D as ma3D
from scipy import constants as const

#crystal parameters
ionmass = 9.012182 #Be
charge = const.elementary_charge
mass = ionmass*const.atomic_mass
B = 4.4566 #T
wcyc = charge*B/mass
vwall = 5 #V 
omega_z = 2*np.pi*1.58e6 #rad/s
frot = 180 #kHz
N = 100 #number of ions
ma3D_instance = ma3D.ModeAnalysis(ionmass=ionmass
                                    ,omega_z=omega_z
                                    ,frot=frot
                                    ,B=B
                                    ,N=N
                                    ,Vwall=vwall
                                    )
ma3D_instance.run()
print(ma3D_instance.pot_energy(ma3D_instance.u0)) 
print(ma3D_instance.pot_energy_3D(np.concatenate((ma3D_instance.u0,np.zeros((N,)))))) 
ma3D_instance.run_3D()
fig,ax = plt.subplots()
print(ma3D.instance.uE_3D )
exit() 
#plotting
ma3D_instance.show_axial_freqs()
ma3D_instance.show_cyc_freqs()
ma3D_instance.show_ExB_freqs()
plt.show()
