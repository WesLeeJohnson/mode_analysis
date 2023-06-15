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
N = 2 #number of ions
ma3D_instance = ma3D.ModeAnalysis(ionmass=ionmass
                                    ,omega_z=omega_z
                                    ,frot=frot
                                    ,B=B
                                    ,N=N
                                    ,Vwall=vwall
                                    )
ma3D_instance.run()
test_pos_2D = ma3D_instance.u0
test_pos_3D = np.concatenate((test_pos_2D,np.zeros((N,))))
#print(ma3D_instance.pot_energy_3D(test_pos_3D)) 
#print(ma3D_instance.pot_energy(test_pos_2D))
#print(ma3D_instance.force_penning(test_pos_2D) - ma3D_instance.force_penning_3D(test_pos_3D)[:2*N])
#print(ma3D_instance.hessian_penning_3D(test_pos_3D))
#print(ma3D_instance.hessian_penning_3D(test_pos_3D)[:2*N,:2*N] - ma3D_instance.hessian_penning(test_pos_2D))
print(ma3D_instance.calc_axial_hessian(test_pos_2D))
print(ma3D_instance.hessian_penning_3D(test_pos_3D)[2*N:,2*N:]) 
exit() 
ma3D_instance.run_3D()
fig,ax = plt.subplots()
print(ma3D.instance.uE_3D )
exit() 
#plotting
ma3D_instance.show_axial_freqs()
ma3D_instance.show_cyc_freqs()
ma3D_instance.show_ExB_freqs()
plt.show()
