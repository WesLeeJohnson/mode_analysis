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
vwall = 20 #V 
omega_z = 2*np.pi*1.58e6 #rad/s
frot = wcyc/(2*np.pi*1000)/2*1.9 #kHz
frot = 180 #kHz
N = 250 #number of ions
ma3D_instance = ma3D.ModeAnalysis(ionmass=ionmass
                                    ,omega_z=omega_z
                                    ,frot=frot
                                    ,B=B
                                    ,N=N
                                    ,Vwall=vwall
                                    ,precision_solving=True
                                    )
ma3D_instance.run()
test_pos_2D = ma3D_instance.u0
test_pos_3D = np.concatenate((test_pos_2D,np.zeros((N,))))
#print(ma3D_instance.pot_energy_3D(test_pos_3D)) 
#print(ma3D_instance.pot_energy(test_pos_2D))
#print(ma3D_instance.force_penning(test_pos_2D) - ma3D_instance.force_penning_3D(test_pos_3D)[:2*N])
#print(ma3D_instance.hessian_penning_3D(test_pos_3D))
#print(ma3D_instance.hessian_penning_3D(test_pos_3D)[:2*N,:2*N] - ma3D_instance.hessian_penning(test_pos_2D))
#print(ma3D_instance.calc_axial_hessian(test_pos_2D)) #scaling is off by a factor of minus 2
#print(ma3D_instance.hessian_penning_3D(test_pos_3D)[2*N:,2*N:]) 
#print(test_pos_2D)
#print(ma3D_instance.u) 
ma3D_instance.run_3D()
test_pos_2D = ma3D_instance.u_3D[:2*N]
test_pos_3D = ma3D_instance.u_3D
ori_pos_3D = np.concatenate((ma3D_instance.u,np.zeros((N,))))
#ori_pos_3D[N:2*N] = -1*ori_pos_3D[N:2*N]
#ori_pos_3D[:N] = -1*ori_pos_3D[:N]
ax = ma3D_instance.show_crystal_3D(pos_vect=test_pos_3D,label='3D, PE = {:.6f} '.format(ma3D_instance.pot_energy_3D(test_pos_3D)),color='b')
ma3D_instance.show_crystal_3D(pos_vect=ori_pos_3D,ax=ax,color='r',label='2D, PE = {:.6f} '.format(ma3D_instance.pot_energy_3D(ori_pos_3D)))
print(ma3D_instance.pot_energy_3D(test_pos_3D))
print(ma3D_instance.pot_energy_3D(ori_pos_3D))
plt.show();exit()
#test_pos_3D[2*N:] = 0
#print(ma3D_instance.pot_energy_3D(test_pos_3D))
#print(ma3D_instance.pot_energy(test_pos_2D))
#print(ma3D_instance.force_penning(test_pos_2D) - ma3D_instance.force_penning_3D(test_pos_3D)[:2*N])
#ma3D_instance.show_crystal_3D()
#plotting
ma3D_instance.show_axial_freqs()
ma3D_instance.show_cyc_freqs()
ma3D_instance.show_ExB_freqs()
plt.show()
