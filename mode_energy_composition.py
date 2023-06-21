"""
Author:     Wes Johnson
Date:       June 21st 20203
Purpose:    This code investigates the energy composition of the modes.
How to run: python mode_energy_composition.py
"""

####5###10###15###20###25###30###35###40###45###50###55###60###65###70##
# Import packages
import numpy as np
import matplotlib.pyplot as plt
import mode_analysis_3D as ma3D
from scipy import constants as const

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
vwall = 10 #V 
omega_z = 2*np.pi*1.58e6 #rad/s
frot = wcyc/(2*np.pi*1000)/2*1.9 #kHz
frot = 200.75 #kHz
N = 20 #number of ions


#run the 3D mode analysis
ma =               ma3D.ModeAnalysis(ionmass=ionmass
                                    ,omega_z=omega_z
                                    ,frot=frot
                                    ,B=B
                                    ,N=N
                                    ,Vwall=vwall
                                    ,precision_solving=True
                                    #,method='newton'
                                    ,method='bfgs'
                                    )
ma.run()
ma.run_3D()

#planarEvects = ma.planarEvects
#axialEvects = ma.axialEvects
#ma.show_crystal_planar_mode(mode=3)
#plt.show()

nrows = 2
ncols = 2
fig,axs = plt.subplots(nrows,ncols,figsize=(8,8))
axs = axs.flatten()
for mode in range(nrows*ncols):
    ma.show_crystal_axial_mode(mode=mode,ax=axs[mode])
    axs[mode].set_title('Mode '+str(mode),fontsize=font_size_title)
plt.tight_layout()

nrows = 2
ncols = 4
fig,axs = plt.subplots(nrows,ncols,figsize=(8,4))
axs = axs.flatten()
for mode in range(nrows*ncols):
    ma.show_crystal_planar_mode(mode=mode,ax=axs[mode])
    axs[mode].set_title('Mode '+str(mode),fontsize=font_size_title)
plt.tight_layout()

def get_planar_mode(ma,mode_index):
    return ma.planarEvects[:,mode_index]

def get_axial_mode(ma,mode_index):
    return ma.axialEvects[:,mode_index]

def get_axial_Mmat(ma): 
    return np.diag(np.ones(ma.Nion,))

def get_planar_Mmat(ma): 
    return np.diag(np.ones(2*ma.Nion,))

def get_planar_hessian(ma): 
    return ma.hessian_penning(ma.u)/2

def get_axial_hession(ma):
    return -ma.calc_axial_hessian(ma.u)

def get_kinetic_energy_planar_mode(ma,mode_index):
    ev = get_planar_mode(ma,mode_index)[2*ma.Nion:].reshape((2*ma.Nion,1))
    ev = np.asmatrix(ev)
    M = get_planar_Mmat(ma)
    M = np.asmatrix(M)
    return np.real(ev.H * M * ev)

def get_potential_energy_planar_mode(ma,mode_index):
    ev = get_planar_mode(ma,mode_index)[:2*ma.Nion].reshape((2*ma.Nion,1))
    ev = np.asmatrix(ev)
    H = get_planar_hessian(ma)
    H = np.asmatrix(H)
    return np.real(ev.H * H * ev)


mode_KE = np.zeros(2*N)
mode_PE = np.zeros(2*N)
mode_TE = np.zeros(2*N)
for mode in range(2*N):
    mode_KE[mode] = get_kinetic_energy_planar_mode(ma,mode)
    mode_PE[mode] = get_potential_energy_planar_mode(ma,mode)
    mode_TE[mode] = get_kinetic_energy_planar_mode(ma,mode)+get_potential_energy_planar_mode(ma,mode)

fig,axs = plt.subplots(2,1,figsize=(12,8)) 
# make a bar plot of he energy composition of the modes
ax = axs[0]
modes = np.arange(2*N)
ax.bar(modes,mode_PE,label='Potential energy',color='red')
ax.bar(modes,mode_KE,bottom=mode_PE,label='Kinetic energy',color='blue')
ax.set_xlabel('Mode index',fontsize=font_size_labels)
ax.set_ylabel('Energy Composition',fontsize=font_size_labels)
ax.legend(fontsize=font_size_legend)
ax.tick_params(axis='both', which='major', labelsize=font_size_ticks)
ax.set_xticks(modes[::10])
ax = axs[1]
ax.plot(modes,ma.planarEvals,'o',label='Planar Mode Frequencies')
ax.set_xlabel('Mode index',fontsize=font_size_labels)
ax.set_ylabel('Frequency ($\omega_z$)',fontsize=font_size_labels)
ax.legend(fontsize=font_size_legend)
ax.tick_params(axis='both', which='major', labelsize=font_size_ticks)
ax.set_xticks(modes[::10])
plt.tight_layout()
plt.show()
