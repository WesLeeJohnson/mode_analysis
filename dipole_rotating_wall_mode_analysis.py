"""
Author:             Wes Johnson
Date:               September 27th, 2024
Purpose:            This script calculates the eigenmode frequencies of
                    a mixed ion species crystal and a dipole rotating wall.
How to use:         python dipole_rotating_wall_mode_analysis.py    
""" 

from penning_trap_mode_analysis import PenningTrapModeAnalysis 
import numpy as np 
import scipy.constants as const
from plotting_settings import * 
import matplotlib.pyplot as plt

# I should just need to change the trap functions 
class DipoleRotatingWallModeAnalysis(PenningTrapModeAnalysis):
    def __init__(self, N=19, XR=3.082, 
                 omega_z = 2*np.pi * 1.58e6, ionmass=9.012182, Z = 1, B=4.4588, frot=180., dipole_wall_strength=.1, initial_equilibrium_guess=None):    
        super().__init__(N=N, XR=XR, omega_z=omega_z, ionmass=ionmass, Z=Z, B=B, frot=frot, Vwall=0., initial_equilibrium_guess=initial_equilibrium_guess)  
        self.kappa = dipole_wall_strength # figure out what this should be ?? 
    
    def potential_trap(self, pos_array):
        V_trap = super().potential_trap(pos_array)
        x = pos_array[0:self.N]
        V_trap += np.sum(self.q * self.kappa * self.wz ** 2 * x)    
        return V_trap

    def force_trap(self, pos_array):
        F_trap = super().force_trap(pos_array)
        F_trap[0:self.N] += self.q * self.kappa * self.wz ** 2 * np.ones(self.N)
        return F_trap   
    
if __name__=="__main__":
    N_Be = 60
    N_BeOH = 1
    N_BeH = 100 - N_Be - N_BeOH 
    mBe = 9.012182
    mBeH = 9.012182 + 1.007825
    mBeOH = 9.012182 + 15.9994 + 1.007825   
    params = {
        'mass_amu': np.hstack(N_Be * [mBe] + N_BeH * [mBeH] + N_BeOH * [mBeOH]),    
        'N': N_Be + N_BeH + N_BeOH, 
        'omega_z': 1.58e6 * 2*np.pi,    
        'B': 4.4588, 
        'frot': 180, 
        'dipole_wall_strength': .2, 
    }
    ma = DipoleRotatingWallModeAnalysis(N=params['N']
                                        ,omega_z=params['omega_z']
                                        ,B=params['B']
                                        ,frot=params['frot']
                                        ,ionmass=params['mass_amu']
                                        ,dipole_wall_strength=params['dipole_wall_strength']
                                        ,Z=np.ones_like(params['mass_amu'])
    ) 
    ma.run()

    # plot the equilibrium positions and mode frequencies   
    fig, axs= plt.subplots(1,2,figsize=(10,4))
    ax = axs[0]
    ax.plot(ma.evals_E/2/np.pi/1e6,'o')
    ax.set_ylabel('Frequency (MHz)')
    ax.set_xlabel('Mode number')
    ax.set_title('Mode frequencies')
    ax = axs[1]
    x = ma.u[:ma.N] * ma.l0 * 1e6
    y = ma.u[ma.N:2*ma.N] * ma.l0 * 1e6
    c = ['b']*N_Be + ['g']*N_BeH + ['r']*N_BeOH 
    ax.scatter(x,y,c=c)    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_aspect('equal')  
    ax.set_xlabel('x ($\mu$m)')
    ax.set_ylabel('y ($\mu$m)')
    ax.set_title('Equilibrium positions')
    # make custom legend
    labels = ['Be', 'BeH', 'BeOH']  
    handles = [plt.Line2D([0], [0]
                          , marker='o'
                          , color='w'
                          , markerfacecolor=color
                          , label=label) for color, label in zip(['blue', 'green', 'red'], labels)
            ]
    ax.legend(handles=handles, labels=labels, loc='lower left')  
    plt.tight_layout()
    pdir =  '../planarCrystalNonlinearCoupling/figures/'
    fig.savefig(pdir + 'dipole_rotating_wall_mode_analysis.pdf')    
    plt.show() 


