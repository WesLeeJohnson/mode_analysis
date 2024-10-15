"""
Author:             Wes Johnson
Date:               October 15th, 2024 
Purpose:            This script calculates the eigenmode frequencies of
                    a mixed ion species crystal and a triangular rotating wall.
How to use:         python dipole_rotating_wall_mode_analysis.py    
""" 

from penning_trap_mode_analysis import PenningTrapModeAnalysis 
import numpy as np 
import scipy.constants as const
from plotting_settings import * 
import matplotlib.pyplot as plt 

class TriangularRotatingWallModeAnalysis(PenningTrapModeAnalysis):
    def __init__(self, N=19, XR=3.082, 
                 omega_z = 2*np.pi * 1.58e6, ionmass=9.012182, Z = 1, B=4.4588, frot=180., triangular_wall_strength=.1, initial_equilibrium_guess=None):    
        super().__init__(N=N, XR=XR, omega_z=omega_z, ionmass=ionmass, Z=Z, B=B, frot=frot, Vwall=0., initial_equilibrium_guess=initial_equilibrium_guess)  
        self.kappa = triangular_wall_strength 
    
    def potential_trap(self, pos_array):
        V_trap = super().potential_trap(pos_array)
        x = pos_array[0:self.N]
        y = pos_array[self.N:2*self.N]  
        V_trap += np.sum(self.q * self.kappa * self.wz ** 2 * (x**3 - 3*x*y**2)  )
        return V_trap

    def force_trap(self, pos_array):
        F_trap = super().force_trap(pos_array)
        x = pos_array[0:self.N]
        y = pos_array[self.N:2*self.N]  
        F_trap[0:self.N] += self.q * self.kappa * self.wz ** 2 * (3 * x**2 - 3*y**2)
        F_trap[self.N:2*self.N] += self.q * self.kappa * self.wz ** 2 * (-6*x*y) 
        return F_trap   

    def hessian_trap(self, pos_array):
        H = super().hessian_trap(pos_array) 
        x = pos_array[0:self.N]
        y = pos_array[self.N:2*self.N]
        H_xx_tri = np.diag(self.q * self.kappa * self.wz ** 2 * (6*x))
        H_yy_tri = np.diag(self.q * self.kappa * self.wz ** 2 * (-6*x))
        H_xy_tri = np.diag(self.q * self.kappa * self.wz ** 2 * (-6*y)) 
        H[:2*self.N,:2*self.N] += np.block([[H_xx_tri, H_xy_tri], [H_xy_tri, H_yy_tri]])    
        return H
    
if __name__== "__main__":
    #N_Be = 100
    #N_BeOH = 0
    #N_BeH = 100 - N_Be - N_BeOH 
    N_Be = 15 
    N_BeH =  15 
    N_BeOH = 0 
    mBe = 9.012182
    mBeH = 9.012182 + 1.007825
    mBeOH = 9.012182 + 15.9994 + 1.007825   
    params = {
        'mass_amu': np.hstack(N_Be * [mBe] + N_BeH * [mBeH] + N_BeOH * [mBeOH]),    
        'N': N_Be + N_BeH + N_BeOH, 
        'omega_z': 1.58e6 * 2*np.pi,    
        'B': 4.4588, 
        'frot': 200, 
        #'triangular_wall_strength': 2.5e-3, 
        #'triangular_wall_strength': 2.e-3
        'triangular_wall_strength': 1.5e-3 # works for N = 50, frot = 200
        #'triangular_wall_strength': 0.5e-3 # works for N = 100, frot = 190 

    }
    initial_guess = np.zeros(3*params['N'])
    initial_guess[:2*params['N']] = np.random.normal(0,.1,2*params['N'])
    ma = TriangularRotatingWallModeAnalysis(N=params['N']
                                        ,omega_z=params['omega_z']
                                        ,B=params['B']
                                        ,frot=params['frot']
                                        ,ionmass=params['mass_amu']
                                        ,triangular_wall_strength=params['triangular_wall_strength']
                                        ,Z=np.ones_like(params['mass_amu']),
                                        initial_equilibrium_guess=initial_guess
    )
    ma.run()

    print('lowest frequency mode: ', ma.evals_E[0]/2/np.pi/1e6*1e3, 'kHz')  
    print(ma.u)
    # plot the equilibrium positions and mode frequencies   
    fig, axs= plt.subplots(1,2,figsize=(10,4))
    ax = axs[0]
    ax.plot(ma.evals_E/2/np.pi/1e6,'o')
    ax.set_ylabel('Frequency (MHz)')
    ax.set_xlabel('Mode number')
    ax.set_title('Mode frequencies')
    ax = axs[1]
    x = ma.u[:ma.N] #* ma.l0 * 1e6
    y = ma.u[ma.N:2*ma.N] #* ma.l0 * 1e6
    c = ['b']*N_Be + ['g']*N_BeH + ['r']*N_BeOH 
    ax.scatter(x,y,c=c)    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_aspect('equal')  
    #ax.set_xlabel('x ($\mu$m)')
    #ax.set_ylabel('y ($\mu$m)')
    ax.set_xlabel('x')
    ax.set_ylabel('y')  
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
    fig.savefig(pdir + 'triangular_rotating_wall_mode_analysis.pdf')    
    plt.show() 


