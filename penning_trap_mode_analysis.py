"""
Author:             Wes Johnson
Date:               September 6th, 2024 
Purpose:            This script calculates the eigenmode frequencies in a Penning trap. 
How to use:         
                    python penning_trap_mode_analysis.py
                    or
                    import penning_trap_mode_analysis as ptma
"""

from generalized_mode_analysis import GeneralizedModeAnalysis 
from generalized_mode_analysis import ensure_numpy_array, characteristic_length
import numpy as np
import scipy.constants as const

def calc_beta(wz, wr, wc): 
    return ( wr*wc - wr**2) / wz**2 - 1/2 

class PenningTrapModeAnalysis(GeneralizedModeAnalysis): 
    k_e = 1/(4*np.pi*const.epsilon_0)
    def __init__(self, N=19, XR=3.082, 
                omega_z = 2*np.pi * 1.58e6, ionmass=9.012182, Z = 1, B=4.4588, frot=180., Vwall=1., initial_equilibrium_guess=None):    
        self.N = N 
        self.XR = XR    
        self.omega_z = omega_z # assume this is the axial frequency in Hz for the first ion species
        self.ionmass = ensure_numpy_array(ionmass, N)
        self.Z = ensure_numpy_array(Z,N)
        assert len(self.ionmass) == len(self.Z), "The number of ion masses and charges must be the same."   
        self.B = B
        self.frot = frot
        self.Vwall = Vwall
        #self.quiet = quiet
        #self.precision_solving = precision_solving
        #self.method = method    
        self.initial_equilibrium_guess = initial_equilibrium_guess

        # calculate the ion properties in experimental units  
        self.q_E = self.Z * const.e  
        self.m_E = self.ionmass * const.u
        self.u_r_sqr = np.sqrt(self.m_E[0] / self.q_E[0]) * self.omega_z 
        self.wz_E = np.sqrt(self.q_E / self.m_E) * self.u_r_sqr
        self.wc_E = self.q_E * self.B / self.m_E    
        self.wr_E = frot * 2*np.pi * 1e3    
        self.wmag_E = 1/2*(self.wc_E - np.sqrt(self.wc_E**2 - 2*self.wz_E**2))
        self.V0 = (0.5 * self.m_E[0] * self.wz_E[0] ** 2) / self.q_E[0] # Find quadratic voltage at trap center, same for all ions  
        self.delta = self.XR*self.Vwall * 1612 / self.V0  # dimensionless coefficient for rotating wall strength 
        self.beta = calc_beta(self.wz_E, self.wr_E, self.wc_E)  
        
        self.hasrun = False 
        


    def trap_is_stable(self): 
        try: 
            assert np.all(self.beta > 0), "Beta must be positive: " + str(self.beta)
            assert np.all(self.delta > 0), "Delta must be positive: " + str(self.delta)
            assert np.all(self.wmag_E > 0), "The magnetron frequency must be positive: " + str(self.wmag_E) 
            return True
        except AssertionError as e:
            print(e)
            return False



    def dimensionless_parameters(self):
        q0_E = self.q_E[0]  
        m0_E = self.m_E[0]  
        wz0_E = self.wz_E[0]
        # mass and charge   
        self.m = self.m_E / m0_E
        self.q = self.q_E / q0_E
        # trap frequencies
        self.wz = self.wz_E / wz0_E 
        self.wr = self.wr_E / wz0_E 
        self.wc = self.wc_E/ wz0_E 

        self.l0 = characteristic_length(q0_E, m0_E, wz0_E)  
        self.t0 = 1 / wz0_E 
        self.v0 = self.l0 / self.t0  # characteristic velocity
        self.E0 = 0.5 * m0_E * self.v0 ** 2 


    def dimensionful_parameters(self):
        self.u0E = self.l0 * self.u0  
        self.uE = self.l0 * self.u  
        self.evals_E = self.wz_E[0] * self.evals



    def run(self):
        super().run()
        self.dimensionful_parameters() 



    def get_B_matrix(self): 
        cross = np.diag( 1/2 * self.m * ( self.wc - 2*self.wr) )    
        B_matrix = np.zeros((3*self.N, 3*self.N))
        B_matrix[self.N:2*self.N, 0:self.N] = -cross    
        B_matrix[0:self.N, self.N:2*self.N] = cross
        return B_matrix 



    def get_transform_matrix(self):
        mass_matrix = np.diag(np.repeat(self.m, 3)) 
        eye = np.eye(3*self.N)  
        zeros = np.zeros((3*self.N, 3*self.N))
        B_matrix = self.get_B_matrix()  
        T = np.block([[eye, zeros], [B_matrix, mass_matrix]])  
        return T    



    def potential_trap(self, pos_array):
        x = pos_array[0:self.N]
        y = pos_array[self.N:2*self.N]
        z = pos_array[2*self.N:]
        V_trap = 0.5 * np.sum((self.q * (self.beta + self.delta) * self.wz ** 2) * x ** 2) + \
            0.5 * np.sum((self.q * (self.beta - self.delta) * self.wz ** 2) * y ** 2) + \
                0.5 * np.sum((self.q * self.wz ** 2) * z ** 2)
        return V_trap



    def force_trap(self, pos_array):
        x = pos_array[0:self.N]
        y = pos_array[self.N:2*self.N]
        z = pos_array[2*self.N:]

        Ftrapx = self.q * (self.beta + self.delta) * self.wz**2 * x
        Ftrapy = self.q * (self.beta - self.delta) * self.wz**2 * y
        Ftrapz = self.q * self.wz**2 * z

        force_trap = np.hstack((Ftrapx, Ftrapy, Ftrapz))
        return force_trap

        

    def hessian_trap(self, pos_array):
        Hxx = np.diag(self.q * (self.beta + self.delta) * (self.wz**2) * np.ones(self.N))
        Hyy = np.diag(self.q * (self.beta - self.delta) * (self.wz**2) * np.ones(self.N))  
        Hzz = np.diag(self.q * (self.wz**2) * np.ones(self.N))  
        zeros = np.zeros((self.N, self.N))  
        H = np.block([[Hxx, zeros, zeros], [zeros, Hyy, zeros], [zeros, zeros, Hzz]])
        return H
















if __name__=='__main__': 

    mBe_amu = 9.012182  
    mYb_amu = 170.936323    
    mBa_amu = 137.327   
    mBeH_amu = mBe_amu + const.m_p / const.u    
    mBeOH_amu = mBe_amu + const.m_p / const.u + 15.999 
    
    N_perturb = 2
    N_Be = 50 
    N_BeOH = 25
    N_BeH = 25 
    N = N_Be + N_BeOH + N_BeH   
    B = 4.4588  
    masses = np.hstack([mBe_amu*np.ones(N_Be), mBeOH_amu*np.ones(N_BeOH), mBeH_amu*np.ones(N_BeH)])
    charges = [1 for _ in range(N)] 
    frot = 180 
    ptma = PenningTrapModeAnalysis(N = N, ionmass=masses, Z=charges, frot=frot, B = B) 
    
    def get_inital_guess(N, N_split):   
        # mean is 0, variance is np.sqrt(N_split), there are N_split samples      
        x_Be = np.random.normal(0, np.sqrt(N_split), N_split)   
        y_Be = np.random.normal(0, np.sqrt(N_split), N_split) 
        r_BeOH = np.random.normal(np.sqrt(N_split), np.sqrt(N - N_split), N - N_split)
        phi_BeOH = np.linspace(0, 2*np.pi, N - N_split)
        x_BeOH = r_BeOH * np.cos(phi_BeOH)
        y_BeOH = r_BeOH * np.sin(phi_BeOH)  
        return np.hstack((x_Be, x_BeOH, y_Be, y_BeOH, np.zeros(N))) 

    #initial_guess = get_inital_guess(N, N_split)
    initial_guess = None
    for it in range(N_perturb): 
        ptma.initial_equilibrium_guess = initial_guess
        ptma.run() 
        initial_guess = ptma.u  
    

    import matplotlib.pyplot as plt 
    fig, axs = plt.subplots(1, 2, figsize=(10, 6)) 
    axs = axs.flatten() 

    ax = axs[0]
    ax.plot(ptma.evals* ptma.omega_z/ 2/ np.pi/1e6, 'o')
    ax.set_title('Eigenfrequencies')
    ax.set_xlabel('Mode number')
    ax.set_ylabel('Frequency (Hz)')

    ax = axs[1]
    x = ptma.u[0:N] 
    y = ptma.u[N:2*N]    
    colors = ['blue']*N_Be + ['green']*N_BeOH + ['red']*N_BeH       
    ax.scatter(x, y, label='Equilibrium positions', c=colors)   
    ax.set_title('Equilibrium positions')
    ax.set_xlabel('x')  
    ax.set_ylabel('y')
    ax.set_aspect('equal')  
    # add custom legend 
    labels = ['Be', 'BeOH', 'BeH']  
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, label=label) for color, label in zip(['blue', 'green', 'red'], labels)]
    ax.legend(handles=handles, labels=labels, loc='upper right')    
    plt.tight_layout()
    plt.show() 
          

