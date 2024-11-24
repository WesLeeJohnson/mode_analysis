"""
Author:         Wes Johnson
Date:           October 20th, 2024
Purpose:        Mode analysis for cyclotron motion
How to run:     python cyclotron_mode_analysis.py
or              import cyclotron_mode_analysis 
"""

#from generalized_mode_analysis import GeneralizedModeAnalysis 
from penning_trap_mode_analysis import PenningTrapModeAnalysis  
from generalized_mode_analysis import ensure_numpy_array
import numpy as np
import matplotlib.pyplot as plt
from scipy import constants as c 

class CyclotronModeAnalysis(PenningTrapModeAnalysis):
    def __init__(self, N=2, ionmass=c.m_p/c.u, Z=c.e, B=1.0):
        self.ionmass_amu = ensure_numpy_array(ionmass, N)
        self.Z = ensure_numpy_array(Z, N)   
        self.B = B  
        self.N = N

        self.m_E = self.ionmass_amu * c.u    
        self.q_E = self.Z * c.e  
        self.B_E = B
        self.wc_E = self.calculate_cyclotron_frequency(self.q_E, self.B_E, self.m_E)    

        self.hasrun = False

    def dimensionless_parameters(self):
        q0_E = self.q_E[0]  
        m0_E = self.m_E[0]  
        w0_E = self.wc_E[0]
        # mass and charge   
        self.m = self.m_E / m0_E
        self.q = self.q_E / q0_E
        # trap frequencies
        self.wc = self.wc_E/ w0_E 

        self.t0 = 1 / w0_E  # characteristic time

    def dimensionful_parameters(self):
        self.evals_E = self.wc_E[0] * self.evals

    def run(self):
        self.dimensionless_parameters()
        self.E_matrix = self.get_E_matrix()
        self.wr = 0 # hack to get correct momentum matrix
        self.T_matrix = super().get_momentum_transform() 
        self.H_matrix = super().get_H_matrix(self.T_matrix, self.E_matrix)   
        self.evals, self.evecs = super().calculate_normal_modes(self.H_matrix)
        self.evecs_vel = super().get_eigen_vectors_xv_coords(self.T_matrix,self.evecs)    
        self.S_matrix = super().get_canonical_transformation() 
        self.dimensionful_parameters()
        self.hasrun = True  

    def get_E_matrix(self):
        E_matrix = np.zeros((6*self.N, 6*self.N))
        PE_matrix = np.zeros((3*self.N, 3*self.N)) # no potential energy for cyclotron motion
        KE_matrix = super().get_mass_matrix(self.m) 
        E_matrix[0:3*self.N, 0:3*self.N] = PE_matrix
        E_matrix[3*self.N:, 3*self.N:] = KE_matrix
        return E_matrix        
    
if __name__=='__main__':
    B = 1.0
    N = 10
    Z = [i+1 for i in range(N)]
    ionmass = [c.m_p/c.u * 100 for _ in range(1, N+1)]    
    cma = CyclotronModeAnalysis(N, ionmass, Z, B)
    cma.run()
    print(cma.evals)    