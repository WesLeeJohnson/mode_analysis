"""
Author:     Wes Johnson
Date:       July 29th, 2024, Edited November 20th, 2024
Purpose:    Perform the normal mode analysis of an ion crystal in a trap 
            with a quartic anharmonicity. 
How to run:
            python quartic_trap_mode_analysis.py
"""
import numpy as np 
from harmonic_trap_mode_analysis import HarmonicTrapModeAnalysis  
from generalized_mode_analysis import GeneralizedModeAnalysis
from scipy import constants as const
from scipy import optimize as opt
from generalized_mode_analysis import ensure_numpy_array

def characteristic_frequency(q, m, l):
    k_e = 1 / (4 * np.pi * const.epsilon_0)  # Coulomb constant
    w0 = (k_e * q ** 2 / (.5 * m * l ** 3)) ** (1 / 2)    
    print(w0/2/np.pi/1e6)   
    return w0 

class InnerIonVarMinimizedChainModeAnalysis(GeneralizedModeAnalysis):    
    """
    Perform a mode analysis of a linear trap with anharmonicity.
    Based on the paper: 10.1209/0295-5075/86/60004
    """
    def __init__(self, N = 2, wx = 2*np.pi*3e6, wy = 2*np.pi*2.5e6, l0 = 4.4e-6, ionmass_amu = 170.936323, N_trim = 0): 
        # for now assume that all the ions are the same mass and charge
        self.N = N
        self.N_trim = N_trim
        self.wx_E = wx
        self.wy_E = wy
        self.l0_unnormalized = l0
        self.ionmass_amu = ensure_numpy_array(ionmass_amu, N)
        self.Z = ensure_numpy_array(1, N)
        self.q_E = self.Z * const.e   
        self.m_E = self.ionmass_amu * const.u
        self.q0 = self.q_E[0]   
        self.m0 = self.m_E[0]
        self.a2 = None # these will need to be set by finding B given N 
        self.a4 = None 
        self.B = None # this will be optimized to minimize the inner ion variance

        self.hasrun = False

    def dimensionless_parameters(self): 

        # this function has to be redefined because the normalizations are different for this problem

        # l0 is given 
        # E_0 = q^2 / ( 4 \pi \epsilon_0 l0) is a characteristic energy of the system
        # ensure the mean spacing of the ions is 1 in dimensionless units
        d_bar = np.mean(self.calc_d(self.u[2*self.N:])) # mean spacing of qubit ions    
        self.l0 = self.l0_unnormalized / d_bar

        m0 = self.m0
        q0 = self.q0
        l0 = self.l0

        # dimensionless parameters
        self.a2 =  np.sign(self.B) # always pm 1
        self.a4 =  np.abs(self.B)

        # dimensionful parameters
        self.a2_E = q0 ** 2 / (4 * np.pi * const.epsilon_0 * l0**3) * self.a2 * 2 # TODO: figure out where the factor of 2 comes from
        self.a4_E = np.abs( self.B * self.a2_E / l0 ** 2 ) 

        w0 = np.sqrt( np.abs(self.a2_E) / m0) # characteristic frequency 

        self.w0 = w0
        self.q0 = q0
        self.m0 = m0

        # ion properties    
        self.m = self.m_E / m0
        self.q = self.q_E / q0  
        # trap frequencies
        self.wy = self.wy_E / w0
        self.wx = self.wx_E / w0
        # system parameters
        self.t0 = 1 / w0  # characteristic time
        self.v0 = self.l0 / self.t0  # characteristic velocity
        self.E0 = m0 * self.v0 ** 2  # characteristic energy  
        self.p0 = self.potential(self.u) # this has to be done after the equilibrium positions are found
        #print(q0**2 / (4 * np.pi * const.epsilon_0 * l0), self.E0)# it is consistent! 
        #print(characteristic_frequency(self.q0, self.m0, self.l0) / 2 / np.pi / 1e6, w0 / 2 / np.pi / 1e6) #TODO: understand why these are different 


    def run(self):
        self.u = self.calculate_equilibrium_positions() # this can be broken into two steps, first given N find B so that the inner ion variance is minimized, then find the equilibrium positions, next normalize rms inner ion variance to 1
        self.dimensionless_parameters() # this has to go after the equilibrium positions are found!! 
        assert self.trap_is_stable()    
        self.E_matrix = self.get_E_matrix(self.u)  
        self.T_matrix = self.get_momentum_transform() 
        self.H_matrix = self.get_H_matrix(self.T_matrix, self.E_matrix)   
        self.evals, self.evecs = self.calculate_normal_modes(self.H_matrix)
        self.evecs_vel = self.get_eigen_vectors_xv_coords(self.T_matrix,self.evecs)    
        self.check_for_zero_modes() 
        self.S_matrix = self.get_canonical_transformation() 
        self.checks() 
        self.hasrun = True  


    def trap_is_stable(self):
        truth1 = self.wx > 0e0 and self.wy > 0e0
        truth2 = self.wx > self.wy
        assert truth1 and truth2, "Trap frequencies must be positive and ordered wx > wy"
        return truth1 and truth2


    def calculate_equilibrium_positions(self):
        self.z0 = np.arange(0, self.N)  - (self.N - 1) / 2
        u = self.find_equilibrium_positions(self.z0)
        return u   

    def calc_d(self, z): 
        N_trim = self.N_trim
        if N_trim == 0:
            z_trim = z
        else:
            z_trim = z[N_trim:-N_trim]  
        d = np.diff(z_trim)
        return d

    def inner_ion_variance(self, z):
        d = self.calc_d(z)
        d_bar = np.mean(d)
        s_z = np.sqrt(np.mean((d/d_bar - 1) ** 2))  
        return s_z 

    def find_axial_equilibrium_positions(self, B, z0):

        def potential_Coulomb_axial(z): 
            dz = z[:, np.newaxis] - z
            rsep = np.sqrt(dz ** 2).astype(np.float64)
            with np.errstate(divide='ignore'):
                V_Coulomb = np.sum( np.where(rsep != 0., 1 / rsep, 0) ) / 2 # divide by 2 to avoid double counting
            V_Coulomb *= .5 
            return V_Coulomb

        def force_Coulomb_axial(z):
            dz = z[:, np.newaxis] - z
            rsep = np.sqrt(dz ** 2).astype(np.float64)
            with np.errstate(divide='ignore', invalid='ignore'):
                rsep3 = np.where(rsep != 0., rsep ** (-3), 0)
            fz = dz * rsep3 
            Fz = -np.sum(fz, axis=1)
            return Fz * 0.5

        def potential_axial_opt(z):
            V_trap = np.sum( 1/2 * z ** 2 + 1/4 * B * z ** 4) * np.sign(B)
            V_Coulomb = potential_Coulomb_axial(z)
            V = V_trap + V_Coulomb
            return V

        def force_axial_opt(z): 
            F_trap = (z + B * z ** 3 ) * np.sign(B)
            F_Coulomb = force_Coulomb_axial(z) 
            return F_trap + F_Coulomb

        bfgs_tolerance = 1e-34
        out = opt.minimize(potential_axial_opt, z0, method='BFGS', jac=force_axial_opt, 
                                    options={'gtol': bfgs_tolerance, 'disp': False})
        z = np.sort(out.x)
        return z 


    def minimize_inner_ion_variance(self, z0):

        def inner_ion_variance_anharmonicity(B): 
            z = self.find_axial_equilibrium_positions(B, z0=z0) 
            z_var = self.inner_ion_variance(z)
            return z_var

        # one way to handle the non-convexity of the problem is to try multiple initial guesses
        initial_guesses = np.array([-3, 3]) 
        b_opts = []
        for initial_guess in initial_guesses: 
            out = opt.minimize(inner_ion_variance_anharmonicity, initial_guess, method='Nelder-Mead',
                                options={'xatol': 1e-6, 'disp': False})
            b_opts.append(out.x)
        # choose the B that minimizes the inner ion variance: 
        s_z_opts = []
        for b_opt in b_opts: 
            z = self.find_axial_equilibrium_positions(b_opt, z0=z0) 
            s_z = self.inner_ion_variance(z)
            s_z_opts.append(s_z)
        min_s_z_index = np.argmin(s_z_opts)
        self.B = b_opts[min_s_z_index]
        z = self.find_axial_equilibrium_positions(self.B , z0=z0)   
        return z 

    def find_equilibrium_positions(self, z0):
        z = self.minimize_inner_ion_variance(z0)
        zeros = np.zeros(self.N)
        u = np.hstack((zeros, zeros, z)) # assume ions are on the z axis   
        return u 









    def potential_trap_axial(self, z):
        return np.sum(0.5 * self.a2 * z **2 + 1/4 * self.a4 * z**4) 

    def potential_trap(self, pos_array):
        x = pos_array[0:self.N]
        y = pos_array[self.N:2*self.N]
        z = pos_array[2*self.N:]
        V_trap = 0.5 * np.sum((self.m * self.wx ** 2) * x ** 2) + \
            0.5 * np.sum((self.m * self.wy ** 2) * y ** 2) + \
                self.potential_trap_axial(z)
        return V_trap
    
    def force_trap_axial(self, z):
        return self.a2 * z + self.a4 * z**3
    
    def force_coulomb_axial(self, z):
        force_coulomb = self.force_coulomb(np.hstack((np.zeros(self.N), np.zeros(self.N), z)))[2*self.N:]
        return force_coulomb

    def force_trap(self, pos_array):
        x = pos_array[0:self.N]
        y = pos_array[self.N:2*self.N]
        z = pos_array[2*self.N:]

        Ftrapx = self.m * self.wx**2 * x
        Ftrapy = self.m * self.wy**2 * y
        Ftrapz = self.force_trap_axial(z)

        force_trap = np.hstack((Ftrapx, Ftrapy, Ftrapz))
        return force_trap

    def hessian_trap_axial(self, z):    
        Hzz = np.diag(self.a2 + 3 * self.a4 * z**2)
        return Hzz

    def hessian_trap(self, pos_array):
        z = pos_array[2*self.N:] 

        Hxx = np.diag(self.m * (self.wx**2) * np.ones(self.N))
        Hyy = np.diag(self.m * (self.wy**2) * np.ones(self.N))  
        Hzz = self.hessian_trap_axial(z)    
        zeros = np.zeros((self.N, self.N))  
        H = np.block([[Hxx, zeros, zeros], [zeros, Hyy, zeros], [zeros, zeros, Hzz]])
        return H
    


















class QuarticTrapModeAnalysis(HarmonicTrapModeAnalysis):
    def __init__(self, N = 2, wx = 2*np.pi*3e6, wy = 2*np.pi*2.5e6, wz = 2*np.pi*.3e6, ionmass_amu = 170.936323, Z = 1, anharmonicity = 0.01): 
        """
        Initialize the class with the number of ions, trap frequencies, ion mass, ion charge, and anharmonicity. 
        """
        super().__init__(N, wx, wy, wz, ionmass_amu, Z)
        self.a_2 = anharmonicity # cubic coefficient is a_1, quartic is a_2, ect.. 

    # overwrite the trap potential function to include the anharmonicity
    def potential_trap(self, pos_array):
        x = pos_array[0:self.N]
        y = pos_array[self.N:2*self.N]
        z = pos_array[2*self.N:]
        V_trap_x = 0.5* self.m * (self.wx ** 2) * np.sum(x ** 2)
        V_trap_y = 0.5* self.m * (self.wy ** 2) * np.sum(y ** 2)
        V_trap_z = 0.5* self.m * (self.wz ** 2) * np.sum(z ** 2 * (1 + .5*self.a_2* z ** 2)) # anharmonicity term  
        V_trap = V_trap_x + V_trap_y + V_trap_z
        return V_trap

    # overwrite the trap force function to include the anharmonicity    
    def force_trap(self, pos_array):
        x = pos_array[0:self.N]
        y = pos_array[self.N:2*self.N]
        z = pos_array[2*self.N:]

        Ftrapx = self.m * self.wx**2 * x
        Ftrapy = self.m * self.wy**2 * y
        Ftrapz = self.m * self.wz**2 * z * (1 + self.a_2 * z ** 2)  

        force_trap = np.hstack((Ftrapx, Ftrapy, Ftrapz))
        return force_trap
    
    # overwrite the trap hessian function to include the anharmonicity  
    def hessian_trap(self, pos_array):
        z = pos_array[2*self.N:]    
        Hxx = np.diag(self.m * (self.wx**2) * np.ones(self.N))
        Hyy = np.diag(self.m * (self.wy**2) * np.ones(self.N))  
        Hzz = np.diag(self.m * (self.wz**2) * (1 + 3 * self.a_2 * z ** 2))
        zeros = np.zeros((self.N, self.N))  
        H = np.block([[Hxx, zeros, zeros], [zeros, Hyy, zeros], [zeros, zeros, Hzz]])
        return H
    
    # overwrite the is trap stable function to include the anharmonicity
    def is_trap_stable(self):
        assert self.wx > 0e0 and self.wy > 0e0 and self.wz > 0e0, "Trap frequencies must be positive"
        assert self.wx > self.wy and self.wy and self.wz, "Trap frequencies must be ordered wx > wy > wz"
        assert self.a_2 >= 0, "Anharmonicity must be positive"    















if __name__ == "__main__": 
    import matplotlib.pyplot as plt
    from plotting_settings import *


    iivmc_obj_test = InnerIonVarMinimizedChainModeAnalysis(N=17, wx = 2*np.pi*2.5e6, wy = 2*np.pi*2.1e6, ionmass_amu = 170.936323, l0 = 5e-6,N_trim=1)
    iivmc_obj_test.run()
    print(iivmc_obj_test.B) 


    # plot the equilibrium positions and frequencies of the normal modes

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    axs = axs.flatten()

    ax = axs[0]
    ax.set_title("Normal mode frequencies")
    mode_numbers = np.arange(0, 3*iivmc_obj_test.N)
    ax.scatter(mode_numbers, iivmc_obj_test.evals * iivmc_obj_test.w0 / 2 / np.pi * 1e-6)
    ax.set_xlabel('Mode number')
    ax.set_ylabel('Frequency (MHz)')
    ax.set_ylim(0, iivmc_obj_test.wx_E / 2 / np.pi * 1e-6 * 1.1)

    ax = axs[1]
    ax.set_title("Equilibrium positions")
    eq_pos_3D = iivmc_obj_test.u.reshape((3, iivmc_obj_test.N))
    eq_pos_3D *= iivmc_obj_test.l0 * 1e6
    y = eq_pos_3D[1]
    z = eq_pos_3D[2]
    ax.scatter(z, y)
    ax.set_xlabel("z ($\mu$m)")
    ax.set_ylabel("y ($\mu$m)")

    plt.tight_layout()
    plt.show()
    exit() 












    N = 6

    htma_obj = HarmonicTrapModeAnalysis(N=N,wz=2*np.pi*.3e6,wy=2*np.pi*2.1e6,wx=2*np.pi*2.5e6)  
    htma_obj.run()
    qtma_obj = QuarticTrapModeAnalysis(N=N,wz=2*np.pi*.3e6,wy=2*np.pi*2.1e6,wx=2*np.pi*2.5e6, anharmonicity=0.1)    
    qtma_obj.run()
    min_diff = np.min(np.abs(np.diff(np.sort(htma_obj.u[2*htma_obj.N:])))) * htma_obj.l0
    print(min_diff) 
    iivmc_obj = InnerIonVarMinimizedChainModeAnalysis(N=N, wx = 2*np.pi*2.5e6, wy = 2*np.pi*2.1e6, ionmass_amu = 170.936323, l0 = min_diff) 
    iivmc_obj.run() 

    # check eigen vectors are orthogonal and normalized wrt Hamiltonian matrix
    def check_normalization(E_matrix,evecs): 
        N6,_ = evecs.shape
        N = N6//6   
        evecs_all = np.empty((6*N, 6*N), dtype=np.complex128)
        check_matrix = np.zeros((6*N, 6*N), dtype=np.complex128)
        evecs_all[:,0:3*N] = evecs
        evecs_all[:,3*N:] = np.conj(evecs)
        for i in range(6*N):
            for j in range(6*N):
                element = evecs_all[:,i].T.conj() @ E_matrix @ evecs_all[:,j]
                assert np.abs(element) < 1e2, f"Element {i},{j} is not zero: {element}"
                check_matrix[i,j] = element
        assert np.allclose(np.eye(6*N), check_matrix, atol=1e-2), "Eigenvectors are not orthogonal"

    def find_center_of_mass_modes(evecs):
        # center of mass modes have equal amplitudes for all ions
        mode_displacement_variances = np.var(evecs/np.linalg.norm(evecs,axis=0), axis=0) 
        # get the three smallest variances
        center_of_mass_modes = np.argsort(mode_displacement_variances)[:3]
        return center_of_mass_modes

    check_normalization(htma_obj.E_matrix, htma_obj.evecs) 
    check_normalization(qtma_obj.E_matrix, qtma_obj.evecs) 
    center_of_mass_mode_indices = find_center_of_mass_modes(htma_obj.evecs) 

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    axs = axs.flatten()

    ax = axs[0]
    ax.set_title("Normal mode frequencies")
    mode_numbers = np.arange(0, 3*htma_obj.N)
    ax.scatter(mode_numbers, htma_obj.evals * htma_obj.wz_E / 2 / np.pi * 1e-6)
    ax.scatter(mode_numbers, qtma_obj.evals * qtma_obj.wz_E / 2 / np.pi * 1e-6, marker='x') 
    ax.scatter(mode_numbers, iivmc_obj.evals * iivmc_obj.w0 / 2 / np.pi * 1e-6, marker='s')

    # draw center of mass modes with star 
    ax.scatter(center_of_mass_mode_indices, htma_obj.evals[center_of_mass_mode_indices] * htma_obj.wz_E / 2 / np.pi * 1e-6, marker='*', color='r')
    ax.set_xlabel('Mode number')
    ax.set_ylabel('Frequency (MHz)')    
    ax.set_ylim(0, htma_obj.wx_E / 2 / np.pi * 1e-6 * 1.1)

    ax = axs[1] 
    ax.set_title("Equilibrium positions")
    eq_pos_3D = htma_obj.u.reshape((3, htma_obj.N))
    eq_pos_3D *= htma_obj.l0 * 1e6  
    eq_pos_3D_q = qtma_obj.u.reshape((3, qtma_obj.N))   
    eq_pos_3D_q *= qtma_obj.l0 * 1e6
    eq_pos_3D_iivmc = iivmc_obj.u.reshape((3, iivmc_obj.N))
    eq_pos_3D_iivmc *= iivmc_obj.l0 * 1e6
    y = eq_pos_3D[1] 
    z = eq_pos_3D[2] 
    y_q = eq_pos_3D_q[1]
    z_q = eq_pos_3D_q[2]
    y_iivmc = eq_pos_3D_iivmc[1]
    z_iivmc = eq_pos_3D_iivmc[2]
    ax.scatter(z, y)    
    ax.scatter(z_q, y_q, marker='x')    
    ax.scatter(z_iivmc, y_iivmc, marker='s')    
    ax.set_xlabel("z ($\mu$m)") 
    ax.set_ylabel("y ($\mu$m)")

    plt.tight_layout() 
    plt.show()
