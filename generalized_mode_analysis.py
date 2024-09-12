"""
Author:            Wes Johnson  
Date:              September 6th, 2024
Purpose:           This script calculates the eigenmode frequencies in a trapped ion system.
How to use:        python generalized_mode_analysis.py
                   or
                   import generalized_mode_analysis as gma
""" 
# Dubin's analysis allows us to calculate the eigenmode frequencies of a trapped ion system. 
# to do this, a canonical transformation is made from the velocity and position coordinates to 
# the position and momentum coordinates. 
# additionally this code will allow for the calculation of the eigenmode frequencies with a mixed species 
# ion crystal. 
# if q is not an array then all ions are assumed to have the same charge.
# if m is not an array then all ions are assumed to have the same mass. 
# this code will have a few parts: 
# 1. 
import numpy as np  
import scipy.constants as const
import scipy.optimize as opt    

def ensure_numpy_array(x,N):
    if not hasattr(x, "__len__"):
        x = np.ones(N) * x 
    else: # make sure its a numpy array
        x = np.array(x) 
    return x    

def characteristic_length(q, m, wz):
    k_e = 1 / (4 * np.pi * const.epsilon_0)  # Coulomb constant
    l0 = ((k_e * q ** 2) / (.5 * m * wz ** 2)) ** (1 / 3)
    return l0

class GeneralizedModeAnalysis:
    def __init__(self, N=2, wz = 2*np.pi*.1e6,wy = 2*np.pi*1e6, wx = 2*np.pi*2e6, Z = 1, ionmass_amu = 9.012182): 
        self.N = N
        self.wz_E = wz
        self.wy_E = wy
        self.wx_E = wx
        self.Z = Z
        self.Z = ensure_numpy_array(Z,N)    
        self.ionmass_amu = ionmass_amu  
        self.ionmass_amu = ensure_numpy_array(ionmass_amu,N)
        self.q_E = self.Z * const.e
        self.m_E = self.ionmass_amu * const.u 
        self.initial_equilibrium_guess = None
    
   
   
    def dimensionless_parameters(self):
        q0 = self.q_E[0]
        m0 = self.m_E[0]    
        # ion properties    
        self.m = self.m_E / m0
        self.q = self.q_E / q0  
        # trap frequencies
        self.wz = 1 
        self.wy = self.wz * (self.wy_E / self.wz_E) 
        self.wx = self.wz * (self.wx_E / self.wz_E)
        # system parameters
        self.l0 = characteristic_length(q0, m0, self.wz_E)  # characteristic length 
        self.t0 = 1 / self.wz_E  # characteristic time
        self.v0 = self.l0 / self.t0  # characteristic velocity
        self.E0 = 0.5 * m0 * self.v0 ** 2  # characteristic energy  

    
    
    def trap_is_stable(self):
        # check that all trap frequencies are positive  
        return self.wz_E > 0 and self.wy_E > 0 and self.wx_E > 0



    def check_for_zero_modes(self):
        assert np.all(self.evals > 0), "All eigenvalues must be positive"   



    def run(self):
        self.dimensionless_parameters()
        assert self.trap_is_stable()    
        
        self.u = self.calculate_equilibrium_positions()
        #self.reindex_ions(self.u)
        self.E_matrix = self.get_E_matrix(self.u)  
        self.T_matrix = self.get_transform_matrix() 
        self.H_matrix = self.get_H_matrix(self.T_matrix, self.E_matrix)   
        self.evals, self.evecs = self.calculate_normal_modes(self.H_matrix)



    def calculate_equilibrium_positions(self):

        if self.initial_equilibrium_guess is None:
            self.initial_equilibrium_guess = self.get_initial_equilibrium_guess()
        else:
            self.u0 = self.initial_equilibrium_guess

        u = self.find_equilibrium_positions(self.u0)
        return u   


    def calculate_normal_modes(self, H_matrix):

       J = self.get_symplectic_matrix()
       D_matrix = J @ H_matrix  

       evals, evecs = np.linalg.eig(D_matrix)
       evals, evecs = self.sort_evals(evals, evecs)
       evecs = self.normalize_evecs(evecs, H_matrix)

       return evals, evecs 

    
    def get_initial_equilibrium_guess(self):
        self.u0 = np.zeros(3*self.N)
        self.u0[:] = (np.random.rand(3*self.N) * 2 - 1) * self.N 
        return self.u0


    def reindex_ions(self, u):  
        # based on the distance from the center of the trap, reindex the ions, smallest index is closest to the center
        x = u[0:self.N]
        y = u[self.N:2*self.N]
        z = u[2*self.N:]
        r = np.sqrt(x**2 + y**2 + z**2)
        idx = np.argsort(r)
        u = np.hstack((x[idx], y[idx], z[idx]))
        self.u = u 
        self.m = self.m[idx]
        self.m_E = self.m_E[idx]
        self.q = self.q[idx]
        self.q_E = self.q_E[idx]
        self.Z = self.Z[idx]

    def find_equilibrium_positions(self, u0):
        bfgs_tolerance = 1e-34
        out = opt.minimize(self.potential, u0, method='BFGS', jac=self.force,
                                    options={'gtol': bfgs_tolerance, 'disp': False})
        return out.x


    def potential_trap(self, pos_array):
        x = pos_array[0:self.N]
        y = pos_array[self.N:2*self.N]
        z = pos_array[2*self.N:]
        V_trap = 0.5 * np.sum((self.q * self.wx ** 2) * x ** 2) + \
            0.5 * np.sum((self.q * self.wy ** 2) * y ** 2) + \
                0.5 * np.sum((self.q * self.wz ** 2) * z ** 2)
        return V_trap
    
    def potential_coulomb(self, pos_array):
        x = pos_array[0:self.N]
        y = pos_array[self.N:2*self.N]
        z = pos_array[2*self.N:]

        dx = x[:, np.newaxis] - x
        dy = y[:, np.newaxis] - y
        dz = z[:, np.newaxis] - z
        rsep = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2).astype(np.float64)
        qq = (self.q * self.q[:, np.newaxis]).astype(np.float64)    

        with np.errstate(divide='ignore'):
            V_Coulomb = np.sum( np.where(rsep != 0., qq / rsep, 0) ) / 2 # divide by 2 to avoid double counting
        V_Coulomb *= .5 
        return V_Coulomb
        
    def potential(self, pos_array):
        return self.potential_trap(pos_array) + self.potential_coulomb(pos_array)   

    def force_trap(self, pos_array):
        x = pos_array[0:self.N]
        y = pos_array[self.N:2*self.N]
        z = pos_array[2*self.N:]

        Ftrapx = self.q * self.wx**2 * x
        Ftrapy = self.q * self.wy**2 * y
        Ftrapz = self.q * self.wz**2 * z

        force_trap = np.hstack((Ftrapx, Ftrapy, Ftrapz))
        return force_trap

    def force_coulomb(self, pos_array): 
        x = pos_array[0:self.N]
        y = pos_array[self.N:2*self.N]
        z = pos_array[2*self.N:]

        dx = x[:, np.newaxis] - x
        dy = y[:, np.newaxis] - y
        dz = z[:, np.newaxis] - z
        rsep = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2).astype(np.float64)
        qq = (self.q * self.q[:, np.newaxis]).astype(np.float64)    

        with np.errstate(divide='ignore', invalid='ignore'):
            rsep3 = np.where(rsep != 0., rsep ** (-3), 0)

        fx = dx * rsep3 * qq
        fy = dy * rsep3 * qq
        fz = dz * rsep3 * qq    

        Fx = -np.sum(fx, axis=1)
        Fy = -np.sum(fy, axis=1)
        Fz = -np.sum(fz, axis=1)

        force_coulomb = np.hstack((Fx, Fy, Fz))
        force_coulomb *= 0.5    
        return force_coulomb

    def force(self, pos_array):
        Force = self.force_coulomb(pos_array) + self.force_trap(pos_array)  
        return Force

    def hessian_trap(self, pos_array):
        Hxx = np.diag(self.q * (self.wx**2) * np.ones(self.N))
        Hyy = np.diag(self.q * (self.wy**2) * np.ones(self.N))  
        Hzz = np.diag(self.q * (self.wz**2) * np.ones(self.N))  
        zeros = np.zeros((self.N, self.N))  
        H = np.block([[Hxx, zeros, zeros], [zeros, Hyy, zeros], [zeros, zeros, Hzz]])
        return H

    def hessian_coulomb(self, pos_array):
        x = pos_array[0:self.N]
        y = pos_array[self.N:2*self.N]
        z = pos_array[2*self.N:]
        
        dx = x[:, np.newaxis] - x
        dy = y[:, np.newaxis] - y
        dz = z[:, np.newaxis] - z
        rsep = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2).astype(np.float64)  
        qq = (self.q * self.q[:, np.newaxis]).astype(np.float64)    

        with np.errstate(divide='ignore'):
            rsep5 = np.where(rsep != 0., rsep ** (-5), 0)

        dxsq = dx ** 2
        dysq = dy ** 2
        dzsq = dz ** 2
        rsep2 = rsep ** 2

        # X derivatives, Y derivatives for alpha != beta
        Hxx = (rsep2 - 3 * dxsq) * rsep5
        Hyy = (rsep2 - 3 * dysq) * rsep5
        Hzz = (rsep2 - 3 * dzsq) * rsep5

        # Above, for alpha == beta
        Hxx[np.diag_indices(self.N)] = -np.sum(Hxx, axis=0)
        Hyy[np.diag_indices(self.N)] = -np.sum(Hyy, axis=0)
        Hzz[np.diag_indices(self.N)] = -np.sum(Hzz, axis=0)

        Hxy = -3 * dx * dy * rsep5
        Hxy[np.diag_indices(self.N)] = 3 * np.sum(dx * dy * rsep5, axis=0)
        Hxz = -3 * dx * dz * rsep5
        Hxz[np.diag_indices(self.N)] = 3 * np.sum(dx * dz * rsep5, axis=0)
        Hyz = -3 * dy * dz * rsep5
        Hyz[np.diag_indices(self.N)] = 3 * np.sum(dy * dz * rsep5, axis=0)
        
        Hxx *= qq
        Hyy *= qq
        Hzz *= qq
        Hxy *= qq
        Hxz *= qq
        Hyz *= qq

        H_coulomb = np.block([[Hxx, Hxy, Hxz], [Hxy, Hyy, Hyz], [Hxz, Hyz, Hzz]])
        H_coulomb /= 2
        return H_coulomb

    def hessian(self, pos_array):
        H = self.hessian_coulomb(pos_array) + self.hessian_trap(pos_array)  
        return H

    def get_E_matrix(self,u): 
        PE_matrix = np.zeros((3*self.N, 3*self.N), dtype=np.complex128)
        KE_matrix = np.zeros((3*self.N, 3*self.N), dtype=np.complex128)
        E_matrix = np.zeros((6*self.N, 6*self.N), dtype=np.complex128)

        PE_matrix = self.hessian(u)
        KE_matrix = np.eye(3*self.N) # TODO: change for different masses
        zeros = np.zeros((3*self.N, 3*self.N)) 
        E_matrix = np.block([[PE_matrix, zeros], [zeros, KE_matrix]])
        return E_matrix

    def get_H_matrix(self, T_matrix, E_matrix):  
        T_matrix_inv = np.linalg.inv(T_matrix)  
        H_matrix = T_matrix_inv.T @ E_matrix @ T_matrix_inv
        return H_matrix 

    def get_transform_matrix(self):
        # assuming no magnetic field
        mass_matrix = np.diag(np.repeat(self.m, 3)) 
        eye = np.eye(3*self.N)  
        zeros = np.zeros((3*self.N, 3*self.N))
        T = np.block([[eye, zeros], [zeros, mass_matrix]])  
        return T    

    def get_symplectic_matrix(self):
        zeros = np.zeros((3*self.N, 3*self.N), dtype=np.complex128)
        I = np.eye(3*self.N, dtype=np.complex128)
        J = np.block([[zeros, I], [-I, zeros]])
        return J    

    def sort_evals(self,evals, evecs):
       evals = np.imag(evals)
       sort_dex = np.argsort(evals)
       evals = evals[sort_dex]
       evecs = evecs[:,sort_dex]
       half = int(len(evals)/2)
       evals = evals[half:]
       evecs = evecs[:,half:]
       return evals, evecs

    def normalize_evecs(self,evecs,E_matrix):
        # vectors are orthogonal with respect to Hamiltonian matrix
        _, modes = np.shape(evecs)
        for i in range(modes): 
           vec = evecs[:,i]
           norm = np.sqrt(vec.T.conj() @ E_matrix @ vec)
           evecs[:,i] = vec / norm
        return evecs

if __name__ == '__main__':
    def get_branch_nums(evecs): 
        num_coords, num_modes = np.shape(evecs)   
        N = num_modes // 3
        branch_nums = np.zeros(num_modes)
        x_indices = np.arange(0,N)  
        y_indices = np.arange(N,2*N)    
        z_indices = np.arange(2*N,3*N)
        for mode in range(num_modes): 
            non_zero_indices = np.where(np.abs(evecs[:num_coords//2,mode]) + abs(evecs[num_coords//2:,mode]) > 1e-6)[0] 
            if np.all(np.isin(non_zero_indices,x_indices)):
                branch_nums[mode] = 0 
            elif np.all(np.isin(non_zero_indices,y_indices)):
                branch_nums[mode] = 1
            elif np.all(np.isin(non_zero_indices,z_indices)):
                branch_nums[mode] = 2
            else:
                branch_nums[mode] = -1 
        return branch_nums
                

    mBe_amu = 9.012182  
    mYb_amu = 170.936323    
    mBa_amu = 137.327   
    charge_1 = 1
    charge_2 = 10
    N = 5
    wz = 2*np.pi*.2e6
    wy = 2*np.pi*1.5e6
    wx = 2*np.pi*3e6   
    ionmass_amu = [mBe_amu, mBa_amu, mYb_amu, mBa_amu, mBe_amu] 
    ionmass_amu = [mYb_amu, mYb_amu, mYb_amu, mYb_amu, mBe_amu] 
    zeros = np.zeros(N)
    zeros += np.random.rand(N) * 1e-6
    initial_equilibrium_guess = np.hstack([zeros, zeros, np.linspace(-N/2,N/2,N)])
    gma_same_ions= GeneralizedModeAnalysis(N = N, wz = wz, wy = wy, wx = wx, ionmass_amu = mYb_amu) 
    gma_same_ions.initial_equilibrium_guess = initial_equilibrium_guess
    gma_different_ions = GeneralizedModeAnalysis(N=N, ionmass_amu=ionmass_amu, wz = wz, wy = wy, wx = wx)      
    gma_different_ions.initial_equilibrium_guess = initial_equilibrium_guess

    gma_same_ions.run() 
    gma_different_ions.run()

    same_branch_nums = get_branch_nums(gma_same_ions.evecs)  
    diff_branch_nums = get_branch_nums(gma_different_ions.evecs)
    print(same_branch_nums)
    print(diff_branch_nums)
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1,1)
    x = np.zeros(gma_same_ions.N)   
    z_same = gma_same_ions.u[2*gma_same_ions.N:] *gma_same_ions.l0 * 1e6
    z_diff = gma_different_ions.u[2*gma_different_ions.N:] * gma_different_ions.l0 * 1e6    
    ax.scatter(x,z_same, label='Same ions', s = gma_same_ions.m) 
    # add text annotation showing the ion number
    for i, txt in enumerate(range(gma_same_ions.N)):
        ax.annotate(txt, (x[i], z_same[i])) 
    ax.scatter(x,z_diff, label='Different ions', s = gma_different_ions.m)  
    for i, txt in enumerate(range(gma_different_ions.N)):
        ax.annotate(txt, (x[i], z_diff[i])) 
    ax.legend() 

    mode_nums = np.arange(1, len(gma_same_ions.evals)+1)
    fig, ax = plt.subplots(1,1)
    # plot the mode frequencies for the two cases
    ax.scatter(mode_nums,gma_same_ions.evals, label='Same ions')
    ax.scatter(mode_nums,gma_different_ions.evals, label='Different ions')
    ax.legend()
    plt.show()


    gma_same_charge = GeneralizedModeAnalysis(N=N, Z=1, wz=wz, wy=wy, wx=wx)    
    gma_same_charge.initial_equilibrium_guess = initial_equilibrium_guess   
    gma_different_charge = GeneralizedModeAnalysis(N=N, Z=[charge_1 if i%2==0 else charge_2 for i in range(N)], wz=wz, wy=wy, wx=wx)    
    gma_different_charge.initial_equilibrium_guess = initial_equilibrium_guess  



    gma_same_charge.run()
    gma_different_charge.run()


    print(get_branch_nums(gma_same_charge.evecs))
    print(get_branch_nums(gma_different_charge.evecs))
    fig, ax = plt.subplots(1,1) 
    z_same = gma_same_charge.u[2*gma_same_charge.N:] * gma_same_charge.l0 * 1e6 
    z_diff = gma_different_charge.u[2*gma_different_charge.N:] * gma_different_charge.l0 * 1e6
    ax.scatter(x,z_same, label='Same charge', s = 10*gma_same_charge.q)
    for i, txt in enumerate(range(gma_same_charge.N)):
        ax.annotate(txt, (x[i], z_same[i]))
    ax.scatter(x,z_diff, label='Different charge', s = 10*gma_different_charge.q)
    for i, txt in enumerate(range(gma_different_charge.N)):
        ax.annotate(txt, (x[i], z_diff[i]))
    ax.legend()

    fig, ax = plt.subplots(1,1)
    # plot the mode frequencies for the two cases
    ax.scatter(mode_nums,gma_same_charge.evals, label='Same charge') 
    ax.scatter(mode_nums,gma_different_charge.evals, label='Different charge')
    ax.legend()
    plt.show()


