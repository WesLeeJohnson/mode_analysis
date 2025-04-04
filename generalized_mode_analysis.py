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
import warnings
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



def get_norm(en,H):
    """
    Get the norm of the eigen vector en w.r.t. the Hamiltonian H.
    """
    norm = np.sqrt(en.T.conj() @ H @ en)
    return norm



def normalize_eigen_vectors(ens,H,evs=None): 
    """
    Rescale the eigen vectors ens w.r.t. the Hamiltonian H.
    """
    ens_rescaled = np.zeros_like(ens,dtype=complex)
    num_coords,num_evs = np.shape(ens)
    if evs is None:
        evs = np.ones(num_evs)
    for i in range(num_evs):
        en = ens[:,i].reshape(num_coords,1)
        norm = get_norm(en,H)
        ens_rescaled[:,i] = en[:,0]/norm
        ens_rescaled[:,i] *= np.sqrt(evs[i])
    return ens_rescaled 



def get_canonical_transformation(H,ens,evs=None):
    """
    Given the eigen-solve of the dynamical matrix, get the transform matrix 
    to the canonical coordinates.
    X = S X', where X' = (Q,P)^T and X = (q,p)^T
    """
    ## TODO: understand the sign in the transformation matrix
    sign = -1
    num_coords, num_evs = np.shape(ens)
    assert num_coords //2 == num_evs    
    T = np.zeros((num_coords,num_coords),dtype=complex) 
    ens = normalize_eigen_vectors(ens,H,evs=evs)
    T = np.sqrt(2)*np.concatenate((np.real(ens), sign*np.imag(ens)), axis=1)
    # the sign is likely due to the convention of writing the time evolution as exp(-iwt)
    return T



class GeneralizedModeAnalysis:
    def __init__(self, N=2, wz = 2*np.pi*.1e6,wy = 2*np.pi*1e6, wx = 2*np.pi*2e6, Z = 1, ionmass_amu = 9.012182): 
        self.N = N
        self.Z = ensure_numpy_array(Z,N)
        self.ionmass_amu = ensure_numpy_array(ionmass_amu,N)
        self.q_E = self.Z * const.e
        self.m_E = self.ionmass_amu * const.u 
        self.initial_equilibrium_guess = None
        self.hasrun = False 
        # trapping frequencies
        self.wz_E = self.calculate_species_trap_frequencies(self.m_E, self.q_E, wz)    
        self.wy_E = self.calculate_species_trap_frequencies(self.m_E, self.q_E, wy)
        self.wx_E = self.calculate_species_trap_frequencies(self.m_E, self.q_E, wx)   
    
    def calculate_species_trap_frequencies(self,m_E, q_E, omega): 
        # assume that the trapping frequency given corresponds to the first ion species 
        u_r_sqr = np.sqrt(m_E[0] / q_E[0]) * omega
        omega_E = np.sqrt(q_E / m_E) * u_r_sqr    
        return omega_E 
   
    def dimensionless_parameters(self):
        # normalize to the first ion species    
        q0 = self.q_E[0]
        m0 = self.m_E[0]    
        w0 = self.wz_E[0]   
        self.q0 = q0
        self.m0 = m0
        self.w0 = w0
        # ion properties    
        self.m = self.m_E / m0
        self.q = self.q_E / q0  
        # trap frequencies
        self.wz = self.wz_E / w0 
        self.wy = self.wy_E / w0 
        self.wx = self.wx_E / w0 
        # system parameters
        self.l0 = characteristic_length(q0, m0, w0)  # characteristic length 
        self.t0 = 1 / w0  # characteristic time
        self.v0 = self.l0 / self.t0  # characteristic velocity
        self.E0 = 0.5 * m0 * self.v0 ** 2  # characteristic energy  

    
    
    def trap_is_stable(self):
        # check that all trap frequencies are positive  
        return np.all(self.wz_E > 0) and np.all(self.wy_E > 0) and np.all(self.wx_E > 0)



    def check_for_zero_modes(self):
        assert np.all(self.evals > 0), "All eigenvalues must be positive"   



    def check_outer_relation(self): 
        H = self.H_matrix.copy()       
        D = self.get_symplectic_matrix() @ H
        Eval, Evec = np.linalg.eig(D)
        _, en = self.sort_modes(Eval,Evec)
        en = self.normalize_eigen_vectors(en,H)
        Outers = np.zeros((6*self.N,6*self.N),dtype=complex)
        for i in range(6*self.N):
            norm = get_norm(en[:,i],H)
            Outers = Outers + np.outer(en[:,i],en[:,i].conj())/norm 
        I_right = H @ Outers
        I_left  = Outers @ H
        eye = np.eye(6*self.N,dtype=complex)
        np.set_printoptions(precision=2, suppress=True) 
        try:
            assert np.allclose(I_left,eye) 
            assert np.allclose(I_right,eye)
        except AssertionError:
            warnings.warn("Outer relation check failed")



    def has_duplicate_evals(self,evals):
        evs = evals.copy()    
        return np.any(np.triu(np.isclose(evs[:, None], evs[None, :], atol=1e-6), k=1))  
    


    def check_diagnolization(self):
        M = np.linalg.inv(self.T_matrix) @ self.S_matrix    
        H_diag = M.T @ self.E_matrix @ M    
        H_diag_check = np.diag(np.tile(self.evals,2)) 
        np.set_printoptions(precision=2, suppress=True) 
        try:
            assert np.allclose(H_diag,H_diag_check)
        except AssertionError:
            warnings.warn("Diagnolization check failed")    
            print("has duplicate evals: ", self.has_duplicate_evals(self.evals))



    def checks(self):   
        self.check_outer_relation()
        self.check_diagnolization()



    def run(self):
        self.dimensionless_parameters()
        assert self.trap_is_stable()    
        
        self.u = self.calculate_equilibrium_positions()
        #self.reindex_ions(self.u)
        self.E_matrix = self.get_E_matrix(self.u)  
        self.T_matrix = self.get_momentum_transform() 
        self.H_matrix = self.get_H_matrix(self.T_matrix, self.E_matrix)   
        self.evals, self.evecs = self.calculate_normal_modes(self.H_matrix)
        self.evecs_vel = self.get_eigen_vectors_xv_coords(self.T_matrix,self.evecs)    
        self.check_for_zero_modes() 
        self.S_matrix = self.get_canonical_transformation() 
        self.checks() 
        self.hasrun = True  
    


    def calculate_equilibrium_positions(self):

        if self.initial_equilibrium_guess is None:
            self.initial_equilibrium_guess = self.get_initial_equilibrium_guess()
        else:
            self.u0 = self.initial_equilibrium_guess

        u = self.find_equilibrium_positions(self.u0)
        self.p0 = self.potential(u)
        return u   



    def get_canonical_transformation(self):
        return get_canonical_transformation(self.H_matrix,self.evecs,evs=self.evals)    



    def normalize_eigen_vectors(self, evecs, H_matrix):
        return normalize_eigen_vectors(evecs,H_matrix) 



    def get_eigen_vectors_xv_coords(self,T,ens): 
        ens_vel = np.zeros_like(ens,dtype=complex)
        ens_vel[:,:] = np.linalg.inv(T) @ ens
        return ens_vel 



    def calculate_normal_modes(self, H_matrix):

        J = self.get_symplectic_matrix()
        D_matrix = J @ H_matrix  
        # u_n(t) = exp(-i w_n t) u_n(0), minus by convention
        evals, evecs = np.linalg.eig(-D_matrix)   
        evals, evecs = self.organize_modes(evals, evecs)
        evecs = self.normalize_eigen_vectors(evecs, H_matrix) 

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
        V_trap = 0.5 * np.sum((self.m * self.wx ** 2) * x ** 2) + \
            0.5 * np.sum((self.m * self.wy ** 2) * y ** 2) + \
                0.5 * np.sum((self.m * self.wz ** 2) * z ** 2)
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

        Ftrapx = self.m * self.wx**2 * x
        Ftrapy = self.m * self.wy**2 * y
        Ftrapz = self.m * self.wz**2 * z

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
        Hxx = np.diag(self.m * (self.wx**2) * np.ones(self.N))
        Hyy = np.diag(self.m * (self.wy**2) * np.ones(self.N))  
        Hzz = np.diag(self.m * (self.wz**2) * np.ones(self.N))  
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

    def get_mass_matrix(self,m):    
        return np.diag(np.tile(m, 3)) 

    def get_E_matrix(self,u): 
        PE_matrix = np.zeros((3*self.N, 3*self.N), dtype=np.complex128)
        KE_matrix = np.zeros((3*self.N, 3*self.N), dtype=np.complex128)
        E_matrix = np.zeros((6*self.N, 6*self.N), dtype=np.complex128)

        PE_matrix = self.hessian(u)
        KE_matrix = self.get_mass_matrix(self.m) 
        zeros = np.zeros((3*self.N, 3*self.N)) 
        E_matrix = np.block([[PE_matrix, zeros], [zeros, KE_matrix]])
        return E_matrix



    def get_H_matrix(self, T_matrix, E_matrix):  
        T_matrix_inv = np.linalg.inv(T_matrix)  
        H_matrix = T_matrix_inv.T @ E_matrix @ T_matrix_inv
        return H_matrix 



    def get_momentum_transform(self):
        # assuming no magnetic field
        mass_matrix = self.get_mass_matrix(self.m)  
        eye = np.eye(3*self.N)  
        zeros = np.zeros((3*self.N, 3*self.N))
        T = np.block([[eye, zeros], [zeros, mass_matrix]])  
        return T    



    def get_symplectic_matrix(self):
        zeros = np.zeros((3*self.N, 3*self.N), dtype=np.complex128)
        I = np.eye(3*self.N, dtype=np.complex128)
        J = np.block([[zeros, I], [-I, zeros]])
        return J    



    def sort_modes(self,evals, evecs):
       evals = np.imag(evals)
       sort_dex = np.argsort(evals)
       evals = evals[sort_dex]
       evecs = evecs[:,sort_dex]
       return evals, evecs  
    


    def split_modes(self,evals, evecs): 
       half = len(evals) // 2
       evals = evals[half:]
       evecs = evecs[:,half:]
       return evals, evecs
    


    def organize_modes(self,evals, evecs):  
        evals, evecs = self.sort_modes(evals, evecs)    
        evals, evecs = self.split_modes(evals, evecs)
        return evals, evecs 






if __name__ == '__main__':
    # set plotting settings

    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 16})
    plt.rcParams.update({'figure.autolayout': True})
    plt.rcParams.update({'figure.figsize': (8, 6)})
    plt.rcParams.update({'lines.linewidth': 2})
    plt.rcParams.update({'axes.linewidth': 2})
    plt.rcParams.update({'xtick.major.width': 2})
    plt.rcParams.update({'ytick.major.width': 2})
    plt.rcParams.update({'xtick.minor.width': 2})
    plt.rcParams.update({'ytick.minor.width': 2})
    plt.rcParams.update({'xtick.direction': 'in'})
    plt.rcParams.update({'ytick.direction': 'in'})
    plt.rcParams.update({'xtick.labelsize': 16})
    plt.rcParams.update({'ytick.labelsize': 16})
    plt.rcParams.update({'axes.labelsize': 16})
    plt.rcParams.update({'legend.fontsize': 16})
    plt.rcParams.update({'savefig.format': 'pdf'})
    plt.rcParams.update({'savefig.bbox': 'tight'})
    plt.rcParams.update({'savefig.pad_inches': 0.1})
    plt.rcParams.update({'savefig.transparent': True})
    # grid 
    plt.rcParams.update({'axes.grid': True})
    plt.rcParams.update({'grid.linestyle': '-'})
    plt.rcParams.update({'grid.linewidth': 1})
    plt.rcParams.update({'grid.alpha': 0.5})
    plt.rcParams.update({'grid.color': 'black'})

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
    split = 3
    wz = 2*np.pi*.2e6
    wy = 2*np.pi*1.5e6
    wx = 2*np.pi*3e6   
    ionmass_amu = [mYb_amu if i<N-split else mBe_amu for i in range(N)] 
    zeros = np.zeros(N)
    initial_equilibrium_guess = np.hstack([zeros, zeros, np.linspace(-N/2,N/2,N)])
    gma_same_ions= GeneralizedModeAnalysis(N = N, wz = wz, wy = wy, wx = wx, ionmass_amu = mYb_amu) 
    gma_same_ions.initial_equilibrium_guess = initial_equilibrium_guess
    gma_different_ions = GeneralizedModeAnalysis(N=N, ionmass_amu=ionmass_amu, wz = wz, wy = wy, wx = wx)      
    gma_different_ions.initial_equilibrium_guess = initial_equilibrium_guess

    gma_same_ions.run() 
    gma_different_ions.run()

    def get_evals_alternate_calculation(gma): 
        N = gma.N
        masses = gma.m
        mass_vec = np.tile(masses,3)
        weights = 1/np.sqrt(np.outer(mass_vec,mass_vec))
        PE_matrix = gma.E_matrix[:3*N,:3*N]  
        PE_weighted = weights * PE_matrix   
        evals, _ = np.linalg.eigh(PE_weighted)
        return np.sqrt(evals)


    same_branch_nums = get_branch_nums(gma_same_ions.evecs)  
    diff_branch_nums = get_branch_nums(gma_different_ions.evecs)
    print(same_branch_nums)
    print(diff_branch_nums)
    import matplotlib.pyplot as plt
    point_size = 20





    def plot_hlines(ax,ma):
        N = ma.N
        w0 = 1/2/np.pi/1e6 
        ax.hlines(ma.wz_E*w0,0,N,color='g',linestyle='--')
        ax.hlines(ma.wy_E*w0,N,2*N,color='b',linestyle='--')
        ax.hlines(ma.wx_E*w0,2*N,3*N,color='r',linestyle='--')
    
    def plot_evals(ax,ma,**kwargs): 
        mode_nums = np.arange(1, len(ma.evals)+1)
        w0 = ma.wz_E[0]/2/np.pi/1e6
        ax.scatter(mode_nums,ma.evals*w0,**kwargs)  
        ax.set_xlabel('Mode number')    
        ax.set_ylabel('Frequency (MHz)')    
        ax.set_title('Mode Spectra')

    def plot_equilibrium_positions(ax,ma,**kwargs): 
        z = ma.u[2*ma.N:] * ma.l0 * 1e6
        x = np.zeros_like(z)
        ax.scatter(x,z,**kwargs)    
        lim = np.abs(z).max()*1.1 
        ax.set_xlim(-lim,lim)
        ax.set_ylim(-lim,lim)
        ax.set_aspect('equal')  
        ax.set_xlabel('x ($\mu$m)')
        ax.set_ylabel('z ($\mu$m)')
        ax.set_title('Equilibrium positions')

    def label_ions(ax,ma):
        z = ma.u[2*ma.N:] * ma.l0 * 1e6
        x = np.zeros_like(z)
        for i, txt in enumerate(range(ma.N)):
            ax.annotate(txt, (x[i], z[i]))





    fig, axs = plt.subplots(1,2,figsize=(10,6)) 

    ax = axs[0] 
    plot_equilibrium_positions(ax,gma_same_ions,label='Same ions',c='b')    
    label_ions(ax,gma_same_ions)    
    plot_equilibrium_positions(ax,gma_different_ions,label='Different ions',c='r')
    label_ions(ax,gma_different_ions)   
    ax.legend() 

    # plot the mode frequencies for the two cases
    ax = axs[1] 
    plot_evals(ax,gma_same_ions,label='Same ions',c='b')    
    plot_hlines(ax,gma_same_ions)
    plot_evals(ax,gma_different_ions,label='Different ions',c='r')  
    plot_hlines(ax,gma_different_ions)
    w0 = gma_different_ions.wz_E[0]/2/np.pi/1e6  
    evals_same = get_evals_alternate_calculation(gma_same_ions) 
    evals_diff = get_evals_alternate_calculation(gma_different_ions)
    ax.scatter(np.arange(1,len(evals_same)+1),evals_same*w0,c='b',label='Same ions (alternate calculation)',marker='x',s=100)    
    ax.scatter(np.arange(1,len(evals_diff)+1),evals_diff*w0,c='r',label='Different ions (alternate calculation)',marker='x',s=100)  
    ax.legend()

    plt.show()
    exit() 


    gma_same_charge = GeneralizedModeAnalysis(N=N, Z=1, wz=wz, wy=wy, wx=wx)    
    gma_same_charge.initial_equilibrium_guess = initial_equilibrium_guess   
    gma_different_charge = GeneralizedModeAnalysis(N=N, Z=[charge_1 if i%2==0 else charge_2 for i in range(N)], wz=wz, wy=wy, wx=wx)    
    gma_different_charge.initial_equilibrium_guess = initial_equilibrium_guess  



    gma_same_charge.run()
    gma_different_charge.run()


    print(get_branch_nums(gma_same_charge.evecs))
    print(get_branch_nums(gma_different_charge.evecs))
    fig, axs = plt.subplots(1,2,figsize=(10,6))  

    ax = axs[0]
    plot_equilibrium_positions(ax,gma_same_charge,label='Same charge',c='b')    
    label_ions(ax,gma_same_charge)
    plot_equilibrium_positions(ax,gma_different_charge,label='Different charge',c='r')
    label_ions(ax,gma_different_charge)
    ax.legend()
    ax.legend()

    # plot the mode frequencies for the two cases
    ax = axs[1] 
    plot_evals(ax,gma_same_charge,label='Same charge',c='b')
    plot_hlines(ax,gma_same_charge)
    plot_evals(ax,gma_different_charge,label='Different charge',c='r')
    plot_hlines(ax,gma_different_charge)
    ax.legend()

    plt.show()



