"""
Author:     Wes Johnson
Date:       May 31st, 2024 
Purpose:    Perform the normal mode analysis for an ion crystal in a harmonic trap. 
How to run: 
            import harmonic_trap_mode_analysis as htma
            htma_obj = htma.HarmonicTrapModeAnalysis(<parameters>)
            htma_obj.run()
"""

import numpy as np
import scipy.constants as const
import scipy.optimize as opt

class HarmonicTrapModeAnalysis:
    def __init__(self, N = 2, wx = 2*np.pi*3e6, wy = 2*np.pi*2.5e6, wz = 2*np.pi*.3e6, ionmass_amu = 170.936323, Z = 1) -> None:
        self.N = N
        self.wx_E = wx
        self.wy_E = wy
        self.wz_E = wz
        self.m_E = ionmass_amu * const.atomic_mass    
        self.q_E = Z * const.e    

        self.hasrun = False
        self.initial_equilibrium_guess = None   
        pass

    def dimensionless_parameters(self):
        self.m = 1
        # trap frequencies
        self.wz = 1 
        self.wy = self.wz * (self.wy_E / self.wz_E) 
        self.wx = self.wz * (self.wx_E / self.wz_E)
        k_e = 1 / (4 * np.pi * const.epsilon_0)  # Coulomb constant
        self.l0 = ((k_e * self.q_E ** 2) / (.5 * self.m_E * self.wz_E ** 2)) ** (1 / 3)
        self.t0 = 1 / self.wz_E  # characteristic time
        self.v0 = self.l0 / self.t0  # characteristic velocity
        self.E0 = 0.5*self.m*(self.wz_E**2)*self.l0**2 # characteristic energy


    def dimensionful_parameters(self):
        self.evals_E = self.evals * self.wz_E / 2 / np.pi
        self.evecs_E = np.zeros((6*self.N, 3*self.N), dtype=np.complex128)
        self.evecs_E[:3*self.N, :] = self.evecs[:3*self.N, :]
        self.evecs_E[3*self.N:, :] = self.evecs[3*self.N:, :]   
        self.uE = self.u * self.l0
        


    def is_trap_stable(self):
        assert self.wx > 0e0 and self.wy > 0e0 and self.wz > 0e0, "Trap frequencies must be positive"
        assert self.wx > self.wy and self.wy and self.wz, "Trap frequencies must be ordered wx > wy > wz"

    def check_for_zero_modes(self):
        # check mode frequencies are non-zero
        assert np.all(np.abs(self.evals) > 1e-10), "Zero frequency mode, ion crystal is not stable."

    def run(self):
        
        self.dimensionless_parameters() 

        self.is_trap_stable() 

        self.calculate_equilibrium_positions()
        self.evals, self.evecs, self.E_matrix = self.calculate_normal_modes()
        self.check_for_zero_modes() 
        self.dimensionful_parameters()
        self.hasrun = True

    def calculate_equilibrium_positions(self):

        if self.initial_equilibrium_guess is None:
            self.initial_equilibrium_guess = self.get_initial_equilibrium_guess()
        else:
            self.u0 = self.initial_equilibrium_guess

        self.u = self.find_equilibrium_positions(self.u0)
        return self.u   

    def calculate_normal_modes(self):
       evals = np.zeros(3*self.N, dtype=np.complex128)
       evecs = np.zeros((6*self.N, 3*self.N), dtype=np.complex128)
       E_matrix = np.zeros((6*self.N, 6*self.N))
       E_matrix = self.get_E_matrix(self.u)
       J = self.get_symplectic_matrix()
       D_matrix = J @ E_matrix 
       evals, evecs = np.linalg.eig(D_matrix)
       evals, evecs = self.sort_evals(evals, evecs)
       evecs = self.normalize_evecs(evecs, E_matrix)
       return evals, evecs, E_matrix


    def get_initial_equilibrium_guess(self):
        # assumes linear chain along z-axis
        # assumes ions are equally spaced
        # vector of equilibrium positions organized as [x1, x2, x3, ..., xn, y1, y2, y3, ..., yn, z1, z2, z3, ..., zn]
        self.u0 = np.zeros(3*self.N)
        self.u0[2*self.N:] = np.linspace(-self.l0*(self.N-1)/2, self.l0*(self.N-1)/2, self.N, endpoint=True)
        return self.u0
        
    def find_equilibrium_positions(self, u0):
        bfgs_tolerance = 1e-34
        out = opt.minimize(self.pot_energy, u0, method='BFGS', jac=self.force,
                                    options={'gtol': bfgs_tolerance, 'disp': False})
        return out.x
    
    def pot_energy(self, pos_array):
        x = pos_array[0:self.N]
        y = pos_array[self.N:2*self.N]
        z = pos_array[2*self.N:]

        dx = x.reshape((x.size, 1)) - x
        dy = y.reshape((y.size, 1)) - y
        dz = z.reshape((z.size, 1)) - z
        rsep = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)

        with np.errstate(divide='ignore'):
            V_Coulomb = np.sum( np.where(rsep != 0., 1 / rsep, 0) ) / 2 # divide by 2 to avoid double counting

        #V = (self.beta + self.delta) * np.sum(self.md * x ** 2) \
        #    + (self.beta - self.delta) * np.sum(self.md * y ** 2) \
        #        + np.sum(self.md * z ** 2) + 0.5 * np.sum(Vc)   
        V_trap = self.m * (self.wx ** 2) * np.sum(x ** 2) + \
            self.m * (self.wy ** 2) * np.sum(y ** 2) + \
                self.m * (self.wz ** 2) * np.sum(z ** 2)
        return V_trap + V_Coulomb

    def force(self, pos_array):
        x = pos_array[0:self.N]
        y = pos_array[self.N:2*self.N]
        z = pos_array[2*self.N:]

        dx = x.reshape((x.size, 1)) - x
        dy = y.reshape((y.size, 1)) - y
        dz = z.reshape((z.size, 1)) - z
        rsep = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2).astype(np.float64)  

        with np.errstate(divide='ignore', invalid='ignore'):
            rsep3 = np.where(rsep != 0., rsep ** (-3), 0)
        
        fx = dx * rsep3
        fy = dy * rsep3
        fz = dz * rsep3

        Ftrapx = 2 * self.m * self.wx**2 * x
        Ftrapy = 2 * self.m * self.wy**2 * y
        Ftrapz = 2 * self.m * self.wz**2 * z

        Fx = -np.sum(fx, axis=1) + Ftrapx
        Fy = -np.sum(fy, axis=1) + Ftrapy
        Fz = -np.sum(fz, axis=1) + Ftrapz

        Force = np.hstack((Fx, Fy, Fz))
        return Force
    
    def hessian(self, pos_array):
        x = pos_array[0:self.N]
        y = pos_array[self.N:2*self.N]
        z = pos_array[2*self.N:]
         
        dx = x.reshape((x.size, 1)) - x
        dy = y.reshape((y.size, 1)) - y
        dz = z.reshape((z.size, 1)) - z
        rsep = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)

        with np.errstate(divide='ignore'):
            rsep5 = np.where(rsep != 0., rsep ** (-5), 0)

        dxsq = dx ** 2
        dysq = dy ** 2
        dzsq = dz ** 2

        # X derivatives, Y derivatives for alpha != beta
        Hxx = np.mat((rsep ** 2 - 3 * dxsq) * rsep5)
        Hyy = np.mat((rsep ** 2 - 3 * dysq) * rsep5)
        Hzz = np.mat((rsep ** 2 - 3 * dzsq) * rsep5)

        # Above, for alpha == beta
        # np.diag usa diagnoal value to form a matrix
        Hxx += np.mat(np.diag(2 * self.m * (self.wx**2) -
                              np.sum((rsep ** 2 - 3 * dxsq) * rsep5, axis=0)))
        Hyy += np.mat(np.diag(2 * self.m * (self.wy**2) -
                              np.sum((rsep ** 2 - 3 * dysq) * rsep5, axis=0)))
        Hzz += np.mat(np.diag(2 * self.m  * (self.wz**2)-
                              np.sum((rsep ** 2 - 3 * dzsq) * rsep5, axis=0)))

        Hxy = np.mat(-3 * dx * dy * rsep5)
        Hxy += np.mat(np.diag(3 * np.sum(dx * dy * rsep5, axis=0)))
        Hxz = np.mat(-3 * dx * dz * rsep5)
        Hxz += np.mat(np.diag(3 * np.sum(dx * dz * rsep5, axis=0)))
        Hyz = np.mat(-3 * dy * dz * rsep5)
        Hyz += np.mat(np.diag(3 * np.sum(dy * dz * rsep5, axis=0)))

        H = np.bmat([[Hxx, Hxy, Hxz], [Hxy, Hyy, Hyz], [Hxz, Hyz, Hzz]])
        H = np.asarray(H)
        H /= 2
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

if __name__ == "__main__":
    import matplotlib.pyplot as plt


    # format matplotlib 
    plt.rcParams.update({'font.size': 16})
    plt.rcParams.update({'figure.figsize': [8, 6]})
    plt.rcParams.update({'axes.grid': True})
    plt.rcParams.update({'grid.alpha': 0.5})
    plt.rcParams.update({'grid.linestyle': '-'})
    plt.rcParams.update({'grid.linewidth': 0.5})
    plt.rcParams.update({'grid.color': 'k'})
    plt.rcParams.update({'legend.fontsize': 14})
    plt.rcParams.update({'legend.frameon': True})
    plt.rcParams.update({'legend.framealpha': 1})
    plt.rcParams.update({'legend.facecolor': 'w'})
    plt.rcParams.update({'legend.edgecolor': 'k'})
    plt.rcParams.update({'legend.shadow': False})
    plt.rcParams.update({'legend.fancybox': True})
    plt.rcParams.update({'lines.linewidth': 2})
    plt.rcParams.update({'lines.markersize': 8})
    plt.rcParams.update({'xtick.labelsize': 14})
    plt.rcParams.update({'ytick.labelsize': 14})
    plt.rcParams.update({'axes.labelsize': 16})
    plt.rcParams.update({'axes.titlesize': 18})
    plt.rcParams.update({'figure.titlesize': 20})
    plt.rcParams.update({'figure.dpi': 200})
    plt.rcParams.update({'savefig.dpi': 200})
    plt.rcParams.update({'savefig.format': 'pdf'})
    plt.rcParams.update({'savefig.bbox': 'tight'})

    htma_obj = HarmonicTrapModeAnalysis(N=6,wz=2*np.pi*.3e6,wy=2*np.pi*2.1e6,wx=2*np.pi*2.5e6)  
    #htma_obj.initial_equilibrium_guess = np.random.rand(3*htma_obj.N) - 0.5
    htma_obj.run()
    # check eigen vectors are orthogonal and normalized wrt Hamiltonian matrix
    def check_normalization(): 
        evecs_all = np.empty((6*htma_obj.N, 6*htma_obj.N), dtype=np.complex128)
        check_matrix = np.zeros((6*htma_obj.N, 6*htma_obj.N), dtype=np.complex128)
        evecs_all[:,0:3*htma_obj.N] = htma_obj.evecs
        evecs_all[:,3*htma_obj.N:] = np.conj(htma_obj.evecs)
        for i in range(6*htma_obj.N):
            for j in range(6*htma_obj.N):
                element = evecs_all[:,i].T.conj() @ htma_obj.E_matrix @ evecs_all[:,j]
                assert np.abs(element) < 1e2, f"Element {i},{j} is not zero: {element}"
                check_matrix[i,j] = element
        assert np.allclose(np.eye(6*htma_obj.N), check_matrix, atol=1e-2), "Eigenvectors are not orthogonal"

    def find_center_of_mass_modes(evecs):
        # center of mass modes have equal amplitudes for all ions
        mode_displacement_variances = np.var(evecs/np.linalg.norm(evecs,axis=0), axis=0) 
        # get the three smallest variances
        center_of_mass_modes = np.argsort(mode_displacement_variances)[:3]
        return center_of_mass_modes
    
    center_of_mass_mode_indices = find_center_of_mass_modes(htma_obj.evecs) 

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    axs = axs.flatten()

    ax = axs[0]
    ax.set_title("Normal mode frequencies")
    mode_numbers = np.arange(0, 3*htma_obj.N)
    ax.scatter(mode_numbers, htma_obj.evals * htma_obj.wz_E / 2 / np.pi * 1e-6)
    # draw center of mass modes with star 
    ax.scatter(center_of_mass_mode_indices, htma_obj.evals[center_of_mass_mode_indices] * htma_obj.wz_E / 2 / np.pi * 1e-6, marker='*', color='r')
    ax.set_xlabel('Mode number')
    ax.set_ylabel('Frequency (MHz)')    
    ax.set_ylim(0, htma_obj.wx_E / 2 / np.pi * 1e-6 * 1.1)

    ax = axs[1] 
    ax.set_title("Equilibrium positions")
    eq_pos_3D = htma_obj.u.reshape((3, htma_obj.N))
    eq_pos_3D *= htma_obj.l0 * 1e6  
    y = eq_pos_3D[1] 
    z = eq_pos_3D[2] 
    ax.scatter(z, y)    
    ax.set_xlabel("z ($\mu$m)") 
    ax.set_ylabel("y ($\mu$m)")

    plt.tight_layout() 
    plt.show()



