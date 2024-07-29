"""
Author:     Wes Johnson
Date:       July 29th, 2024 
Purpose:    Perform the normal mode analysis of an ion crystal in a trap 
            with a quartic anharmonicity. 
How to run:
            python quartic_trap_mode_analysis.py
"""
import numpy as np 
from harmonic_trap_mode_analysis import HarmonicTrapModeAnalysis    

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
        assert self.a_2 > 0, "Anharmonicity must be positive"    

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
    htma_obj.run()
    qtma_obj = QuarticTrapModeAnalysis(N=6,wz=2*np.pi*.3e6,wy=2*np.pi*2.1e6,wx=2*np.pi*2.5e6, anharmonicity=0.1)    
    qtma_obj.run()

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
    y = eq_pos_3D[1] 
    z = eq_pos_3D[2] 
    y_q = eq_pos_3D_q[1]
    z_q = eq_pos_3D_q[2]
    ax.scatter(z, y)    
    ax.scatter(z_q, y_q, marker='x')    
    ax.set_xlabel("z ($\mu$m)") 
    ax.set_ylabel("y ($\mu$m)")

    plt.tight_layout() 
    plt.show()
