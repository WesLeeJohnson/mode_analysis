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
        self.wx = 1 
        self.wy = self.wy_E / self.wx_E
        self.wz = self.wz_E / self.wx_E
        k_e = 1 / (4 * np.pi * const.epsilon_0)  # Coulomb constant
        self.l0 = ((k_e * self.q_E ** 2) / (.5 * self.m_E * self.wx ** 2)) ** (1 / 3)
        self.t0 = 1 / self.wx  # characteristic time
        self.v0 = self.l0 / self.t0  # characteristic velocity
        self.E0 = 0.5*self.m*(self.wx**2)*self.l0**2 # characteristic energy


    def dimensionful_parameters(self):
        pass

    def is_trap_stable(self):
        print(self.wx, self.wy, self.wz)
        assert self.wx > 0e0 and self.wy > 0e0 and self.wz > 0e0, "Trap frequencies must be positive"
        assert self.wx > self.wy and self.wy and self.wz, "Trap frequencies must be ordered wx > wy > wz"

    def run(self):
        
        self.dimensionless_parameters() 

        self.is_trap_stable() 

        self.calculate_equilibrium_positions()

        self.calculate_normal_modes()

    def calculate_equilibrium_positions(self):

        if self.initial_equilibrium_guess is None:
            self.initial_equilibrium_guess = self.get_initial_equilibrium_guess()
        else:
            self.u0 = self.initial_equilibrium_guess

        self.u = self.find_equilibrium_positions(self.u0)

    def calculate_normal_modes(self):
        pass

    def get_initial_equilibrium_guess(self):
        # assumes linear chain along z-axis
        # assumes ions are equally spaced
        # vector of equilibrium positions organized as [x1, x2, x3, ..., xn, y1, y2, y3, ..., yn, z1, z2, z3, ..., zn]
        self.u0 = np.zeros(3*self.N)
        self.u0[2*self.N:] = np.linspace(-self.l0*(self.N-1)/2, self.l0*(self.N-1)/2, self.N, endpoint=True)
        print(self.l0 * self.wz**(-2/3));exit()
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

if __name__ == "__main__":
    htma_obj = HarmonicTrapModeAnalysis()
    htma_obj.run()
    print(htma_obj.u0)
    print(htma_obj.u)
    print(htma_obj.pot_energy(htma_obj.u))
    print(htma_obj.force(htma_obj.u))
