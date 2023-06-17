from __future__ import division, with_statement
from scipy.constants import pi
import scipy.constants as cons
import numpy as np
import scipy.optimize as optimize
import matplotlib.pyplot as plt
#import scipy.linalg as LA
__author__ = 'Wes Johnson'

# -*- coding: utf-8 -*-
"""
Contains the ModeAnalysis class, which can simulate the positions of ions in a crystal
of desired size. The class contains methods for the generation of a crystal,
relaxation to a minimum potential energy state, and determination of axial and (eventually) planar modes of motion
by methods derived by Wang, Keith, and Freericks in 2013.

Translated from MATLAB code written by Adam Keith by Justin Bohnet.
Standardized and slightly revised by Steven Torrisi.

Be careful. Sometimes when using the exact same parameters this
code will make different crystals with the same potential energy. That is,
crystal are degenerate when reflecting over the axes.

Wes Johnson, 2023 
The code has been modified to work for 3D crystals.
"""

class ModeAnalysis:
    """
    The code solves the eigenvalue problem for a 3D ion crystal in a Penning trap.
    The code first finds the equilibrium positions of the ions in the crystal.
    The the code uses these posistions to find the stiffness matrix for the crystal.
    The code then finds the eigenvalues and eigenvectors of the stiffness matrix.
    """
    #Establish fundamental physical constants as class variables
    q = cons.elementary_charge
    amu = cons.atomic_mass
    k_e = 1/(4*pi*cons.epsilon_0)
    def __init__(self, N=19, XR=3.082, 
                omega_z = 2*np.pi * 1.58e6, ionmass=9.012182, B=4.4588, frot=180., Vwall=1., 
                quiet=True, precision_solving=True,
                method = 'bfgs'):
        """
        This class solves the eigenvalue problem for a 3D ion crystal in a Penning trap.

        The code solves the eigenvalue problem for a 3D ion crystal in a Penning trap.
        The code first finds the equilibrium positions of the ions in the crystal.
        The the code uses these posistions to find the stiffness matrix for the crystal.
        The code then finds the eigenvalues and eigenvectors of the stiffness matrix.

        Parameters:
        -----------
         
        N : int
            Number of ions in the crystal
             
        XR : float
            Geometric factor for the rotating wall potential, Bryce Bullock @ NIST found it to be 3.082
             
        omega_z : float
            Axial frequency of the trap, in Hz
             
        ionmass : float
            Mass of the ions, in amu
             
        B : float
            Magnetic field strength, in Tesla
             
        frot : float
            Rotation frequency of the trap, in kHz
             
        Vwall : float
            Voltage on the rotating wall electrode, in volts
             
        quiet : bool
            If True, will not print anything
             
        precision_solving : bool
            If True, will perturb the crystal to find a lower energy state
             
        method : str
            Method to use for optimization. Either 'bfgs' or 'newton'
             
        Returns:
        --------

        None

        Examples:
        ---------
        >>> import mode_analysis as ma
        >>> ma_instance = ma.ModeAnalysis(N=19, XR=1, 
        ...                               omega_z = 1.58e6, ionmass=9.012182, B=4.4588, frot=180., Vwall=1.,
        ...                               quiet=True, precision_solving=True,
        ...                               method = 'bfgs')
        >>> ma_instance.run()
        >>> print(ma_instance.axialEvalsE)
        """

        self.method = method
        self.ionmass = ionmass
        self.m_Be = self.ionmass * self.amu
        self.quiet = quiet
        self.precision_solving = precision_solving
        self.Nion = N

        # for array of ion positions first half is x, last is y
        self.u0 = np.empty(2 * self.Nion)  # initial lattice
        self.u = np.empty(2 * self.Nion)  # equilibrium positions
        self.u0_3D = np.empty(3 * self.Nion)  # initial lattice
        self.u_3D = np.empty(3 * self.Nion)  # equilibrium positions

        # trap definitions
        self.B = B
        self.wcyc = self.q * B / self.m_Be  # Beryllium cyclotron frequency

        #  potentials at trap center
        self.omega_z = omega_z
        self.wz = self.omega_z
        self.wrot = 2 * pi * frot * 1e3  # Rotation frequency in units of angular fre   quency

        # if no input masses, assume all ions are beryllium
        self.m = self.m_Be * np.ones(self.Nion)

            # mass order is irrelevant and don't assume it will be fixed
            # FUTURE: heavier (than Be) ions will be added to outer shells

        # magnetron frequnecy in the lab frame, used to check if frot confining
        self.wmag= 1/2*(self.wcyc - np.sqrt(self.wcyc**2 - 2*self.wz**2))

        self.V0 = (0.5 * self.m_Be * self.wz ** 2) / self.q  # Find quadratic voltage at trap center
        self.XR=XR
        self.delta = self.XR*Vwall * 1612 / self.V0  # dimensionless coefficient for rotating wall strength 
        self.dimensionless()  # Make system dimensionless
        self.beta = (self.wr*self.wc - self.wr ** 2) -1/2 # dimensionless coefficient for planar confinement

        #TODO: initialize the size of the arrays based on the number of ions
        self.axialEvals = np.array([])  # Axial eigenvalues
        self.axialEvects = np.array([])  # Axial eigenvectors
        self.planarEvals = np.array([])  # Planar eigenvalues
        self.planarEvects = np.array([])  # Planar eigenvectors
        self.Evects_3D = np.array([])  # Eigenvectors
        self.Evals_3D = np.array([])  # Eigenvalues

        self.axialEvalsE = np.array([])  # Axial eigenvalues in experimental units
        self.axialEvals_raw = np.array([])  # Axial eigenvalues without ordering
        self.planarEvalsE = np.array([])  # Planar eigenvalues in experimental units
        self.Evects_3DE = np.array([])  # Eigenvectors in experimental units
        self.Evals_3DE = np.array([])  # Eigenvalues in experimental units

        self.p0 = 0    # dimensionless potential energy of equilibrium crystal
        self.p0_3D = 0    # dimensionless potential energy of equilibrium crystal
        
        self.hasrun = False

    def dimensionless(self):
        """
        Converts all variables to dimensionless units

        Parameters:
        -----------
        None

        Returns:
        --------
        None
        """
        # characteristic length
        self.l0 = ((self.k_e * self.q ** 2) / (.5 * self.m_Be * self.wz ** 2)) ** (1 / 3)
        self.t0 = 1 / self.wz  # characteristic time
        self.v0 = self.l0 / self.t0  # characteristic velocity
        self.E0 = 0.5*self.m_Be*(self.wz**2)*self.l0**2 # characteristic energy
        self.wr = self.wrot / self.wz  # dimensionless rotation
        self.wc = self.wcyc / self.wz  # dimensionless cyclotron
        self.md = np.ones(self.Nion)#self.m / self.m_Be  # dimensionless mass

    def expUnits(self):
        """
        Converts all variables to experimental units

        Parameters:
        -----------
        None

        Returns:
        --------
        None
        """
        self.u0E = self.l0 * self.u0  # Seed lattice
        self.uE = self.l0 * self.u  # Equilibrium positions
        self.u0E_3D = self.l0 * self.u0_3D  # Seed lattice
        self.uE_3D = self.l0 * self.u_3D  # Equilibrium positions
        self.axialEvalsE_raw = self.wz * self.axialEvals_raw
        self.axialEvalsE = self.wz * self.axialEvals
        self.planarEvalsE = self.wz * self.planarEvals
        self.Evals_3DE = self.wz * self.Evals_3D 
        # eigenvectors are dimensionless anyway

    def is_trap_stable(self):
        """
        Checks if the trap is stable. 

        Parameters:
        -----------
        None

        Returns:
        --------
        stable : bool
            True if the trap is stable, False if the trap is unstable.
        """ 
        stable = True 

        if self.wcyc < 2*self.wz:
            print('Warning: cyclotron frequency of %1.2f MHz is below two times the axial frequency of %1.2f MHz' % (self.wcyc/(2*pi*1e6), self.wz/(pi*1e6)))
            stable = False
        
        if self.beta < 0: 
            if self.wmag > self.wrot:
                print("Warning: rotation frequency of %1.2f kHz is below magnetron frequency of %1.2f kHz" % (self.wrot/(2*pi*1e3), self.wmag/(2*pi*1e3)))
                stable = False

            if self.wrot > self.wcyc - self.wmag:
                print("Warning: rotation frequency of %1.2f kHz is too high must be less than %1.2f kHz" % (self.wrot/(2*pi*1e3), (self.wcyc - self.wmag)/(2*pi*1e3)))
                stable = False

        else: 
            if self.beta < self.delta:
                print("Warning: rotating wall strength of %1.2f is too high must be less than %1.2f" % (self.delta, self.beta)) 
                stable = False
        
        return stable

    def run(self):
        """
        Generates a crystal from the generate_crystal method (by the find_scalled_lattice_guess method,
        adjusts it into an eqilibirium position by find_eq_pos method)
        and then computes the eigenvalues and eigenvectors of the axial modes by calc_axial_modes.

        Sorts the eigenvalues and eigenvectors and stores them in self.Evals, self.Evects.
        Stores the radial separations as well.

        Parameters:
        -----------
        None

        Returns:
        --------
        None
        """
        assert self.is_trap_stable(), "Trap is not stable"
         
        self.generate_crystal()

        self.axialEvals_raw, self.axialEvals, self.axialEvects = self.calc_axial_modes(self.u)
        self.planarEvals, self.planarEvects, self.V = self.calc_planar_modes(self.u)
        self.expUnits()  # make variables of outputs in experimental units
        self.hasrun = True

        if not self.is_plane_stable():
            print('Warning: 2D planar crystal is not stable. (Axial modes with calculated zero frequency)')

    def run_3D(self):
        """
        Generates a 3D crystal. Starts by creating a guess lattice in 3D. Then solves for the equilibrium position of the crystal.
        Next it will perturb the crystal to try to find a lower energy state. Finally it will calculate the
        final potential energy of the crystal. Then it will compute the eigenvalues and eigenvectors of the axial modes by calc_modes_3D.

        Parameters:
        -----------
        None

        Returns:
        --------
        None
        """
        assert self.is_trap_stable(), "Trap is not stable"

        self.generate_crystal_3D()

        self.Evals_3D, self.Evects_3D, self.V_3D = self.calc_modes_3D(self.u_3D)
        self.expUnits()  # make variables of outputs in experimental units
        self.hasrun_3D = True

    def generate_crystal(self):
        """
        Finds the equilibrium position of the crystal by first generating a guess lattice, then solving and perturbing

        Parameters:
        -----------
        None

        Returns:
        --------
        u : array
            The planar equilibrium position vector of the crystal. The first N elements are the x positions,
            the last N elements are the y positions.
        """

        #Generate a lattice in dimensionless units
        self.u0 = self.find_scaled_lattice_guess(mins=1, res=50)

        # if masses are not all beryllium, force heavier ions to be boundary
        # ions, and lighter ions to be near center
        # ADD self.addDefects()

        #Solve for the equilibrium position
        self.u = self.find_eq_pos(self.u0,self.method)



        # Will attempt to nudge the crystal to a slightly lower energy state via some
        # random perturbation.
        # Only changes the positions if the perturbed potential energy was reduced.

        #Will perturb less for bigger crystals, as it takes longer depending on how many ions
        #there are.
        if self.precision_solving is True:
            if self.quiet is False:
                print("Perturbing crystal...")

            if self.Nion <= 62:
                for attempt in np.linspace(.05, .5, 50):
                    self.u = self.perturb_position(self.u, attempt)
            if 62 < self.Nion <= 126:
                for attempt in np.linspace(.05, .5, 25):
                    self.u = self.perturb_position(self.u, attempt)
            if 127 <= self.Nion <= 200:
                for attempt in np.linspace(.05, .5, 10):
                    self.u = self.perturb_position(self.u, attempt)
            if 201 <= self.Nion:
                for attempt in np.linspace(.05, .3, 5):
                    self.u = self.perturb_position(self.u, attempt)

            if self.quiet is False:
                pass

        self.p0 = self.pot_energy(self.u)
        return self.u

    def generate_crystal_3D(self,attempts=np.linspace(.1, 1, 25)[::-1]):
        """
        Starts by creating a guess lattice in 3D. Then solves for the equilibrium position of the crystal.
        Next it will perturb the crystal to try to find a lower energy state. Finally it will calculate the
        final potential energy of the crystal.
        
        Parameters:
        -----------
        attempts : array
            The array of perturbation attempts. The gaussian is centered at 1 but has a variance of attempt. 
        
        Returns:
        --------
        u : array
            The planar equilibrium position vector of the crystal. The first N elements are the x positions,
        """

        #Generate a lattice in dimensionless units
        self.u0_3D = self.find_scaled_lattice_guess_3D()
        self.u_3D = self.find_eq_pos_3D(self.u0_3D,self.method)

        # Will attempt to nudge the crystal to a slightly lower energy state via some
        # random perturbation.    
        if self.precision_solving is True:
            if self.quiet is False:
                print("Perturbing crystal...")
        
            if self.Nion <= 62:
                for attempt in attempts:
                    self.u_3D = self.perturb_position_3D(self.u_3D, attempt)
            if 62 < self.Nion <= 126:
                for attempt in attempts:
                    self.u_3D = self.perturb_position_3D(self.u_3D, attempt)
            if 127 <= self.Nion <= 200:
                for attempt in attempts:
                    self.u_3D = self.perturb_position_3D(self.u_3D, attempt)
            if 201 <= self.Nion:
                for attempt in attempts:
                    self.u_3D = self.perturb_position_3D(self.u_3D, attempt)

            if self.quiet is False:
                pass

        self.p0_3D = self.pot_energy_3D(self.u_3D)
        return self.u_3D

    def generate_lattice(self):
        """
        Generate lattice for an arbitrary number of ions (self.Nion)
        The lattice is a hexagonal lattice with a number of closed shells
        calculated from the number of ions.

        Parameters:
        -----------
        None

        Returns:
        --------
        u : array
            The planar equilibrium guess position vector of the crystal. The first N elements are the x positions,
            the last N elements are the y positions.

        """
        # number of closed shells
        S = int((np.sqrt(9 - 12 * (1 - self.Nion)) - 3) / 6)
        u0 = self.generate_2D_hex_lattice(S)
        N0 = int(u0.size / 2)
        x0 = u0[0:N0]
        y0 = u0[N0:]
        Nadd = self.Nion - N0  # Number of ions left to add
        self.Nion = N0

        pair = self.add_hex_shell(S + 1)  # generate next complete shell
        xadd = pair[0::2]
        yadd = pair[1::2]

        for i in range(Nadd):
            # reset number of ions to do this calculation
            self.Nion += 1

            # make masses all one (add defects later)
            self.md = np.ones(self.Nion)

            V = []  # list to store potential energies from calculation

            # for each ion left to add, calculate potential energy if that
            # ion is added
            for j in range(len(xadd)):
                V.append(self.pot_energy(np.hstack((x0, xadd[j], y0,
                                                    yadd[j]))))
            ind = np.argmin(V)  # ion added with lowest increase in potential

            # permanently add to existing crystal
            x0 = np.append(x0, xadd[ind])
            y0 = np.append(y0, yadd[ind])

            # remove ion from list to add
            xadd = np.delete(xadd, ind)
            yadd = np.delete(yadd, ind)

        # Restore mass array
        self.md = self.m / self.m_Be  # dimensionless mass
        return np.hstack((x0, y0))

    def generate_lattice_3D(self):
        """
        Generate lattice for an arbitrary number of ions (self.Nion)
        The lattice is a hexagonal lattice with a number of closed shells
        calculated from the number of ions.

        Parameters:
        -----------
        None

        Returns:
        --------
        lattice : array
            The planar equilibrium guess position vector of the crystal. 
        """
        # number of closed shells
        num_shell  = 0
        num_in_shells = []
        num_in_shell = 0
        sum_num_in_shells = 0
        while sum_num_in_shells < self.Nion:
            num_shell +=1
            num_in_shell = num_shell **2
            num_in_shells.append(num_in_shell)
            sum_num_in_shells += num_in_shell
        num_in_shells[-1] = self.Nion - np.sum(num_in_shells[:-1])
        num_shells = len(num_in_shells)

        lattice = np.empty((0,3))
        for i in range(num_shells):
            R = (i + 1)  
            shell = self.generate_shell_3D(R, num_in_shells[i])
            lattice = np.concatenate((lattice,shell)) 
        return np.hstack((lattice[:,0],lattice[:,1],lattice[:,2]))

    def generate_shell_3D(self,R, N):
        """
        populates a shell of with equally spaced points on a sphere of radius R. 

        Parameters:
        -----------
        R : float
            The radius of the sphere.
        
        N : int
            The number of points to be placed on the sphere.
        
        Returns:    
        --------
        points : array
            The points on the sphere.
        """
        points = []
        increment = np.pi * (3 - np.sqrt(5))  # Golden angle increment

        if N == 1:
            # avoid placing leftover at same place 
            x,y,z = np.random.random()-1/2,np.random.random()-1/2,np.random.random()-1/2

            points.append([x,y,z])
            return np.array(points)

        for i in range(N):
            y = (1 - (i / float(N - 1)) * 2)
            radius = np.sqrt(1 - y * y)
            theta = i * increment

            x = np.cos(theta) * radius
            z = np.sin(theta) * radius

            x *= R
            y *= R
            z *= R

            points.append([x, y, z])

        return np.array(points)

    def pot_energy(self, pos_array):
        """
        Computes the potential energy of the ion crystal,
        taking into consideration:
            Coulomb repulsion
            qv x B forces
            Trapping potential
        
        Parameters:
        -----------
        pos_array : array
            The planar equilibrium position vector of the crystal. 
        
        Returns:
        --------
        V : float
            The potential energy of the crystal.
        """
        x = pos_array[0:self.Nion]
        y = pos_array[self.Nion:]

        dx = x.reshape((x.size, 1)) - x
        dy = y.reshape((y.size, 1)) - y
        rsep = np.sqrt(dx ** 2 + dy ** 2)

        with np.errstate(divide='ignore'):
            Vc = np.where(rsep != 0., 1 / rsep, 0)

        V = -np.sum((self.md * self.wr ** 2 + 0.5 * self.md - self.wr * self.wc) * (x ** 2 + y ** 2)) \
            + np.sum(self.md * self.delta * (x ** 2 - y ** 2)) + 0.5 * np.sum(Vc)

        return V

    def pot_energy_3D(self, pos_array):
        """
        Computes the potential energy of the ion crystal,
        taking into consideration:
            Coulomb repulsion
            qv x B forces
            Trapping potential
        
        Parameters:
        -----------
        pos_array : array
            The planar equilibrium position vector of the crystal. 
            The first N elements are the x positions,
            the next N elements are the y positions,
            the last N elements are the z positions.
        
        Returns:
        --------
        V : float
            The potential energy of the crystal.
        """
        x = pos_array[0:self.Nion]
        y = pos_array[self.Nion:2*self.Nion]
        z = pos_array[2*self.Nion:]

        dx = x.reshape((x.size, 1)) - x
        dy = y.reshape((y.size, 1)) - y
        dz = z.reshape((z.size, 1)) - z
        rsep = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)

        with np.errstate(divide='ignore'):
            Vc = np.where(rsep != 0., 1 / rsep, 0)

        V = np.sum(self.md * self.beta * (x ** 2 + y ** 2)) \
            + np.sum(self.md * self.delta * (x ** 2 - y ** 2)) \
                + 0.5 * np.sum(Vc) + np.sum(self.md * z ** 2)

        return V
         
    def force_penning(self, pos_array):
        """
        Computes the net forces acting on each ion in the crystal;
        used as the jacobian by find_eq_pos to minimize the potential energy
        of a crystal configuration.

        Parameters:
        -----------
        pos_array : array
            The planar equilibrium position vector of the crystal.
        
        Returns:
        --------
        Force : array
            The net force acting on each ion in the crystal.
        """

        x = pos_array[0:self.Nion]
        y = pos_array[self.Nion:]

        dx = x.reshape((x.size, 1)) - x
        dy = y.reshape((y.size, 1)) - y
        rsep = np.sqrt(dx ** 2 + dy ** 2)

        # Calculate coulomb force on each ion
        with np.errstate(divide='ignore'):
            Fc = np.where(rsep != 0., rsep ** (-2), 0)

        with np.errstate(divide='ignore', invalid='ignore'):
            fx = np.where(rsep != 0., np.float64((dx / rsep) * Fc), 0)
            fy = np.where(rsep != 0., np.float64((dy / rsep) * Fc), 0)

        Ftrapx = -2 * self.md * (self.wr ** 2 - self.wr * self.wc + 0.5 -
                                 self.delta) * x
        Ftrapy = -2 * self.md * (self.wr ** 2 - self.wr * self.wc + 0.5 +
                                 self.delta) * y

        Fx = -np.sum(fx, axis=1) + Ftrapx
        Fy = -np.sum(fy, axis=1) + Ftrapy

        Force = np.hstack((Fx, Fy))

        return Force

    def force_penning_3D(self, pos_array):
        """
        Computes the net forces acting on each ion in the crystal;
        used as the jacobian by find_eq_pos to minimize the potential energy
        of a crystal configuration.

        Parameters:
        -----------
        pos_array : array
            The planar equilibrium position vector of the crystal. The first N elements are the x positions,
            the next N elements are the y positions,
            the last N elements are the z positions.
        
        Returns:
        --------
        Force : array
            The net force acting on each ion in the crystal.
        """

        x = pos_array[0:self.Nion]
        y = pos_array[self.Nion:2*self.Nion]
        z = pos_array[2*self.Nion:]

        dx = x.reshape((x.size, 1)) - x
        dy = y.reshape((y.size, 1)) - y
        dz = z.reshape((z.size, 1)) - z
        rsep = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)

        # Calculate coulomb force on each ion
        with np.errstate(divide='ignore'):
            Fc = np.where(rsep != 0., rsep ** (-2), 0)

        with np.errstate(divide='ignore', invalid='ignore'):
            fx = np.where(rsep != 0., np.float64((dx / rsep) * Fc), 0)
            fy = np.where(rsep != 0., np.float64((dy / rsep) * Fc), 0)
            fz = np.where(rsep != 0., np.float64((dz / rsep) * Fc), 0)

        Ftrapx = -2 * self.md * ( - self.beta - self.delta) * x
        Ftrapy = -2 * self.md * ( - self.beta + self.delta) * y
        Ftrapz = 2 * self.md * z

        Fx = -np.sum(fx, axis=1) + Ftrapx
        Fy = -np.sum(fy, axis=1) + Ftrapy
        Fz = -np.sum(fz, axis=1) + Ftrapz

        Force = np.hstack((Fx, Fy, Fz))

        return Force

    def hessian_penning(self, pos_array):
        """
        Calculate Hessian of potential energy for a crystal defined by pos_array.

        Parameters:
        -----------
        pos_array : array
            The planar equilibrium position vector of the crystal.
        
        Returns:
        --------
        H : array
            The Hessian of the potential energy of the crystal.
        """

        x = pos_array[0:self.Nion]
        y = pos_array[self.Nion:]
        dx = x.reshape((x.size, 1)) - x
        dy = y.reshape((y.size, 1)) - y
        rsep = np.sqrt(dx ** 2 + dy ** 2)
        with np.errstate(divide='ignore'):
            rsep5 = np.where(rsep != 0., rsep ** (-5), 0)
        dxsq = dx ** 2
        dysq = dy ** 2

        # X derivatives, Y derivatives for alpha != beta
        Hxx = np.mat((rsep ** 2 - 3 * dxsq) * rsep5)
        Hyy = np.mat((rsep ** 2 - 3 * dysq) * rsep5)

        # Above, for alpha == beta
        # np.diag usa diagnoal value to form a matrix
        Hxx += np.mat(np.diag(-2 * self.md * (self.wr ** 2 - self.wr * self.wc + .5 -
                                              self.delta) -
                              np.sum((rsep ** 2 - 3 * dxsq) * rsep5, axis=0)))
        Hyy += np.mat(np.diag(-2 * self.md * (self.wr ** 2 - self.wr * self.wc + .5 +
                                              self.delta) -
                              np.sum((rsep ** 2 - 3 * dysq) * rsep5, axis=0)))

        # Mixed derivatives
        Hxy = np.mat(-3 * dx * dy * rsep5)
        Hxy += np.mat(np.diag(3 * np.sum(dx * dy * rsep5, axis=0)))

        H = np.bmat([[Hxx, Hxy], [Hxy, Hyy]])
        H = np.asarray(H)
        return H

    def hessian_penning_3D(self, pos_array):
        """
        Calculate Hessian of potential energy for a crystal defined by pos_array.

        Parameters:
        -----------
        pos_array : array
            The planar equilibrium position vector of the crystal.
        
        Returns:
        --------
        H : array
            The Hessian of the potential energy of the crystal.
        """
        x = pos_array[0:self.Nion]
        y = pos_array[self.Nion:2*self.Nion]
        z = pos_array[2*self.Nion:]
         
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
        Hxx += np.mat(np.diag(-2 * self.md * (-self.beta - self.delta) -
                              np.sum((rsep ** 2 - 3 * dxsq) * rsep5, axis=0)))
        Hyy += np.mat(np.diag(-2 * self.md * (-self.beta + self.delta) -
                              np.sum((rsep ** 2 - 3 * dysq) * rsep5, axis=0)))
        Hzz += np.mat(np.diag(2 * self.md  -
                              np.sum((rsep ** 2 - 3 * dzsq) * rsep5, axis=0)))

        Hxy = np.mat(-3 * dx * dy * rsep5)
        Hxy += np.mat(np.diag(3 * np.sum(dx * dy * rsep5, axis=0)))
        Hxz = np.mat(-3 * dx * dz * rsep5)
        Hxz += np.mat(np.diag(3 * np.sum(dx * dz * rsep5, axis=0)))
        Hyz = np.mat(-3 * dy * dz * rsep5)
        Hyz += np.mat(np.diag(3 * np.sum(dy * dz * rsep5, axis=0)))

        H = np.bmat([[Hxx, Hxy, Hxz], [Hxy, Hyy, Hyz], [Hxz, Hyz, Hzz]])
        H = np.asarray(H)
        return H


    def find_scaled_lattice_guess(self, mins, res):
        """
        Will generate a 2d hexagonal lattice based on the shells intialiization parameter.
        Guesses initial minimum separation of mins and then increases spacing until a local minimum of
        potential energy is found.

        This doesn't seem to do anything. Needs a fixin' - AK

        :param mins: the minimum separation to begin with.
        :param res: the resizing parameter added onto the minimum spacing.
        :return: the lattice with roughly minimized potential energy (via spacing alone).
        """

        # Make a 2d lattice; u represents the position
        uthen = self.generate_lattice()
        uthen = uthen * mins
        # Figure out the lattice's initial potential energy
        pthen = self.pot_energy(uthen)

        # Iterate through the range of minimum spacing in steps of res/resolution
        for scale in np.linspace(mins, 10, res):
            # Quickly make a 2d hex lattice; perhaps with some stochastic procedure?
            uguess = uthen * scale
            # Figure out the potential energy of that newly generated lattice
            pnow = self.pot_energy(uguess)

            # And if the program got a lattice that was less favorably distributed, conclude
            # that we had a pretty good guess and return the lattice.
            if pnow >= pthen:
                return uthen
            # If not, then we got a better guess, so store the energy score and current arrangement
            # and try again for as long as we have mins and resolution to iterate through.
            uthen = uguess
            pthen = pnow
        # If you're this far it means we've given up
        # self.scale = scale
        return uthen

    def find_scaled_lattice_guess_3D(self): 
        """
        TODO: this function could scale the lattice to find the minimum potential energy. i

        Parameters:
        -----------
        None

        Returns:
        --------
        u : array
            The equilibrium position vector guess for the crystal. The first N elements are the x positions,
            the next N are the y positions, and the last N are the z positions.
        """
        return self.generate_lattice_3D()

    def find_eq_pos(self, u0, method="bfgs"):
        """
        Runs optimization code to tweak the position vector defining the crystal to a minimum potential energy
        configuration.

        Parameters:
        -----------
        u0 : array
            The planar equilibrium guess position vector of the crystal. The first N elements are the x positions,
            the last N elements are the y positions.
        method : str
            Method to use for optimization. Either 'bfgs' or 'newton'
        
        Returns:
        --------
        u : array
            The planar equilibrium position vector of the crystal. The first N elements are the x positions,
            the last N elements are the y positions.
        """
        newton_tolerance = 1e-34
        bfgs_tolerance = 1e-34
        if method == "newton":

            out = optimize.minimize(self.pot_energy, u0, method='Newton-CG', jac=self.force_penning,
                                    hess=self.hessian_penning,
                                    options={'xtol': newton_tolerance, 'disp': not self.quiet})
        if method == 'bfgs':
            out = optimize.minimize(self.pot_energy, u0, method='BFGS', jac=self.force_penning,
                                    options={'gtol': bfgs_tolerance, 'disp': False})  # not self.quiet})
        if (method != 'bfgs') & (method != 'newton'):
            print('method, '+method+', not recognized')
            exit()
        return out.x

    def find_eq_pos_3D(self, u0, method="bfgs"): 
        """
        Runs optimization code to find the equillibrium position of the crystal.

        Parameters:
        -----------
        u0 : array
            The planar equilibrium guess position vector of the crystal. The first N elements are the x positions,
            the next N are the y positions, and the last N are the z positions.
        method : str
            Method to use for optimization. Either 'bfgs' or 'newton'

        Returns:
        --------
        u : array
            The planar equilibrium position vector of the crystal. The first N elements are the x positions,
            the next N are the y positions, and the last N are the z positions.
        """
        newton_tolerance = 1e-34
        bfgs_tolerance = 1e-34
        if method == "newton":

            out = optimize.minimize(self.pot_energy_3D, u0, method='Newton-CG', jac=self.force_penning_3D,
                                    hess=self.hessian_penning_3D,
                                    options={'xtol': newton_tolerance, 'disp': not self.quiet})
        if method == 'bfgs':
            out = optimize.minimize(self.pot_energy_3D, u0, method='BFGS', jac=self.force_penning_3D,
                                    options={'gtol': bfgs_tolerance, 'disp': False})  # not self.quiet})
        if (method != 'bfgs') & (method != 'newton'):
            print('method, '+method+', not recognized')
        return out.x

    def calc_axial_hessian(self, pos_array):
        """
        Calculate the axial hessian matrix for a crystal defined
        by pos_array.

        THIS MAY NEED TO BE EDITED FOR NONHOMOGENOUS MASSES

        Parameters:
        -----------
        pos_array : array
            The planar equilibrium position vector of the crystal.
        
        Returns:
        --------
        K : array
            The axial hessian matrix of the crystal.
        """

        x = pos_array[0:self.Nion]
        y = pos_array[self.Nion:]

        dx = x.reshape((x.size, 1)) - x
        dy = y.reshape((y.size, 1)) - y
        rsep = np.sqrt(dx ** 2 + dy ** 2)

        with np.errstate(divide='ignore'):
            rsep3 = np.where(rsep != 0., rsep ** (-3), 0)

        K = np.diag((-1 + 0.5 * np.sum(rsep3, axis=0)))
        K -= 0.5 * rsep3
        return K 

    def calc_axial_modes(self, pos_array):
        """
        Calculate the modes of axial vibration for a crystal defined
        by pos_array.

        THIS MAY NEED TO BE EDITED FOR NONHOMOGENOUS MASSES

        Parameters:
        -----------
        pos_array : array
            The planar equilibrium position vector of the crystal.

        Returns:
        --------
        Eval_raw : array
            The raw eigenvalues of the crystal. Contains imaginary values and is unsorted, has 2N eigenvalues.
        Eval : array
            The sorted eigenvalues of the crystal. Contains only real values and has N eigenvalues.
        Evect : array
            The eigenvectors of the crystal. 
        """

        K = self.calc_axial_hessian(pos_array)
        # Make first order system by making space twice as large
        Zn = np.zeros((self.Nion, self.Nion))
        eyeN = np.identity(self.Nion)
        Mmat = np.diag(self.md)
        Minv = np.linalg.inv(Mmat)
        firstOrder = np.bmat([[Zn, eyeN], [np.dot(Minv,K), Zn]])
        Eval, Evect = np.linalg.eig(firstOrder)
        Eval_raw = Eval
        # make eigenvalues real.
        Eval = np.imag(Eval)
        # sort eigenvalues
        ind = np.argsort(Eval)
        Eval = Eval[ind]
        # toss the negative eigenvalues
        Eval = Eval[self.Nion:]
        # sort eigenvectors accordingly
        Evect = Evect[:, ind] 
        # Normalize by energy of mode
        for i in range(2*self.Nion):
            pos_part = Evect[:self.Nion, i]
            vel_part = Evect[self.Nion:, i]
            norm = vel_part.H*Mmat*vel_part - pos_part.H*K*pos_part

            with np.errstate(divide='ignore',invalid='ignore'):
                Evect[:, i] = np.where(np.sqrt(norm) != 0., Evect[:, i]/np.sqrt(norm), 0)

        Evect = np.asarray(Evect)
        return Eval_raw, Eval, Evect

    def calc_planar_modes(self, pos_array):
        """
        Calculate Planar Mode Eigenvalues and Eigenvectors

        THIS MAY NEED TO BE EDITED FOR NONHOMOGENOUS MASSES

        Parameters:
        -----------
        pos_array : array
            The planar equilibrium position vector of the crystal.

        Returns:
        --------
        Eval : array
            The sorted eigenvalues of the crystal. Contains only real values and has 2N eigenvalues.
        Evect : array
            The eigenvectors of the crystal.
        V : array
            The hessian matrix of the potential.
        """

        V = -self.hessian_penning(pos_array)  # -Hessian
        Zn = np.zeros((self.Nion, self.Nion)) #Nion, number of ions
        Z2n = np.zeros((2 * self.Nion, 2 * self.Nion))
        offdiag = (2 * self.wr - self.wc) * np.identity(self.Nion) # np.identity: unitary matrix
        A = np.bmat([[Zn, offdiag], [-offdiag, Zn]])
        Mmat = np.diag(np.concatenate((self.md,self.md))) #md =1
        Minv = np.linalg.inv(Mmat)
        firstOrder = np.bmat([[Z2n, np.identity(2 * self.Nion)], [np.dot(Minv,V/2), A]])

        Eval, Evect = np.linalg.eig(firstOrder) 

        # make eigenvalues real.
        Eval = np.imag(Eval)
        # sort eigenvalues
        ind = np.argsort(Eval)
        Eval = Eval[ind]
        # toss the negative eigenvalues
        Eval = Eval[2*self.Nion:]
        Evect = Evect[:, ind]    # sort eigenvectors accordingly

        # Normalize by energy of mode
        for i in range(4*self.Nion):
            pos_part = Evect[:2*self.Nion, i]
            vel_part = Evect[2*self.Nion:, i]
            norm = vel_part.H*Mmat*vel_part - pos_part.H*(V/2)*pos_part

            with np.errstate(divide='ignore'):
                Evect[:, i] = np.where(np.sqrt(norm) != 0., Evect[:, i]/np.sqrt(norm), 0)

        # if there are extra zeros, chop them
        Eval = Eval[(Eval.size - 2 * self.Nion):]
        return Eval, Evect, V

    def calc_modes_3D(self, pos_array):
        """Calculate Planar Mode Eigenvalues and Eigenvectors

        THIS MAY NEED TO BE EDITED FOR NONHOMOGENOUS MASSES

        :param pos_array: Position vector which defines the crystal
                          to be analyzed.
        :return: Array of eigenvalues, Array of eigenvectors
        """

        V = -self.hessian_penning_3D(pos_array)  # -Hessian
        Zn = np.zeros((self.Nion, self.Nion)) #Nion, number of ions
        Z3n = np.zeros((3 * self.Nion, 3 * self.Nion))
        offdiag = (2 * self.wr - self.wc) * np.identity(self.Nion) # np.identity: unitary matrix
        A = np.bmat([[Zn, offdiag, Zn], [-offdiag, Zn, Zn], [Zn, Zn, Zn]])
        Mmat3 = np.diag(np.tile(self.md,3))
        Minv3 = np.linalg.inv(Mmat3)
        firstOrder = np.bmat([[Z3n, np.identity(3 * self.Nion)], [np.dot(Minv3,V/2), A]])

        Eval, Evect = np.linalg.eig(firstOrder) 

        # make eigenvalues real.
        Eval = np.imag(Eval)
        ind = np.argsort(Eval)
        Eval = Eval[ind]
        Eval = Eval[3*self.Nion:]      # toss the negative eigenvalues
        Evect = Evect[:, ind]    # sort eigenvectors accordingly

        # Normalize by energy of mode
        for i in range(6*self.Nion):
            pos_part = Evect[:3*self.Nion, i]
            vel_part = Evect[3*self.Nion:, i]
            norm = vel_part.H*Mmat3*vel_part - pos_part.H*(V/2)*pos_part

            with np.errstate(divide='ignore'):
                Evect[:, i] = np.where(np.sqrt(norm) != 0., Evect[:, i]/np.sqrt(norm), 0)

        Evect = np.asarray(Evect)
        return Eval, Evect, V

    def show_crystal(self, pos_vect=None,ax = None,label='Ion Positions',color='blue'):
        """
        Makes a pretty plot of the crystal with a given position vector.

        Parameters:
        -----------
        pos_vect : array
            The planar equilibrium position vector of the crystal. The first N elements are the x positions,
            the last N elements are the y positions. The units are in meters.
            If None, the equilibrium position vector of the crystal class is used.
        ax : matplotlib.axes object
            The axes to plot the crystal on. If None, a new figure is created.
        label : str
            The label to use for the plot legend.

        Returns:
        --------
        ax : matplotlib.axes object
        """
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)
        if pos_vect is None:
            pos_vect = self.uE
        x = pos_vect[:self.Nion]
        y = pos_vect[self.Nion:]
        x = x*1e6
        y = y*1e6
        ax.scatter(x,y,color=color,label=label)
        ax.set_xlabel('x ($\mu$m)')
        ax.set_ylabel('y ($\mu$m)')
        ax.set_aspect('equal')
        ax.legend()
        ax.set_title('Ion Positions')
        return ax

    def show_crystal_3D(self, pos_vect=None,ax = None,label='Ion Positions',color='blue'):
        """
        Makes a 3D plot of the ion crystal given the position vector.

        Parameters:
        -----------
        pos_vect : array
            The planar equilibrium position vector of the crystal. The first N elements are the x positions,
            the next N are the y positions, and the last N are the z positions. The units are in meters.
            If None, the equilibrium position vector of the crystal class is used.
        ax : matplotlib.axes object
            The axes to plot the crystal on. If None, a new figure is created.
        label : str
            The label to use for the plot legend.

        Returns:
        --------
        ax : matplotlib.axes object
        """
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111,projection='3d')
        if pos_vect is None:
            pos_vect = self.u0_3D
        x = pos_vect[:self.Nion]
        y = pos_vect[self.Nion:2*self.Nion]
        z = pos_vect[2*self.Nion:]
        ax.scatter(x,y,z,color=color,label=label)
        limits = np.max(np.abs(pos_vect))
        ax.set_xlim(-limits,limits)
        ax.set_ylim(-limits,limits)
        ax.set_zlim(-limits,limits)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()
        ax.set_box_aspect([1,1,1]) # make axes equal 
        return ax


    def show_crystal_modes(self, pos_vect=None, Evects=None, mode = 0, ax=None,label=None):
        """
        Plots the axial modes of the crystal, using a color map to show displacement.

        Parameters:
        -----------
        pos_vect : array
            The planar equilibrium position vector of the crystal. The first N elements are the x positions,
            the last N elements are the y positions. The units are in meters.
            If None, the equilibrium position vector of the crystal class is used.
        Evects : array
            The eigenvectors of the crystal. If None, the eigenvectors of the crystal class are used.
        mode : int
            The mode to plot.
        ax : matplotlib.axes object
            The axes to plot the crystal on. If None, a new figure is created.

        Returns:
        --------
        ax : matplotlib.axes object 
        """
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)
        if pos_vect is None:
            pos_vect = self.uE
        if Evects is None:
            Evect = self.axialEvects[:, 2*mode]
        x = pos_vect[:self.Nion]
        y = pos_vect[self.Nion:]
        x = x*1e6
        y = y*1e6
        z = np.real(Evect)[:self.Nion]
        clim = np.max(np.abs(Evect))
        cmap = plt.get_cmap('seismic')
        ax.scatter(x,y,c=z,cmap=cmap,vmin=-clim,vmax=clim)
        ax.set_xlabel('x ($\mu$m)')
        ax.set_ylabel('y ($\mu$m)')
        ax.set_aspect('equal')
        ax.set_title('Axial Mode %d' % mode)
        return ax 

    def perturb_position(self, pos_vect, strength=.1):
        """
        Slightly displaces each ion by a random proportion (determined by 'strength' parameter)
        and then solves for a new equilibrium position.

        If the new configuration has a lower global potential energy, it is returned.
        If the new configuration has a higher potential energy, it is discarded and
            the previous configuration is returned.
        
        Parameters:
        -----------
        pos_vect : array
            The planar equilibrium position vector of the crystal. The first N elements are the x positions,
            the last N elements are the y positions. The units are in meters.
        strength : float
            The maximum proportion of the ion separation to displace each ion by.

        Returns:
        --------
        pos_vect : array
        """
        unudge = self.find_eq_pos([coord * abs(np.random.normal(1, strength)) for coord in pos_vect])
        if self.pot_energy(unudge) < self.pot_energy(pos_vect):
            return unudge
        else:
            return pos_vect

    def perturb_position_3D(self, pos_vect, strength=.1):
        """
        This function is the same as perturb_position, but for 3D crystals.

        Parameters:
        -----------
        pos_vect : array
            The planar equilibrium position vector of the crystal. The first N elements are the x positions,
            the next N are the y positions, and the last N are the z positions. 
        strength : float
            The variance of a normal distribution to sample the proportion to displace each ion by.
        
        Returns:
        --------
        pos_vect : array
        """
        unudge = self.find_eq_pos_3D([coord * abs(np.random.normal(1, strength)) for coord in pos_vect])
        if self.pot_energy_3D(unudge) < self.pot_energy_3D(pos_vect):
            return unudge
        else:
            return pos_vect

    def show_axial_freqs(self,ax = None): 
        """
        Plots the mode frequencies of the crystal.

        Parameters:
        -----------
        ax : matplotlib.axes object
            The axes to plot the crystal on. If None, a new figure is created.        

        Returns:
        --------
        ax : matplotlib.axes object 
        """
        if ax is None: 
            fig = plt.figure()
            ax = fig.add_subplot(111)
        
        mode_freqs = self.axialEvalsE
        ax.scatter(np.arange(len(mode_freqs)),mode_freqs/(2*np.pi*1e6))
        xticks = np.arange(len(mode_freqs))
        if len(xticks) > 20: 
            skip = int(len(xticks)/10)
            xticks = xticks[::skip]
        ax.set_xticks(xticks)
        ax.set_xlabel('Mode Number')
        ax.set_ylabel('Frequency (MHz)')
        ax.set_title('Axial Mode Frequencies')
        return ax

    def show_ExB_freqs(self,ax = None):
        """
        Plots the ExB mode frequencies of the crystal.

        Parameters:
        -----------
        ax : matplotlib.axes object
            The axes to plot the crystal on. If None, a new figure is created.
        
        Returns:
        --------
        ax : matplotlib.axes object 
        """
        if ax is None: 
            fig = plt.figure()
            ax = fig.add_subplot(111)
        mode_freqs = self.planarEvalsE[:self.Nion]
        ax.scatter(np.arange(len(mode_freqs)),mode_freqs/(2*np.pi*1e3))
        xticks = np.arange(len(mode_freqs))
        if len(xticks) > 20:
            skip = int(len(xticks)/10)
            xticks = xticks[::skip]
        ax.set_xticks(xticks)
        ax.set_xlabel('Mode Number')
        ax.set_ylabel('Frequency (kHz)')
        ax.set_title('ExB Mode Frequencies')
        return ax
    
    def show_cyc_freqs(self,ax = None):
        """
        Plots the cyclotron mode frequencies of the crystal.

        Parameters:
        -----------
        ax : matplotlib.axes object
            The axes to plot the crystal on. If None, a new figure is created.

        Returns:
        --------
        ax : matplotlib.axes object 
        """
        if ax is None: 
            fig = plt.figure()
            ax = fig.add_subplot(111)
        mode_freqs = self.planarEvalsE[self.Nion:]
        ax.scatter(np.arange(len(mode_freqs)),mode_freqs/(2*np.pi*1e6))
        xticks = np.arange(len(mode_freqs))
        if len(xticks) > 20:
            skip = int(len(xticks)/10)
            xticks = xticks[::skip]
        ax.set_xticks(xticks)
        ax.set_xlabel('Mode Number')
        ax.set_ylabel('Frequency (MHz)')
        ax.set_title('Cyclotron Mode Frequencies')
        return ax

    def get_x_and_y(self, pos_vect):
        """
        Hand it a position vector and it will return the x and y vectors
        :param pos_vect:
        :return: [x,y] arrays
        """
        return [pos_vect[:self.Nion], pos_vect[self.Nion:]]

    def is_plane_stable(self):
        """
        Checks if the plane is stable against single to double plane transitions.
        These occur when any of the axial modes have a frequency of zero.
        Parameters:
        -----------
        None

        Returns:
        --------
        True if the plane is stable, False if it is not. (zero frequency modes)
        """
        if self.hasrun is False: self.run()

        axial_modes = self.axialEvals
        for mode_freq in axial_modes:
            if np.abs(mode_freq) < 1e-14:
                return False

        return True

    def rotate_crystal(self, pos_vect, theta, Nion=None):
        """
        Given a position vector defining a crystal, rotates it by the angle theta
        counter-clockwise.

        :param pos_vect: Array of length 2*Nion defining the crystal to be rotated.
        :param theta: Theta defining the angle to rotate the crystal around.
        :param Nion: Number of ions in the crystal (can be optionally defined,
            but will default to the number of ions in the class)
        :return: The returned position vector of the new crystal
        """
        if Nion is None:
            Nion = self.Nion

        x = pos_vect[:Nion]
        y = pos_vect[Nion:]

        xmod = x * np.cos(theta) - y * np.sin(theta)
        ymod = x * np.sin(theta) + y * np.cos(theta)
        newcrys = np.concatenate((xmod, ymod))
        return newcrys

    @staticmethod
    def nan_to_zero(my_array):
        """
        Converts all elements of an array which are np.inf or nan to 0.

        :param my_array: array to be  filtered of infs and nans.
        :return: the array.
        """
        my_array[np.isinf(my_array) | np.isnan(my_array)] = 0
        return my_array

    @staticmethod
    def save_positions(u):
        """
        Takes a position vector and saves it as a text file.
        :param u: position vector to store.
        :return: nothing
        """
        np.savetxt("py_u.csv", u, delimiter=",")

    @staticmethod
    def crystal_spacing_fit(r, offset, curvature):
        """
        """
        return np.sqrt(2 / (np.sqrt(3) * offset * np.sqrt(1 - (r * curvature) ** 2)))

    @staticmethod
    def find_radial_separation(pos_array):
        """
        When given the position array of a crystal,
        returns 4 arrays:
        N radii, N^2 x separations, N^2 y separations, and N^2 radial separations.

        :param pos_array: position array of a crystal.

        :return: radius, x separations, y separations, radial separations
        """
        N = int(pos_array.size / 2)
        x = pos_array[0:N]
        y = pos_array[N:]
        r = np.sqrt(x ** 2 + y ** 2)

        sort_ind = np.argsort(r)

        r = r[sort_ind]
        x = x[sort_ind]
        y = y[sort_ind]

        dx = x.reshape((x.size, 1)) - x
        dy = y.reshape((y.size, 1)) - y
        rsep = np.sqrt(dx ** 2 + dy ** 2)

        return r, dx, dy, rsep

    @staticmethod
    def generate_2D_hex_lattice(shells=1, scale=1):
        """Generate closed shell hexagonal lattice with shells and scale spacing.

        :param scale: scales lattice
        :return: a flattened xy position vector defining the 2d hexagonal lattice.
        """
        posvect = np.array([0.0, 0.0])  # center ion at [0,0]

        for s in range(1, shells + 1):
            posvect = np.append(posvect, ModeAnalysis.add_hex_shell(s))
        posvect *= scale
        return np.hstack((posvect[0::2], posvect[1::2]))

    @staticmethod
    # A slave function used to append shells onto a position vector
    def add_hex_shell(s):
        """
        A method used by generate_2d_hex_lattice to add the s-th hex shell to the 2d lattice.
        Generates the sth shell.
        :param s: the sth shell to be added to the lattice.

        :return: the position vector defining the ions in sth shell.
        """
        a = list(range(s, -s - 1, -1))
        a.extend(-s * np.ones(s - 1))
        a.extend(list(range(-s, s + 1)))
        a.extend(s * np.ones(s - 1))

        b = list(range(0, s + 1))
        b.extend(s * np.ones(s - 1))
        b.extend(list(range(s, -s - 1, -1)))
        b.extend(-s * np.ones(s - 1))
        b.extend(list(range(-s, 0)))

        x = np.sqrt(3) / 2.0 * np.array(b)
        y = 0.5 * np.array(b) + np.array(a)
        pair = np.column_stack((x, y)).flatten()
        return pair

########################################################################################

if __name__ == "__main__":
    # For reference the following ion number correspond the closed shells:
    # 1  2  3  4  5   6   7   8   9  10  11  12  13  14
    # 7 19 37 61 91 127 169 217 271 331 397 469 547 631...

    a = ModeAnalysis(N=37, Vwall=1, frot=180)
    a.run()
