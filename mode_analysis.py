from __future__ import division, with_statement
from scipy.constants import pi
import scipy.constants as cons
import numpy as np
import scipy.optimize as optimize
import matplotlib.pyplot as plt
import scipy.linalg as LA
__author__ = 'sbt'

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
"""

class ModeAnalysis:
    """
    Simulates a 2-dimensional ion crystal, determining an equilibrium plane configuration given
    Penning trap parameters, and then calculates the eigenvectors and eigenmodes.

    For reference the following ion number correspond the closed shells:
    1  2  3  4  5   6   7   8   9  10  11  12  13  14
    7 19 37 61 91 127 169 217 271 331 397 469 547 631...


    """
    #Establish fundamental physical constants as class variables
    q = 1.602176565E-19
    amu = 1.66057e-27
    k_e = 8.9875517873681764E9 # electrostatic constant k_e = 1 / (4.0 pi epsilon_0)

    def __init__(self, N=19, XR=3.082, 
                omega_z = 2*np.pi * 1.58e6, ionmass=9.012182, B=4.4588, frot=180., Vwall=1., 
                quiet=True, precision_solving=True,
                method = 'bfgs'):
        """
        This class solves the eigenvalue problem for a 2D ion crystal in a Penning trap.

        The class find the normal vectors and frequencies along with the equilibrium positions. 
        The class assumes that the ion crystal is planar, which may not be true for all trap parameters.
        The class separates the axial and planar modes of motion.

        Parameters:
        -----------
        N : int
            Number of ions in the crystal
        XR : float
            Geometric factor for the rotating wall potential, Bryce Bullock found it to be 3.082
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
        # Initialize basic variables such as physical constants
        self.Nion = N

        # for array of ion positions first half is x, last is y
        self.u0 = np.empty(2 * self.Nion)  # initial lattice
        self.u = np.empty(2 * self.Nion)  # equilibrium positions

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

        self.axialEvals = []  # Axial eigenvalues
        self.axialEvects = []  # Axial eigenvectors
        self.planarEvals = []  # Planar eigenvalues
        self.planarEvects = []  # Planar Eigenvectors

        self.axialEvalsE = []  # Axial eigenvalues in experimental units
        self.planarEvalsE = []  # Planar eigenvalues in experimental units

        self.p0 = 0    # dimensionless potential energy of equilibrium crystal
        self.r = []
        self.rsep = []
        self.dx = []
        self.dy = []

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
        self.axialEvalsE_raw = self.wz * self.axialEvals_raw
        self.axialEvalsE = self.wz * self.axialEvals
        self.planarEvalsE = self.wz * self.planarEvals
        # eigenvectors are dimensionless anyway

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
        if self.wmag > self.wrot:
            print("Warning: rotation frequency of %1.2f kHz is below magnetron frequency of %1.2f kHz" % (self.wrot/(2*pi*1e3), self.wmag/(2*pi*1e3)))
            print('This will not provide confinement ')
            return 0

        self.generate_crystal()

        self.axialEvals_raw, self.axialEvals, self.axialEvects = self.calc_axial_modes(self.u)
        self.planarEvals, self.planarEvects, self.V = self.calc_planar_modes(self.u)
        self.expUnits()  # make variables of outputs in experimental units
        self.axial_hessian = -self.calc_axial_hessian(self.u)
        self.planar_hessian= -self.V/2 
        self.axial_Mmat    = np.diag(self.md)
        self.planar_Mmat   = np.diag(np.tile(self.md,2))
        self.hasrun = True

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

        # This check hasn't been working properly, and so wmag has been set to
        # 0 for the time being (July 2015, SBT)
        if self.wmag > self.wrot:
            print("Warning: Rotation frequency", self.wrot/(2*pi),
                  " is below magnetron frequency of", float(self.wrot/(2*pi)))
            return 0

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

        self.r, self.dx, self.dy, self.rsep = self.find_radial_separation(self.u)
        self.p0 = self.pot_energy(self.u)
        return self.u

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
            # print(uguess)
            pnow = self.pot_energy(uguess)

            # And if the program got a lattice that was less favorably distributed, conclude
            # that we had a pretty good guess and return the lattice.
            if pnow >= pthen:
                # print("find_scaled_lattice: Minimum found")
                # print "initial scale guess: " + str(scale)
                # self.scale = scale
                # print(scale)
                return uthen
            # If not, then we got a better guess, so store the energy score and current arrangement
            # and try again for as long as we have mins and resolution to iterate through.
            uthen = uguess
            pthen = pnow
        # If you're this far it means we've given up
        # self.scale = scale
        # print "find_scaled_lattice: no minimum found, returning last guess"
        return uthen

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

        x = pos_array[0:self.Nion]
        y = pos_array[self.Nion:]

        dx = x.reshape((x.size, 1)) - x
        dy = y.reshape((y.size, 1)) - y
        rsep = np.sqrt(dx ** 2 + dy ** 2)

        with np.errstate(divide='ignore'):
            rsep3 = np.where(rsep != 0., rsep ** (-3), 0)

        K = np.diag((-1 + 0.5 * np.sum(rsep3, axis=0)))
        K -= 0.5 * rsep3
        # Make first order system by making space twice as large
        Zn = np.zeros((self.Nion, self.Nion))
        eyeN = np.identity(self.Nion)
        Mmat = np.diag(self.md)
        Minv = np.linalg.inv(Mmat)
        firstOrder = np.bmat([[Zn, eyeN], [np.dot(Minv,K), Zn]])
        Eval, Evect = np.linalg.eig(firstOrder)
        Eval_raw = Eval
        # Convert 2N imaginary eigenvalues to N real eigenfrequencies
        ind = np.argsort(np.absolute(np.imag(Eval)))
        Eval = np.imag(Eval[ind])
        Eval = Eval[Eval >= 0]      # toss the negative eigenvalues
        Evect = Evect[:, ind]     # sort eigenvectors accordingly
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
        ind = np.argsort(np.absolute(np.imag(Eval)))
        Eval = np.imag(Eval[ind])
        Eval = Eval[Eval >= 0]      # toss the negative eigenvalues
        Evect = Evect[:, ind]    # sort eigenvectors accordingly

        # Normalize by energy of mode
        for i in range(4*self.Nion):
            pos_part = Evect[:2*self.Nion, i]
            vel_part = Evect[2*self.Nion:, i]
            norm = vel_part.H*Mmat*vel_part - pos_part.H*(V/2)*pos_part

            with np.errstate(divide='ignore'):
                Evect[:, i] = np.where(np.sqrt(norm) != 0., Evect[:, i]/np.sqrt(norm), 0)
            #Evect[:, i] = Evect[:, i]/np.sqrt(norm)

        # if there are extra zeros, chop them
        Eval = Eval[(Eval.size - 2 * self.Nion):]
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
        
    def show_crystal_axial_mode(self, pos_vect=None, Evects=None, mode = 0, ax=None,label=None):
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
        print(np.shape(self.axialEvects));print(self.axialEvects)
        cmap = plt.get_cmap('seismic')
        ax.scatter(x,y,c=z,cmap=cmap,vmin=-clim,vmax=clim)
        ax.set_xlabel('x ($\mu$m)')
        ax.set_ylabel('y ($\mu$m)')
        ax.set_aspect('equal')
        ax.set_title('Axial Mode %d' % mode)
        return ax 


    def show_crystal_planar_mode(self,mode=0,ax=None,theta=0):
        """
        Plots the planar modes of the crystal, using arrows to show displacement 
        in velocity and position.

        Parameters:
        -----------
        mode : int
            The mode to plot.
        ax : matplotlib.axes object
            The axes to plot the crystal on. If None, a new figure is created.
        theta : float
            The phase angle of the mode to plot.
        
        Returns:
        --------
        ax : matplotlib.axes object
        fig : matplotlib.figure object
        """
        if (ax is None): 
            fig = plt.figure()
            ax = fig.add_subplot(111)
        else: 
            fig = ax.get_figure()


        N             = self.Nion
        x             = self.uE[:N]*1e6
        y             = self.uE[N:]*1e6
        evs = self.planarEvects
        om  = self.planarEvalsE
        ev = -evs[:,mode*2]*np.exp(complex(0,theta))

        ax.scatter(x=x,y=y,color='royalblue',zorder = 3)
        ax.set_aspect('equal', 'box')
        ax.set_title("n = %d, "%(mode+1)+r"$\omega_n$"+"=%1.2e[Hz]" %(om[mode]))
        lim = np.max(np.abs([x,y]))*1.25
        ax.set_xlim(-lim,lim)
        ax.set_ylim(-lim,lim)
        dx  = np.real(ev)[0*N:1*N]
        dy  = np.real(ev)[1*N:2*N]
        dvx = np.real(ev)[2*N:3*N]
        dvy = np.real(ev)[3*N:4*N]
        norm = np.linalg.norm(   np.concatenate(  (dx,dy,dvx,dvy)   )  )/(2*N/2.5)
        if norm>0:
            dx /= norm
            dy /= norm
            dvx/= norm
            dvy/= norm
        for ion in range(N):
            ax.arrow(x[ion],y[ion],dx[ion,0],dy[ion,0],width = 1,head_length=2,fc='black', ec='black')
            ax.arrow(x[ion],y[ion],dvx[ion,0],dvy[ion,0],width = 1,head_length=2,fc='red', ec='red')

        fig.set_size_inches(8, 8)
        leg = fig.legend([r'$\mathbf{\delta x}$'
                            ,r'$\mathbf{\delta v}$'
                            ,r'$\mathbf{X_0}$']
                            ,labelcolor = ['black','red','royalblue'],
                         loc = 'upper left')
        leg.legendHandles[0].set_color('black')
        leg.legendHandles[1].set_color('red')
        leg.legendHandles[2].set_color('royalblue')
        ax.set_xlabel(r"x [$\mu$m]")
        ax.set_ylabel(r"y [$\mu$m]")
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
        Checks to see if any of the axial eigenvalues in the current configuration of the crystal
        are equal to zero. If so, this indicates that the one-plane configuration is unstable
        and a 1-2 plane transistion is possible.

        :return: Boolean: True if no 1-2 plane transistion mode exists, false if it does
        (Answers: "is the plane stable?")

        """
        if self.hasrun is False:
            self.run()

        for x in self.axialEvals:
            if x == 0.0:
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
