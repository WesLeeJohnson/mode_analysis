"""
Author:     Wes Johnson 
Date:       June 14th 2023
Purpose:    This script investigates the single-double plane 
            transition of the ion crystal in a penning trap 
            as a function of the rotating wall frequency. 
How to use: python mode_analysis_single_double_plane.py
""" 
import mode_analysis as ma
import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as constants

# parameters
ionmass = 9.012182
omega_z = 2*np.pi*1.58e6
B = 4.4588
q = constants.e
m = ionmass*constants.atomic_mass
wcyc = q*B/m
wmag = 1/2 * ( wcyc - np.sqrt(wcyc**2 - 2*omega_z**2) )
Vwall = 1
N = 4
XR = 3.082
num_ma = 5
ma_list = [None]*num_ma
frot_list = np.linspace(wmag*1.01,wmag*1.5,num_ma)

# run mode analysis
for i in range(num_ma):
    ma_list[i] = ma.ModeAnalysis(ionmass=ionmass
                                    ,omega_z=omega_z
                                    ,frot=frot_list[i]
                                    ,B=B
                                    ,N=N
                                    ,XR=XR
                                    ,Vwall=Vwall
                                )
    ma_list[i].run()

