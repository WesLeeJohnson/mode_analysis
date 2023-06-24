"""
Author:     Wes Johnson
Date:       June 23rd, 2023
Purpose:    Compare the eigenvectors of the 2D and 3D calculations. 
            The eigenvectors are plotted using the helper functions in
            the mode_analysis_3D.py file. 
How to Run: python compare_eigenvectors.py
            make sure the mode_analysis_3D.py file is in the same directory 
            or in your python path.
"""

# Import statements
import numpy as np
import matplotlib.pyplot as plt
from scipy import constants
import mode_analysis_3D as mode_analysis
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset


# trap parameters 
wz          = 2*np.pi*1.58e6
B           = 4.4588
m_Be        = 9.012182 * constants.u
q_Be        = constants.e
wc          = q_Be * B / m_Be   
Vwall       = 5
frot        = 180
N           = 50 


# ploting parameters
pdir        = '../plots/'
title_size  = 20
label_size  = 18 
tick_size   = 16
legend_size = 16
line_width  = 3
point_size  = 10
font_size_labels = label_size
font_size_ticks = tick_size
font_size_legend = legend_size
font_size_title = title_size
fig_size    = (12,8)

# limits
ratio_lims  = [0,500]
beta_lims   = [0,.085]

# run mode analysis
ma = mode_analysis.ModeAnalysis(N = N
                                ,omega_z = wz
                                ,ionmass = m_Be/constants.u
                                ,B = B
                                ,frot = frot 
                                ,Vwall = Vwall
                                )
ma.run()
ma.run_3D()




#get the mode frequencuies of the 2D and 3D calculations
planar_freqs = ma.planarEvalsE
axial_freqs = ma.axialEvalsE
modes_2D = np.hstack((planar_freqs[:N],axial_freqs))
modes_3D = ma.Evals_3DE
vectors_3D = ma.Evects_3D
Evects_xy = np.zeros((4*N,2*N))    
Evects_z = np.zeros((2*N,N))

# fill the Evects_xy and Evects_z with the appropriate vectors
for mode in range(N):
    for ion in range(N): 
        x1  = ion + 0*N 
        y1  = ion + 1*N
        vx1 = ion + 2*N
        vy1 = ion + 3*N
        z1  = ion + 0*N
        vz1 = ion + 1*N

        x2  = ion + 0*N
        y2  = ion + 1*N
        z2  = ion + 2*N
        vx2 = ion + 3*N
        vy2 = ion + 4*N
        vz2 = ion + 5*N

        ExB1 = mode
        ExB2 = mode 

        axl1 = mode 
        axl2 = mode + N

        cyc1 = mode + N 
        cyc2 = mode + 2*N

        # fill the axial modes Evects
        Evects_z[z1,axl1] = np.real(vectors_3D[z2,axl2])
        Evects_z[vz1,axl1] = np.real(vectors_3D[vz2,axl2])
        # fill the ExB modes Evects
        Evects_xy[x1,ExB1] = np.real(vectors_3D[x2,ExB2])
        Evects_xy[y1,ExB1] = np.real(vectors_3D[y2,ExB2])
        Evects_xy[vx1,ExB1] = np.real(vectors_3D[vx2,ExB2])
        Evects_xy[vy1,ExB1] = np.real(vectors_3D[vy2,ExB2])
        # fill the cyc modes Evects
        Evects_xy[x1,cyc1] = np.real(vectors_3D[x2,cyc2])
        Evects_xy[y1,cyc1] = np.real(vectors_3D[y2,cyc2])
        Evects_xy[vx1,cyc1] = np.real(vectors_3D[vx2,cyc2])
        Evects_xy[vy1,cyc1] = np.real(vectors_3D[vy2,cyc2])

for mode in range(N): 
    axl_mode = N - mode - 1
    ExB_mode = mode
    cyc_mode = N + mode
    fig,axs = plt.subplots(3,2,figsize=(8,8))
    axs = axs.T
    axs = axs.flatten()
    ax = axs[0]
    ma.show_crystal_axial_mode(ax=ax,mode=axl_mode)
    ax = axs[1]
    ma.show_crystal_planar_mode(ax=ax,mode=ExB_mode)
    ax = axs[2]
    ma.show_crystal_planar_mode(ax=ax,mode=cyc_mode) 


    ax = axs[3]
    ma.show_crystal_axial_mode(ax=ax,mode=axl_mode
                               ,Evects=Evects_z
                               ,pos_vect=ma.uE_3D[:2*N]
                               )
    ax = axs[4]
    ma.show_crystal_planar_mode(ax=ax,mode=ExB_mode
                                ,Evects=Evects_xy
                                ,pos_vect=ma.uE_3D[:2*N]
                                )
    ax = axs[5]
    ma.show_crystal_planar_mode(ax=ax,mode=cyc_mode
                                ,Evects=Evects_xy
                                ,pos_vect=ma.uE_3D[:2*N]
                                )


    plt.tight_layout()
    plt.show()
exit() 