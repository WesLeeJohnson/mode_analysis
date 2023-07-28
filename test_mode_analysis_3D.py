"""
Author:     Wes Johnson 
Date:       June 15th 2023
Purpose:    This code tests the mode_analysis_3D script. 
How to run: python test_mode_analysis_3D.py
"""
import numpy as np
import matplotlib.pyplot as plt
import mode_analysis_3D as ma3D
from scipy import constants as const
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset


#function to calculate the similarity between two vectors
def calculate_similarity(array1, array2):
    # Mean Squared Error (MSE)
    mse = np.mean((array1 - array2)**2)

    # Root Mean Squared Error (RMSE)
    rmse = np.sqrt(mse)

    # Pearson Correlation Coefficient
    correlation = np.corrcoef(array1, array2)[0, 1]

    # Cosine Similarity
    dot_product = np.dot(array1, array2)
    norm1 = np.linalg.norm(array1)
    norm2 = np.linalg.norm(array2)
    cosine_similarity = dot_product / (norm1 * norm2)

    # Euclidean Distance
    euclidean_distance = np.linalg.norm(array1 - array2)

    return mse, rmse, correlation, cosine_similarity, euclidean_distance

#plotting parameters
font_size_ticks = 16
font_size_labels = 20
font_size_title = 24
font_size_legend = 20 
font_size_annotation = 18
point_size = 10

#crystal parameters
ionmass = 9.012182 #Be
charge = const.elementary_charge
mass = ionmass*const.atomic_mass
B = 4.4566 #T
wcyc = charge*B/mass
vwall = 10 #V 
omega_z = 2*np.pi*1.58e6 #rad/s
frot = wcyc/(2*np.pi*1000)/2*1.9 #kHz
frot = 200.75 #kHz
N = 61 #number of ions


#run the 3D mode analysis
ma3D_instance = ma3D.ModeAnalysis(ionmass=ionmass
                                    ,omega_z=omega_z
                                    ,frot=frot
                                    ,B=B
                                    ,N=N
                                    ,Vwall=vwall
                                    ,precision_solving=True
                                    #,method='newton'
                                    ,method='bfgs'
                                    )
ma3D_instance.run()
ma3D_instance.run_3D()






#compare the positions of the 2D and 3D crystals
test_pos_3D = ma3D_instance.u_3D
ori_pos_3D = np.concatenate((ma3D_instance.u,np.zeros((N,))))
ax = ma3D_instance.show_crystal_3D(pos_vect=test_pos_3D,label='3D, PE = {:.6f} '.format(ma3D_instance.pot_energy_3D(test_pos_3D)),color='b')
ma3D_instance.show_crystal_3D(pos_vect=ori_pos_3D,ax=ax,color='r',label='2D, PE = {:.6f} '.format(ma3D_instance.pot_energy_3D(ori_pos_3D)))
plt.show()
#ma3D_instance.show_axial_freqs()
#ma3D_instance.show_cyc_freqs()
#ma3D_instance.show_ExB_freqs()
#plt.show()

#get the mode frequencuies of the 2D and 3D calculations
planar_freqs = ma3D_instance.planarEvalsE
axial_freqs = ma3D_instance.axialEvalsE
modes_2D = np.hstack((planar_freqs[:N],axial_freqs))
modes_3D = ma3D_instance.Evals_3DE
modes_2D = modes_2D/2/np.pi/1e6 
modes_3D = modes_3D/2/np.pi/1e6
#modes_2D = np.sort(modes_2D)
modes_3D = np.sort(modes_3D)
ExB_2D = modes_2D[:N]
ExB_3D = modes_3D[:N]
axial_2D = modes_2D[N:2*N]
axial_3D = modes_3D[N:2*N]


#compare the ExB and axial mode branch frequencies for the 2D and 3D calculations: 
fig,ax = plt.subplots(figsize=(8,8))
ExB_modes_nums_2D = np.arange(0,len(ExB_2D))
ExB_modes_nums_3D = np.arange(0,len(ExB_3D))
axial_modes_nums_2D = np.arange(0,len(axial_2D)) + len(ExB_2D)
axial_modes_nums_3D = np.arange(0,len(axial_3D)) + len(ExB_3D)
ax.plot(ExB_modes_nums_3D,ExB_3D,'o',label='3D ExB',color='royalblue',markersize=point_size)
ax.plot(ExB_modes_nums_2D,ExB_2D,'x',label='2D ExB',color='orange',markersize=point_size)
ax.plot(axial_modes_nums_3D,axial_3D,'o',label='3D axial',color='blue',markersize=point_size)
ax.plot(axial_modes_nums_2D,axial_2D,'x',label='2D axial',color='red',markersize=point_size)
ax.set_xlabel('Mode Number',fontsize=font_size_labels)
ax.set_ylabel('Frequency (MHz)',fontsize=font_size_labels)
ax.legend(fontsize=font_size_legend)
ax.set_title('ExB and Axial Modes, $f_r$ = {:.2f} kHz'.format(frot)
             ,fontsize=font_size_title)
ax.axes.tick_params(labelsize=font_size_ticks)
inset_ax = inset_axes(ax, width="30%", height="30%", loc='lower right')

# Plotting the inset data
inset_ax.plot(ExB_modes_nums_3D[N-5:], ExB_3D[N-5:], 'o', color='royalblue', markersize=point_size)
inset_ax.plot(ExB_modes_nums_2D[N-5:], ExB_2D[N-5:], 'x', color='orange', markersize=point_size)
inset_ax.plot(axial_modes_nums_3D[:5], axial_3D[:5], 'o', color='blue', markersize=point_size)
inset_ax.plot(axial_modes_nums_2D[:5], axial_2D[:5], 'x', color='red', markersize=point_size)
inset_ax.tick_params(which='both', bottom=False, top=False, left=False, right=False,labelbottom=False, labelleft=True)
ax.indicate_inset_zoom(inset_ax, edgecolor="black")
mark_inset(ax, inset_ax, loc1=1, loc2=3, fc="none", ec="0.5")
plt.show()




##compare the eigenfrequencies of the 2D and 3D crystals
#fig,ax = plt.subplots(figsize=(10,10))
#modes_nums_2D = np.arange(0,len(modes_2D))
#modes_nums_3D = np.arange(0,len(modes_3D))
#assert len(modes_2D) == len(modes_3D), 'The number of modes in the 2D and 3D crystals are not the same'
#ax.plot(modes_nums_3D,modes_3D,'o',label='3D',color='r',markersize=point_size)
#ax.plot(modes_nums_2D,modes_2D,'x',label='2D',color='b',markersize=point_size)
#ax.set_xlabel('Mode Number',fontsize=font_size_labels)
#ax.set_ylabel('Frequency (MHz)',fontsize=font_size_labels)
#ax.legend(fontsize=font_size_legend)
#ax.set_title('2D and 3D Mode Calculation, $f_r$ = {:.2f} kHz'.format(frot)
#             ,fontsize=font_size_title)
#ax.axes.tick_params(labelsize=font_size_ticks)
#mse, rmse, correlation, cosine_similarity, euclidean_distance = calculate_similarity(modes_2D,modes_3D)
#ax.annotate('MSE = {:.2f}'.format(mse)+
#            '\nCorrelation = {:.2f}'.format(correlation)+\
#                '\nCosine Similarity = {:.2f}'.format(cosine_similarity)+\
#                    '\nEuclidean Distance = {:.2f}'.format(euclidean_distance)
#                    ,xy=(0.05,0.5),xycoords='axes fraction',fontsize=font_size_annotation
#            )
#plt.show();exit()




