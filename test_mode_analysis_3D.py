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

#crystal parameters
ionmass = 9.012182 #Be
charge = const.elementary_charge
mass = ionmass*const.atomic_mass
B = 4.4566 #T
wcyc = charge*B/mass
vwall = 10 #V 
omega_z = 2*np.pi*1.58e6 #rad/s
frot = wcyc/(2*np.pi*1000)/2*1.9 #kHz
frot = 180 #kHz
frot = 201.25 #kHz
N = 61 #number of ions
ma3D_instance = ma3D.ModeAnalysis(ionmass=ionmass
                                    ,omega_z=omega_z
                                    ,frot=frot
                                    ,B=B
                                    ,N=N
                                    ,Vwall=vwall
                                    ,precision_solving=True
                                    )
ma3D_instance.run()
ma3D_instance.run_3D()

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

#compare the eigenfrequencies of the 2D and 3D crystals
planar_freqs = ma3D_instance.planarEvalsE
axial_freqs = ma3D_instance.axialEvalsE
modes_2D = np.hstack((planar_freqs,axial_freqs))
modes_3D = ma3D_instance.Evals_3DE
font_size_ticks = 16
font_size_labels = 20
font_size_title = 24
font_size_legend = 20 
font_size_annotation = 18
fig,ax = plt.subplots(figsize=(10,10))
modes_nums_2D = np.arange(0,len(modes_2D))
modes_nums_3D = np.arange(0,len(modes_3D))
modes_2D = modes_2D/2/np.pi/1e6 
modes_3D = modes_3D/2/np.pi/1e6
modes_2D = np.sort(modes_2D)
modes_3D = np.sort(modes_3D)
ax.plot(modes_nums_3D,modes_3D,'o',label='3D',color='r')
ax.plot(modes_nums_2D,modes_2D,'x',label='2D',color='b')
ax.set_xlabel('Mode Number',fontsize=font_size_labels)
ax.set_ylabel('Frequency (MHz)',fontsize=font_size_labels)
ax.legend(fontsize=font_size_legend)
ax.set_title('2D and 3D Mode Calculation, $f_r$ = {:.2f} kHz'.format(frot)
             ,fontsize=font_size_title)
ax.axes.tick_params(labelsize=font_size_ticks)
mse, rmse, correlation, cosine_similarity, euclidean_distance = calculate_similarity(modes_2D,modes_3D)
ax.annotate('MSE = {:.2f}'.format(mse)+
            '\nCorrelation = {:.2f}'.format(correlation)+\
                '\nCosine Similarity = {:.2f}'.format(cosine_similarity)+\
                    '\nEuclidean Distance = {:.2f}'.format(euclidean_distance)
                    ,xy=(0.05,0.5),xycoords='axes fraction',fontsize=font_size_annotation
            )
print(calculate_similarity(modes_2D,modes_3D))
plt.show();exit()




#compare the positions of the 2D and 3D crystals
test_pos_3D = ma3D_instance.u_3D
ori_pos_3D = np.concatenate((ma3D_instance.u,np.zeros((N,))))
ax = ma3D_instance.show_crystal_3D(pos_vect=test_pos_3D,label='3D, PE = {:.6f} '.format(ma3D_instance.pot_energy_3D(test_pos_3D)),color='b')
ma3D_instance.show_crystal_3D(pos_vect=ori_pos_3D,ax=ax,color='r',label='2D, PE = {:.6f} '.format(ma3D_instance.pot_energy_3D(ori_pos_3D)))
print(ma3D_instance.pot_energy_3D(test_pos_3D))
print(ma3D_instance.pot_energy_3D(ori_pos_3D))
plt.show();exit()
ma3D_instance.show_axial_freqs()
ma3D_instance.show_cyc_freqs()
ma3D_instance.show_ExB_freqs()
plt.show()
