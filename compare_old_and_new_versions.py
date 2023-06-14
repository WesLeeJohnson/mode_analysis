"""
The purpose of this script is to compare the results of the new mode analysis
"""

import mode_analysis as ma_new 
import mode_analysis_code_original as ma_old
import matplotlib.pyplot as plt
import numpy as np

ma_new_instance = ma_new.ModeAnalysis()
ma_old_instance = ma_old.ModeAnalysis(ionmass=9.012182)

ma_new_instance.run()
ma_old_instance.run()

fig,axs = plt.subplots(nrows=1,ncols=1,figsize=(10,5))
ax1 = axs
x = ma_new_instance.uE[:ma_new_instance.Nion]
y = ma_new_instance.uE[ma_new_instance.Nion:]
ax1.scatter(x,y,color='blue',label='new')
x = ma_old_instance.uE[:ma_old_instance.Nion]
y = ma_old_instance.uE[ma_old_instance.Nion:]
ax1.scatter(x,y,color='red',label='old')
ax1.set_xlabel('x ($\mu$m)')
ax1.set_ylabel('y ($\mu$m)')
ax1.set_title('Positions of ions')
ax1.set_aspect('equal')
plt.show()
