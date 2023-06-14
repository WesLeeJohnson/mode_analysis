"""
The purpose of this script is to compare the results of the new mode analysis
"""

import mode_analysis as ma_new 
import mode_analysis_code_original as ma_old

ma_new_instance = ma_new.ModeAnalysis()
ma_old_instance = ma_old.ModeAnalysis()

ma_new_instance.show_crystal(ma_new_instance.uE)
ma_old_instance.show_crystal(ma_old_instance.uE)
