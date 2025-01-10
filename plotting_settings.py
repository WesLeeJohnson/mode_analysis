"""
Author:     Wes Johnson
Date:       July 24th, 2023
Purpose:    Sets the value of matplotlib fonts and parameters. 
How to use: import plot_settings * 
            Make sure to have in current working dir 
""" 

import matplotlib.pyplot as plt

# plotting parameters and fonts
plotting_params = {
    #'font_size' : 18,
    'tick_font_size' : 16,
    'label_font_size' : 18,
    'legend_font_size' : 16,
    'title_font_size' : 20,
    'line_width' : 3,
    'point_size' : 10,
    'fig_size' : (8,8), 
    'tick_major_width' : 1.5, 
    'tick_minor_width' : 1,
}

def set_plot_params(params):
    plt.rcParams['font.family'] = 'serif'
    #plt.rcParams['font.size'] = params['font_size']
    plt.rcParams['axes.linewidth'] = 2
    plt.rcParams['xtick.major.size'] = params['tick_font_size']
    plt.rcParams['xtick.major.width'] = params['tick_major_width']
    plt.rcParams['xtick.minor.size'] = params['tick_font_size'] - 2
    plt.rcParams['xtick.minor.width'] = params['tick_minor_width']
    plt.rcParams['ytick.major.size'] = params['tick_font_size']
    plt.rcParams['ytick.major.width'] = params['tick_major_width']
    plt.rcParams['ytick.minor.size'] = params['tick_font_size'] - 2
    plt.rcParams['ytick.minor.width'] = params['tick_minor_width']
    plt.rcParams['legend.fontsize'] = params['legend_font_size']
    plt.rcParams['axes.labelsize'] = params['label_font_size']
    plt.rcParams['axes.titlesize'] = params['title_font_size']
    plt.rcParams['lines.linewidth'] = params['line_width']
    plt.rcParams['lines.markersize'] = params['point_size']
    plt.rcParams['figure.figsize'] = params['fig_size']
    plt.rcParams['figure.titlesize'] = params['title_font_size']
    plt.rcParams['axes.grid'] = True
    plt.rcParams['axes.axisbelow'] = True

set_plot_params(plotting_params) 