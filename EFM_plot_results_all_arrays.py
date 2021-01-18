#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 21 23:35:05 2019

@author: eeinga
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from F_EFM import linear_least_squares
from vel_models import get_velocity_model, get_velocity_data
from matplotlib.ticker import MultipleLocator, FormatStrFormatter, AutoMinorLocator, ScalarFormatter


network = 'AU'
arrays = ['ASAR', 'WRA', 'PSA']
comps = 3

# Frequencies to be used:
fbands = {'A':[0.5,1], 'B':[0.75,1.5], 'C':[1,2], 'D':[1.5,3], 'E':[2,4], 'F':[2.5,5], 'G':[3,6], 'H':[3.5,7]}
#fcs = []
#for fband in fbands:
#    fcs.append( (fbands[fband][0] + fbands[fband][1])/2 )

# Load EFM results for all arrays:
EFM_results = {}
s_envs = {}; deltas = {}

# Quality factors:
Qs = {}; Qs_std = {}; Qs_inv_vals = {}; Qs_inv_vals_std = {}; theor_Qs_invs = {};
Qs_lists = {}; Qs_std_lists = {}
Qi = {}; Qi_std = {}; Qi_lists = {}; Qi_std_lists = {}
Qdiff = {}; Qdiff_std = {}; Qdiff_lists = {}; Qdiff_std_lists = {}
Qd0 = {}; Qi0 = {};

# Scattering parameters:
a1s = {}; a0s = {}; a1s_std = {}; a0s_std = {}; theor_a1s = {}
a = {}; a_std = {}; E = {}; E_std = {}
alpha = {}; L = {}

# Other:
tP_inds = {}; coda_inds = {}
fcs = {}

for array in arrays:
    fopen = open( '/nfs/a9/eeinga/Results/AU/EFM/' + array + '/EFM_' + array \
                 + '_' + str(comps) + 'comps_final_results.pckl', 'rb')
    EFM_res = pickle.load( fopen)
    fopen.close()

    EFM_results[array] = EFM_res[array]

    # Get all quality factor values and uncertainties for all arrays:
    Qs[array] = EFM_res[array]['Qs']
    Qs_std[array] = EFM_res[array]['Qs_std']
    Qs_inv_vals[array] = EFM_res[array]['Qs_inv_vals']
    Qs_inv_vals_std[array] = EFM_res[array]['Qs_inv_vals_std']
    theor_Qs_invs[array] = EFM_res[array]['theor_Qs_invs']
    tP_inds[array] = EFM_res[array]['tP_inds']
    coda_inds[array] = EFM_res[array]['coda_inds']

    Qi[array] = EFM_res[array]['Qi']; Qi_std[array] = EFM_res[array]['Qi_std']
    Qdiff[array] = EFM_res[array]['Qdiff']; Qdiff_std[array] = EFM_res[array]['Qdiff_std']

    # Transform Q factors into arrays:
    Qs_lists[array] = []; Qdiff_lists[array] = []; Qi_lists[array] = []
    Qs_std_lists[array] = []; Qdiff_std_lists[array] = []; Qi_std_lists[array] = []
    for fband in fbands:
        Qs_lists[array].append( Qs[array][fband])
        Qi_lists[array].append( Qi[array][fband])
        Qdiff_lists[array].append( Qdiff[array][fband])
        Qs_std_lists[array].append( Qs_std[array][fband])
        Qi_std_lists[array].append( Qi_std[array][fband])
        Qdiff_std_lists[array].append( Qdiff_std[array][fband])

    # Get linear fit coefficients:
    a1s[array] = EFM_res[array]['a1s']; a0s[array] = EFM_res[array]['a0s']
    a1s_std[array] = EFM_res[array]['a1s_std']; a0s_std[array] = EFM_res[array]['a0s_std']
    theor_a1s[array] = EFM_res[array]['theor_a1s']
    #print( array + ', ' + str( theor_a1s[array]))

    # Get structural parameters:
    a[array] = EFM_res[array]['a']/1000; a_std[array] = EFM_res[array]['a_std']/1000
    E[array] = EFM_res[array]['E']*100; E_std[array] = EFM_res[array]['E_std']*100

    # Get diffusion and intrinsic Q, alpha and L:
    alpha[array] = EFM_res[array]['alpha']
    Qi0[array] = EFM_res[array]['Qi0']
    Qd0[array] = EFM_res[array]['Qd0']
    Qi[array] = EFM_res[array]['Qi']
    Qi_std[array] = EFM_res[array]['Qi_std']
    Qdiff[array] = EFM_res[array]['Qdiff']
    Qdiff_std[array] = EFM_res[array]['Qdiff_std']
    L[array] = EFM_res[array]['L']

    # Define dataset path and file name:
    s_envs_fname = '/nfs/a9/eeinga/Results/' + network + '/EFMD/' + array + '/EFMD_' \
    + array + '_' + str( comps ) + 'comps_s_envs_all_fbands.pckl'

    # Load the non normalised coda envelopes for all arrays and frequency bands:
    fopen = open( s_envs_fname, 'rb')
    s_envs0 = pickle.load(fopen)
    fopen.close()
    s_envs[array] = s_envs0[array]

    # We need the inverse of the sampling rate:
    fopen2 = open('/nfs/a9/eeinga/Results/' + network + '/EFM/' + array + '/' + array \
                  + '_delta.pckl','rb')
    delta0 = pickle.load(fopen2)
    fopen2.close()
    # Define delta:
    deltas[array] = delta0[array]

    # Frequencies should be the same for all arrays:
    fcs = EFM_res[array]['freqs']



####################################################################################################

# Calculate Qtot values for all arrays and frequencies: it is the sum of the inverses of
# the Q factors.
Qtot = {}; Qtot_lists = {}
Qtot_std = {}; Qtot_std_lists = {}

for array in arrays:
    Qtot[array] = {}; Qtot_std[array] = {}
    Qtot_lists[array] = []; Qtot_std_lists[array] = []

    for fband in fbands:

        # Calculate Q tot value:
        Qtot_val = 1 / (( 1/Qs[array][fband] ) + ( 1/Qdiff[array][fband] ) + ( 1/Qi[array][fband] ))
        Qtot[array][fband] = Qtot_val
        Qtot_lists[array].append( Qtot_val)

        # Calculate the standard deviation of Q tot:
        Qtot_std_val = np.sqrt( ( Qs_std[array][fband]**2 / (Qs[array][fband] **4) ) \
                     + ( Qdiff_std[array][fband]**2 / (Qdiff[array][fband]**4) ) \
                     + ( Qi_std[array][fband]**2 / (Qi[array][fband]**4) ) )
        Qtot_std[array][fband] = Qtot_std_val
        Qtot_std_lists[array].append( Qtot_std_val)

####################################################################################################


# Prepare to plot results:
colors = {'ASAR': ['orange', 'orangered', 'gold', 'maroon'],
          'WRA': ['dodgerblue', 'mediumblue', 'lightskyblue', 'midnightblue'],
          'PSA': ['mediumseagreen', 'darkolivegreen', 'springgreen', 'darkgreen']}
markers = {'ASAR': 'ko',
           'WRA': 'sk',
           'PSA': '^k'}

####################################################################################################



# # Fig. 1: Qs with errorbars vs. theoretical
# f, ax1 = plt.subplots(figsize= (16, 8))
# ax2 = ax1.twiny()

# #plt.title('Qs vs. theoretical curve, ' + str(comps) + ' components', fontsize= 20)
# ax1.set_xscale('log'); ax1.set_yscale('log')
# ax2.set_xscale('log'); ax2.set_yscale('log')

# for array in arrays:

#     ax2.plot( fcs, theor_Qs_invs[array], color = colors[array][0],
#               linewidth= 3, label= array + ', theoretical $Q_s^{-1}$: a= [' \
#               + str(round(a[array], 2)) + '+/-' + str(round(a_std[array], 2)) \
#               + '] km, $\epsilon$ = [' + str(round(E[array], 2)) + '+/-' \
#               + str(round(E_std[array], 2)) + ']%')
#     ax1.errorbar( fcs, Qs_inv_vals[array], yerr = Qs_inv_vals_std[array],
#                   fmt = markers[array], markersize= 15, ecolor = colors[array][1],
#                   capsize = 10, markerfacecolor = colors[array][1],
#                   markeredgecolor = 'k', label = '/')

# ax1.set_xlabel('Frequency (Hz)', fontsize= 24)
# ax1.set_ylabel('$Q_s^{-1}$', fontsize= 24)
# ax1.grid()
# ax1.set_ylim( [0.6e-3, 7e-3])
# ax2.set_ylim( [0.6e-3, 7e-3])

# # Make a plot with major ticks that are multiples of 20 and minor ticks that
# # are multiples of 5.  Label major ticks with '%d' formatting but don't label
# # minor ticks.
# ax1.xaxis.set_major_locator(MultipleLocator(1))
# ax1.xaxis.set_major_formatter(FormatStrFormatter('%d'))

# # For the minor ticks, use no labels; default NullFormatter.
# ax1.yaxis.set_major_locator(MultipleLocator(1e-3))
# #ax1.yaxis.set_minor_locator(MultipleLocator(7.5e-4))
# ax1.yaxis.set_major_formatter(FormatStrFormatter('%.1E'))
# #ax1.yaxis.set_minor_formatter(FormatStrFormatter('%.1E'))

# # Change size of ticks labels:
# ax1.tick_params(axis= 'both', labelsize= 20)
# ax1.tick_params(which= 'minor', labelsize= 20)
# ax2.get_xaxis().set_visible(False)
# #plt.legend(loc= 'lower right', fontsize= 20)

# handles1, labels1 = ax1.get_legend_handles_labels()
# handles2, labels2 = ax2.get_legend_handles_labels()
# handles = handles1 + handles2
# labels = labels1 + labels2
# ax2.legend( handles, labels, loc = 'lower right', ncol = 2, handletextpad = 0.4,
#            columnspacing = 0.4, fontsize = 18)

# fname = '/nfs/a9/eeinga/Results/AU/EFM/Qs_vs_theoretical_curve_ALL_ARRAYS'
# plt.savefig(fname + '.png', bbox_inches= 'tight')
# plt.savefig(fname + '.pdf', bbox_inches= 'tight')
# plt.close()




####################################################################################################




# # Figure 2: data derived vs. theoretical a1s we use to obtain Qi0 and Qd0.
# fig, ax1 = plt.subplots(figsize = (16, 8))
# ax2 = ax1.twinx()

# for array in arrays:

#     #if array != 'PSA': x = fcs[1:]
#     #else: x = fcs
#     x = fcs[1:]

#     # Plot a1 vs. angular frequency with the Qi0 and Qdiff values:
#     ax2.plot(x, theor_a1s[array], color= colors[array][0], linewidth= 3,
#               label= array + r'$,  \alpha $= ' + str(round(alpha[array], 1)) \
#               + ', $Q_{i0}$= ' + str(round(Qi0[array], 2)) + ' , $Q_{diff0}$= ' \
#               + str(round(Qd0[array], 2)))# + ', L= ' + str(np.round(L[array]/1000, 1)) + ' km')
#     for i, fband in enumerate(fbands):
#         if i==0:
#             ax1.errorbar(fcs[i], a1s[array][fband], yerr= a1s_std[array][fband], fmt = markers[array],
#                           markersize = 15, ecolor= colors[array][1], capsize= 10,  markeredgecolor = 'k',
#                           markerfacecolor = colors[array][1], label= '/')
#         else:
#             ax1.errorbar(fcs[i], a1s[array][fband], yerr= a1s_std[array][fband], fmt = markers[array],
#                           markersize = 15, ecolor= colors[array][1], capsize= 10,  markeredgecolor = 'k',
#                           markerfacecolor = colors[array][1])

# ax1.set_xlim( [fcs[0]-0.1, x[-1]+0.1])
# ax2.set_xlim( [fcs[0]-0.1, x[-1]+0.1])
# ax1.set_ylim( [0.0022, 0.018])
# ax2.set_ylim( [0.0022, 0.018])
# ax1.set_xlabel('Frequency (Hz)', fontsize= 24)
# ax1.set_ylabel('Coda decay coefficient ($a_1$)', fontsize= 24)
# ax1.grid()
# ax1.tick_params(axis= 'x', labelsize= 20)
# ax1.tick_params(axis= 'y', labelsize= 20)
# ax2.get_xaxis().set_visible(False)
# ax2.get_yaxis().set_visible(False)
# #ax2.tick_params( right = False, labelright=False)
# #plt.legend(loc= 'lower right', fontsize= 20)

# handles1, labels1 = ax1.get_legend_handles_labels()
# handles2, labels2 = ax2.get_legend_handles_labels()
# handles = handles1 + handles2
# labels = labels1 + labels2
# ax2.legend( handles, labels, loc = 'lower right', ncol = 2, handletextpad = 0.4,
#             columnspacing = 0.4, fontsize = 16)

# fname = '/nfs/a9/eeinga/Results/AU/EFM/a1s_vs_theoretical_curve_ALL_ARRAYS.pdf'
# plt.savefig(fname, bbox_inches= 'tight')
# fname = '/nfs/a9/eeinga/Results/AU/EFM/a1s_vs_theoretical_curve_ALL_ARRAYS.png'
# plt.savefig(fname, bbox_inches= 'tight')
# plt.close()


####################################################################################################



# Figure 3: comparison of Qs, Qdiff and Qi for all three arrays.
markers = {'ASAR': 'ro',
           'WRA': 'bs',
           'PSA': '^g'}

# Paper version of the figure:

f, (ax1, ax2, ax3) = plt.subplots( 1, 3, sharey = 'all', figsize = (10, 10))

for array in arrays:

    if array == 'ASAR': ax = ax1
    elif array == 'WRA': ax = ax2
    elif array == 'PSA': ax = ax3

    # Plot markers for legend:
    if array == 'ASAR':
        l1 = ax.plot( -10, -10, 'p', color = 'white', markeredgecolor = 'k',
                     markersize = 12, label = '$Q_{diff}$')
        l2 = ax.plot(  -10, -10, 'o', color = 'white', markeredgecolor = 'k',
                     markersize = 12, label = '$Q_i$')
        l3 = ax.plot(  -10, -10, '^', color = 'white', markeredgecolor = 'k',
                     markersize = 12, label = '$Q_s$')
        l4 = ax.plot(  -10, -10, 's', color = 'white', markeredgecolor = 'k',
                     markersize = 12, label = '$Q_{tot}$')

    # Plot Q factors with both lines and markers:
    ax.plot( fcs, Qdiff_lists[array], 'p-', color = colors[array][0],
            markeredgecolor = 'k', markersize = 10)
    ax.plot( fcs, Qi_lists[array], 'o-', color = colors[array][2],
            markeredgecolor = 'k', markersize = 10)
    ax.plot( fcs, Qs_lists[array], '^-', color = colors[array][1],
            markeredgecolor = 'k', markersize = 10)
    ax.plot( fcs, Qtot_lists[array], 's-', color = colors[array][3],
            markeredgecolor = 'k', markersize = 10)
    ax.errorbar( fcs, Qdiff_lists[array], yerr = Qdiff_std_lists[array],
                fmt = 'None', ecolor = 'k', elinewidth = 2, capsize = 5)
    ax.errorbar( fcs, Qi_lists[array], yerr = Qi_std_lists[array],
                fmt = 'None', ecolor = 'k', elinewidth = 2, capsize = 5)
    ax.errorbar( fcs, Qs_lists[array], yerr = Qs_std_lists[array],
                fmt = 'None', ecolor = 'k', elinewidth = 2, capsize = 5)
    ax.errorbar( fcs, Qtot_lists[array], yerr = Qtot_std_lists[array],
                fmt = 'None', ecolor = 'k', elinewidth = 2, capsize = 5)

    ax.grid( )
    #ax.set_title( array, fontsize = 24)
    ax2.set_xlabel( 'Frequency (Hz)', fontsize = 18)
    ax1.set_ylabel('Quality factors', fontsize = 18)
    ax.set_xlim( 0.25, 6)
    # ax.set_xticks( fcs)
    ax.set_yscale('log')
    ax.set_yticks( np.arange( 0, 2400, 300))
    ax.set_ylim( 1*10**2, 3.0*10**3)
    #ax.legend( loc='upper left', fontsize = 18)
    ax.tick_params(axis= 'both', labelsize= 16)
    # ax.tick_params(axis= 'y', labelsize= 16)
    #ax.xaxis.tick_top()
    #ax.xaxis.set_label_position('top')
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))

    #for side in ax.spines.keys():
    #    ax.spines[side].set_linewidth(2)

plt.subplots_adjust( wspace = .05)
# Labels at the bottom:
#f.text( 0.16, 0.12, 'ASAR', fontsize = 24)
#f.text( 0.44, 0.12, 'WRA', fontsize = 24)
#f.text( 0.7, 0.12, 'PSA', fontsize = 24)
# Labels at the top:
f.text( 0.205, 0.84, 'ASAR', fontsize = 24)
f.text( 0.47, 0.84, 'WRA', fontsize = 24)
f.text( 0.735, 0.84, 'PSA', fontsize = 24)

# Labels to use in the legend for each quality factor:
legend_labels = ['$Q_{diff}$', '$Q_i$', '$Q_s$', '$Q_{tot}$']

# Create the legend
f.legend([l1, l2, l3, l4],     # The line objects
         labels = legend_labels,   # The labels for each line
         loc = (0.325, 0.92),   # Position of legend
         borderaxespad = 0.1,    # Small spacing around legend box
         ncol = 4,
         columnspacing = 0.6,
         handletextpad = 0,
         fontsize = 16)

fname = '/nfs/a9/eeinga/Results/AU/EFM/EFM_all_Qfactors_ALL_ARRAYS.pdf'
plt.savefig(fname, bbox_inches= 'tight')
fname = '/nfs/a9/eeinga/Results/AU/EFM/EFM_all_Qfactors_ALL_ARRAYS.png'
plt.savefig(fname, bbox_inches= 'tight', transparent = False)



# # Plot the Q factors: Qi, Qdiff, Qs and Qtot, one in each subplot.

# f, (( ax1, ax2),(ax3, ax4)) = plt.subplots( 2, 2, sharex = 'all', sharey = 'all',
#                                            figsize = (10, 10))

# for array in arrays:

#     # Plot Q factors with both markers and lines in each subplot:
#     ax1.plot( fcs, Qdiff_lists[array], 'p--', color = colors[array][0],
#             markeredgecolor = 'k', markersize = 12, label = array)
#     ax2.plot( fcs, Qi_lists[array], 'o--', color = colors[array][0],
#             markeredgecolor = 'k', markersize = 12, label = array)
#     ax3.plot( fcs, Qs_lists[array], '^--', color = colors[array][0],
#             markeredgecolor = 'k', markersize = 12, label = array)
#     ax4.plot( fcs, Qs_lists[array], 's--', color = colors[array][0],
#             markeredgecolor = 'k', markersize = 12, label = array)

#     # Draw grid:
#     ax1.grid(); ax2.grid(); ax3.grid(); ax4.grid()

#     # Plot 1:
#     ax1.set_ylabel('Quality factors', fontsize = 20)
#     ax1.set_xlim( 0.25, 6)#; ax3.set_xlim( 0.25, 6)
#     ax1.tick_params( labelsize= 16)
#     ax1.set_ylim( 1*10**1, 2.6*10**3)
#     ax1.legend( title = '$Diffusion Q (Q_{diff}$)', title_fontsize = 18,
#                fontsize = 12, ncol = 1)

#     # Plot 2:
#     # ax2.set_xlim( 0.25, 6); ax4.set_xlim( 0.25, 6)
#     # ax2.tick_params( labelleft = False, labelright = True, labelsize = 16)
#     ax2.legend( title = 'Intrinsic Q ($Q_i$)', title_fontsize = 18,
#                ncol = 3, fontsize = 12)

#     # Plot 3:
#     ax3.set_xlabel( 'Frequency (Hz)', fontsize = 20)
#     ax3.set_ylabel('Quality factors', fontsize = 20)
#     ax3.tick_params( labelsize= 16)
#     ax3.legend( title = 'Scattering Q ($Q_s$)', title_fontsize = 18,
#                ncol = 3, fontsize = 12)

#     # Plot 4:
#     ax4.set_xlabel( 'Frequency (Hz)', fontsize = 20)
#     # ax4.tick_params( labelleft = False, labelright = True, labelsize = 16)
#     ax4.legend( title = '$Q_{tot}$', title_fontsize = 18,
#                ncol = 3, fontsize = 12)


#     #ax.set_yscale('log')
#     #ax.yaxis.set_major_formatter(FormatStrFormatter('%.0E'))

#     # for side in ax.spines.keys():
#         # ax.spines[side].set_linewidth(2)

# plt.subplots_adjust( wspace = .05, hspace = 0.05)

# fname = '/nfs/a9/eeinga/Results/AU/EFM/EFM_all_Qfactors_ALL_ARRAYS_b.pdf'
# plt.savefig(fname, bbox_inches= 'tight')
# fname = '/nfs/a9/eeinga/Results/AU/EFM/EFM_all_Qfactors_ALL_ARRAYS_b.png'
# plt.savefig(fname, bbox_inches= 'tight', transparent = False)
# plt.close()



####################################################################################################




# # Plot envelopes and their fitting functions together:
# #fbc = {'A':'r', 'B':'navy', 'C':'gray', 'D':'c', 'E':'darkviolet', 'F':'orange', 'G':'lime', 'H':'brown'}

# # Get mean tP_ind and coda_ind (for the coda time window shading):
# coda_time = []; max_times = []
# for array_nm in arrays:
#     for fband in fbands:

#         # Get inverse of the sampling rate:
#         delta = deltas[array_nm]

#         # Extract normalised coda envelope for this frequency band:
#         s_env = s_envs[array_nm][fband]['s_envs']

#         # Time vector for the fit:
#         t = np.arange(0, len(s_env) * delta, delta)

#         coda_time.append( t[coda_inds[array_nm][fband]] )
#         max_times.append( t[-100])

# min_coda_time = np.array( coda_time).min()
# max_time = np.array( max_times).max()

# f, axes = plt.subplots( 4, 2, sharex = 'all', sharey = 'all', figsize = (17, 20))
# gs = [0, 2, 4, 6]# Indices for left column panels

# tw_lengths = []

# for g, fband in enumerate(fbands):

#     if g<2: ax = axes[0][g]
#     elif g>=2 and g<4: ax = axes[1][g-2]
#     elif g>=4 and g<6: ax = axes[2][g-4]
#     else: ax = axes[3][g-6]

#     for array_nm in arrays:

#         delta = deltas[array_nm]

#         # Extract normalised coda envelope for this frequency band:
#         s_env = s_envs[array_nm][fband]['s_envs']

#         # Time vector for the fit:
#         t = np.arange(0, len(s_env) * delta, delta)
#         tw_lengths.append( t[-100] - t[ coda_inds[array_nm][fband] ])

#         # Plot shaded areas and divisions:
#         if array_nm == 'ASAR':
#             ax.axvspan( min_coda_time, max_time, color = 'rosybrown', alpha = 0.3)

#         # Data envelopes:
#         ax.plot(t[tP_inds[array_nm][fband] - 100:],
#              np.log10(s_env**2)[tP_inds[array_nm][fband] - 100:],
#              color = colors[array_nm][0], linewidth = 2, label = '/')
#         # Linear fits:
#         lab = array_nm + ', ' +  str(np.round(a0s[array_nm][fband], 3)) + '+/-' \
#                 + str(np.round(a0s_std[array_nm][fband], 3)) + ']+[' \
#                 + str(np.round(a1s[array_nm][fband], 6)) + '+/-' \
#                 + str(np.round(a1s_std[array_nm][fband], 6)) + '] *t'
#         yfit = a0s[array_nm][fband] + t[coda_inds[array_nm][fband]:-100]*-a1s[array_nm][fband]
#         ax.plot(t[coda_inds[array_nm][fband]:-100], yfit, '--', color = colors[array_nm][1],
#                 linewidth = 2, label = lab)

#         ax.grid(color= 'dimgrey', linewidth= 1)

#         #          AXES
#         if g>=6: ax.set_xlabel('Time (s)', fontsize = 18)
#         if g in gs: ax.set_ylabel('$log~(Amplitude^2)$', fontsize= 18)
#         ax.set_xlim( t[coda_inds[array_nm][fband]]-10, t[-10]);
#         ax.set_ylim( -5, -2.0)
#         ax.tick_params( axis = 'y', labelsize= 16, length= 10, width= 1)#, right= True, labelright= True, left= False, labelleft= False)
#         ax.tick_params(axis= 'x', labelsize= 16, length= 10, width= 1)
#         #plt.gca().yaxis.set_label_position('right')
#         #plt.axis([-0, 3*tJ, -4, 0.5])
#         for side in plt.gca().spines.keys():
#             ax.spines[side].set_linewidth(3)

#     #         LEGEND
#     tit = str(fbands[fband][0]) + '-' + str(fbands[fband][1]) + '$~$Hz'

#     handles1, labels1 = ax.get_legend_handles_labels()
#     handles11 = [handles1[0], handles1[2], handles1[4]]
#     handles12 = [handles1[1], handles1[3], handles1[5]]
#     labels11 = [labels1[0], labels1[2], labels1[4]]
#     labels12 = [labels1[1], labels1[3], labels1[5]]
#     handles = handles11 + handles12
#     labels = labels11 + labels12

#     ax.legend( handles, labels, loc = 'best', title = tit, title_fontsize = 16, ncol = 2,
#               handletextpad = 0.4, columnspacing = 0.4, fontsize= 12, fancybox= True,
#               framealpha= 1)

# #f.text( 0.08, 0.4, '$log (Amplitude^2)$', rotation = 90, fontsize= 18)
# f.tight_layout( rect = [0.1, 0.01, 0.99, 0.99])

# fname = '/nfs/a9/eeinga/Results/AU/EFM/EFM_all_envs_linfits_ALL_ARRAYS_portrait'
# plt.savefig( fname + '.pdf', bbox_inches= 'tight')
# plt.savefig( fname + '.png', bbox_inches= 'tight')
# plt.close('all')

# print('')
# print( 'Minimum time window length is ' + str( min(tw_lengths) ))
# print( 'Maximum time window length is ' + str( max(tw_lengths) ))
# print('')




# # Repeat the same figure but in landscape format:
# f, axes = plt.subplots( 2, 4, sharex = 'all', sharey = 'all', figsize = ( 30.5, 15))
# gs = [0, 2, 4, 6]# Indices for left column panels

# for g, fband in enumerate(fbands):

#     if g<4: ax = axes[0][g]
#     elif g>=4: ax = axes[1][g-4]

#     for array_nm in arrays:

#         delta = deltas[array_nm]

#         # Extract normalised coda envelope for this frequency band:
#         s_env = s_envs[array_nm][fband]['s_envs']

#         # Time vector for the fit:
#         t = np.arange(0, len(s_env) * delta, delta)

#         # Plot shaded areas and divisions:
#         if array_nm == 'ASAR':
#             ax.axvspan( min_coda_time, max_time, color = 'rosybrown', alpha = 0.3)

#         # Data envelopes:
#         ax.plot(t[tP_inds[array_nm][fband] - 100:],
#              np.log10(s_env**2)[tP_inds[array_nm][fband] - 100:],
#              color = colors[array_nm][0], linewidth = 2, label = '/')
#         # Linear fits:
#         lab = array_nm + ', ' +  str(np.round(a0s[array_nm][fband], 3)) + '+/-' \
#                 + str(np.round(a0s_std[array_nm][fband], 3)) + ']+[' \
#                 + str(np.round(a1s[array_nm][fband], 6)) + '+/-' \
#                 + str(np.round(a1s_std[array_nm][fband], 6)) + '] *t'
#         yfit = a0s[array_nm][fband] + t[coda_inds[array_nm][fband]:-100]*-a1s[array_nm][fband]
#         ax.plot(t[coda_inds[array_nm][fband]:-100], yfit, '--', color = colors[array_nm][1],
#                 linewidth = 2, label = lab)

#         ax.grid(color= 'dimgrey', linewidth= 1)

#         #          AXES
#         if g>=4: ax.set_xlabel('Time (s)', fontsize = 18)
#         if g==0 or g==4: ax.set_ylabel('$log~(Amplitude^2)$', fontsize= 18)
#         ax.set_xlim( t[coda_inds[array_nm][fband]]-10, t[-10]);
#         ax.set_ylim( -5.0, -2.0)
#         ax.tick_params( axis = 'y', labelsize= 16, length= 10, width= 1)#, right= True, labelright= True, left= False, labelleft= False)
#         ax.tick_params(axis= 'x', labelsize= 16, length= 10, width= 1)
#         #plt.gca().yaxis.set_label_position('right')
#         #plt.axis([-0, 3*tJ, -4, 0.5])
#         for side in plt.gca().spines.keys():
#             ax.spines[side].set_linewidth(3)

#     #         LEGEND
#     tit = str(fbands[fband][0]) + '-' + str(fbands[fband][1]) + '$~$Hz'

#     handles1, labels1 = ax.get_legend_handles_labels()
#     handles11 = [handles1[0], handles1[2], handles1[4]]
#     handles12 = [handles1[1], handles1[3], handles1[5]]
#     labels11 = [labels1[0], labels1[2], labels1[4]]
#     labels12 = [labels1[1], labels1[3], labels1[5]]
#     handles = handles11 + handles12
#     labels = labels11 + labels12

#     ax.legend( handles, labels, loc = 'best', title = tit, title_fontsize = 16, ncol = 2,
#               handletextpad = 0.4, columnspacing = 0.4, fontsize= 12, fancybox= True,
#               framealpha= 1)

# #f.text( 0.08, 0.4, '$log (Amplitude^2)$', rotation = 90, fontsize= 18)
# f.tight_layout( rect = [0.1, 0.01, 0.99, 0.99])

# fname = '/nfs/a9/eeinga/Results/AU/EFM/EFM_all_envs_linfits_ALL_ARRAYS_landscape'
# plt.savefig( fname + '.pdf', bbox_inches= 'tight')
# plt.savefig( fname + '.png', bbox_inches= 'tight')
# plt.close('all')


#          ---------------------------------------------------------          #

# Plot scattering Q results from the EFM and EFMD together:
num_layers = 2
scattering_layer = 'all'
EFMD_fbands = {'D': [1.5, 3], 'E': [2, 4], 'F': [2.5, 5], 'G': [3, 6], 'H': [3.5, 7]}

Qs_EFMD = {}; Qs_tot_EFMD = {}

for array in arrays:

    comb_results_fname = '/nfs/a9/eeinga/Results/AU/EFMD/' + array \
                        + '/Final_results/2layer_9M_3x3M_HighFreqs/EFMD_' \
                        + array + '_Bayesian_results_' + str(num_layers) \
                        + 'layers_model_all_layer_scattering_all_MCMCS.pckl'
    fopen = open( comb_results_fname, 'rb')
    EFMD_comb_results = pickle.load( fopen)
    fopen.close()

    # Extract Qs_EFMD from EFMD results:
    Qs_EFMD[array] = EFMD_comb_results['Qs_EFMD']
    Qs_tot_EFMD[array] = EFMD_comb_results['Qs_tot']

# Clean EFMD_comb_results from memory (they're HUGE!)
EFMD_comb_results = {}

# Turn them into lists:
Qs_EFMD_lists = {}; Qs_tot_EFMD_lists = {}
for array in arrays:
    Qs_EFMD_lists[array] = {}
    Qs_tot_EFMD_lists[array] = {}
    Qs_EFMD_lists[array]['L1'] = {}
    Qs_EFMD_lists[array]['L2'] = {}

    Qs_EFMD_lists[array]['L1']['min'] = []
    Qs_EFMD_lists[array]['L2']['min'] = []
    Qs_EFMD_lists[array]['L1']['max'] = []
    Qs_EFMD_lists[array]['L2']['max'] = []

    Qs_tot_EFMD_lists[array]['min'] = []
    Qs_tot_EFMD_lists[array]['max'] = []

    for fband in EFMD_fbands:

        Qs_EFMD_lists[array]['L1']['min'].append( Qs_EFMD[array][fband][1,0])
        Qs_EFMD_lists[array]['L2']['min'].append( Qs_EFMD[array][fband][2,0])

        Qs_EFMD_lists[array]['L1']['max'].append( Qs_EFMD[array][fband][1,1])
        Qs_EFMD_lists[array]['L2']['max'].append( Qs_EFMD[array][fband][2,1])

        Qs_tot_EFMD_lists[array]['min'].append( Qs_tot_EFMD[array][fband][0])
        Qs_tot_EFMD_lists[array]['max'].append( Qs_tot_EFMD[array][fband][1])

# Plot EFM and EFMD Qs values together (with error bars):

f, (ax1, ax2, ax3) = plt.subplots( 1, 3, sharey = 'all', figsize = (10, 10))

for array in arrays:

    if array == 'ASAR': ax = ax1
    elif array == 'WRA': ax = ax2
    elif array == 'PSA': ax = ax3

    # Plot Q factors with both lines and markers:
    ax.plot( fcs[3:], Qs_lists[array][3:], '^-', color = colors[array][1],
            markeredgecolor = 'k', markersize = 10, label = 'EFM Qs values')
    ax.errorbar( fcs[3:], Qs_lists[array][3:], yerr = Qs_std_lists[array][3:],
                fmt = 'None', ecolor = 'k', elinewidth = 2, capsize = 5)

    # Plot min and max Qs from EFMD:
    ax.fill_between( fcs[3:], Qs_tot_EFMD_lists[array]['min'],
                    Qs_tot_EFMD_lists[array]['max'], alpha = 0.7,
                    color = 'khaki', label = 'EFMD Qs range')
    # ax.plot( fcs[3:], Qs_tot_EFMD_lists[array]['min'], 'o--', color = 'k',
    #         linewidth = 2, label = 'EFMD min Qs')
    # ax.plot( fcs[3:], Qs_tot_EFMD_lists[array]['max'], 'o--', color = 'mediumpurple',
    #         linewidth = 2, label = 'EFMD max Qs')
    # ax.plot( fcs[3:], Qs_EFMD_lists[array]['L1']['min'], 'o--', color = 'k',
    #         linewidth = 2, label = 'EFMD min Qs, L1')
    # ax.plot( fcs[3:], Qs_EFMD_lists[array]['L1']['max'], 'o--', color = 'darkgray',
    #         linewidth = 2, label = 'EFMD max Qs, L1')
    # ax.plot( fcs[3:], Qs_EFMD_lists[array]['L2']['min'],'o--',  color = 'indigo',
    #         linewidth = 2, label = 'EFMD min Qs, L2')
    # ax.plot( fcs[3:], Qs_EFMD_lists[array]['L2']['max'], 'o--', color = 'mediumpurple',
    #         linewidth = 2, label = 'EFMD max Qs, L2')

    ax.grid( )
    # ax.set_yscale( 'log')
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    #ax.set_title( array, fontsize = 24)
    ax2.set_xlabel( 'Frequency (Hz)', fontsize = 18)
    ax1.set_ylabel('$Q_s$', fontsize = 18)
    ax.set_xlim( 2, 6)
    ax.set_ylim( 65, 1200)
    ax.tick_params(axis= 'both', labelsize= 16)
    ax3.legend( loc = 'lower right', fontsize = 12)

plt.subplots_adjust( wspace = .05)

# Labels at the top:
f.text( 0.205, 0.84, 'ASAR', fontsize = 24)
f.text( 0.47, 0.84, 'WRA', fontsize = 24)
f.text( 0.735, 0.84, 'PSA', fontsize = 24)


fname = '/nfs/a9/eeinga/Results/AU/EFM/EFM_vs_EFMD_Qs_ALL_ARRAYS'
plt.savefig( fname + '.pdf', bbox_inches= 'tight')
plt.savefig( fname + '.png', bbox_inches= 'tight', transparent = False)



#          ---------------------------------------------------------          #












#plt.figure( figsize = (20, 20) )
#for array in arrays:
#    for i, fband in enumerate( fbands):
#        if i == 0:
#            plt.errorbar( fcs[i], Qs[array][fband], yerr = Qs_std[array][fband], fmt = '.',
#                         color = colors[array], ecolor = colors[array], elinewidth = 2,
#                         capsize = 10, markersize = 15, label = array )
#        else:
#            plt.errorbar( fcs[i], Qs[array][fband], yerr = Qs_std[array][fband], fmt = '.',
#                         color = colors[array], ecolor = colors[array], elinewidth = 2,
#                         capsize = 10, markersize = 15 )
#
#plt.grid()
#plt.legend( loc = 'best', ncol = 3, fontsize = 24)
#plt.xlabel( 'Frequency (Hz)', fontsize = 24)
#plt.ylabel( '$Q_s$', fontsize = 24)
#plt.gca().tick_params( axis = 'both', labelsize = 24 )
#figname1 = '/nfs/a9/eeinga/Results/AU/EFM/Qs_all_arrays.png'
#plt.savefig( figname1, bbox_inches = 'tight')
#
#####################################################################################################
#
#plt.figure( figsize = (20, 20) )
#for array in arrays:
#    for i, fband in enumerate( fbands):
#        if i == 0:
#            plt.errorbar( fcs[i], Qi[array][fband], yerr = Qi_std[array][fband], fmt = '^',
#                         color = colors[array], ecolor = colors[array], elinewidth = 2,
#                         capsize = 10, markersize = 15, label = array )
#        else:
#            plt.errorbar( fcs[i], Qi[array][fband], yerr = Qi_std[array][fband], fmt = '^',
#                         color = colors[array], ecolor = colors[array], elinewidth = 2,
#                         capsize = 10, markersize = 15 )
#
#plt.grid()
#plt.legend( loc = 'best', ncol = 3, fontsize = 24)
#plt.xlabel( 'Frequency (Hz)', fontsize = 24)
#plt.ylabel( '$Q_{i}$', fontsize = 24)
#plt.gca().tick_params( axis = 'both', labelsize = 24 )
#figname1 = '/nfs/a9/eeinga/Results/AU/EFM/Qi_all_arrays.png'
#plt.savefig( figname1, bbox_inches = 'tight')
#
#####################################################################################################
#
#plt.figure( figsize = (20, 20) )
#for array in arrays:
#    for i, fband in enumerate( fbands):
#        if i == 0:
#            plt.errorbar( fcs[i], Qdiff[array][fband], yerr = Qd_std[array][fband], fmt = '*',
#                         color = colors[array], ecolor = colors[array], elinewidth = 2,
#                         capsize = 10, markersize = 15, label = array )
#        else:
#            plt.errorbar( fcs[i], Qdiff[array][fband], yerr = Qd_std[array][fband], fmt = '*',
#                         color = colors[array], ecolor = colors[array], elinewidth = 2,
#                         capsize = 10, markersize = 15 )
#
#plt.grid()
#plt.legend( loc = 'best', ncol = 3,  fontsize = 24)
#plt.xlabel( 'Frequency (Hz)', fontsize = 24)
#plt.ylabel( '$Q_{diff}$', fontsize = 24)
#plt.gca().tick_params( axis = 'both', labelsize = 24 )
#figname1 = '/nfs/a9/eeinga/Results/AU/EFM/Qd_all_arrays.png'
#plt.savefig( figname1, bbox_inches = 'tight')
#
