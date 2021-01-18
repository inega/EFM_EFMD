#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Created on Tue Mar 12 14:31:12 2019

@author: eeinga

This script takes my implementation of the EFM, EFMD and TFWM and combines them
so the whole analysis can be done together.
'''

import pickle
from glob import glob
from datetime import datetime
from vel_models import get_velocity_model
from EFM_EFMD_tools import datetime_string
from EFM import EFM_get_envelopes, EFM_analysis
from EFMD_Bayesian_Modelling import EFMD_Bayesian_modelling, EFMD_combine_results
from F_EFMD import EFMD_plot_results_summary, EFMD_plot_results_summary_simple,\
                   EFMD_plot_x_histograms

#      ***************            INITIALIZE           ******************     #

# I want to time this script:
t0=datetime.now()

# Generate datetime string so it can be used in file names (the main purpose is
# avoiding overwriting results from different runs):
datetime_str = datetime_string( datetime.now() )

# Initial settings:
network = ''
array_nm = ''

# Set data source:
data_source = ''

# Get vertical traveltimes through the model, so we can trim the seismograms to
# the time window we use for the EFM/EFMD:
units = 'm' # They're either 'm' or 'km'
vel_source = ''

# Set number of layers:
num_layers = 2
# Set number of frequency bands to use in the EFMD (the EFM requires all of them):
num_fbands = 5
# Define number of iterations of the MCMC for the EFMD inversion:
N_iters = 100

EFM1 = False     # Calculate the normalised and non-normalised coda envelopes
EFM2 = False      # Run the EFM analysis
EFMD1 = False       # Run Bayesian EFMD
syn_test = False
EFMD2 = False          # Combine/plot EFMD results from multiple chains
combine_results = False    # Combine results only (EFMD2 needs to be set to true)
plot_comb_results = False    # Plot results only (EFMD2 needs to be set to true)

# Define scattering layer:
if syn_test == True:
    scattering_layer = 'all' # Either "all" or string corresponding to layer
                             # number ('1', '2')
else:
    scattering_layer = 'all'

#                     ****************************************                #


# Get velocity model and tJ. All variables are given in SI units:
vel_model_fname = '/path/to/file/with/velocity/data/vel_model.csv'

vel_model = get_velocity_model( array_nm, vel_source, vel_model_fname, num_layers,
                       units = 'm')
v = vel_model['v']
tJ = vel_model['tJ']
layers_bottoms = vel_model['layers_bottoms']

# Get velocity model and tJ for the 1 layer case (needed for the EFM):
vmodel1 = get_velocity_model(array_nm, vel_source, vel_model_fname, 1, units = 'm')
v1 = vmodel1['v']
tJ1 = vmodel1['tJ']

# We need the inverse of the sampling rate:
fopen = open('/path/to/file/with/delta/value.pckl')
delta = pickle.load(fopen)
fopen.close()

#                  ****************************************                   #

# Full set of frequencies to be used: we ALWAYS need to use ALL of them for the
# EFM. For the EFMD, however, sometimes we'll use all of them and sometimes we
# will use only some.
fbands = {'A':[0.5,1], 'B':[0.75,1.5], 'C':[1,2], 'D':[1.5,3],
          'E':[2,4], 'F':[2.5,5], 'G':[3,6], 'H':[3.5,7]}

#                  ****************************************                   #

# Define paths and file names for data, results and figures:
EFM_sac_path = '/path/to/directory/where/the/GQ/SAC/data/live'

EFM_path = '/path/to/directory/for/EFM/Results/'
EFMD_path = '/path/to/directory/for/EFMD/Results/'
EFM_fname = EFM_path + 'EFM_' + array_nm + '_'
EFMD_fname = EFMD_path + 'EFMD_' + array_nm + '_'

# Define file names for stacked non-normalised envelopes and normalised coda
# envelopes BEFORE stacking (call them DATA_*** to avoid confusions during
# synthetic tests):
DATA_s_envs_fname = EFMD_fname + 's_envs_all_fbands.pckl'
DATA_nces0_fname = EFM_fname + 'nces_all_fbands.pckl'
DATA_s_nces_fname = EFM_fname + 's_nces_all_fbands.pckl'

# Define EFM results and figures file names:
EFM_results_fname = EFM_fname + 'final_results.pckl'
EFM_figs_fname = EFM_fname[:-1]

if syn_test == False:

    # Define dataset path and file name:
    s_nces_fname = DATA_s_nces_fname

    # Define file name for the EMFD modelling results and figures:
    EFMD_results_fname = EFMD_fname + str( num_layers ) + 'layers_' \
                        + str(N_iters) + 'iters_' + datetime_str + '.pckl'
    EFMD_figs_fname = EFMD_results_fname[:-5]

elif syn_test == True:

    if scattering_layer == 'all':

        # Define synthetic dataset path and file name:
        s_nces_fname = EFMD_path + 'EFMD_' + array_nm + '_syn_dataset_' \
                      + str( num_layers ) + 'layers_model_scatt_all_layers.pckl'

        # Define file name #
        EFMD_results_fname = EFMD_path + 'EFMD_' + array_nm \
                            + '_SYN_TEST_Bayesian_results_' + str( num_layers ) \
                            + 'layers_model_scatt_all_layers_' + str(N_iters) \
                            + 'iters_' + datetime_str + '.pckl'

        # Define figures path and name:
        EFMD_figs_fname = EFMD_results_fname[:-5]

    else:

        # Define synthetic dataset path and file name:
        s_nces_fname = EFMD_path + 'EFMD_' + array_nm + '_syn_dataset_' \
                        + str( num_layers ) + 'layers_model_scatt_layer' \
                        + scattering_layer + '.pckl'

        # Define file name for the modelling results:
        EFMD_results_fname = EFMD_path + 'EFMD_' + array_nm \
                            + '_SYN_TEST_Bayesian_results_' + str( num_layers ) \
                            + 'layers_model_L' + scattering_layer \
                            + '_layer_scattering_' + str(N_iters) + 'iters_' \
                            + datetime_str + '.pckl'

        # Define figures path and name:
        EFMD_figs_fname = EFMD_results_fname[:-5]

#                  ****************************************                   #

print(' ')
print('                         ARRAY:     '+array_nm)
print('********************************************************************* ')
print(' ')
print('Layering is: ' + str(layers_bottoms))

##########                 STEP 2: EFM                     ####################

if EFM1 == True:
    # Run part 1 of the EFM to obtain the normalised and not-normalised coda
    # envelopes required for the EFM and EFMD respectively (the results are
    # saved into files). It takes a long time to run this step of the process,
    # as it needs to get the normalised coda envelope for each event and frequency
    # band and stack them together.
    t_ini = datetime.now()
    EFM_get_envelopes( array_nm, EFM_sac_path, EFM_path, EFMD_path, fbands, tJ1)
    print('It took the script ', str(datetime.now() - t_ini), ' to run EFM part 1')

if EFM2 == True:
    # Run part 2 of the EFM to obtain correlation length, velocity fluctuations,
    # Qs^-1, Qi0, Qdiff0, L and alpha estimations. It will use a single scattering
    # layer with the same background velocity our lithosphere has.
    t_ini = datetime.now()
    EFM_results = EFM_analysis( array_nm, fbands, v1, tJ1, delta, DATA_nces0_fname,
                               DATA_s_envs_fname, EFM_results_fname,
                               EFM_figs_fname, units = 'm', syn_test = False)

    t_1 = datetime.now()
    Q_i = EFM_results [array_nm]['Qi']
    print('It took the script ', str( t_1 - t_ini), ' to run EFM part 2')

##############                STEP 3: EFMD                 ####################

# Run part the EFMD to create the models based on combinations of the structural
# parameters, then calculate synthetic envelopes for each one of them and obtain
# the structural parameters for the best fitting model.

# Redefine fbands depending on how many of them we want to include in the EFMD:
if num_fbands == 8:
    EFMD_fbands = {'A':[0.5,1], 'B':[0.75,1.5], 'C':[1,2], 'D':[1.5,3],
                   'E':[2,4], 'F':[2.5,5], 'G':[3,6], 'H':[3.5,7]}
else:
    EFMD_fbands = {'D':[1.5,3], 'E':[2,4], 'F':[2.5,5], 'G':[3,6], 'H':[3.5,7]}

if EFMD1 == True:

    # EFMD Part 1:
    EFMD_results = EFMD_Bayesian_modelling ( array_nm, EFMD_fbands,
                                            units, delta, Q_i, N_iters,
                                            scattering_layer, vel_model,
                                            datetime_str, s_nces_fname,
                                            DATA_s_envs_fname,
                                            EFMD_results_fname,
                                            EFMD_figs_fname,
                                            syn_test = syn_test,
                                            showplots = False )
    t_2 = datetime.now()
    print('It took the script ', str(t_2 - t_1), ' to run EFMD part 1')

###############################################################################

# Part 2 of the EFMD consists on combining the results from the multiple MCMCs
# we run into a single set of results:

if EFMD2 == True:

    print( '---------------------------------------------------------------')
    print('')

    # Define time strings of the results we want to combine ( all results files
    # are called the same except for the time stamp):

    if syn_test == True:

        if num_layers == 1:
            subdir = 'SYN_TEST_1layer_Scatt_All_Layers_3M_3x1M_AllFreqs'

        if num_layers == 2  and scattering_layer == 'all':
            subdir = 'SYN_TEST_2layer_Scatt_All_Layers_9M_3x3M_AllFreqs'

        if num_layers == 2 and scattering_layer == '1':
            subdir = 'SYN_TEST_2layer_Scatt_L1_9M_3x3M_AllFreqs'

        if num_layers == 2 and scattering_layer == '2':
            subdir = 'SYN_TEST_2layer_Scatt_L2_9M_3x3M_AllFreqs'

        if num_layers == 3:
            subdir = 'SYN_TEST_3layer_Scatt_All_Layers_15M_3x5M_AllFreqs'

    if syn_test == False and len(EFMD_fbands) == 8:

        if num_layers == 1: subdir = '1layer_3M_3x1M_AllFreqs'

        if num_layers == 2: subdir = '2layer_9M_3x3M_AllFreqs'

    if syn_test == False and len(EFMD_fbands) == 5:

        if num_layers == 2: subdir = '2layer_9M_3x3M_HighFreqs'

        if num_layers == 3: subdir = '3layer_15M_3x5M_HighFreqs'

    # Get results filenames:
    path = EFMD_path + 'Final_results/' + subdir + '/'
    results_fnames = glob( path + '*.pckl')

    # Define parameter to be used to define the RM:
    parameter = 'Mode'

    # Define results path and filename:
    if syn_test == False:
        comb_results_fname = path + 'EFMD_' + array_nm \
                            + '_Bayesian_results_' \
                            + str( num_layers ) + 'layers_model_' \
                            + scattering_layer + '_layer_scattering_'
    elif syn_test == True:
        comb_results_fname = path + 'EFMD_' + array_nm \
                            + '_SYN_TEST_Bayesian_results_' + str( num_layers ) \
                            + 'layers_model_' + scattering_layer \
                            + '_layer_scattering_'

    # Sanity check in case we run this part of the code more than once:
    for res_fname in results_fnames:
        if comb_results_fname in res_fname:
            # print( 'File from previously combined results removed')
            # print('')
            results_fnames.remove( res_fname)

    # Define figures filename:
    figs_fname = comb_results_fname + parameter + '_'

    # Print summery of the case we're combining data for:
    print('')
    print( array_nm + ', ' + str( num_layers )
            + ' layers')
    print( 'Synthetic test: ' + str(syn_test))
    print( 'Scattering layer: ' + scattering_layer )
    print('')
    print( 'Frequency bands: ')
    print( EFMD_fbands)
    print('')

    # Combine results:
    if combine_results == True:

        print( '---------------------------------------------------------------')
        print('')
        print( 'Combining results from EFMD part 2...')

        EFMD_comb_results = EFMD_combine_results( array_nm, EFMD_fbands, units,
                                                 delta, Q_i, scattering_layer,
                                                 vel_model, parameter,
                                                 EFMD_path, s_nces_fname,
                                                 DATA_s_envs_fname,
                                                 results_fnames,
                                                 comb_results_fname,
                                                 syn_test = syn_test)

        print( 'Results sucessfully combined!')

    if plot_comb_results == True:

        print( '')
        print( 'Plotting combined results...')

        # If we just combined the EFMD results, we want to use the EFMD_comb_results.
        # However, if we are just repeating the figures and don't want to combine
        # the results again, we will need to load those results to re-create the
        # figures.
        if combine_results == True:
            EFMD_comb_results = EFMD_comb_results
        else:
            fopen = open( comb_results_fname + 'all_MCMCS.pckl', 'rb')
            EFMD_comb_results = pickle.load( fopen)
            fopen.close()

        EFMD_plot_results_summary ( array_nm, EFMD_fbands, units, vel_model,
                                    EFMD_comb_results, figs_fname, scattering_layer,
                                    parameter, syn_test = syn_test, comb_results = True,
                                    showplots = False)

        EFMD_plot_results_summary_simple ( array_nm, EFMD_fbands, units,
                                          vel_model, EFMD_comb_results, figs_fname,
                                          scattering_layer, parameter, syn_test = syn_test,
                                          comb_results = True, showplots = False)

        if num_layers != 1:
            EFMD_plot_x_histograms ( array_nm, s_nces_fname, EFMD_comb_results,
                                    EFMD_fbands, figs_fname, syn_test = syn_test,
                                    comb_results = True, showplots = False)

#                ********************************************                            #

print('Total time it took this script to run is ' + str(datetime.now() - t0))







