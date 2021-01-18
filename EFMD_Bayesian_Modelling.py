'''
First created on Wed Mar 27 2019
@author: Itahisa Gonzalez Alvarez

    This script contains the functions with my implementation of the Bayesian
    EFMD approach. The first one runs a single MCMC for an array, saves the
    results into a file and plots them. The second function combines the results
    from multiple chains and plots and saves the results.

'''

import copy
import pickle
import numpy as np
from F_EFMD import *
from datetime import datetime


def EFMD_Bayesian_modelling ( array_nm, fbands, units, delta, Q_i, N_iters,
                             scattering_layer, vel_model, datetime_str,
                             s_nces_fname, s_envs_fname, results_fname,
                             figs_fname, syn_test = False, showplots = False):

    '''
    This function performs a Bayesian inversion of normalised coda envelopes
    according to the EFMD (Korn, 1997). Data can be either the stacked and
    normalised coda envelopes computed by the EFM_get_envelopes function or
    some form of synthetic data with similar processing.
    The processing steps it carries out are, for each frequency band:
        1- Get the normalised and stacked coda envelopes for each event and
           fband from the EFM results or synthetic data (dictionaries saved in
           pickle files with the same format).
        2- Define correlation length (a) and RMS velocity fluctuations (E)
           minimum and maximum accepted values, as well as the step sizes for
           the inversion.
        3- Create an initial model as a combination of random values of a and E
           for each layer.
        4- Initialize and start Markov Chain Monte Carlo, in which models are
           updated randomly and their likelihoods are calculated by comparing
           their respective synthetic envelopes with input data. Some steps
           carried out within the MCMC are:
               4a- Calculate Qs for each layer in the model
               4b- Calculate direct wave spectral energy density
               4c- Solve system of ODEs to get coda spectral energy density
                   for each layer
               4d- Calculate synthetic normalised coda envelope
               4e- Accept/Reject model based on its likelihood
               4f- Update model parameters and repeat
    Then, it calculates the "representative" (RM) and maximum likelihood  (MLM)
    models, as well as the 5-95 percentile range for each parameter and layer,
    which can all be used to explore the results. The RM is built using the
    mode of each parameter and within each layer, while the MLM is the model
    with the highest likelihood found during the inversion. Finally, the
    function saves the results into a pickle file and plots the results using
    the EFMD_plot_results function.

    Arguments:
        - array_nm: (str) name of the seismic array.
        - fbands: (dict) Frequencies to be used in the analysis. Keys are names
                  assigned to each fband and each element is a list with
                  freqmin and freqmax for each frequency band (in Hz) of interest.
                  (Example: fband = {'A': [0.5,1], 'B': [1,2]}).
        - units: (str) Units for distances and velocities (either 'm' or 'km').
        - delta: (float) Inverse of the sampling rate of the data.
        - Q_i: (dict) Dictionary with intrinsic quality factor Q_i values for
                all fbands. Keys should be the same from fbands dictionary.
        - N_iters: (int) Number of iterations in the MCMC.
        - scattering_layer: (str) Either 'all' or layer number ('1', '2'), to
                            indicate the layer(s) in the model that contain
                            significant scattering.
        - vel_model: (dict) Characteristics of the velocity model to be used in
                     the EFMD analysis. For each layer, it contains thickness,
                     mean P wave velocity, traveltimes through each layer,
                     cumulative traveltimes from the bottom of the model to the
                     top of each layer, number of layers in the model, velocities
                     This dictionary is created by the get_velocity_model
                     function from vel_models.
        - datetime_str: (str) String from time stamp representing starting
                        moment of the MCMC that identifies files associated
                        with it.
        - s_nces_fname: (str) Path and file name of dataset consisting on
                        normalised coda envelopes and their standard deviation.
                        For synthetic tests, the input model is also included.
                        File created in part 1 of the EFM.
        - s_envs_fname: (str) Path and file name of dataset consisting on
                        NON-normalised envelopes and their standard deviation.
                        File created in part 1 of the EFM.
        - results_fname: (str) Path and file name where we want to store the
                         results of the modelling.
        - fig_name: (str) Path and common part of figures' file name.
        - syn_test: (bool) True for synthetic data inversion, false otherwise.
        - showplots: (bool, optional) True for showing the results plots after
                     the code is done (default is False, figures are saved to
                     files but closed immediately).

    Output:
        - A pickle file is saved at the end of the analysis, containing all
         accepted models, their likelihoods, histogram matrices representing
         the synthetic envelopes of all accepted models at all fbands, the time
         it took each iteration to run, the number of unexpected errors, the
         percentage of iterations completed at the time of saving the file, a
         and E values for each layer both in units/decimal form and km/percentage,
         total number of times a and E were accepted, total number of times a
         and E were updated, total acceptance rate, total number of iterations,
         structural parameters of the best fitting and minimum loglikelihood
         models, their synthetic envelopes at all frequencies and their
         loglikelihoods.

        - Plots produced and saved into files by the EFMD_plot_syn_test_results
          function:
            * Histograms and values of the likelihoods for kept models.
            * Histograms of the values of the structural parameters for all
              layers in the model.
            * 2D Histograms of the values of the structural parameters for all
              layers in the model.
            * Data normalised coda envelopes vs. synthetic envelopes for all
              accepted models.

            Figures are saved both in .png and .pdf formats.

    '''

    ###########################################################################
    #                          EFMD  PART 1                                   #
    ###########################################################################

    # Use the EFMD_preprocess_data function to load and preprocess data and
    # velocity information: data is downsampled in this step to 10 sps and the
    # new sampling rate is returned with the rest of the results.
    if syn_test == True:
        num_layers, N, L, tjs, dtjs, \
        v, tJ, vJ, thicks, lambda_min, \
        lambda_max, input_model, \
        s_nces, data_envs, s_envs, \
        maxamp_inds, DW_end_inds, cfs, \
        inv_Qi_vals, inv_Qds, delta, \
        s_nces_dataset, envs_dataset = EFMD_preprocess_data( array_nm,
                                                            fbands, s_nces_fname,
                                                            s_envs_fname,
                                                            vel_model, delta,
                                                            Q_i, resample = True,
                                                            syn_test = syn_test)
        # Input model string:
        input_model_str = ''
        for i in range( num_layers - 1, N, 1):
            input_model_str += '[ ' + str( input_model[i][0]/1000 ) + ', ' \
                                + str( 100*input_model[i][1]) + ' ]'

    else:
        num_layers, N, L, tjs, dtjs, \
        v, tJ, vJ, thicks, lambda_min, \
        lambda_max, s_nces, data_envs, \
        s_envs, maxamp_inds, DW_end_inds, \
        cfs, inv_Qi_vals, inv_Qds, delta, \
        s_nces_dataset, envs_dataset = EFMD_preprocess_data( array_nm,
                                                            fbands, s_nces_fname,
                                                            s_envs_fname,
                                                            vel_model, delta,
                                                            Q_i, resample = True,
                                                            syn_test = syn_test)

    ###########################################################################
    ###########################################################################

    print(' ')
    print('******************************************************************')
    print(' ')
    print('                Running EFMD, ' + str( num_layers ) + ' layers')
    print(' ')
    print('Fbands are: ' + str(fbands))
    print(' ')
    if syn_test == True:
        print('     Synthetic test. Scattering_layer(s): ' + scattering_layer )
        print(' ')
        print( 'Input model is: ')
        print( input_model_str )
        print(' ')
    print('******************************************************************')
    print(' ')
    print('                      EFMD modelling                              ')
    print('******************************************************************')
    print(' ')

    ###########################################################################
    ###########################################################################

    # Define model parameters step size and initial value. They will be different
    # depending on whether we are inverting real data or doing a synthetic test.
    # For the correlation length, both initial value and step size will depend
    # on the units being used.

    if syn_test == True:
        if num_layers == 1:
            a_step_size = 5.0;     E_step_size = 0.0001
        elif num_layers == 2:
            if scattering_layer == 'all':
                a_step_size = 10.0;    E_step_size = 0.0001
            elif scattering_layer == '1':
                a_step_size = 50.0;    E_step_size = 0.00025
            elif scattering_layer == '2':
                a_step_size = 50.0;    E_step_size = 0.0001
        elif num_layers == 3:
            a_step_size = 250.0;     E_step_size = 0.001

        if units == 'km':
            a_step_size = a_step_size/1000

    else:
        if num_layers == 1:
            if array_nm == 'PSA':
                a_step_size = 10000.0;     E_step_size = 0.00001
            elif array_nm == 'ASAR':
                a_step_size = 10000.0;   E_step_size = 0.001
            else:
                a_step_size = 10000.0; E_step_size = 0.00001

        elif num_layers == 2:
            if len(fbands) == 8:
                if array_nm == 'PSA':
                    a_step_size = 15000.0;   E_step_size = 0.0001

            elif len(fbands) == 5:
                if array_nm == 'PSA':
                    a_step_size = 800.0;    E_step_size = 0.006
                elif array_nm == 'ASAR':
                    a_step_size = 2500.0;    E_step_size = 0.008
                elif array_nm == 'WRA':
                    a_step_size = 1000.0;    E_step_size = 0.006
                else:
                    a_step_size = 800.0;    E_step_size = 0.006

        elif num_layers == 3:
            if array_nm == 'PSA':
                a_step_size = 1900.0;   E_step_size = 0.007
            if array_nm == 'ASAR':
                a_step_size = 1500.0;   E_step_size = 0.008
            if array_nm == 'WRA':
                a_step_size = 2000.0;   E_step_size = 0.01



    # Define minimum and maximum allowed values of the parameters
    min_a = lambda_min / 5; max_a = lambda_max * 2
    min_E = np.exp( -10 ); max_E = 0.1

    # Create initial model as combinations of random values of a and E
    # (correlation length and RMS velocity fluctuations): each model has a
    # number of layers that are duplicated (have the exact same structural
    # parameters values) on the other side of the free surface to take into
    # account that the direct wave is totally reflected at the free surface.

    # Randomly initialize the models (starting model for the MCMC):
    initial_model = np.zeros((N, 2))
    for i in range( num_layers ):
        if i == num_layers - 1:
            initial_model[i][0] = np.random.uniform( min_a[i], max_a[i], 1)[0]
            initial_model[i][1] = np.random.uniform( min_E, max_E, 1 )[0]
        else:
            initial_model[i][0] = np.random.uniform( min_a[i], max_a[i], 1)[0]
            initial_model[i][1] = np.random.uniform( min_E, max_E, 1 )[0]
            initial_model[-(1+i)][0] = initial_model[i][0]
            initial_model[-(1+i)][1] = initial_model[i][1]

    # Initialize current_model:
    current_model = np.zeros( (N, 2) )

    print('Initial model is:')
    print( initial_model )
    print('')
    print('Correlation length step size is ' + str(a_step_size) \
          + ' metres, and minimum and maximum accepted values \nin each layer are ' \
          + str( np.round( min_a, 3)) + ' and ' + str( np.round( max_a, 3)) \
          + ' metres.')
    print('RMS velocity fluctuations step size is ' + str(E_step_size*100) \
          + '%, and minimum and maximum accepted values in each layer are ' \
          + str( np.round (min_E*100, 3)) + '% and ' + str(max_E*100) + '%.')
    print(' ')

    ###########################################################################

    # Print number of iterations and initialise stuff for the Markov Chain
    # (random walk):
    print('Number of iterations for the Markov Chain is ' + str(N_iters))
    print(' ')

    #    *************      PREPARE MCMC LOOP         *********************                   #

    # Predefine list that will contain kept models:
    kept_models = []; kept_logliks = []; kept_syn_envs = []

    # Initialize number of rejections:
    rejections = 0
    good_accepts = 0
    rand_accepts = 0
    a_updates = 0
    E_updates = 0
    a_accepts = 0
    E_accepts = 0
    times = []
    model_update_times = []
    L_ratios = []
    unexpected_errors = []

    # Create matrices to store 2D histograms of the synthetic envelopes:
    nrows = 1000
    hist_range = np.linspace( -0.02, 4.0, nrows )
    hists_2D_syn_envs = {}

    for fband in fbands:
        hists_2D_syn_envs[fband] = np.zeros(( nrows, len( s_nces[fband])))

    # Choose percentages at which we want a summary of the results printed to
    # screen and the results saved into a file:
    percs = [15, 30, 45, 60, 75, 90, 99]

    kept_results = []

    for k in range(N_iters):

        # Print summary of the results when reaching percentages of iterations
        # in percs:
        if ( 100*k / N_iters) in percs:
            # Save dictionary with key results:
            EFMD_results = {'kept_models': kept_models,
                            'kept_logliks': np.array(kept_logliks),
                            'kept_results': kept_results,
                            'kept_syn_envs': kept_syn_envs,
                            'hists_2D_syn_envs': hists_2D_syn_envs,
                            'a_accepts': a_accepts,
                            'a_updates': a_updates,
                            'E_accepts': E_accepts,
                            'E_updates': E_updates,
                            'times': np.array(times),
                            'Unexpected_errors': unexpected_errors,
                            'perc_iters_completed': (100 * k / N_iters)}
            h = open( results_fname, 'wb')
            pickle.dump(EFMD_results, h)
            h.close()
            EFMD_modelling_summary( k, N_iters, num_layers, units, kept_models,
                                   kept_logliks, kept_syn_envs, times, rejections,
                                   good_accepts, rand_accepts, a_accepts,
                                   a_updates, E_accepts, E_updates )

        t1 = datetime.now()

        # Define current_model by randomly updating structural parameters ONE
        # PARAMETER AT A TIME. CAREFUL! The models need to be symmetric with
        # respect to the free surface!.
        if k == 0:
            current_model = initial_model.copy()

        elif k != 0:

            # Update model parameters and get index pointing to which parameter
            # was updated:
            current_model, param = EFMD_model_update( model, num_layers,
                                                     a_step_size, E_step_size,
                                                     min_a, max_a, min_E, max_E)
            if param == 0: a_updates += 1
            elif param == 1: E_updates += 1

            tdiff0 = ( datetime.now() - t1 )
            model_update_times.append( tdiff0.seconds + (tdiff0.microseconds / 1e6) )


        try:

            # Predefine liks list to save loglikelihoods for all frequency bands:
            logliks = []

            # Predefine dictionary that will contain the synthetic envelopes
            # for each frequency band:
            syn_envs = {}

            # Iterate over all frequency bands:
            for fband in fbands:

                # Get indices for the maximum amplitude of the data normalised
                # coda envelope for this frequency band and the end of the
                # direct wave time window:
                maxamp_ind = maxamp_inds[fband]
                DW_end_ind = DW_end_inds[fband]

                # Get central frequency and intrinsic quality factor for this
                # frequency band:
                cf = cfs[fband]
                inv_Qi_val = inv_Qi_vals[fband]

                # Get stacked not-normalised coda envelope for this frequency
                # band:
                s_env = s_envs[fband]

                ###############################################################

                # Calculate synthetic envelope:
                syn_env_results = get_synthetic_envelope( current_model, s_env,
                                                          delta, tjs, dtjs, cf,
                                                          N, v, tJ, vJ,
                                                          inv_Qi_val, maxamp_ind)
                syn_env = syn_env_results['syn_env']
                syn_envs[fband] = syn_env

                ###############################################################

                # Get data, syn_data and data_vars to be used for the calculation
                # of the likelihood (we only use the coda for this, since the
                # synthetic envelope does NOT reproduce the direct wave arrival):
                data = data_envs[fband]
                syn_data = syn_env[DW_end_ind:-150]

                # Get inverse variance-covariance matrix:
                inv_Qd = inv_Qds[fband]

                # Calculate loglikelihood for this frequency band:
                loglik = get_loglik ( data, syn_data, inv_Qd )

                # Add loglikelihood to list to average over all fbands:
                logliks.append( loglik )

            # Calculate the mean loglikelihood for this model:
            current_loglik = np.mean( np.array( logliks ))

            # Calculate the L ratio:
            if k == 0:
                model = current_model.copy()
                model_loglik = current_loglik.copy()

            else:
                # We need to minimise the loglikelihood in order to obtain the
                # maximum likelihood.
                L_ratio = np.exp( model_loglik - current_loglik )
                L_ratios.append(L_ratio)

                if current_loglik <= model_loglik:
                    model = current_model.copy()
                    model_loglik = current_loglik.copy()
                    kept_models.append( model )
                    kept_logliks.append( model_loglik )
                    #kept_syn_envs.append( syn_envs )
                    # Add synthetic envelopes to 2D histograms matrices:
                    for fband in fbands:
                        hists_2D_syn_envs[fband] = update_hists_2D_syn_envs( hists_2D_syn_envs[fband],
                                                                             syn_envs[fband],
                                                                             hist_range )
                    good_accepts = good_accepts + 1
                    kept_results.append( [model, model_loglik] )
                    if param == 0: a_accepts += 1
                    elif param == 1: E_accepts += 1

                elif L_ratio > np.random.uniform(0, 1, 1):
                    model = current_model.copy()
                    model_loglik = current_loglik.copy()
                    kept_models.append( model )
                    kept_logliks.append( model_loglik )
                    #kept_syn_envs.append( syn_envs )
                    # Add synthetic envelopes to 2D histograms matrices:
                    for fband in fbands:
                        hists_2D_syn_envs[fband] = update_hists_2D_syn_envs( hists_2D_syn_envs[fband],
                                                                             syn_envs[fband],
                                                                             hist_range )
                    rand_accepts = rand_accepts + 1
                    kept_results.append( [model, model_loglik] )
                    if param == 0: a_accepts += 1
                    elif param == 1: E_accepts += 1

                else:
                    rejections = rejections + 1

        except:
            unexpected_errors.append( k )
            print('Weird error!')

        #print( 'Loop ' + str(k) + ' took ' + str( datetime.now() - t1) + ' seconds')
        tdiff = ( datetime.now() - t1 )
        times.append( tdiff.seconds + (tdiff.microseconds / 1e6) )


    # *********************************************************************** #
    #                           END OF LOOP                                   #
    # *********************************************************************** #

    print(' ')
    print('   *************    END OF THE MARKOV CHAIN       *************   ')
    print(' ')
    print('Analysing results... ')
    print('Number of kept models is ' + str(len(kept_models)))

    print('Number of good models accepted is ' + str(good_accepts))
    print('Number of random models accepted is ' + str(rand_accepts))

    # Print ACCEPTANCE rate:
    accepts_rate = np.round( ( (N_iters - rejections) / N_iters ) * 100, 4)
    if accepts_rate > 60:
        print('Acceptance rate is ' + str(accepts_rate) + '%, too high!')
    elif accepts_rate < 30 :
        print('Acceptance rate is ' + str(accepts_rate) + '%, too low!')
    else: print('Acceptance rate is ' + str(accepts_rate) + '%')

    a_accepts_rate = np.round( ( a_accepts / a_updates ) * 100, 2)
    E_accepts_rate = np.round( ( E_accepts / E_updates ) * 100, 2)
    print( 'Correlation length acceptance rate is ' + str( a_accepts_rate ) + '%')
    print( 'Velocity fluctuations acceptance rate is ' + str( E_accepts_rate ) + '%')
    print(' ')
    print('Average time per iteration was = ' + str( np.mean(times)) )
    print(' ')

    # Get total number of accepted models:
    num_models = len( kept_models )

    # Save structural parameters for each layer from kept models (we are only
    # interested in the first half of our model). I also create a_km and E_perc
    # to save a values in km and E values in percentage, so they are easier to
    # interpret in histograms and scatter plots.
    a = {}; a_km = {}; E = {}; E_perc = {}; layer_labels = []

    for i in range(num_layers):
        layer_labels.append( 'L' + str(i+1) )
    for key in layer_labels:
        a[key] = []
        a_km[key] = []
        E[key] = []
        E_perc[key] = []
    for mod in kept_models:
        for i, key in enumerate( layer_labels ):
            a[key].append( mod[num_layers - 1 + i][0] )
            if units == 'm': a_km[key].append( mod[num_layers - 1 + i][0] / 1000)
            elif units == 'km': a_km[key] = copy.deepcopy( a[key] )
            E[key].append( mod[num_layers - 1 + i][1] )
            E_perc[key].append( mod[num_layers - 1 + i][1]*100 )
    for key in layer_labels:
        a[key] = np.array( a[key] )
        a_km[key] = np.array( a_km[key] )
        E[key] = np.array( E[key] )
        E_perc[key] = np.array( E_perc[key] )

    ###########################################################################
    #          *****************************************************          #
    ###########################################################################

    # Get Representative Model, Minimum Loglikelihood Model, their loglikelihoods
    # and synthetic envelopes:
    parameter = 'Mode'
    RM, RM_loglik, RM_syn_envs, RM_percs, \
    MLM, MLM_loglik, MLM_syn_envs = EFMD_get_RM_MLM ( array_nm, fbands, data_envs,
                                                     s_envs, delta, vel_model,
                                                     kept_models, kept_logliks,
                                                     a, E, maxamp_inds, DW_end_inds,
                                                     cfs, inv_Qi_vals, inv_Qds,
                                                     parameter = parameter)

    # RM string:
    RM_str = ''
    for i in range( num_layers - 1, N, 1):
        RM_str += '[ ' + str( np.round( RM[i][0]/1000, 3) ) + ', ' \
                  + str( np.round( 100 * RM[i][1], 3)) + ' ]'

    # RM string:
    RM_percs_str = ''
    for i in range( num_layers - 1, N, 1):
        RM_percs_str += '[ ' + str( np.round( RM_percs[i][0]/1000, 3) ) + ', ' \
                        + str( np.round( 100 * RM_percs[i][1], 3)) + ' ]'

    # MLM model string:
    MLM_str = ''
    for i in range( num_layers - 1, N, 1):
        MLM_str += '[ ' + str( np.round( MLM[i][0]/1000, 3) ) + ', ' \
                    + str( np.round( 100*MLM[i][1], 3)) + ' ]'

    # Print input model, best_fitting_model and misfit to screen:
    if syn_test == True:
        print('Input model was: ')
        print( input_model_str)
    print(' ')
    print( 'Representative Model is:')
    print( RM_str )
    print( 'RM parameters 95% Confidence Interval is:')
    print( RM_percs_str )
    print(' ')
    print( 'RM loglikelihood is ' + str( RM_loglik ))
    print(' ')
    print('Minimum loglikelihood model is:')
    print( MLM_str )
    print( 'MLM loglikelihood after recalculating synthetic envelopes is ' \
          + str( MLM_loglik ))
    print(' ')

    ###########################################################################
    #          *****************************************************          #
    ###########################################################################

    # Create dictionary with key results:
    EFMD_results = {'kept_models': kept_models,
                    'kept_logliks': np.array(kept_logliks),
                    'kept_results': kept_results,
                    'kept_syn_envs': kept_syn_envs,
                    'hists_2D_syn_envs': hists_2D_syn_envs,
                    'hists_2D_range': hist_range,
                    'times': np.array(times),
                    'Unexpected_errors': unexpected_errors,
                    'perc_iters_completed': (100 * (k+1) / N_iters),
                    'a_step_size': a_step_size,
                    'E_step_size': E_step_size,
                    'a': a,
                    'a_km': a_km,
                    'E': E,
                    'E_perc': E_perc,
                    'a_accepts': a_accepts,
                    'a_updates': a_updates,
                    'E_accepts': E_accepts,
                    'E_updates': E_updates,
                    'total_accepts_rate': accepts_rate,
                    'a_accepts_rate': a_accepts_rate,
                    'E_accepts_rate': E_accepts_rate,
                    'N_iters': N_iters,
                    'RM_syn_envs': RM_syn_envs,
                    'RM_structural_params': RM,
                    'RM_loglik': RM_loglik,
                    'RM_percs': RM_percs,
                    'MLM_syn_envs': MLM_syn_envs,
                    'MLM_structural_params': MLM,
                    'MLM_loglik': MLM_loglik,
                    'delta': delta}

    # Save the results into a file:
    h = open( results_fname,'wb')
    pickle.dump(EFMD_results, h)
    h.close()

    ###########################################################################
    #          *****************************************************          #
    ###########################################################################

    # Plot results:
    if scattering_layer == 'all': scat_layer_label = ''
    else: scat_layer_label = scat_layer_label = 'L' + str(scattering_layer) \
                             + '_scattering'

    EFMD_plot_results( array_nm, fbands, units, vel_model, s_nces_dataset,
                      DW_end_inds, delta, EFMD_results, figs_fname,
                      scat_layer_label, parameter, syn_test = syn_test,
                      comb_results = False, showplots = False)

    return EFMD_results

###############################################################################
#          *                *                  *                    *         #
###############################################################################
#                                                                             #
#                       END OF PART 1 OF THE EFMD                             #
#                                                                             #
###############################################################################
#          *                *                  *                    *         #
###############################################################################




def EFMD_combine_results ( array_nm, fbands, units, delta, Q_i, scattering_layer,
                          vel_model, parameter, EFMD_path, s_nces_fname,
                          s_envs_fname, results_fnames, comb_results_fname,
                          syn_test = False):

    '''
    This function takes the EFMD results from multiple MCMCs and combines them
    into a single set of results. Then, it plots these results in different ways
    so we can analyse them.

    Arguments:
        - array_nm: (str) name of the seismic array.
        - fbands: (dict) Frequencies to be used in the analysis. Keys are names
                  assigned to each fband and each element is a list with
                  freqmin and freqmax for each frequency band (in Hz) of interest.
                  (Example: fband = {'A': [0.5,1], 'B': [1,2]}).
        - units: (str) Units for distances and velocities (either 'm' or 'km').
        - delta: (float) Inverse of the sampling rate for the data.
        - Q_i: (dict) Dictionary with intrinsic quality factor Q_i values for
                all fbands. Keys should be the same from fbands dictionary.
        - scattering_layer: (str) Either 'all' or layer number ('1', '2'), to
                            indicate the layer(s) in the model that contain
                            significant scattering.
        - vel_model: (dict) Characteristics of the velocity model to be used in
                     the EFMD analysis. For each layer, it contains thickness
                     and mean P wave velocity. Traveltimes through each layer,
                     cumulative traveltimes from the bottom of the model to the
                     top of each layer, number of layers in the model, velocities
                     for the TFWM and EFM are also included. Dictionary created
                     by the get_velocity_model function from F_V_models.
        - parameter: (str) parameter to be used to obtain the RM, options are
                     mean, mode and median.
        - EFMD_path: (str) Path to parent directory containing all EFMD results.
        - s_nces_fname: (str) Path and file name of dataset consisting on
                        normalised coda envelopes and their standard deviation.
                        For synthetic tests, the input model is also included.
        - s_envs_fname: (str) Path and file name of dataset consisting on
                        NON-normalised envelopes and their standard deviation.
        - results_fname: (str) Path and file name where we want to store the
                         results of the modelling.
        - comb_results_fname: (str) Path to directory used to store combined
                              EFMD results.
        - syn_test: (bool) Indication of whether real data is being inverted or
                    a synthetic test is being conducted.

    Return:

        - Pickle file with the most relevant results:
            - all accepted models
            - all accepted models' loglikelihoods
            - total number of iterations
            - all tested values of a for each layer in "units"
            - all tested values of a for each layer in km
            - all tested values of E for each layer in decimal form
            - all tested values of E for each layer in percentage form
            - total number of times a was updated
            - total number of times the proposed change in a was accepted
            - total number of times E was updated
            - total number of times the proposed change in E was updated
            - acceptance rates for each MCMC
            - 2D histograms of the synthetic envelopes from all accepted models
            - y axis for 2D histograms

    '''

    #                      ****   LOAD AND PREPROCESS DATA   ****                                  #

    # Use the EFMD_preprocess_data function to load and preprocess data and
    # velocity information:
    # data is downsampled in this step to 10 sps and the new sampling rate is
    # returned with the rest of the results.
    if syn_test == True:
        num_layers, N, L, tjs, \
        dtjs, v, tJ, vJ, thicks, \
        lambda_min, lambda_max, \
        input_model, s_nces, \
        data_envs, s_envs, \
        maxamp_inds, DW_end_inds, \
        cfs, inv_Qi_vals, \
        inv_Qds, delta, \
        resampled_s_nces_dataset, \
        resampled_s_envs_dataset = EFMD_preprocess_data( array_nm, fbands,
                                                        s_nces_fname, s_envs_fname,
                                                        vel_model, delta, Q_i,
                                                        syn_test = syn_test)
        # Input model string:
        input_model_str = ''
        for i in range( num_layers - 1, N, 1):
            input_model_str += '[ ' + str( input_model[i][0]/1000 ) + ', ' \
                               + str( 100*input_model[i][1]) + ' ]'

    else:
        num_layers, N, L, tjs, \
        dtjs, v, tJ, vJ, thicks, \
        lambda_min, lambda_max, \
        s_nces, data_envs, s_envs, \
        maxamp_inds, DW_end_inds, \
        cfs, inv_Qi_vals, \
        inv_Qds, delta, \
        resampled_s_nces_dataset, \
        resampled_s_envs_dataset = EFMD_preprocess_data( array_nm, fbands,
                                                        s_nces_fname, s_envs_fname,
                                                        vel_model, delta, Q_i,
                                                        syn_test = syn_test)

    #                      ****   LOAD MODELLING RESULTS   ****                                    #

    # Create lists that will contain results from the MCMCs:
    models = []; logliks = []; logliks_all = []; hists_2D_all = {}; hist_ranges = []
    for fband in fbands:
        hists_2D_all[fband] = []

    # a, a_km, E and E_perc need to be dictionaries:
    a_all = {}; a_km_all = {}; E_all = {}; E_perc_all = {}; layer_labels = []
    for i in range(num_layers):
        layer_labels.append( 'L' + str(i+1) )
    for key in layer_labels:
        a_all[key] = []
        a_km_all[key] = []
        E_all[key] = []
        E_perc_all[key] = []

    total_N_iters = 0; num_models = 0; total_accepts = []
    total_a_accepts = 0; total_a_updates = 0; total_E_accepts = 0; total_E_updates = 0

    # Pre-define list to keep burn-in lengths:
    burnin_lengths = []

    # Load results files:
    for w, res_fname in enumerate( results_fnames):

        fopen = open( res_fname, 'rb')
        EFMD_res = pickle.load( fopen )
        fopen.close()

        # Extract most relevant results and add these results to new lists:
        # REMOVE MODELS CORRESPONDING TO THE BURN-IN PHASE! I won't use the
        # first 10k models from each chain.

        # Calculate the mean value of the loglikelihood in the second part of
        # the MCMC:
        kept_logliks = EFMD_res['kept_logliks']
        len_logliks = len( kept_logliks)
        mean_loglik = np.mean( kept_logliks [ int(len_logliks/2) :] )
        # Define loglik value that will establish the end of the burn in phase
        # (it will finish when that value is reached for the first time).
        burnin_loglik = mean_loglik + mean_loglik * 0.05
        for i, val in enumerate( kept_logliks ):
            if val < burnin_loglik:
                print('Burn in length for MCMC number ' + str(w) + ' is ' + str(i))
                burnin_length = i
                burnin_lengths.append( i )
                break

        kept_models = EFMD_res['kept_models']
        num_models += (len( kept_models ) - burnin_length)
        for i in range( burnin_length, len( kept_models )):
            models.append( kept_models[i] )
            logliks.append( kept_logliks[i] )

        for fband in fbands:
            hists_2D_all[fband].append( EFMD_res['hists_2D_syn_envs'][fband] )
        a = EFMD_res['a']; a_km = EFMD_res['a_km']
        E = EFMD_res['E']; E_perc = EFMD_res['E_perc']
        for key in layer_labels:
            for r1, val1 in enumerate( a[key] ):
                if r1 >= burnin_length: a_all[key].append( val1 )
            for r2, val2 in enumerate( a_km[key] ):
                if r2 >= burnin_length: a_km_all[key].append( val2 )
            for r3, val3 in enumerate( E[key] ):
                if r3 >= burnin_length: E_all[key].append( val3 )
            for r4, val4 in enumerate( E_perc[key] ):
                if r4 >= burnin_length: E_perc_all[key].append( val4 )

        logliks_all.append( kept_logliks[burnin_length:] )
        hist_ranges.append( EFMD_res['hists_2D_range'] ) # They should all be the same!
        total_N_iters += EFMD_res['N_iters']
        total_a_accepts += EFMD_res['a_accepts']
        total_a_updates += EFMD_res['a_updates']
        total_E_accepts += EFMD_res['E_accepts']
        total_E_updates += EFMD_res['E_updates']
        total_accepts.append( EFMD_res['total_accepts_rate'] )

    # Convert logliks, a, a_km, E and E_perc into arrays (for convenience):
    for key in layer_labels:
        a_all[key] = np.array( a_all[key] )
        a_km_all[key] = np.array( a_km_all[key] )
        E_all[key] = np.array( E_all[key] )
        E_perc_all[key] = np.array( E_perc_all[key] )
    logliks = np.array( logliks )

    # Add 2D histograms into a single matrix for each fband:
    hists_2D = {}
    for fband in fbands:
        temp_mat = np.zeros( (len( hists_2D_all[fband][0]),
                              len( hists_2D_all[fband][0][0]) ))
        for mat in hists_2D_all[fband]:
            temp_mat += mat
        hists_2D[fband] = temp_mat

    # Calculate total a and E acceptance rates and global acceptance rate:
    total_accepts_rate = np.round( ( num_models / total_N_iters ) * 100, 2)
    a_accepts_rate = np.round( ( total_a_accepts / total_a_updates ) * 100, 2)
    E_accepts_rate = np.round( ( total_E_accepts / total_E_updates ) * 100, 2)
    hist_range = hist_ranges[0]

    ###########################################################################
    print('')

    # Get Representative Model, Minimum Loglikelihood Model, their loglikelihoods
    # and synthetic envelopes:
    RM, RM_loglik, RM_syn_envs, RM_percs, \
    MLM, MLM_loglik, MLM_syn_envs = EFMD_get_RM_MLM ( array_nm, fbands, data_envs,
                                                     s_envs, delta, vel_model,
                                                     models, logliks, a_all, E_all,
                                                     maxamp_inds, DW_end_inds,
                                                     cfs, inv_Qi_vals, inv_Qds,
                                                     parameter)

    # RM string:
    RM_str = ''
    for i in range( num_layers - 1, N, 1):
        RM_str += '[ ' + str( np.round( RM[i][0]/1000, 3) ) + ', ' \
                  + str( np.round( 100 * RM[i][1], 3)) + ' ]'

    # RM string:
    RM_percs_str = ''
    for i in range( num_layers - 1, N, 1):
        RM_percs_str += '[ ' + str( np.round( RM_percs[i][0]/1000, 4) ) + ', ' \
        + str( np.round( 100 * RM_percs[i][1], 4)) + ' ]'

    # MLM model string:
    MLM_str = ''
    for i in range( num_layers - 1, N, 1):
        MLM_str += '[ ' + str( np.round( MLM[i][0]/1000, 3) ) + ', ' \
                   + str( np.round( 100*MLM[i][1], 3)) + ' ]'

    # Print input model, best_fitting_model and misfit to screen:
    if syn_test == True:
        print('Input model was: ')
        print( input_model_str)
    print(' ')
    print( 'Representative Model is:')
    print( RM_str )
    print( 'RM parameters 95% Confidence Interval is:')
    print( RM_percs_str )
    print(' ')
    print( 'RM loglikelihood is ' + str( RM_loglik ))
    print(' ')
    print('Minimum loglikelihood model is:')
    print( MLM_str )
    print( 'MLM loglikelihood after recalculating synthetic envelopes is ' \
          + str( MLM_loglik ))
    print(' ')
    print(' Acceptance rates are:')
    print(' a = ' + str( a_accepts_rate ) + '%, E = ' + str( E_accepts_rate ) + '%' )
    print(' ')


    #           -----------------------------------------------            #

    # Calculate the maximum and minimum Qs using the 5-95 percentile range of
    # the parameters and equation from Fang and Muller (1996):
    Qs_EFMD = {}
    for fband in fbands:
        Qs_EFMD[fband] = np.zeros((3,2))

        # Get central frequency:
        cf = cfs[fband]

        for i, vec in enumerate(RM_percs):

            # Get minimum and maximum values of the parameters:
            a_min = vec[0,0]; a_max = vec[0,1]
            E_min = vec[1,0]; E_max = vec[1,1]

            # Get velocity of the layer:
            v_layer = v[i]

            # Calculate aw/v factor:
            fact_min = ( a_min * 2 * np.pi * cf ) / v_layer
            fact_max = ( a_max * 2 * np.pi * cf ) / v_layer

            # Define numerical coefficients in the equation:
            c1 = 28.73; c2 = 16.77; c3 = 2.40

            # Calculate the inverse of the scattering quality factor. Normally,
            # Qs inv max --> E_max and fact_min
            # Qsinv  min --> E_min and fact_max
            min_num = ( c1 * (fact_min**3) )
            max_num = ( c1 * (fact_max**3) )
            min_denom = ( 1 + ( c2 * (fact_min**2) ) + ( c3 * (fact_min**4) ) )
            max_denom = ( 1 + ( c2 * (fact_max**2) ) + ( c3 * (fact_max**4) ) )
            Qs_inv_min = ( E_min**2 ) * ( min_num / max_denom )
            Qs_inv_max = ( E_max**2 ) * ( max_num / min_denom )

            # Get Qs value (instead of its inverse):
            Qs_min = 1 / Qs_inv_max
            Qs_max = 1 / Qs_inv_min

            # Add values to the results:
            Qs_EFMD[fband][i,0] = Qs_min
            Qs_EFMD[fband][i,1] = Qs_max

    # Calculate minimum and maximum total scattering Q in each layer: we only
    # need to calculate this for the 2 layer model (3 layer model won't be
    # included in the results).
    Qs_tot = {}
    if num_layers == 2:
        for fband in fbands:
            Qs_tot_min_num = ( Qs_EFMD[fband][1,0] * Qs_EFMD[fband][2,0] * tJ )
            Qs_tot_max_num = ( Qs_EFMD[fband][1,1] * Qs_EFMD[fband][2,1] * tJ )
            Qs_tot_min_denom = ( (tjs[1] * Qs_EFMD[fband][2,0]) \
                                + (tjs[2] * Qs_EFMD[fband][1,0]) )
            Qs_tot_max_denom = ( (tjs[1] * Qs_EFMD[fband][2,1]) \
                                + (tjs[2] * Qs_EFMD[fband][1,1]) )

            Qs_tot_min = Qs_tot_min_num / Qs_tot_max_denom
            Qs_tot_max = Qs_tot_max_num / Qs_tot_min_denom
            Qs_tot[fband] = [Qs_tot_min, Qs_tot_max]

    #           --------------------------------------------------            #

    # Save combined results into a dictionary and save to disk:
    EFMD_comb_results = {
            'kept_models': models,
            'kept_logliks': logliks,
            'logliks_all': logliks_all,
            'N_iters': total_N_iters,
            'a': a_all,
            'a_km': a_km_all,
            'E': E_all,
            'E_perc': E_perc_all,
            'a_accepts': total_a_accepts,
            'a_updates': total_a_updates,
            'E_accepts': total_E_accepts,
            'E_updates': total_E_updates,
            'total_accepts_rate': total_accepts_rate,
            'a_accepts_rate': a_accepts_rate,
            'E_accepts_rate': E_accepts_rate,
            'hists_2D_syn_envs': hists_2D,
            'hists_2D_range': hist_range,
            'RM_syn_envs': RM_syn_envs,
            'RM_structural_params': RM,
            'RM_structural_params_percs': RM_percs,
            'RM_loglik': RM_loglik,
            'MLM_syn_envs': MLM_syn_envs,
            'MLM_structural_params': MLM,
            'MLM_loglik': MLM_loglik,
            'burn_in_length': burnin_lengths,
            'resampled_s_nces_dataset': resampled_s_nces_dataset,
            'resampled_delta': delta,
            'DW_end_inds': DW_end_inds,
            'Qs_EFMD': Qs_EFMD,
            'Qs_tot': Qs_tot
            }

    results_fname = comb_results_fname + 'all_MCMCS.pckl'

    # Save the results into a file:
    h = open( results_fname,'wb')
    pickle.dump(EFMD_comb_results, h)
    h.close()

    return EFMD_comb_results


###############################################################################
#         *                *                  *                    *          #
###############################################################################

