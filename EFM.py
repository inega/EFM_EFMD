'''
First created on August 9, 2018
@author: Itahisa Gonzalez Alvarez

    This script contains my implementation of the EFM for all events and arrays.

    EFM STEPS:
        1- Calculate the normalised coda envelope for each event and fband
        2- Stack all normalised coda envelopes for each fband
        3- For each fband, fit the logarithm of the square envelope to a linear function, get a0 and a1
        4- Use w and a0 to obtain Qs-1
        5- Use the Qs-1 values at each fband and w to obtain the structural parameters a and E
        6- Use a1 and w to get Qd and Qi


'''

import os
import pickle
import numpy as np
from glob import glob
from pathlib import Path
import matplotlib.pyplot as plt
from trace_alignment import stack_single
from matplotlib.ticker import ScalarFormatter
from F_EFM import create_nce3c_EFM, plot_envelopes, linear_least_squares, \
                  EFM_least_squares, QiQd_least_squares

###############################################################################
#                                 EFM                                         #
###############################################################################


def EFM_get_envelopes(array_nm, sac_path, EFM_path, EFMD_path, fbands, tJ):

    '''
    First created on August 9, 2018
    @author: Itahisa Gonzalez Alvarez

    This function obtains the normalised and non-normalised 3-component coda
    envelopes required for the EFM and EFMD respectively. Specifically, this
    function carries out the first two steps in the EFM:
        1- Calculate the normalised coda envelope for each event and fband
        2- Stack all normalised coda envelopes for each fband

    To calculate them, it uses the create_nce3c_EFM and plot_envelopes functions
    hosted in the F_EFM module and the stack_single function from F_stacking.

    Arguments:
        - array_nm: (str) name of the seismic array.
        - sac_path: (str) path to the main directory containing the SAC files
                    for all frequency bands
        - EFM_path: (str) path to directory where EFM results are to be stored
        - EFMD_path: (str) path to directory where EFMD results are to be stored
        - fbands: (dict) containing a key name and the freqmin, freqmax for
                  each frequency band of interest
                  (Example: fband = {'A': [0.5, 1]})
        - tJ: (float) one-way traveltime through the lithosphere

    Output:
        The results are saved into several pickle files:
            * Normalised coda envelopes for every event and frequency band
            * Not-normalised coda envelopes for every event and frequency band
            * Ids: results of the normalization integrals for every event and
                   frequency band
            * Stacked normalised coda envelopes for every frequency band and
              their standard deviations
            * Stacked not-normalised coda envelopes for every frequency band
              and their standard deviations
        Plots produced by this script:
            * Alignment plot, to check whether the traces are completely aligned
            * Envelope section, just a visualization of the individual envelopes
              before stacking

    '''

    # Create a dictionary to store the values of Qs for each fband:
    nces_3c = {array_nm: {}}; envs_3c = {array_nm: {}};
    Ids_3c = {array_nm: {}}; no_data_events = {array_nm: {}}

    print(' ')
    print('***************************************************************** ')
    print(' ')
    print('                           Running EFM')
    print(' ')
    print('Fbands are: ' + str(fbands))
    print(' ')
    print('***************************************************************** ')
    print(' ')
    print('             EFM PART 1: get coda envelopes                       ')
    print('***************************************************************** ')

    for fband in fbands:

        print(' ')
        print('Array: ' + array_nm + ', fband ' + fband)
        print(' ')
        print('#                  *****************                         #')

        # Required paths and file names:
        path = sac_path + fband + '/'

        # List of events (from the directories names):
        ds = glob(path + '*/')

        # Not all directories in ds contain enough traces, so we filter them:
        dirs = []
        for directory in ds:
            dirs.append(directory)

        print('Number of events with at least one 3-component trace for fband '
              , fband, ' is: ', len(dirs))
        print(' ')

        #######################################################################

        ################    PART 1: INPUT ARGUMENTS    ########################

        # Predefine variables that will contain:
        # A) The normalised coda envelope for each event
        # B) The coda
        # C) The structural parameters for each array and event
        nces_3c[array_nm][fband] = [] # This contains the nce for each event
                                      # and fband. All nces for the same
        envs_3c[array_nm][fband] = [] # fband are stored in the same dictionary
        Ids_3c[array_nm][fband] = []  # We need to save the value of the
                                      # integrals for the EFMD as well
        no_data_events[array_nm][fband] = []

        ###############    PART 2: EFM FOR THE ENTIRE DATASET    ##############

        for directory in dirs:

            # Get and print event date from directory: it will be the key in our
            # dictionaries.
            ev_date = directory[-16:-1]

            print('Event on ' + str(ev_date))

            # 1. Calculate the normalised coda envelope: the create_nce1c_EFM
            # and create_nce3c_EFM functions return, in this order, for each
            # event and fband: a stream with all the envelopes (correction
            # factor has already been applied to 1-comp envelopes), an array
            # with the normalised coda envelope, the time vector, the index of
            # the beginning of the coda and the integral over the envelope.
            try:
                fname = EFM_path + 'Alignment_plots/' + fband + '/' + array_nm \
                        + '_' + ev_date + '_' + fband + '_3comp_alignment_check.png'
                envs3c, nce3c, t3c, coda_ind_3c, Id3c = create_nce3c_EFM(array_nm,
                                                                         directory,
                                                                         fband,
                                                                         tJ,
                                                                         fname = fname)
                ###   FINE UNTIL HERE!!!
                nces_3c[array_nm][fband].append(nce3c)
                Ids_3c[array_nm][fband].append(Id3c)
                # Stack the non-normalised envelopes so we can use them for the EFMD:
                nn_envs_3c = stack_single(envs3c)[0]
                envs_3c[array_nm][fband].append(nn_envs_3c)
                fname2 = EFM_path + 'Envelope_sections/' + array_nm \
                         + '_envs_' + fband + '_' + ev_date + '.png'
                plot_envelopes(array_nm, ev_date, envs3c, nce3c, fband, tJ,
                               filename = fname2, show_plot = False)
                print('')

            except:
                print(' ')
                print('********     NO USABLE DATA FOUND FOR THIS EVENT!     *********')
                print('None of the stations had three valid traces for the analysis   ')
                print('      or there were less than five GQ vertical traces...       ')
                print(' ')
                print('***************************************************************')
                print(' ')
                no_data_events[array_nm][fband].append(ev_date)

    print(' ================================================================ ')

    ###########################################################################

    # 2. Rename objects so we can save them into a file: these objects contain
    # the results from the EFM for ALL ARRAYS AND ALL FBANDS, no need to separate
    # by array!
    envs0 = envs_3c; nces0 = nces_3c; Ids0 = Ids_3c

    ###########################################################################

    # Save these normalised and non-normalised coda envelopes so we don't have
    # to run that part of the code every time! For some reason, this bit doesn't
    # work unless I save them twice!
    fname = EFM_path + 'EFM_' + array_nm + '_nces_all_fbands' + '.pckl'
    path = Path(fname)
    if not os.path.exists(path.parent):
        os.makedirs(path.parent)
    h = open(fname, 'wb')
    pickle.dump(nces0, h)
    h.close()

    fname3 = 'EFM_' + array_nm + '_Ids_all_fbands'
    h = open(EFM_path + fname3 + '.pckl', 'wb')
    pickle.dump(Ids0, h)
    h.close()

    fname4 = EFMD_path + 'EFMD_' + array_nm + '_env_streams_all_fbands' + '.pckl'
    path = Path(fname4)
    if not os.path.exists(path.parent):
        os.makedirs(path.parent)
    h = open(fname4, 'wb')
    pickle.dump(envs0, h)
    h.close()

    ###########################################################################

    # Stack all normalised coda envelopes for ALL events and fbands:
    s_nces = {}; s_envs = {}; nces = {}; ncelens = {}; new_ncelens = {};
    envlens = {}; new_envlens = {}

    # For some events the normalisation returned infinites (division by 0). To
    # remove these and keep the others:
    nces[array_nm] = {}
    for fband in fbands:
        nces[array_nm][fband] = []
        for vec in nces0[array_nm][fband]:
            if np.inf not in vec:
                nces[array_nm][fband].append(vec)
            else:
                print('Inifinity found in the array for fband ' + fband)

    # Some envelopes have + /-1 sample difference in length. We can't stack them
    # if they have different length:
    ncelens[array_nm] = []; new_ncelens[array_nm] = []
    envlens[array_nm] = []; new_envlens[array_nm] = []
    for fband in fbands:
        N = len(nces[array_nm][fband])
        for i in range(N):
            ncelens[array_nm].append(len(nces[array_nm][fband][i]))
            envlens[array_nm].append(len(envs0[array_nm][fband][i]))

    # Take the minimum number of samples as the size of the envelope: remove the
    # last sample from the longest envelopes.
    for fband in fbands:
        N = len(nces[array_nm][fband])
        for i in range(N):
            nce = nces[array_nm][fband][i]
            env = envs0[array_nm][fband][i]
            if len(nce)>min(ncelens[array_nm]):
                nces[array_nm][fband][i] = nce[:(min(ncelens[array_nm]))]
            if len(env)>min(envlens[array_nm]):
                envs0[array_nm][fband][i] = env[:(min(envlens[array_nm]))]
            new_ncelens[array_nm].append(len(nces[array_nm][fband][i]))
            new_envlens[array_nm].append(len(envs0[array_nm][fband][i]))

    # Sanity check:
    for val in new_ncelens[array_nm]:
        if val != min(ncelens[array_nm]):
            print('Number of samples is not the same for all nces!')
    for val in new_envlens[array_nm]:
        if val != min(envlens[array_nm]):
            print('Number of samples is not the same for all envs!')

    ###########################################################################

    # Stacking: we need the mean normalised coda envelope, non-normalised coda
    # envelopes and their standard deviations.
    s_nces[array_nm] = {}; s_envs[array_nm] = {}
    for fband in fbands:
        s_nces[array_nm][fband] = {}
        s_envs[array_nm][fband] = {}

    for fband in fbands:

        # Stack normalised coda envelopes and get standard deviation:
        s_nces[array_nm][fband]['s_nces'] = np.mean([nce for nce in nces[array_nm][fband]], axis = 0)
        s_nces[array_nm][fband]['s_nces_std'] = np.std([nce for nce in nces[array_nm][fband]], axis = 0)

        # Stack also the non-normalised envelopes for each event and fband and
        # get the standard deviation:
        s_envs[array_nm][fband]['s_envs'] = np.mean([env for env in envs0[array_nm][fband]], axis = 0)
        s_envs[array_nm][fband]['s_envs_std'] = np.std([env for env in envs0[array_nm][fband]], axis = 0)

        # Remove noise level from the stacked envelopes and from the standard
        # deviations:
        noise_level_nce = np.mean(s_nces[array_nm][fband]['s_nces'][0:500])
        noise_level_nce_std = np.mean(s_nces[array_nm][fband]['s_nces_std'][0:500])
        s_nces[array_nm][fband]['s_nces'] = s_nces[array_nm][fband]['s_nces'] - noise_level_nce
        s_nces[array_nm][fband]['s_nces_std'] = s_nces[array_nm][fband]['s_nces_std'] - noise_level_nce_std

        noise_level_env = np.mean(s_envs[array_nm][fband]['s_envs'][0:500])
        noise_level_env_std = np.mean(s_envs[array_nm][fband]['s_envs_std'][0:500])
        s_envs[array_nm][fband]['s_envs'] = s_envs[array_nm][fband]['s_envs'] - noise_level_env
        s_envs[array_nm][fband]['s_envs_std'] = s_envs[array_nm][fband]['s_envs_std'] - noise_level_env_std

    ###########################################################################

    # Save the normalised and non-normalised stacked coda envelopes into new files:
    fname5 = 'EFM_' + array_nm + '_s_nces_all_fbands'
    h = open(EFM_path + fname5 + '.pckl', 'wb')
    pickle.dump(s_nces, h)
    h.close()

    fname6 = 'EFMD_' + array_nm + '_s_envs_all_fbands'
    h = open(EFMD_path + fname6 + '.pckl', 'wb')
    pickle.dump(s_envs, h)
    h.close()


###############################################################################
####                  END OF THE PART 1 OF THE EFM                         ####
###############################################################################





def EFM_analysis (array_nm, fbands, v, tJ, delta, nces0_fname, s_envs_fname,
                  results_fname, figs_fname, units = 'm', syn_test = False):

    '''
    This function takes the normalised coda envelopes computed by the
    EFM_get_envelopes function and process them according to the EFM. The
    processing steps it carries out are:
        3- For each fband, fit the logarithm of the squared envelope to a linear
          function, get a0 and a1
        4- Use w and a0 to obtain Qs^-1
        5- Use the Qs^-1 values at each fband and w to obtain the structural
           parameters a and E
        6- Use a1 and w to get Qd and Qi

    It uses the linear_least_squares, EFM_least_squares and QiQd_least_squares
    functions from F_EFM and the get_model_traveltimes function from F_V_models
    to obtain the velocity model.

    Arguments:
        - array_nm: (str) name of the seismic array.
        - fbands: (dict) containing a key name and the freqmin, freqmax for
                  each frequency band of interest
                  (Example: fband = {'A': [0.5, 1]})
        - v: (np.array) mean P wave velocity in the lithosphere
        - tJ: (float) one-way traveltime through the lithosphere
        - delta: (float) inverse of the sampling rate of the data
        - nces0_fname: (str) path and filename of the normalised coda envelopes
                       BEFORE stacking obtained from part 1 of the EFM
        - s_envs_fname: (str) path and filename of the stacked non-normalised
                        envelopes obtained from part 1 of the EFM
        - results_fname: (str) path and filename for saving results
        - figs_fname: (str) path and filename for saving figures
        - units: (str) units for distances and velocities (either 'm' or 'km')
        - syn_test: (bool) True for input synthetic data.

    Output:
        - EFM_best_fits: dictionary containing the results for the correlation
          length, the RMS velocity fluctuations, Qs^-1 at different frequencies,
          Qi0, Qi, Qdiff0, Qdiff, alpha and a very rough estimation of the
          thickness of the scattering layer. The standard deviations (errors)
          for all these results are also given in this dictionary.
        - Plots produced and saved into files by this script:
            * Results of the fit of the coda decay to a linear function for
              each freq. band
            * Results of the fit of the coda decay to a linear function for all
              freqs. together
            * Qs as obtained from a0 vs. theoretical Qs, with obtained a and E
              values
            * a1 vs. theoretical a1 plot with Qi0, Qd0, L, alpha values

    '''

    try:
        if len(v) == 1:

            # Load saved normalised, stacked and corrected coda envelopes:
            fopen = open(nces0_fname, 'rb')
            nces0 = pickle.load(fopen)
            fopen.close()

            fopen = open(s_envs_fname, 'rb')
            s_envs = pickle.load(fopen)
            fopen.close()

            # Create dictionaries that I will need for my results:
            coda_ind = {}; tP_ind = {}; Id = {}

            Qss = {}; Qs_invs = {}; Qs_invs_std = {}; Qss_std = {}
            Qs_vals = []; Qs_inv_vals = []; Qs_vals_std = [];Qs_inv_vals_std = []

            a0s = {}; a1s = {}; a0s_std = {}; a1s_std = {}

            num_envs = {}; fcs = []
            EFM_results = {}

            print(' ')
            print('   ******************************************************** ')
            print(' ')
            print('EFM PART 2: getting structural parameters and quality factors')
            print(' ')
            print('   ******************************************************** ')
            print(' ')
            print('Fbands are: ' + str(fbands))
            print(' ')

            for fband in fbands:

                # Extract normalised coda envelope for this frequency band:
                s_env = s_envs[array_nm][fband]['s_envs']

                ###############################################################
                #    CODA DETERMINATION AND INTEGRAL OVER THE DIRECT WAVE ARRIVAL

                # Time vector for the fit:
                t = np.arange(0, len(s_env) * delta, delta)

                # Maximum amplitude index for the fit: we take the integral over
                # the time window of the direct arrival as the integral from the
                # beginning of the trace to the maximum amplitude and multiplying
                # by 2.
                maxamp_ind = (np.abs(s_env - max(s_env))).argmin() + 1

                # Direct wave arrival index (only for plots): since we don't
                # have a tP time for the stacked nce, we take it from the
                # maximum amplitude index. Direct wave goes from 5 seconds
                # before maximum amplitude to (5+tJ) seconds after.
                tP_ind[fband] = int(round(np.mean(maxamp_ind)) - 5/delta)
                coda_ind[fband] = tP_ind[fband] + int((5+tJ) * 1/delta)

                # Define x (time):
                x = t[:maxamp_ind]; dx = delta
                # Define y (coda data):
                y = (s_env[:maxamp_ind])**2
                # Integrate:
                Id[fband] = 2 * np.trapz(y, x, dx = dx)

                #*************************************************************#
                #          EFM PART 1: OBTAIN STRUCTURAL PARAMETERS
                #*************************************************************#

                #          FIT CODA DECAY TO A LINEAR FUNCTION:
                lsr = linear_least_squares(t, s_env, coda_ind[fband])
                # lsr.x contains a0 and a1 values, in that order.
                a0s[fband] = lsr.x[0]; a1s[fband] = -lsr.x[1]

                #####    LINEAR FIT COEFFICIENTS ERROR CALCULATIONS!    #######

                # Calculate residuals: I need them to be vertical arrays to use
                # dot product later, that is why I create it this way.
                res = lsr.fun
                N = len(res)

                # I need to transpose res:
                res_linlsq = np.empty((N, 1))
                for h, val in enumerate(res):
                    res_linlsq[h] = val

                # Get uncertainty in the estimation of the parameters:
                # Calculate MSE (mean square error): (2 = num. of parameters)
                MSE_a0a1 = np.dot(res_linlsq.transpose(), res_linlsq) / (N-2)
                # Get the Jacobian:
                J = lsr.jac
                # Get the Variance Covariance Matrix (VCM):
                VCM_a0a1 = np.linalg.inv(np.dot(J.transpose(), J)) * MSE_a0a1
                a0s_std[fband] = np.sqrt(VCM_a0a1[0][0])
                a1s_std[fband] = np.sqrt(VCM_a0a1[1][1])

                ####  Sanity check: plot results of the linear fitting.  ######
                num_envs[fband] = len(nces0[array_nm][fband])

                plt.figure(figsize = (10, 5))
                plt.title(array_nm + ', ' + fband + ', num.envs. = ' \
                          + str(num_envs[fband]), fontsize = 14)

                plt.plot(t[tP_ind[fband] - 100:],
                     np.log10(s_env**2)[tP_ind[fband] -100 :], 'k',
                     linewidth = 2, label = 'Log of normalised coda envelope squared')
                plt.plot(t[coda_ind[fband]:-100],
                     np.log10(s_env[coda_ind[fband]:-100]**2), 'r--',
                     label = 'Coda')
                plt.plot(t[tP_ind[fband]:coda_ind[fband]],
                     np.log10(s_env[tP_ind[fband]:coda_ind[fband]]**2),
                     'g--', label = 'Direct wave')

                # Fitting function:
                yfit = a0s[fband] + t[coda_ind[fband]:-100]*-a1s[fband]
                plt.plot(t[coda_ind[fband]:-100], yfit, 'b',
                       label = 'Linear fit = [' + str(np.round(a0s[fband], 4)) \
                           + ' + /-' + str(np.round(a0s_std[fband], 4)) \
                           + '] + [' + str(np.round(-a1s[fband], 4)) + ' + /-' \
                           + str(np.round(a1s_std[fband], 4)) + '] *t')
                plt.xlabel('Time (s)', fontsize = 14)
                plt.ylabel('log($nce^2$)', fontsize = 14)
                plt.ylim([-7, -0.7])
                plt.grid()
                plt.legend(loc = 'best', fontsize = 14)

                fname = figs_fname + '_envs_linfit_' + fband
                plt.savefig(fname + '.pdf', bbox_inches = 'tight')
                plt.savefig(fname + '.png', bbox_inches = 'tight')
                plt.close('all')

                ###############################################################

                #     GET QS-1 FROM THE FIRST LINEAR FIT COEFFICIENT, a0

                # Get the central frequency:
                cf = ((fbands[fband][1] + fbands[fband][0])/2)
                fcs.append(cf)

                # Calculate Qs-1:
                Qs_inv = (10**a0s[fband]) / (2* (2 * np.pi * cf) * Id[fband])
                Qs = 1 / Qs_inv
                Qs_invs[fband] = Qs_inv; Qss[fband] = Qs; Qs_vals.append(Qs)
                Qs_inv_vals.append(Qs_inv)

                # Calculate the errors for each Qs and add them to Qs dictionary:
                Qs_inv_std = (np.log(10) * (10**a0s[fband]) * a0s_std[fband]) \
                            / (2* Id[fband]* 2* np.pi* cf)
                Qs_std = 2 * Id[fband] * 2*np.pi*cf * np.log(10) \
                        * (10**-a0s[fband]) * a0s_std[fband]
                Qs_invs_std[fband] = Qs_inv_std; Qss_std[fband] = Qs_std
                Qs_vals_std.append(Qs_std); Qs_inv_vals_std.append(Qs_inv_std)

            # Convert Qss and fcs into arrays so we don't have problems using
            # them in Numba functions later on:
            Qs_vals = np.array(Qs_vals); Qs_inv_vals = np.array(Qs_inv_vals)
            fcs = np.array(fcs)
            Qs_vals_std = np.array(Qs_vals_std)
            Qs_inv_vals_std = np.array(Qs_inv_vals_std)

            ###################################################################

            # PLOTS NEEDED BEFORE MOVING ON:

            # Plot all linfits of the envelopes together:
            fbc = {'A':'r', 'B':'navy', 'C':'gray', 'D':'c', 'E':'darkviolet',
                   'F':'orange', 'G':'lime', 'H':'brown'}

            # Get mean tP_ind and coda_ind (for the plot shading):
            tP_inds = []; coda_inds = []
            for fband in fbands:
                tP_inds.append(tP_ind[fband])
                coda_inds.append(coda_ind[fband])
            mean_tP_ind = int(round(np.mean(np.array(tP_inds))))
            mean_coda_ind = int(round(np.mean(np.array(coda_inds))))

            plt.figure(figsize = (17, 8))

            # Plot shaded areas and divisions:
            plt.gca().axvspan(t[mean_tP_ind], t[mean_coda_ind],
                               color = 'whitesmoke')
            plt.gca().axvspan(t[mean_coda_ind], t[-100], color = 'seashell')
            plt.gca().axvline(t[mean_tP_ind], color = 'k', linewidth = 2)
            plt.axvline(t[mean_coda_ind], color = 'k', linewidth = 2)
            plt.axvline(t[-100], color = 'k', linewidth = 2)
            # Plot envelopes and linear fits:
            for fband in fbands:
                yfit = a0s[fband] + t[coda_ind[fband]:-100]*-a1s[fband]
                plt.plot(t[tP_ind[fband] -100 :],
                     np.log10(s_envs[array_nm][fband]['s_envs']**2)[tP_ind[fband] -100 :],
                     color = fbc[fband], linewidth = 1)
                plt.plot(t[coda_ind[fband]:-100], yfit, '--', color = fbc[fband],
                     linewidth = 2, label = str(fbands[fband][0]) + '-' \
                     + str(fbands[fband][1]) + 'Hz')
            plt.xlabel('Time (s)', fontsize = 24)
            plt.ylabel('$log (Amplitude^2)$', fontsize = 24)
            plt.grid(color = 'dimgrey', linewidth = 1)
            plt.xlim(20, 100); plt.ylim(-5, -0.5)
            plt.legend(loc = 'best', title = array_nm, ncol = 3,
                       title_fontsize = 28, fontsize = 20, fancybox = True,
                       framealpha = 1)
            plt.gca().tick_params(axis = 'y', labelsize = 24, length = 10,
                                   width = 2)
            plt.gca().tick_params(axis = 'x', labelsize = 24, length = 10,
                                   width = 2)

            for side in plt.gca().spines.keys():
                plt.gca().spines[side].set_linewidth(3)
            fname = figs_fname + '_all_envs_linfits'
            plt.savefig(fname + '.pdf', bbox_inches = 'tight')
            plt.savefig(fname + '.png', bbox_inches = 'tight')
            plt.close('all')

            ###################################################################

            #      GET STRUCTURAL PARAMETERS FROM QS-1 FOR ALL FBANDS         #

            # IMPORTANT NOTE: The Qs-1 value I get for fband A is an outlier,
            # don't use it! (Don't use it either in next steps of the analysis!)
            vel = v[0]

            # Initial estimate of my parameters:
            if units == 'km': params0 = np.array([1.7, 0.02])
            if units == 'm': params0 = np.array([1700, 0.02])

            EFM_fit = EFM_least_squares(Qs_inv_vals, fcs, vel, params0)
            # Parameters:
            a = EFM_fit.x[0]; E = EFM_fit.x[1]

            ####        STRUCTURAL PARAMETERS ERROR CALCULATIONS!          ####

            # Calculate residuals: I need them to be vertical arrays to use
            # dot product later, that is why I create it this way.
            res = EFM_fit.fun
            N = len(res)

            res_EFM = np.empty((N, 1))
            for h, val in enumerate(res):
                res_EFM[h] = val

            # Get uncertainty in the estimation of the parameters:
            #Calculate mean square error MSE: (2 is the number of parameters)
            MSE_EFM = np.dot(res_EFM.transpose(), res_EFM) / (N-2)
            # Get the Jacobian:
            J = EFM_fit.jac
            # Get the Variance Covariance Matrix (VCM):
            VCM_EFM = np.linalg.inv(np.dot(J.transpose(), J))*MSE_EFM
            a_std = np.sqrt(VCM_EFM[0][0]); E_std = np.sqrt(VCM_EFM[1][1])

            units_str = 'km'
            if units == 'm': a_km = a/1000; a_std_km = a_std/1000
            print('')
            print('The correlation length is ' + str(np.round(a_km, 4)) \
                  + ' +/-' + str(np.round(a_std_km, 4)) + ' ' + units_str \
                  + ' and \nthe RMS velocity variations are ' \
                  + str(np.round(E*100, 4)) + ' +/-' \
                  + str(np.round(E_std*100, 4)) + '%')

            #   ----------------    PLOT RESULT      ---------------------    #

            # Fang and Muller's equation contains the factor (aw/v) multiple
            # times. It is better to define it outside the equation.
            # Constants from Fang and Muller for the exponential ACF:
            c1 = 28.73; c2 = 16.77; c3 = 2.40
            x = np.array(fcs)
            factor = a*2*np.pi*x/vel
            # This are the theoretical scattering Q values:
            theor_Qs_invs = (E**2) * (c1 * (factor**3)) \
                            / (1 + c2 * (factor**2) + c3 * (factor**4))

            # Function to plot (prediction of my data):
            y = np.array(Qs_inv_vals)# This are my data derived Qs

            # Sanity check: plot results of the fitting.
            plt.figure(figsize = (15, 10))
            plt.title(array_nm +  ': Qs vs. theoretical curve', fontsize = 20)
            plt.plot(x, theor_Qs_invs, 'r', linewidth = 3,
                     label = 'Theoretical curve = > a = [' + str(round(a_km, 1)) \
                     + ' + /-' + str(round(a_std_km, 1)) + '] km, $\epsilon$ = [' \
                     + str(round(E*100, 1)) + ' + /-' + str(round(E_std*100, 1)) \
                     + ']%')
            for fband in fbands:
                cf = (fbands[fband][1] + fbands[fband][0])/2
                plt.errorbar(cf, Qs_invs[fband], yerr = Qs_invs_std[fband],
                             fmt = '.k', markersize = 3, ecolor = 'k',
                             capsize = 10)
            plt.xlabel('Frequency (Hz)', fontsize = 20)
            plt.ylabel('$Q_s^{-1}$', fontsize = 20)
            plt.grid(axis = 'both')
            plt.gca().xaxis.set_major_formatter(ScalarFormatter())
            plt.gca().xaxis.set_minor_formatter(ScalarFormatter())
            plt.gca().yaxis.set_major_formatter(ScalarFormatter())
            plt.gca().yaxis.set_minor_formatter(ScalarFormatter())
            plt.gca().tick_params(axis = 'x', labelsize = 16)
            plt.gca().tick_params(axis = 'y', labelsize = 16)
            plt.legend(loc = 'lower right', fontsize = 16)
            fname = figs_fname + '_Qs_vs_theoretical_curve'
            plt.savefig(fname + '.pdf', bbox_inches = 'tight')
            plt.savefig(fname + '.png', bbox_inches = 'tight')
            plt.close('all')

            ###################################################################

            #*****************************************************************#
            #   EFM PART 2: OBTAIN INTRINSIC AND DIFFUSION QUALITY FACTORS
            #*****************************************************************#

            #    GET Qd AND Qi FROM THE SECOND LINEAR FIT COEFFICIENTS, a1    #

            # I need a0s and a1s as arrays:
            a0_vals = []; a1_vals = []
            for fband in fbands:
                a0_vals.append(a0s[fband])
                a1_vals.append(a1s[fband])
            a0_vals = np.array(a0_vals); a1_vals = np.array(a1_vals)

            # Set the power for the frequency dependency of Qi0/Qdiff: we are
            # going to test different values of alpha and use the one yielding
            # minimum misfits.
            alphas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]

            # Let's do a grid search for alpha:
            Qi0Qd0s = {}

            # DATA:
            # Define constant for our function:
            c = 2*np.pi*np.log10(np.e)
            x = fcs[1:]; y = a1_vals[1:]

            # Create list to store the value of the squared residuals:
            sqrs = []

            for alpha in alphas:

                # Fit a1s to get Qi0^-1, Qd0^-1, in that order:
                Qi0Qd0_fit = QiQd_least_squares(x, y, alpha)
                res = Qi0Qd0_fit.fun
                J = Qi0Qd0_fit.jac

                # These are the values we were looking for (Qi0 and Qdiff0),
                # the function above calculates their inverses:
                inv_Qi0 = Qi0Qd0_fit.x[0]
                inv_Qd0 = Qi0Qd0_fit.x[1]
                Qi0 = 1/Qi0Qd0_fit.x[0]
                Qd0 = 1/Qi0Qd0_fit.x[1]
                L = Qd0*vel/(8*np.pi)

                if Qi0 > 0 and Qd0 > 0:

                    # Get theoretical values:
                    syn_y = c*((1/Qd0) + ((1/Qi0) * (x**(1-alpha))))

                    # Get sum of residuals squared:
                    rsq = np.sum(res**2)
                    sqrs.append(rsq)

                    # Add results to Qi0Qd0s, we'll have Qi0, Qd0, alpha, L,
                    # rsq, chisq:
                    Qi0Qd0s[str(alpha)] = {}
                    Qi0Qd0s[str(alpha)]['inv_Qi0'] = inv_Qi0
                    Qi0Qd0s[str(alpha)]['inv_Qd0'] = inv_Qd0
                    Qi0Qd0s[str(alpha)]['Qi0'] = Qi0
                    Qi0Qd0s[str(alpha)]['Qd0'] = Qd0
                    Qi0Qd0s[str(alpha)]['L'] = L
                    Qi0Qd0s[str(alpha)]['res'] = res
                    Qi0Qd0s[str(alpha)]['sqres'] = rsq
                    Qi0Qd0s[str(alpha)]['J'] = J
                    Qi0Qd0s[str(alpha)]['alpha'] = alpha
                    Qi0Qd0s[str(alpha)]['syn_y'] = syn_y

            # Get best fit parameters:
            Qi = {}; Qdiff = {}; Qi_std = {}; Qdiff_std = {}

            for key in Qi0Qd0s:

                if Qi0Qd0s[key]['sqres'] == min(sqrs):

                    # Redefine quantities:
                    inv_Qi0 = Qi0Qd0s[key]['inv_Qi0']
                    inv_Qd0 = Qi0Qd0s[key]['inv_Qd0']
                    Qi0 = Qi0Qd0s[key]['Qi0']
                    Qd0 = Qi0Qd0s[key]['Qd0']
                    alpha = Qi0Qd0s[key]['alpha']
                    L = Qi0Qd0s[key]['L']
                    J = Qi0Qd0s[key]['J']
                    a1s_theor = Qi0Qd0s[key]['syn_y']
                    res = Qi0Qd0s[key]['res']
                    rsq = Qi0Qd0s[key]['sqres']

                    # Get final Qdiff and Qi:
                    for fband in fbands:
                        fc = (fbands[fband][0] + fbands[fband][1]) / 2
                        Qdiff[fband] = Qd0 * fc
                        Qi[fband] = Qi0 * (fc**alpha)

                    ############         ERROR CALCULATIONS          ##########

                    # Add uncertainty in the estimation of Qi0Qd0s:
                    # Get standard deviation from syn_y for the best fit
                    # parameters.
                    N = len(res)

                    r = np.empty((N, 1))
                    for h, val in enumerate(res):
                        r[h] = val

                    #Calculate MSE: (3 is the number of parameters)
                    MSE_Qi0Qd0 = np.dot(r.transpose(), r)/(N-3)

                    # Get the Variance Covariance Matrix (VCM): this is the
                    # variance of the INVERSE of Qi0 and Qd0.
                    VCM_Qi0Qd0 = np.linalg.inv(np.dot(J.transpose(), J))*MSE_Qi0Qd0
                    invQi0_std = np.sqrt(VCM_Qi0Qd0[0][0])
                    invQd0_std = np.sqrt(VCM_Qi0Qd0[1][1])
                    Qi0_std = (1 / (inv_Qi0**2)) * invQi0_std
                    Qd0_std = (1 / (inv_Qd0**2)) * invQd0_std

                    # Get error on L:
                    L_std = (vel/(8*np.pi)) * Qd0_std

                    # Get error for Qdiff and Qi:
                    for fband in fbands:
                        fc = (fbands[fband][0] + fbands[fband][1]) / 2
                        Qdiff_std[fband] = fc * Qd0_std
                        Qi_std[fband] = Qi0_std * (fc**alpha)

                    print('')
                    print('Minimum misfit corresponds to \n\nQi0 = ' \
                          + str(round(Qi0, 2)) \
                          + ' +/- ' + str(round(Qi0_std, 2)) + ',\nQdiff0 = ' \
                          + str(round(Qd0, 2)) + ' +/- ' + str(round(Qd0_std, 2)) \
                          + ',\nalpha = ' + str(alpha) + ', \nL = ' \
                          + str(np.round((L/1000), 1)) + ' +/- ' \
                          + str(round((L_std/1000), 1)) \
                          + ' ' + units_str + ',\nrsq = ' + str(rsq))
                    print('')

            #######              PLOT RESULTS                         #########

            # Sanity check: plot the results with the fitting!
            plt.figure(figsize = (15, 10))

            # Plot a1 vs. angular frequency with the Qi0 and Qdiff values:
            plt.title(array_nm + ': $Q_i$, $Q_{diff}$ vs. theoretical curve',
                      fontsize = 20)
            plt.plot(x, syn_y, color = 'purple', linewidth = 3,
                     label = "Theoretical coda decay rate")
            plt.plot(fcs, a1_vals, 'k.', markersize = 20,
                     label = 'Coda decay rate')

            for fband in fbands:
                cf = (fbands[fband][1] + fbands[fband][0])/2
                plt.errorbar(cf, a1s[fband], yerr = a1s_std[fband],
                             ecolor = 'k', capsize = 15)

            plt.xlabel('Frequency (Hz)', fontsize = 20)
            plt.ylabel('Coda decay coefficient ($a_1$)', fontsize = 20)
            plt.grid()
            plt.gca().tick_params(axis = 'x', labelsize = 20)
            plt.gca().tick_params(axis = 'y', labelsize = 20)
            legtitle = r'$ \alpha $ = ' \
                     + str(round(alpha, 1)) + ', \n$Q_{i}$ = ' \
                     + str(round(Qi0, 2)) + ' +/- ' + str(round(Qi0_std, 2)) \
                     + ', \n$Q_{diff}$ = ' + str(round(Qd0, 2)) + ' +/- ' \
                     + str(round(Qd0_std, 2)) + ', \nL = ' \
                     + str(np.round(L/1000, 1)) + ' +/- ' \
                     + str(round(L_std/1000, 1)) + units_str
            plt.legend(loc = 'lower right', fontsize = 12, title = legtitle,
                       title_fontsize = 14)
            fname = figs_fname + '_Qi0_Qdiff_a1s_vs_theoretical_curve'
            plt.savefig(fname + '.pdf', bbox_inches = 'tight')
            plt.savefig(fname + '.png', bbox_inches = 'tight')
            plt.close()

            print('---------------------------------------------------------')


            EFM_results [array_nm] = { 'a':a,
                                       'E':E,
                                       'a_std':a_std,
                                       'E_std':E_std,
                                       'freqs': fcs,
                                       'tP_inds': tP_ind,
                                       'coda_inds': coda_ind,
                                       'Qs': Qss, #dictionary form
                                       'Qs_vals': Qs_vals,#array_form
                                       'Qs_invs': Qs_invs,#dictionary form
                                       'Qs_inv_vals': Qs_inv_vals,#array form
                                       'theor_Qs_invs': theor_Qs_invs,#inv Qs from eq. 3
                                                                      #in Fang & Muller (1996)
                                       'Qs_std': Qss_std,#dictionary form
                                       'Qs_vals_std': Qs_vals_std, #array form
                                       'Qs_invs_std': Qs_invs_std,#dictionary form
                                       'Qs_inv_vals_std': Qs_inv_vals_std,#array form
                                       'inv_Qi0': inv_Qi0,
                                       'inv_Qi0_std': invQi0_std,
                                       'Qi0': Qi0,
                                       'Qi0_std': Qi0_std,
                                       'Qi': Qi,
                                       'Qi_std': Qi_std,
                                       'inv_Qd0': inv_Qd0,
                                       'inv_Qd0_std': invQd0_std,
                                       'Qd0': Qd0,
                                       'Qd0_std': Qd0_std,
                                       'Qdiff': Qdiff,
                                       'Qdiff_std': Qdiff_std,
                                       'alpha': alpha,
                                       'L': L,
                                       'L_std': L_std,
                                       'a0s': a0s, #dictionary form
                                       'a0_vals': a0_vals,#array form
                                       'a0s_std': a0s_std,
                                       'a1s': a1s, #dictionary form
                                       'a1_vals': a1_vals,#array form
                                       'a1s_std': a1s_std,
                                       'theor_a1s': a1s_theor}

            h = open(results_fname, 'wb')
            pickle.dump(EFM_results, h)
            h.close()

            return EFM_results

    except:

        if len(v) != 1:
            print('EFM part 2 can only run for single layer models')
            print('Please check velocity model...')
        else:
            print('Unexpected error, quitting...')

        EFM_results = {}

        return EFM_results

###############################################################################
####                         END OF THE EFM                                ####
###############################################################################



