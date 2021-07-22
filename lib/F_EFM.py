'''
First created on August 9, 2018
@author: Itahisa Gonzalez Alvarez

    Set of functions I need for the EFM.

'''

import os
import copy
import numpy as np
#matplotlib.use('agg')
from glob import glob
from numba import jit
from pathlib import Path
import matplotlib.pyplot as plt
from obspy import read, Stream, Trace
from scipy.optimize import least_squares
from matplotlib.pyplot import plot as plot
from fk_analysis import get_Pwave_arrivals
from obspy.signal.filter import envelope as envelope
from trace_alignment import stack_single, align_stream_3c_fk




def create_nce3c_EFM(array_nm, sac_path, fband, tJ, fname = None):

    '''
    This function computes the 3-component normalised coda envelopes required
    for the EFM/EFMD analysis. First, it creates a stream of aligned, normalized
    and filtered 3-component envelopes for a given event and array. The envelopes
    for the horizontal components need to be aligned and normalized with respect
    to the vertical component (it uses the align_stream_3c function from
    F_trace_alignment for the alignment and get_Pwave_arrivals from
    F_trace_env_stream_creation to get the theoretical P wave arrival times).
    Then, it computes three component seismograms, takes their envelopes, stacks
    them and normalises again using the integral over the direct wave arrival
    time window, as in Korn (1997)/Hock et al (2000).

    Steps (in order):
        1 - Normalize traces to vertical component
        2 - Align to vertical component
        3 - Filter
        4 - Get 1-component envelopes
        5 - Get 3-component envelopes
        6 - Stack 3-component envelopes
        7 - Normalize according to eq. 12 from Hock et al. (2000)

    Arguments:
         - array_nm: (str) array name/code
         - sac_path: (str) path to SAC files
         - fband: (str) frequency band we want to filter the traces into (capital
                        letter from A to H, see fbands dictionary below to check
                        frequency range)
         - tJ: (str) time it takes P waves to reach the free surface of our model
         - fname: (str) path with filename for the alignment plots (just sanity
                        check, optional)

    Output:
         - envs3c: (Obspy stream) of filtered, aligned and normalized 3-component
                                  (vertical) envelopes
         - nce_3c: (np.array) global 3-component normalised coda envelope for
                              the event
         - t: (np.array) time vector
         - coda_ind: (int) index indicating when the coda starts
         - Id3c: (float) value of the integral over the envelope for the length
                         of the direct wave arrival
         - alignment plot (optional, created by create_stream_EFM function)

    '''

    fbands = {'A':[0.5, 1], 'B':[0.75, 1.5], 'C':[1, 2], 'D':[1.5, 3],
              'E':[2, 4], 'F':[2.5, 5], 'G':[3, 6], 'H':[3.5, 7]}
    print('Creating three-component envelope stream for the ' \
          + str(fbands[fband][0]) + ' to ' + str(fbands[fband][1]) \
          + ' Hz frequency band...')

    ###########################################################################

    # Get ALL traces we have for a given event and array:
    files = glob(sac_path + '*.SAC')

    # 0 - Create a separate stream for each component, but host them inside a
    # dictionary so it is easier to work with them. Further on, I will also
    # need to separate all traces by station, so create those streams too.
    sts = {'st_z':Stream(), 'st_t':Stream(), 'st_r':Stream()}
    station_sts = {}

    for f1 in files:
        if 'BHR' in f1:
            a = read(f1)
            a_station = a[0].stats.station
            sts['st_r'].append(a[0])
            station_sts[a_station] = Stream()
            for f2 in files:
                b = read(f2)
                b_station = b[0].stats.station
                b_channel = b[0].stats.channel
                if b_station == a_station and b_channel == 'BHZ':
                    sts['st_z'].append(b[0])
                if b_station == a_station and b_channel == 'BHT':
                    sts['st_t'].append(b[0])

    print('Number of 3-component traces for this event is ' \
          + str(len(sts['st_z'])))

    # For PSA, we are using streams only if they have 5 or more traces for the
    # vertical component. For ASAR and WRA, we will use as many traces as we have.
    if len(sts['st_z']) >= 5 or array_nm != 'PSA':

        #######################################################################

        # 1 - Normalize traces with respect to the maximum amplitude present in
        # the VERTICAL component stream:
        z_amp = []
        for trace in sts['st_z']:
            minamp = min(trace.data); maxamp = max(trace.data)
            if abs(minamp)>abs(maxamp):
                z_amp.append(abs(minamp))
            elif abs(maxamp)>abs(minamp):
                z_amp.append(abs(maxamp))

        max_z_amp = max(z_amp)

        for key in sts:
            for trace in sts[key]:
                trace.data = trace.data/max_z_amp

        #######################################################################

        # 2 - 3 - Filter into the desired frequency band and get the envelope:
        #print('Filtering traces...')
        for key in sts:
            for trace in sts[key]:
                trace.filter('bandpass', freqmin = fbands[fband][0],
                             freqmax = fbands[fband][1], corners = 2,
                             zerophase = True)
                trace.data = envelope(trace.data)

        #######################################################################

        # 4 - Align traces:

        # The alignment and trimming needs to be done according to the VERTICAL
        # component.
        # Get event date from path:
        ev_date = sac_path[-16:-1]
        # Get theoretical P wave arrival times:
        t_P, t_P_date = get_Pwave_arrivals(sts['st_z'], ev_date, model = 'prem')

        t_P_max = max(t_P)

        # Get streams of aligned non-trimmed and trimmed traces:
        full_sts3, trimsts3 = align_stream_3c_fk (array_nm, sts, ev_date,
                                                  tJ, fname = None)

        # From now on, we will use only the trimmed streams:
        sts3 = trimsts3

        # Sanity check plot:
        if fname != None:
            plt.figure(figsize = (20, 10))
            for trace in sts3['st_z']:
                plot(trace.data, 'k', linewidth = 0.8)
            plt.title(str(trace.stats.starttime) + ', ' + str(trace.stats.channel))
            plt.grid()

            # Save plot in the parent directory of fname:
            path = Path(fname)
            if not os.path.exists(path.parent):
                os.makedirs(path.parent)
            plt.savefig(fname, dpi = 300, bbox_inches = 'tight')
            plt.close()
            #print('Plotting alignment section...')

        #######################################################################

        # 5 - Get three component envelopes for each station from the RMS value
        # of the single-component envelopes (Hock et al (2000), eq. 15). This
        # needs to be done station by station and for some stations there may be
        # only one or two envelopes, instead of three. Don't use those cases!
        # The easiest then is to create a new stream with station
        # names as key.
        e3c = {}

        # Create stream for each station:
        for key in sts3:
            for trace in sts3[key]:
                for key2 in station_sts:
                    if trace.stats.station == key2:
                        station_sts[key2].append(trace)

        # Get three component envelopes:
        for key in station_sts:
            if len(station_sts[key]) == 3:
                # Add a trace from the old stream to the new one:
                e3c[key] = station_sts[key][0]
                # RMS value of the single-component envelopes:
                env3c = np.sqrt((station_sts[key][0].data)**2 \
                                + (station_sts[key][1].data)**2 \
                                + (station_sts[key][2].data)**2)
                # Replace trace data:
                e3c[key].data = env3c
                e3c[key].stats.channel = '3C_ENV'

        # However, it is not very practical to work with dictionaries. Let's
        # save this in a normal stream we can easily use afterwards.
        envs3c = Stream()

        for key in e3c:
            envs3c.append(e3c[key])

        #######################################################################

        # For some events, it may happen that none of the station streams has
        # three traces (for ASAR we may have more than 5 vertical traces for a
        # given event but no 3-comp traces at all).
        # We need to include that possibility in the code:

        if envs3c != Stream():

            ###################################################################


            # 6- Stack all 3-component envelopes.

            env3c = stack_single(envs3c)[0]

            ###################################################################

            # 7- Normalise using the integral over the direct wave arrival.

            # 7.1. We want our t to be in seconds FROM THE EVENT, not from the
            # beginning of the trace! That is why we subtract 120 from t_P (our
            # traces start two minutes before the event happened).
            npts = sts3['st_z'][0].stats.npts
            srate = sts3['st_z'][0].stats.sampling_rate
            delta = sts3['st_z'][0].stats.delta
            t = (t_P_max-120 - tJ) + np.arange(0, npts / srate, delta)

            # 7.2. Indices of the maximum amplitude and P wave arrival time.
            tP_ind = (np.abs(t - (t_P_max - 120))).argmin()# P wave arrival, ok

            maxamp3c_ind = (np.abs(env3c - max(env3c))).argmin()#ok

            # 7.3. Index of the beginning of the coda. We consider the direct
            #      arrival to end tJ seconds after the maximum amplitude (this
            #      is when the EFM/EFMD can be applied).
            coda_ind = (tP_ind + tJ * envs3c[0].stats.sampling_rate)#ok

            # b. Calculate the normalised coda envelope from the squared
            # envelopes by dividing by the integral over the length of the
            # direct arrival for each one of them. As described in Korn (1997),
            # we take the integral from the beginning of the trace to the
            # maximum amplitude and then multiply by 2.
            # b.1. Define x (time):
            x = t[:maxamp3c_ind]; dx = envs3c[0].stats.delta
            # b.2. Define y (coda data):
            y3c = (env3c[:maxamp3c_ind])**2
            # b.3. Integrate:
            Id3c = 2 * np.trapz(y3c, x, dx = dx)

            # b.4. Replace non - normalized envelopes with the new ones:
            nce_3c = np.sqrt(env3c**2 / Id3c)

            return envs3c, nce_3c, t, coda_ind, Id3c

        else:

            print('No usable data found for this event...')
            pass

    else:
        print('Only ' + str(len(sts['st_z'])) + ' traces for this array and \
              event were found.')
        print('Analysis not possible for this event')


###############################################################################
####                    END OF THE FUNCTION                                ####
###############################################################################




@jit
def lin_misfits(params, time, envelope, coda_ind):
    '''
    This function computes the residuals between my linear function and my
    actual data. I need this as an input for the least_squares function. The
    parameters of my fittingfunction need to be defined as elements of the same
    array (x[0], x[1], x[2], ...)

    Arguments:
         - params: (np.array) array containing the initial guess for the
           parameters (2) of my linear fitting function (ax + b)
         - time: (np.array) time vector in seconds from origin time
         - envelope: (np.array) stacked envelope
         - coda ind: (float) index for coda start

    '''

    # We only fit the envelope from the beginning of the coda til the end of it.
    # Parabolic fitting:
    #s_rate = int(1/(time[1] - time[0]))
    x = time[coda_ind:-100]
    y = np.log10((envelope[coda_ind:-100])**2)# THIS IS MY DATA.

    syn_y = params[0] + params[1]*x# THIS IS MY PREDICTION OF MY DATA

    # Misfits: these are the differences between my data and my fitting line.
    misf = syn_y - y

    return misf

###############################################################################
####                    END OF THE FUNCTION                                ####
###############################################################################




def linear_least_squares(time, envelope, coda_ind):
    '''
    This function uses scipy.optimize.least_squares to fit our data to a linear
    function.

    Arguments:
         - time: (np.array) time vector in seconds from origin time
         - envelope: (np.array) stacked envelope
         - coda ind: (int) index for coda start

    Output:
        -least_sq_res: (dict) dictionary containing the results of the linear
                       fitting. Residuals are under key 'fun', jacobian under
                       key 'jac', coefficient(s) under key 'x'. See documentation
                       from scipy.optimize.least_squares for more information.

    '''

    # Initial estimate of my parameters:
    # We know b should be around the minimum of ln(ratio + 1) and that a is
    # always positive (we may use a more restrictive condition further on). We
    # also know that it is smaller than 10 (from our plots of parabolic functions
    # and mean saturation values of f and ln(ratio + 1).
    # Let's just use a = 0.5 as the initial estimate.

    params0 = np.array([1, -1e-3])

    # Run standard least squares:
        # INPUTS:
            # misfits = The function that calculates the misfits
            # params0 = Initial guess for the parameters
            # loss = Type of loss function ('linear' (default), 'soft_l1',
            #        'huber', 'cauchy', 'arctan')
            # max_nfev = number of iterations
            # f_scale = Margin between inlier and outlier residuals (inlier
            #           residuals should not significantly exceed this value
            # args = Additional arguments passed to fun
    least_sq_res = least_squares(lin_misfits, params0, loss = 'soft_l1',
                                 max_nfev = 500, f_scale = 0.8,
                                 args = (time, envelope, coda_ind))

    return least_sq_res

###############################################################################
####                    END OF THE FUNCTION                                ####
###############################################################################




@jit
def EFM_misfits(params, Qs, fcs, v):
    '''
    This function computes the residuals between the analytical approximation
    obtained by Fang and Muller (1996) and the Qs values obtained from my data.
    I need this as an input for the least_squares function. The parameters of
    my fitting function need to be defined as elements of the same array
    (x[0], x[1], x[2], ...)

    Arguments:
         - params: (np.array) array containing the initial guess for the
           parameters (2) of the fitting function
         - Qs: (list) list of Qs values obtained from my data
         - fcs: (list) frequencies list
         - v: (float) average velocity of my scattering layer

    '''

    # Constants from Fang and Muller for the exponential ACF in 3d elastic media:
    c1 = 28.73; c2 = 16.77; c3 = 2.40

    # Parameters:
    a = params[0]; E = params[1]
    x = fcs
    y = Qs# THIS IS MY DATA
    # Fang and Muller's equation contains the factor (aw/v) multiple times. It
    # is better to define it outside the equation:
    factor = a * 2 * np.pi * x / v

    # Theoretical Qs values:
    syn_y = (E**2) * (c1 * (factor**3)) / (1 + c2 * (factor**2) + c3 * (factor**4))

    # Misfits: these are the differences between my data and my fitting line.
    misf = syn_y- y

    return misf

###############################################################################
####                    END OF THE FUNCTION                                ####
###############################################################################




def EFM_least_squares(Qs, fcs, v, params0):
    '''
    This function uses scipy.optimize.least_squares to fit our data to a linear
    function.

    Arguments:
         - params0: (np.array) array containing the initial guess for the
           parameters (2) of the fitting function
         - Qs: (list) list of Qs values obtained from my data
         - fcs: (list) frequencies list
         - v: (float) average velocity of my scattering layer

    Output:
        -least_sq_res: (dict) dictionary containing the results of the linear
                       fitting. Residuals are under key 'fun', jacobian under
                       key 'jac', coefficient(s) under key 'x'. See documentation
                       from scipy.optimize.least_squares for more information.
    '''


    # Run standard least squares:
        # INPUTS:
            # misfits = The function that calculates the misfits
            # params0 = Initial guess for the parameters
            # loss = Type of loss function ('linear' (default), 'soft_l1',
            #       'huber', 'cauchy', 'arctan')
            # f_scale = Margin between inlier and outlier residuals (inlier
            #           residuals should not significantly exceed this value
            # args = Additional arguments passed to fun
    least_sq_res = least_squares(EFM_misfits, params0, loss = 'soft_l1',
                                 max_nfev = 500, f_scale = 0.8,
                                 args = (Qs, fcs, v))

    return least_sq_res

###############################################################################
####                    END OF THE FUNCTION                                ####
###############################################################################




@jit
def QiQd_misfits(params, fcs, a1s, alpha):
    '''
    This function computes the residuals between the analytical approximation
    obtained by Korn (1990) and the coda decay coefficients (a1) obtained from
    my data. I need this as an input for the QiQd_least_squares function. The
    parameters of my fitting function need to be defined as elements of the
    same array (x[0], x[1], x[2], ...)

    Arguments:
         - params: (np.array) array containing the parameters (2) of my fitting
                   function
         - a1s: (np.array) list of a1 values obtained from my data
         - fcs: (list) frequencies list
         - alpha: (float) power for the frequency dependency of Qi/Qdiff

    '''

    # Parameters:
    Qi = params[0]; Qd = params[1]
    x = fcs
    y = a1s# THIS IS MY DATA

    # Equation to fit:
    c = 2 * np.pi * np.log10(np.e)
    syn_y = c * (Qd + Qi * (x**(1 - alpha)))# THIS IS MY PREDICTION OF MY DATA.

    # Misfits: these are the differences between my data and my fitting line.
    misf = syn_y - y

    return misf

###############################################################################
####                    END OF THE FUNCTION                                ####
###############################################################################




def QiQd_least_squares(fcs, a1s, alpha):
    '''
    This function uses scipy.optimize.least_squares to fit our data to a linear
    function.

    Arguments:
         - fcs: (list) frequencies list
         - a1s: (np.array) list of a1 values obtained from my data
         - alpha: (float) power for the frequency dependency of Qi/Qdiff

    Output:
        -least_sq_res: (dict) dictionary containing the results of the linear
                       fitting. Residuals are under key 'fun', jacobian under
                       key 'jac', coefficient(s) under key 'x'. See documentation
                       from scipy.optimize.least_squares for more information.
    '''

    # Initial estimate of my parameters:
    params0 = np.array([1e3, 1e3])

    # Run standard least squares:
        # INPUTS:
            # misfits = The function that calculates the misfits
            # params0 = Initial guess for the parameters
            # loss = Type of loss function ('linear' (default), 'soft_l1',
            #       'huber', 'cauchy', 'arctan')
            # f_scale = Margin between inlier and outlier residuals (inlier
            #          residuals should not significantly exceed this value
            # args = Additional arguments passed to fun
    least_sq_res = least_squares(QiQd_misfits, params0, loss = 'soft_l1',
                                 max_nfev = 500, f_scale = 0.8,
                                 args = (fcs, a1s, alpha))

    return least_sq_res

###############################################################################
####                    END OF THE FUNCTION                                ####
###############################################################################



def plot_envelopes(array_nm, ev_date, envs, nce, fband, tJ, filename = None,
                   show_plot = False):

    '''
    This function plots a section with the 1-comp or 3-comp envelopes registered
    at each station for a given event and array, as well as the global normalised
    coda envelope that we will use for the EFM/EFMD.

    Arguments:
         - array_nm: (str) array name/code (for plot title)
         - ev_date: (str) event date in format DDMMYYTHHMM (for plot title)
         - envs: (Obspy stream) stream of envelopes for the event
         - nce: (np.array) normalised coda envelope for the event
         - fband: (str) frequency band (capital letter from A to H, only for plot
                  title)
         - tJ: (float) time it takes P waves to reach the free surface of the model
         - filename: (str) path + filename where we want to store our figure (optional)
         - show_plot: (bool) If False, plot will be closed without displaying it
    '''

    # We need theoretical P wave arrival times ONLY to calculate the time vector
    # for plots.
    t_P, t_P_date = get_Pwave_arrivals(envs, ev_date, model = 'prem')

    ###########################################################################

    # Add the normalised coda envelope as a trace to a copy of the envelope stream.
    envels = copy.deepcopy(envs)
    envels.append(Trace())
    envels[- 1].stats.network = envels[0].stats.network
    envels[- 1].stats.station = 'TOTAL_NCE'
    envels[- 1].stats.starttime = envels[0].stats.starttime
    envels[- 1].stats.delta = envels[0].stats.delta
    envels[- 1].stats.sampling_rate = envels[0].stats.sampling_rate
    envels[- 1].data = nce

    ################          PLOT SECTION         ############################
    ###########################################################################

    # PREPARE PLOT:
    #print('Plotting normalised coda envelope sections...')
    #print(' ')

    # Time vector for plots: we want it to be in seconds FROM THE EVENT, not
    # from the beginning of the trace. That is why we subtract 120 from t_P.
    t = (max(t_P) - 120 - tJ) \
        + np.arange(0, envels[0].stats.npts/envels[0].stats.sampling_rate,
                    envels[0].stats.delta)

    colors = ['mediumaquamarine', 'darkslategrey', 'seagreen', 'forestgreen']

    # Station labels of the traces present in the stream:
    stlbs = []
    for trace in envels:
        stlbs.append(trace.stats.station)

    # Y axis ticks:
    a = 0.18; b = 0.1
    y_tks = np.arange(0, a*len(stlbs), a); y_tks[- 1] = y_tks[- 1]

    f = plt.figure(figsize = (20, 15))

    suptit = 'Normalised coda envelopes'
    f.suptitle(suptit, fontsize = 20)
    #plt.text(20, 5, suptit, horizontalalignment = 'center', fontsize = 24)

    ###########################################################################

    ev_mag = envels[0].stats.sac.mag
    ev_dep = envels[0].stats.sac.evdp/1000
    dist_degs = envels[0].stats.sac.gcarc

    tit = array_nm + ', ' + fband + ', ' + str(ev_date) + ', M = ' + str(ev_mag) \
        + ', dpth = ' + str(ev_dep) + 'km' + ', dist = ' + str(dist_degs) + 'degs.'

    ax1 = plt.gca()
    ax1.set_yticks(y_tks)
    ax1.set_title(tit, fontsize = 20)
    ax1.set_xlabel('Time (seconds)', fontsize = 18)
    #ax1.set_xticks(x_tks); #ax.set_yticks(y_tks)
    ax1.set_yticklabels(stlbs, fontsize = 18)
    ax1.tick_params(axis = 'x', labelsize = 18)
    ax1.grid()

    for v in range(len(envels)):
        if envels[v].stats.station == 'TOTAL_NCE':
            ax1.plot(t, (a*v) + envels[v].data, 'k', linewidth = 1)
        else:
            ax1.plot(t, (a*v) + envels[v].data, color = colors[0], linewidth = 1)#Ok

    ax1.axis([max(t_P)-120-tJ, max(t_P)-120 + 3*tJ, -0.13,
              max((a*v + b) + envels[v].data) + b])

    # Save plot in the parent directory of fname:
    if filename is not None:
        path = Path(filename)
        if not os.path.exists(path.parent):
            os.makedirs(path.parent)
        plt.savefig(filename, dpi = 300, bbox_inches = 'tight')
    if show_plot == False: plt.close('all')

###############################################################################
####                    END OF THE FUNCTION                                ####
###############################################################################


