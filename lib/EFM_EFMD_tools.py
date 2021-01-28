'''
Created on Dec 10, 2017
@author: Itahisa Gonzalez Alvarez

This script contains auxiliary functions needed to run the data preprocessing
or EFM/EFMD analysis.

'''

import copy
import obspy
import numpy as np
from glob import glob
from obspy import read, Stream
import matplotlib.pyplot as plt
from obspy.taup import TauPyModel
from fk_analysis import get_Pwave_arrivals
from trace_alignment import align_stream_fk, stack_single



def create_stream(array_nm, main_sac_path, comp):

    '''

    Creates a stream with all the UNFILTERED and UNALIGNED but NORMALISED traces
    available for the seismic array for a particular frequency band (even if this
    function WILL NOT filter the traces).
    The number of good traces for each frequency band varies for each event, so
    we need to take that into account, especially since this function will be
    used to create the basic streams that I will later filter, align and
    normalize for future analysis. Then, it sorts them by distance to the event.

    Arguments:
        -array_nm (str) name of the array
        -main_sac_path (str) path to the directory where our sac files are
        -comp (str) component we want to get traces from

    It returns:
        -a stream with all the UNFILTERED traces

    '''

    # Create streams and variables I will need later:
    st=Stream()

    # We are taking the traces for just one of the frequency bands
    sac_path=main_sac_path

    if comp=='Z':
        files=glob(sac_path+'/*HZ*.SAC')
    elif comp=='R':
        files=glob(sac_path+'/*HR*.SAC')
    elif comp=='T':
        files=glob(sac_path+'/*HT*.SAC')
    n=len(files)

    # Set a condition, we can't do the analysis if we don't have enough traces!
    if n < 5 and array_nm == 'PSA':
        print('Only '+str(n)+' traces were found for this array, component and \
              event.')
        print('Analysis not possible for this event')

    else:

        print(str(n)+' traces were found for this array, component and event.')
        #print('Creating stream...')

        for q in range(n):

            tr=read(files[q])

            # We need to detrend and taper before filtering to remove any linear
            # trend and to make sure our traces start with zero.
            tr.detrend('linear')
            tr.taper(max_percentage=0.05,type='hann')
            tr[0].stats.sac.dist=1000*tr[0].stats.sac.dist

            # Previous filtering, apply to all traces:
            tr.filter('highpass',freq=0.33333)

            # Append trace to our unfiltered stream:
            st.append(tr[0])


        # Sort the traces within the streams so the first one corresponds to the
        # station closest to the source and the last one to the furthest one.
        st.traces.sort(key=lambda x: x.stats.sac.dist)
        #

        # All traces should be exactly 144000/72000 samples long. However, some
        # of them have 144001 and some are shorter than they should. Delete the
        # last one in these cases:

        if array_nm!='AS':
            for trace in st:
                if len(trace.data)>144000:
                    trace.data=trace.data[0:144000]
                if len(trace.data)<144000:
                    diff=144000-len(trace.data)
                    zers=np.zeros((diff,1))
                    new_data=np.append(trace.data,zers)

        else:
            for trace in st:
                if len(trace.data)>72000:
                    trace.data=trace.data[0:72000]
                if len(trace.data)<72000:
                    diff=72000-len(trace.data)
                    zers=np.zeros((diff,1))
                    new_data=np.append(trace.data,zers)
                    trace.data=new_data

        # Call the normalize_traces function:
        st=normalize_traces(st)

    return st

###############################################################################
####                       END OF THE FUNCTION                             ####
###############################################################################




def create_stream_EFM(array_nm, ev_date, main_sac_path, comp, fband, tJ,
                      fname = None):

    '''

    Creates a stream with all the FILTERED, ALIGNED and NORMALIZED traces
    available for an array. It uses the create_stream and get_Pwave_arrivals
    functions from F_trace_env_stream_creation to create the initial stream of
    unfiltered and unaligned but normalized traces and calculate the theoretical
    P wave arrival times, respectively. It also uses align_stream, align_traces,
    align_traces_Pwave_arrivals and align_traces_xcorr_TFWM from F_trace_alignment
    to align tracesin the initial stream using different methods.

    Arguments:
         - array_nm: (str) name/code of the array
         - main_sac_path: (str) path to the directory where our sac files are
         - comp: (str) component we want to get traces from
         - fband: (str) frequency band we want to filter the traces into
                        (capital letter from A to H, see fbands dictionary below
                         to check frequency range)
         - tJ: (float) time it takes P waves to reach the free surface
         - fname: (str) path and filename for the alignment plot (optional)

    Output:
         - a stream with all the FILTERED, ALIGNED and NORMALIZED traces
         - alignment plot (optional)

    '''

    # Inputs:
    fbands = {'A':[0.5, 1], 'B':[0.75, 1.5], 'C':[1, 2], 'D':[1.5, 3],
              'E':[2, 4], 'F':[2.5, 5], 'G':[3, 6], 'H':[3.5, 7]}

    ###########################################################################

    # 1. Create a stream of unaligned and unfiltered but normalised traces:
    try:

        st = create_stream(array_nm, main_sac_path, comp)
        starttime = st[0].stats.starttime

        #######################################################################

        # 3. Align traces:

        # Getting theoretical P wave arrivals to trim the traces to the time
        # window of interest.
        t_P, t_P_date = get_Pwave_arrivals(st, ev_date, model='prem')

        # Use fk analysis to align the traces:
        #In degrees:
        st2 =  align_stream_fk (array_nm, ev_date, st)

        for trace in st2:
            trace.stats.starttime = starttime

        #######################################################################

        # 2. Filter traces:
        #srate = int(st[0].stats.sampling_rate)
        st.filter('bandpass', freqmin = fbands[fband][0],
                    freqmax = fbands[fband][1], corners = 2, zerophase = True)

        #######################################################################

        #                           TRACE TRIMMING                            #
        # Trim traces to the time window of interest:
        t1 = tJ; t2 = 3*tJ# Time window of interest goes from t_P - t1 to t_P + t2
        #print('Trimming stream to the time window of interest...')
        m = len(st2)
        mean_t_P = np.mean(t_P); mean_t_P_date = starttime + mean_t_P
        for i in range(m):
            st2[i].trim(starttime = mean_t_P_date-t1, endtime = mean_t_P_date + t2)
            # Make it start and end in zero again:
            st2[i].taper(max_percentage = 0.05, type = 'hann')

        if fname != None:
            #print('Plotting alignment section...')
            plt.figure(figsize = (20, 10))
            for trace in st2:
                plt.plot(trace.data, 'k', linewidth = 0.8)
            plt.title(str(trace.stats.starttime) + ', ' + str(trace.stats.channel))
            plt.savefig(fname, bbox_inches = 'tight')
            plt.close()

        print('... stream successfully created')

        return st2

    except:

        print('Aborting analysis for this event...')

###############################################################################
####                       END OF THE FUNCTION                             ####
###############################################################################




def envelope_traces(stream):

    '''
    Function that replaces the traces within a stream for their respective
    envelopes.

    Arguments:
        -stream

    Output:
        -stream containing the envelopes of the traces

    '''

    #print('Getting envelopes...')

    # Calculate the envelope of the traces:
    for trace in stream:
        trace.data=obspy.signal.filter.envelope(trace.data)
    #

    return stream

###############################################################################
####                       END OF THE FUNCTION                             ####
###############################################################################





def filter_stream(stream,fmin,fmax):

    '''
    This function bandspass filters the traces within a stream from fmin to fmax.

    Arguments:
        -stream
        -fmin = minimum frequency for the filter
        -fmax = maximum frequency for the filter

    Output:
        -stream of filtered traces

    Note:
        This is a very basic function but it will help me when I am creating
        dictionaries of streams later on.

    '''

    #print('Filtering traces...')

    stream.filter('bandpass',freqmin=fmin,freqmax=fmax,corners=3,zerophase=True)

    return stream

###############################################################################
####                       END OF THE FUNCTION                             ####
###############################################################################





def normalize_traces(stream):

    '''
    Function that takes the traces within an Obspy stream an normalizes their
    amplitudes to the maximum amplitude in the entire stream.

    Arguments:
        -stream (Obspy stream)

    Output:
        -stream of normalized traces

    '''

    max_amp = []
    for trace in stream:
        max_a = abs(trace.data.max())
        min_a = abs(trace.data.min())
        if max_a > min_a: max_amp.append(trace.data.max())
        else: max_amp.append(abs(trace.data.min()))

    # Normalize streams:
    #print('Normalizing traces...')
    for trace in stream:
        trace.data = trace.data / max(max_amp)

    return stream

###############################################################################
####                       END OF THE FUNCTION                             ####
###############################################################################




def datetime_string (datetime):

    '''
    Function that creates a string out of a given datetime object, so it can
    be used in file and directory names and avoid overwriting results.

    Arguments:
        -datetime: datetime.datetime or UTCDateTime object

    Returns:
        -datetime_string (str): String of the chosen date and time in
                                YYYYMMDDTHHMMSS format

    '''

    year = str(datetime.year); month = str(datetime.month)
    day = str(datetime.day); hour = str(datetime.hour)
    minute = str(datetime.minute); second = str(datetime.second)
    if len(month) < 2: month = '0' + month
    if len(day) < 2: day = '0' + day
    if len(hour) < 2: hour = '0' + hour
    if len(minute) < 2: minute = '0' + minute
    if len(second) < 2: second = '0' + second
    datetime_str = year + month + day + 'T' + hour + minute + second

    return datetime_str

###############################################################################
####                       END OF THE FUNCTION                             ####
###############################################################################




def stream_plotting_trim_stack(array_nm, ev_date, stream, comp, tJ,
                               filename = None, show_plots = False):

    '''

    Function that plots a section with the filtered, trimmed and aligned traces
    for a given array, event and frequency band. The theoretical P wave arrivals
    are also marked on the plot. The traces will NOT be aligned and this script is
    written under the assumption that the traces are trimmed to the time window
    of interest.
    This function plots the stacked seismogram at the top of the section.

    Arguments:
        -array name
        -event date (format 'yyyymmddThhmm')
        -stream
        -component (need it for the title)

        *Optional:
            -filename (with path!) if we want to automatically save the figure

    Output:
        -plot containing the section for the, constrained within the time
         window of interest and marking the theoretical P wave arrival

    Note:
        The traces need to be normalized!

    '''

    # Create a copy of the original stream so we don't modify it.
    st = copy.deepcopy(stream)

    # Normalize traces:
    st = normalize_traces(st)

    # Get stack (just for plot):
    stack, stack_sd = stack_single(st)

    # Trace information:
    for trace in st:
        ev_lat = trace.stats.sac.evla
        ev_lon = trace.stats.sac.evlo
        ev_dep = trace.stats.sac.evdp/1000# In km
        ev_mag = trace.stats.sac.mag

    # Choose model to calculate theoretical traveltimes:
    model = TauPyModel(model = 'prem')

    # Create variable to store theoretical P arrival times:
    P_arrs = {}
    # Save other arrivals too:
    pP_arrs = {}; PcP_arrs = {}; PP_arrs = {}; sP_arrs = {}; ScP_arrs = {}

    # Get theoretical P wave arrival times so we can plot a red line marking it:
    # We can get them just for one f band, they should be the same for the others.
    t_P = []; t_pP = []; t_PcP = []; t_PP = []; t_sP = []; t_ScP = []

    for trace in st:

        dist_degs = trace.stats.sac.gcarc
        # We are only interested in P wave arrivals:
        arrivals = model.get_travel_times(source_depth_in_km = ev_dep,
                                          distance_in_degree = dist_degs)

        for arrival in arrivals:
            # Save travel times:
            if arrival.name == 'P':
                P_arrs[trace.stats.station] = 120 + arrival.time
                t_P.append(P_arrs[trace.stats.station])
            if arrival.name == 'pP':
                pP_arrs[trace.stats.station] = 120 + arrival.time
                t_pP.append(pP_arrs[trace.stats.station])
            if arrival.name == 'PcP':
                PcP_arrs[trace.stats.station] = 120 + arrival.time
                t_PcP.append(PcP_arrs[trace.stats.station])
            if arrival.name == 'PP':
                PP_arrs[trace.stats.station] = 120 + arrival.time
                t_PP.append(PP_arrs[trace.stats.station])
            if arrival.name == 'sP':
                sP_arrs[trace.stats.station] = 120 + arrival.time
                t_sP.append(sP_arrs[trace.stats.station])
            if arrival.name == 'ScP':
                ScP_arrs[trace.stats.station] = 120 + arrival.time
                t_ScP.append(ScP_arrs[trace.stats.station])

    # Get time shifts from theoretical P wave arrivals:
    max_t = max(t_P);P_tt_diffs = {}
    if t_pP != []: max_pP_t = max(t_pP); pP_tt_diffs = {}
    if t_PcP != []:max_PcP_t = max(t_PcP); PcP_tt_diffs = {}
    if t_PP != []:max_PP_t = max(t_PP); PP_tt_diffs = {}
    if t_sP != []:max_sP_t = max(t_sP); sP_tt_diffs = {}
    if t_ScP != []:max_ScP_t = max(t_ScP); ScP_tt_diffs = {}

    for station in P_arrs:
        P_tt_diffs[station] = P_arrs[station] - max_t
        if t_pP != []:pP_tt_diffs[station] = pP_arrs[station] - max_pP_t
        if t_PcP != []:PcP_tt_diffs[station] = PcP_arrs[station] - max_PcP_t
        if t_PP != []:PP_tt_diffs[station] = PP_arrs[station] - max_PP_t
        if t_ScP != []:ScP_tt_diffs[station] = ScP_arrs[station] - max_ScP_t
        if t_sP != []:sP_tt_diffs[station] = sP_arrs[station] - max_sP_t

    ev_time = st[0].stats.starttime - max(t_P) + 120 + 10 # The traces were
                                                          # trimmed and the
                                                          # starttime was changed
                                                          # to when they actually
                                                          # start now, but that
                                                          # is NOT the event
                                                          # time! The event time
                                                          # is that time minus
                                                          # max(t_P)

    ###############          PLOT SECTION         #############################
    ###########################################################################

    # PREPARE PLOT:

    colors = ['limegreen','green']

    # Time vector for plots: we want it to give time in seconds FROM THE EVENT,
    # not from the start of the trace.
    t = (max(t_P) - 120 - tJ) + np.arange(0, st[0].stats.npts \
                                          / st[0].stats.sampling_rate,
                                          st[0].stats.delta)

    #X axis ticks for plots (common for all plots):
    #x_tks = np.arange(450,475,5)
    x_tks = np.arange(max(t_P) - 120 - tJ, max(t_P) - 120 + 3 * tJ, 10)
    x_tks_labels = []
    for tk in x_tks:
        x_tks_labels.append(int(tk))

    f = plt.figure(figsize = (20,10))

    suptit = array_nm + ', ' + comp + ', ' + str(ev_time) + ', loc: ' \
             + str(ev_lat) + ',' + str(ev_lon)
    f.suptitle(suptit, fontsize = 30)

    ###########################################################################

    tit = 'Mag: ' + str(ev_mag) + ', depth: ' + str(ev_dep) + ' km' + ', dist :' \
        + str(dist_degs) + ' degrees'

    # Station labels of the traces present in the stream:
    stlbs = []
    for trace in st:
        stlbs.append(trace.stats.station)
    stlbs.append('Stack')
    # This contains ONLY the stations whose envelopes are contained in the stream.

    # Y axis ticks:
    y_tks = np.arange(0, 0.6 * (len(stlbs) + 1), 0.6)

    ax1 = plt.gca()
    ax1.set_yticks(y_tks)
    ax1.set_title(tit, fontsize = 30)
    ax1.axis([x_tks[0], x_tks[-1], min(st[0].data)-0.3, 0.63*(len(stlbs)+1)])
    ax1.set_xlabel('Time (seconds)', fontsize = 24)
    ax1.set_xticks(x_tks);#ax.set_yticks(y_tks)
    ax1.set_yticklabels(stlbs, fontsize = 24)
    ax1.set_xticklabels(x_tks_labels, fontsize = 24)
    #ax1.yaxis.tick_right()

    for v in range(len(stlbs) - 1):
        # Expression for the phases vertical markers:
        vm = np.arange(0.6 * v - 0.5, 0.6 * v + 0.5, 0.005)
        Parrs = (P_arrs[stlbs[v]]-120-P_tt_diffs[stlbs[v]])*np.ones(len(vm))
        if t_pP != []:
            pParrs = (pP_arrs[stlbs[v]] - 120 - pP_tt_diffs[stlbs[v]]) * np.ones(len(vm))
        if t_PcP != []:
            PcParrs = (PcP_arrs[stlbs[v]] - 120 - PcP_tt_diffs[stlbs[v]]) * np.ones(len(vm))
        if t_PP != []:
            PParrs = (PP_arrs[stlbs[v]] - 120 - PP_tt_diffs[stlbs[v]]) * np.ones(len(vm))
        if t_sP != []:
            sParrs = (sP_arrs[stlbs[v]] - 120 - sP_tt_diffs[stlbs[v]]) * np.ones(len(vm))
        if t_ScP != []:
            ScParrs = (ScP_arrs[stlbs[v]] - 120 - ScP_tt_diffs[stlbs[v]]) * np.ones(len(vm))

        if st[v].stats.station in stlbs:
            if v == 0:
                ax1.plot(t, (0.6 * v) + (st[v].data), color = colors[0],
                         linewidth = 1, label = 'Data')
                ax1.plot(Parrs, vm, 'r', linewidth = 1, label = 'P')
                if t_pP != []:
                    ax1.plot(pParrs, vm, 'b', linewidth = 1, label = 'pP')
                if t_PcP != []:
                    ax1.plot(PcParrs, vm, 'g', linewidth = 1, label = 'PcP')
                if t_PP != []:
                    ax1.plot(PParrs, vm, color = 'orange', linewidth = 1, label = 'PP')
                if t_ScP != []:
                    ax1.plot(ScParrs, vm, color = 'purple', linewidth = 1, label = 'ScP')
                if t_sP != []:
                    ax1.plot(sParrs, vm, color = 'brown', linewidth = 1, label = 'sP')
            else:
                ax1.plot(t, (0.6 * v) + (st[v].data), color = colors[0], linewidth = 1)
                ax1.plot(Parrs, vm, 'r', linewidth = 1)
                if t_pP != []: ax1.plot(pParrs, vm, 'b', linewidth = 1)
                if t_PcP != []: ax1.plot(PcParrs, vm, 'g', linewidth = 1)
                if t_PP != []: ax1.plot(PParrs, vm, color = 'orange', linewidth = 1)
                if t_ScP != []: ax1.plot(ScParrs, vm, color = 'purple', linewidth = 1)
                if t_sP != []: ax1.plot(sParrs, vm, color = 'brown', linewidth = 1)

    ax1.plot(t, (0.6 * (v + 1)) + (stack), color = 'k', linewidth = 2, label = 'Stack')
    plt.legend(loc = 'upper right', fontsize = 18)

    #Save figure:
    if filename != None:
        fname = filename
        plt.savefig(fname, transparent = False)

    if show_plots == False: plt.close()


###############################################################################
####                       END OF THE FUNCTION                             ####
###############################################################################
