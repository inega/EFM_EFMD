'''
Created on Nov 16, 2017
@author: Itahisa Gonzalez Alvarez

Functions related to the alignment of traces required for data preprocessing
and the EFM/EFMD analysis.

'''

import copy
import numpy as np
import matplotlib.pyplot as plt
from obspy.signal.cross_correlation import xcorr_max
from fk_analysis import get_timeshifts, get_Pwave_arrivals
from obspy.signal.cross_correlation import correlate as correlate




def align_stream_fk (array_nm, ev_date, stream):

    '''
    Function to align the traces within a stream according to the time shifts
    obtained from the fk analysis.

    Arguments:
        -array_nm: (str) name of the array
        -ev_date: (str) Event date in DDMMYYYYThhmmss format
        -stream: (Obspy stream) traces for the event and array

    Output:
        -stream containing the aligned traces of the input stream

    '''

    num_traces = len(stream)

    # Run fk analysis to get time shifts for each trace in the stream:
    smin = -11.12; smax = 11.12; s_int = 0.1; tmin = 1; tmax = 5; units = 'degrees'
    tshifts = get_timeshifts(stream, smin, smax, s_int, tmin, tmax, units)

    # Define sampling rate and its inverse:
    sampling_rate = stream[0].stats.sampling_rate
    delta = stream[0].stats.delta

    # Get number of samples we need to shift each trace:
    num_sam = []; num_secs = []
    for v in range(num_traces):
        num_sam.append(int(sampling_rate * tshifts[v]))
        num_secs.append(num_sam[v] * delta)

    ###########################################################################

    # Create a copy of the stream:
    st = copy.deepcopy(stream)

    # IMPORTANT NOTE:
    #   - If the time shift is POSITIVE, the trace will be shifted towards the
    #     RIGHT (future time)
    #   - If time shift is NEGATIVE, trace shifted to the LEFT (back in time)

    # Apply shift to traces:
    tr_length = []
    for i in range(num_traces):
        if num_sam[i] > 0:
            add_zeros = np.zeros(num_sam[i])
            st[i].data = np.concatenate((add_zeros, st[i].data))
            st[i].stats.starttime = st[i].stats.starttime - num_secs[i]
        elif num_sam[i] < 0:
            st[i].data = st[i].data[abs(num_sam[i]):]
            st[i].stats.starttime = st[i].stats.starttime + num_secs[i]
        elif num_sam[i] == 0:
            pass
        tr_length.append(st[i].stats.npts)

    # All traces should have the same length, so get maximum length of a trace
    # in the current stream:
    max_length = max(tr_length)

    for trace in st:
        # Add zeros at the end of the shorter traces:
        if trace.stats.npts < max_length:
            num_zeros = np.zeros(abs(max_length - trace.stats.npts))
            trace.data = np.concatenate((trace.data, num_zeros))

    return st

###############################################################################
####                       END OF THE FUNCTION                             ####
###############################################################################





def align_stream_3c_fk (array_nm, stream_3c, ev_date, tJ, fname = None):

    '''
    This function uses fk analysis to obtain the timeshifts we need to apply to
    each trace in the 3-component stream to align it with respect to a reference
    trace (the one with the latest theoretical P wave arrival). Then, it uses
    cross correlation to correct from possible traces with reversed polarity.
    The horizontal traces will be aligned with respect to the vertical one.

    Arguments:
        -stream_3c = dictionary with an Obspy stream for each component. The
                     structure needs to be: {'st_z':z_stream, 'st_t':t_stream,
                     'st_r':r_stream}
        -t_P: (list) theoretical P wave traveltimes for the event, in seconds
                      from the beginning of the trace
    Output:
        -aligned stream

    '''

    # Save the event date from the original stream so we can recover it later:
    starttime = stream_3c['st_z'][0].stats.starttime
    # Number of traces:
    num_traces = len(stream_3c['st_z'])
    # Get t_P:
    t_P, t_P_date = get_Pwave_arrivals(stream_3c['st_z'], ev_date, model = 'prem')
    # To correlate the whole traceS does not work for every event. Trim traces
    # to a narrow time window around the theoretical P wave arrival before doing
    # this.

    # Create a copy of st to trim:
    trimst = copy.deepcopy(stream_3c)

    # Trim traces to the time window of interest:
    t1 = tJ; t2 = 3 * tJ# The chosen time window goes from t_P-t1 to t_P + t2
    mean_t_P = np.mean(t_P); mean_t_P_date = starttime + mean_t_P
    for st in trimst:
        for i in range(num_traces):
            trimst[st][i].trim(starttime = mean_t_P_date - t1,
                                endtime = mean_t_P_date + t2)


    ###########################################################################
    #           STEP 1: FK ANALYSIS TO OBTAIN AN INITIAL ALIGNMENT                           #
    ###########################################################################

    # Run fk analysis to obtain the number of samples we need to shift each
    # trace of our FILTERED streams.

    #print('Running fk analysis to get time shifts and align traces...')

    # Get fk analysis:
    # Get time shifts for each trace in the stream:
    smin = -11.12; smax = 11.12; s_int = 0.1; tmin = 1; tmax = 5; units = 'degrees'
    tshifts = get_timeshifts(stream_3c['st_z'], smin, smax, s_int, tmin, tmax,
                             units)

    # IMPORTANT NOTE:
    #   - If the time shift is POSITIVE, the trace will be shifted towards the
    #     RIGHT (future time)
    #   - If time shift is NEGATIVE, trace shifted backwards in time

    # Get number of samples we need to shift each trace:
    sampling_rate = stream_3c['st_z'][0].stats.sampling_rate
    delta = stream_3c['st_z'][0].stats.delta

    num_sam = []; num_secs = []
    for v in range(num_traces):# Those keys are the name of the stations
        num_sam.append(int(sampling_rate * tshifts[v]))
        num_secs.append(num_sam[v] * delta)
    #

    # Apply time shifts to a copy of the stream:
    st_3c = copy.deepcopy(stream_3c)

    tr_length = []
    for st in st_3c:
        for i in range(num_traces):
            if num_sam[i] > 0:
                add_zeros = np.zeros(num_sam[i])
                st_3c[st][i].data = np.concatenate((add_zeros, st_3c[st][i].data))
            elif num_sam[i] < 0:
                st_3c[st][i].data = st_3c[st][i].data[abs(num_sam[i]):]
            elif num_sam[i] == 0:
                pass
    for trace in st_3c['st_z']:
        tr_length.append(trace.stats.npts)
    #

    # All traces should have the same length:
    # Get maximum length of a trace in the current stream:
    max_length = max(tr_length)

    for st in st_3c:
        for trace in st_3c[st]:
            # Add zeros at the end of the shorter traces:
            if trace.stats.npts < max_length:
                num_zeros = np.zeros(abs(max_length - trace.stats.npts))
                trace.data = np.concatenate((trace.data, num_zeros))
                #print('Final trace length is ', trace.stats.npts)
    #

    # st is now my pre-aligned stream. The timeshifts I obtain from next steps
    # will be applied to st.

    ################################################################################################

    # Apply shifts to trimmed traces as well:
    tr_length = []
    for st in trimst:
        for i in range(num_traces):
            if num_sam[i] > 0:
                add_zeros = np.zeros(num_sam[i])
                trimst[st][i].data = np.concatenate((add_zeros, trimst[st][i].data))
            elif num_sam[i] < 0:
                trimst[st][i].data = trimst[st][i].data[abs(num_sam[i]):]
            elif num_sam[i] == 0:
                pass
    for trace in trimst['st_z']:
        tr_length.append(trace.stats.npts)
    #

    # All traces should have the same length:
    # Get maximum length of a trace in the current stream:
    max_length = max(tr_length)

    for st in trimst:
        for trace in trimst[st]:
            # Add zeros at the end of the shorter traces:
            if trace.stats.npts < max_length:
                num_zeros = np.zeros(abs(max_length - trace.stats.npts))
                trace.data = np.concatenate((trace.data, num_zeros))
                #print('Final trace length is ', trace.stats.npts)

    # SANITY CHECK:

    if fname != None:

        print('Plotting alignment...')

        stack, stack_stdv = stack_single(st_3c['st_z'])

        t = np.arange(st_3c['st_z'][0].stats.npts) * st_3c['st_z'][0].stats.delta
        ev_date = str(st_3c['st_z'][0].stats.starttime)

        plt.figure(figsize = (20, 10))
        for trace in st_3c['st_z']:
            if trace == st_3c['st_z'][0]:
                plt.plot(t, trace.data, color = 'gray', linewidth = 0.8,
                         label = 'Traces')
            else:
                plt.plot(t, trace.data, color = 'gray', linewidth = 0.8)
        plt.plot(t, stack + stack_stdv, 'k', label = 'Stack  +/- st.dev')
        plt.plot(t, stack - stack_stdv, 'k')
        plt.grid()
        plt.title(array_nm + ', ' + ev_date + ', aligned using fk + xcorr')
        plt.legend(loc = 'upper right')
        plt.axis([np.mean(t_P) - 5, np.mean(t_P) + 20, -1, 1])
        plt.savefig(fname, bbox_inches = 'tight')
        plt.close('all')

    return st_3c, trimst

###############################################################################
####                       END OF THE FUNCTION                             ####
###############################################################################




def align_stream_3c_fk_xcorr (array_nm, stream_3c, ev_date, tJ, fname = None):

    '''
    This function uses fk analysis to obtain the timeshifts we need to apply to
    ach trace in the 3-component stream to align it with respect to a reference
    trace (the one with the latest theoretical P wave arrival). Then, it uses
    cross correlation to correct from possible traces with reversed polarity.
    The horizontal traces will be aligned with respect to the vertical one.

    Arguments:
        -stream_3c = dictionary with an Obspy stream for each component. The
                     structure needs to be:
                     {'st_z':z_stream, 'st_t':t_stream, 'st_r':r_stream}
        -t_P: (list) theoretical P wave traveltimes for the event, in seconds
                      from the beginning of the trace
    Output:
        -aligned stream

    '''

    # Save the event date from the original stream so we can recover it later:
    starttime = stream_3c['st_z'][0].stats.starttime
    # Number of traces:
    num_traces = len(stream_3c['st_z'])
    # Get t_P:
    t_P, t_P_date = get_Pwave_arrivals(stream_3c['st_z'], ev_date, model = 'prem')
    # To correlate the whole traceS does not work for every event. Trim traces
    # to a narrow time window around the theoretical P wave arrival before doing
    # this.

    # Create a copy of st to trim:
    trimst = copy.deepcopy(stream_3c)

    # Trim traces to the time window of interest:
    t1 = tJ; t2 = 3 * tJ# The chosen time window goes from t_P - t1 to t_P + t2
    mean_t_P = np.mean(t_P); mean_t_P_date = starttime + mean_t_P
    for st in trimst:
        for i in range(num_traces):
            trimst[st][i].trim(starttime = mean_t_P_date - t1,
                               endtime = mean_t_P_date + t2)

    ###########################################################################
    #          STEP 1: FK ANALYSIS TO OBTAIN AN INITIAL ALIGNMENT                           #
    ###########################################################################

    # Run fk analysis to obtain the number of samples we need to shift each trace of
    # our FILTERED streams.

    #print('Running fk analysis to get time shifts and align traces...')

    # Get fk analysis:
    # Get time shifts for each trace in the stream:
    smin = -11.12; smax = 11.12; s_int = 0.1; tmin = 1; tmax = 5; units = 'degrees'
    tshifts = get_timeshifts(array_nm, ev_date, stream_3c['st_z'], smin, smax,
                             s_int, tmin, tmax, units, stat = 'power')
    # This gives a dictionary with the time shift for each station.

    # IMPORTANT NOTE:
    #   - If the time shift is POSITIVE, the trace will be shifted towards the
    #     RIGHT (future time)
    #   - If time shift is NEGATIVE, trace shifted backwards in time

    # Get number of samples we need to shift each trace:
    sampling_rate = stream_3c['st_z'][0].stats.sampling_rate
    delta = stream_3c['st_z'][0].stats.delta

    num_sam = []; num_secs = []
    for v in range(num_traces):# Those keys are the name of the stations
        num_sam.append(int(sampling_rate * tshifts[v]))
        num_secs.append(num_sam[v] * delta)
    #

    # Apply time shifts to a copy of the stream:
    st_3c = copy.deepcopy(stream_3c)

    tr_length = []
    for st in st_3c:
        for i in range(num_traces):
            if num_sam[i] > 0:
                add_zeros = np.zeros(num_sam[i])
                st_3c[st][i].data = np.concatenate((add_zeros, st_3c[st][i].data))
            elif num_sam[i] < 0:
                st_3c[st][i].data = st_3c[st][i].data[abs(num_sam[i]):]
            elif num_sam[i] == 0:
                pass
    for trace in st_3c['st_z']:
        tr_length.append(trace.stats.npts)
    #

    # All traces should have the same length:
    # Get maximum length of a trace in the current stream:
    max_length = max(tr_length)

    for st in st_3c:
        for trace in st_3c[st]:
            # Add zeros at the end of the shorter traces:
            if trace.stats.npts < max_length:
                num_zeros = np.zeros(abs(max_length - trace.stats.npts))
                trace.data = np.concatenate((trace.data, num_zeros))
                #print('Final trace length is ', trace.stats.npts)
    #

    # st is now my pre-aligned stream. The timeshifts I obtain from next steps
    # will be applied to st.

    ###########################################################################

    # Apply shifts to trimmed traces as well:
    tr_length = []
    for st in trimst:
        for i in range(num_traces):
            if num_sam[i] > 0:
                add_zeros = np.zeros(num_sam[i])
                trimst[st][i].data = np.concatenate((add_zeros, trimst[st][i].data))
            elif num_sam[i] < 0:
                trimst[st][i].data = trimst[st][i].data[abs(num_sam[i]):]
            elif num_sam[i] == 0:
                pass
    for trace in trimst['st_z']:
        tr_length.append(trace.stats.npts)
    #

    # All traces should have the same length:
    # Get maximum length of a trace in the current stream:
    max_length = max(tr_length)

    for st in trimst:
        for trace in trimst[st]:
            # Add zeros at the end of the shorter traces:
            if trace.stats.npts < max_length:
                num_zeros = np.zeros(abs(max_length - trace.stats.npts))
                trace.data = np.concatenate((trace.data, num_zeros))
                #print('Final trace length is ', trace.stats.npts)
    #

    ###########################################################################
    #           STEP 2: REFINE ALIGNMENT BY USING CROSS CORRELATION                                #
    ###########################################################################

    # Stack the traces in trimst['st_z'] to obtain the first reference trace:
    stack, stack_stdv = stack_single(trimst['st_z'])

    # Get number of samples for shifts
    num_sam = []; corrs = []; num_secs = []

    for i in range(num_traces):
        trace = trimst['st_z'][i]

        cc = correlate(stack, trace, 500, demean = False)
        shift, value = xcorr_max(cc)

        # Get number of samples for shifts:
        num_sam.append(int(shift))
        num_secs.append(abs(shift * trimst['st_z'][0].stats.delta))
        corrs.append(value)
        #print('Number of samples to shift is ', num_sam[i])
    #

    # Check for traces with reversed polarity:
    for i in range(num_traces):
        if corrs[i] < -0.8:
            for st in trimst:
                trimst[st][i].data = trimst[st][i].data * (-1)
                st_3c[st][i].data = st_3c[st][i].data * (-1)
            corrs[i] = corrs[i] * (-1)

    # SANITY CHECK:

    if fname != None:

        print('Plotting alignment...')

        stack, stack_stdv = stack_single(st_3c['st_z'])

        t = np.arange(st_3c['st_z'][0].stats.npts) * st_3c['st_z'][0].stats.delta
        ev_date = str(st_3c['st_z'][0].stats.starttime)

        plt.figure(figsize = (20, 10))
        for trace in st_3c['st_z']:
            if trace == st_3c['st_z'][0]:
                plt.plot(t, trace.data, color = 'gray', linewidth = 0.8,
                         label = 'Traces')
            else:
                plt.plot(t, trace.data, color = 'gray', linewidth = 0.8)
        plt.plot(t, stack + stack_stdv, 'k', label = 'Stack  +/- st.dev')
        plt.plot(t, stack - stack_stdv, 'k')
        plt.grid()
        plt.title(array_nm + ', ' + ev_date + ', aligned using fk + xcorr')
        plt.legend(loc = 'upper right')
        plt.axis([np.mean(t_P) - 5, np.mean(t_P) + 20, -1, 1])
        plt.savefig(fname, bbox_inches = 'tight')
        plt.close('all')

    return st_3c, trimst

###############################################################################
####                       END OF THE FUNCTION                             ####
###############################################################################




def improve_alignment_xcorr(stream, t_P):

    '''
    Function that improves the alignment obtained from fk or traveltime by cross
    correlating pairs of traces with the stack or beam in order to correct traces
    with reversed polarity.

    Arguments:
        -stream: (Obspy stream) traces recorded at a given array for an event
        -t_P: (list) theoretical P wave arrival times for each trace

    '''

    # To correlate the whole unfiltered traces does not work for every event.
    # Filter into low freqs and trim traces to a narrow time window around the
    # theoretical P wave arrival before doing this.

    starttime = stream[0].stats.starttime

    # Create a copy of stream to trim and filter:
    trimst = copy.deepcopy(stream)
    trimst.filter('bandpass', freqmin = 0.5, freqmax = 1, corners = 2,
                  zerophase = True)

    # Trim traces to the time window of interest:
    t1 = 1; t2 = 6# The chosen time window goes from t_P - t1 to t_P + t2
    m = len(trimst)
    mean_t_P = np.mean(t_P); mean_t_P_date = starttime + mean_t_P
    for i in range(m):
        trimst[i].trim(starttime = mean_t_P_date - t1, endtime = mean_t_P_date + t2)
    #

    ###########################################################################

    # Stack the traces in st to obtain the first reference trace:
    stack, stack_stdv = stack_single(trimst)

    # Number of traces:
    num_traces = len(trimst)

    # Get number of samples for shifts
    num_sam = []; corrs = []; num_secs = []

    for i in range(num_traces):
        trace = trimst[i]

        cc = correlate(stack, trace, 500, demean = False)
        shift, value = xcorr_max(cc)

        # Get number of samples for shifts:
        num_sam.append(int(shift))
        num_secs.append(abs(shift * trimst[0].stats.delta))
        corrs.append(value)
        #print('Number of samples to shift is ', num_sam[i])
    #

    # Check and correct traces with reversed polarity: they tend to be very
    # similar to all the others, only upside down.
    for i in range(num_traces):
        if corrs[i] < -0.8:
            trimst[i].data = trimst[i].data * (-1)
            stream[i].data = stream[i].data * (-1)
            corrs[i] = corrs[i] * (-1)

    return stream

###############################################################################
####                       END OF THE FUNCTION                             ####
###############################################################################




def stack_single(stream):

    '''
    This function stacks all the traces in a single stream by taking their
    sample-by-sample mean and standard deviation.

    Arguments:
        -stream (Obspy stream): traces recorded by an array for a seismic event

    Returns:
        -List containing Numpy arrays for the mean wavefield and its standard
         deviation respectively

    '''

    for trace in stream:
        stack = np.mean([trace.data for trace in stream], axis = 0)
        stack_stdv = np.std([trace.data for trace in stream], axis = 0)
        stack = np.mean([trace.data for trace in stream] , axis = 0)
        stack_stdv = np.std([trace.data for trace in stream] , axis = 0)

    return [stack, stack_stdv]

###############################################################################
####                       END OF THE FUNCTION                             ####
###############################################################################







