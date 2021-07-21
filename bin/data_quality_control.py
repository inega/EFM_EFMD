'''
Created on Dec 10, 2017
@author: Itahisa Gonzalez Alvarez

    This script contains a set of functions required to remove bad quality data
    from the dataset used for the EFM/EFMD, as well as the necessary code to
    perform the quality test. The minimum SNR used is 5.

    Good data for each fband and array are sent to separate directories for each
    event and fband.

'''


import shutil
import numpy as np
from glob import glob
import os, copy, pickle
from datetime import datetime
import matplotlib.pyplot as plt
from obspy.taup import TauPyModel
from fk_analysis import get_Pwave_arrivals
from EFM_EFMD_tools import datetime_string
from obspy import UTCDateTime, read, Stream



def raw_traces_streams_EFM( network, array_nm, data_source, raw_sac_path, fband):

    '''

    Creates two streams with both the filtered and unfiltered traces available
    for all the stations within a given seismic array.

    Arguments:
        -network (str): name of the network the stations are part of (for path).
        -array_nm (str): name of the array the stations are part of (for path).
        -data_source (str): name of the source of our data (for path).
        -raw_sac_path (str): path to the specific directory where our raw sac
                             files are.
        -fband (str): frequency band we want to filter the data into.

    It returns:
        -a stream with all the unfiltered traces for the specified array.
        -a stream with the traces filtered into the specified frequency band.

    '''

    # Define frequency bands of interest and frequency ranges:
    fbands = {'A':[0.5,1], 'B':[0.75,1.5], 'C':[1,2], 'D':[1.5,3], 'E':[2,4],
              'F':[2.5,5], 'G':[3,6], 'H':[3.5,7]}
    print('Creating stream for ' + array_nm + ', for the ' + str(fbands[fband][0]) \
          + ' to ' + str(fbands[fband][1]) + ' Hz frequency band...')

    #                         GET RAW SAC FILES                               #

    all_files = glob( raw_sac_path + '*.SAC')# List of all files inside the wvf
                                             # directory
    print('Total number of raw SAC files is ', len(all_files))

    # Let's separate only those files we are interested in (BHZ, SHZ, BHR and
    # BHT ones):
    files = []
    for file in all_files:
        if 'BHZ' in file or 'BHR' in file or 'BHT' in file or 'SHZ':
            files.append(file)
    N = len(files)
    print('Number of raw SAC files of interest is ' + str(N))

    ###############################################################################

    # Create a stream with the filtered traces we have for each event and station:
    # this is a dictionary whose keys are the date of the event and the code of
    # the station.

    # Create another stream with the UNFILTERED traces, because those are the
    # ones we want to save after the QT.

    unfilt_streams = {}; streams = {}

    for q in range(N):

        tr = read(files[q])

        # Define number of data points, sampling rate and channel of the trace:
        npts = tr[0].stats.npts
        srate = tr[0].stats.sampling_rate
        channel = tr[0].stats.channel

        # Let's create a string with the event date that we will use as dict
        # keys, to avoid duplicities in our directories:
        event_date = UTCDateTime(tr[0].stats.starttime + 120)
        event_date_string = datetime_string ( event_date )

        key = event_date_string + ',' + tr[0].stats.station

        # Add entries to the dictionaries:
        if key in streams: pass
        else:
            streams[key] = Stream()
            unfilt_streams[key] = Stream()

        # We only want BHZ/SHZ, BHR and BHT traces (no BHN/BHE/BH1/BH2):
        if channel  != 'BHN' and  channel  != 'BHE' \
        and  channel  != 'BH2' and channel  != 'BH1':
            # We need to detrend and taper before filtering to remove any linear \
            # trend and to make sure our traces start with zero.
            tr.detrend( 'linear')
            tr.taper( max_percentage = 0.05, type = 'hann')

            # Change distance units to m:
            tr[0].stats.sac.dist = 1000*tr[0].stats.sac.dist

            # Previous filtering, apply to all traces:
            tr.filter( 'highpass', freq = 0.33333)

            # For 40 Hz sampling rates, traces should be exactly 144000 samples
            # long (72000 for 20 Hz s.rate). However, I've seen some of them
            # have 144001 (72001) and some are shorter than they should.
            # Delete the last sample in case they're too long or add zeros at
            # the end if they're too short:

            if srate ==  40:
                if npts > 144000:
                    tr[0].data = tr[0].data[0:144000]
                if npts < 144000:
                    diff = 144000 - npts
                    zers = np.zeros((diff,1))
                    new_data = np.append( tr[0].data, zers)
                    tr[0].data = new_data

            elif srate ==  20:
                if npts > 72000 and channel ==  'SHZ':
                        tr[0].data = tr[0].data[0:72000]
                if npts > 72000 and \
                (channel == 'BHR' or channel == 'BHT' or channel == 'BHZ'):
                    tr.resample(sampling_rate = 20.0, no_filter = True,
                                strict_length = False)
                if npts < 72000:
                    diff = 72000 - npts
                    zers = np.zeros(( diff, 1))
                    new_data = np.append( tr[0].data, zers)
                    tr[0].data = new_data

            unfilt_streams[key].append( tr[0])

            # Filter trace into the desired frequency band:
            tr2 = copy.deepcopy(tr)
            tr2.filter( 'bandpass', freqmin = fbands[fband][0],
                       freqmax = fbands[fband][1], corners = 2,
                       zerophase = True)
            streams[key].append( tr2[0])

    print('Streams successfully created!')

    return unfilt_streams, streams

###############################################################################
####                    END OF THE FUNCTION                                ####
###############################################################################



def quality_test_EFM( network, array_nm, data_source, main_path, raw_sac_path,
                     fband):

    '''
    This functions uses the raw_traces_stream function to get streams with all
    the raw filtered and unfiltered SAC traces for each array and fband. Then,
    it compares the peak-to-peak amplitude of the signal before the first arrival
    with the peak to peak amplitude within the first arrival time window to help
    decide whether the trace is noisy and/or usable. If the signal-to-noise ratio
    is lower than 5, then this function stores the trace into a new directory.

    Arguments:
        -network (str): name of the network the stations are part of (for path).
        -array_nm (str): name of the array the stations are part of (for path).
        -data_source (str): name of the source of our data (for path).
        -main_path (str): path to the root directory where we want to save our
                          quality controlled traces.
        -raw_sac_path (str): path to the specific directory where our raw sac
                             files are.
        -fband (str): frequency band we want to filter the data into.

    Output:
        -Good quality traces saved into separate directories for each array,
         frequency band and even.
        -Inverse of the sampling rate for the array (necessary later for the
         rest of the analysis).

    '''

    # I want to time this function:
    stime = datetime.now()

    # Call the raw_traces_stream function to get all the traces for this array
    # and component. This returns two dictionaries containing streams with the
    # traces we have for each event and station, unfiltered and also filtered
    # into frequency band fband. The keys of the dictionary are the start time
    # and the station code in the format 'UTCDateTime(starttime), station.code'.
    unfilt_streams, filt_streams = raw_traces_streams_EFM( network, array_nm,
                                                          data_source,
                                                          raw_sac_path, fband)
    # n = len(filt_streams)
    unusable_streams = []

    print('Getting SNR and saving good traces into new directories...')

    for key in filt_streams:

        print( key)

        # Some streams only had radial and transverse traces and some had no
        # traces at all (not all stations recorded some events). I can't use
        # those (I need 3 comp data) so remove all streams with only two traces:
        if len(filt_streams[key]) != 0 and len(filt_streams[key]) != 2:

            # Call the get_P_wave arrivals to get the theoretical P wave arrival
            # for each trace.
            ev_time = key[:15]
            t_P,t_P_date = get_Pwave_arrivals( filt_streams[key], ev_time,
                                               model = 'prem')

            # Signal time window: each stream in filt_streams contains the
            # traces registered at ONE station and ONE event. t_p is the
            # theoretical P wave arrival time for that event and station, t_0
            # the beginning of the P wave time window (for QT purposes) and t_1
            # the end of the signal time window used for quality control.
            t_p = np.array(t_P[0])
            t_0 = np.array(t_P[0]) - 1
            t_1 = np.array(t_P[0]) + 40

            # Time vector:
            npts = filt_streams[key][0].stats.npts
            srate = filt_streams[key][0].stats.sampling_rate
            t = np.arange( npts) / srate

            # Pre-define signal-to-noise ratio (SNR) as a dictionary:
            SNR = {}; SNR[array_nm] = []

            # Get indices for the p wave arrival, start and end of the time window:
            p_arr_index = (np.abs(t - t_p)).argmin()
            t0_index = (np.abs(t - t_0)).argmin()
            t1_index = (np.abs(t - t_1)).argmin()

            # We take the noise window from the beginning of the trace up to 5
            # seconds before the theoretical P wave arrival (number of
            # samples = time * sampling_rate):
            noise_index = int( p_arr_index - ( 5 * srate))

            #        ################################################         #

            # We have data for the vertical, radial and transverse components
            # of the ground velocity. We CAN'T APPLY THE SAME QUALITY TEST TO
            # ALL OF THEM because we generally have less energy on the
            # horizontal components and most of the energy is on the verticals.
            # We will do the quality test ONLY ON THE VERTICALS and save the
            # horizontals too if the verticals pass it.

            for trace in filt_streams[key]:
                if trace.stats.channel ==  'BHZ' or trace.stats.channel ==  'SHZ':
                    # Define data:
                    data = trace.data

            # Peak to peak amplitude of the data in the P wave time window:
            #print('Getting mean signal level...')
            signal = data[t0_index:t1_index]
            pp_signal = max(signal) - min(signal)

            # Define noise:
            #print('Getting mean noise level...')
            noise_data = data[0:noise_index]
            # Peak to peak amplitude of the noise:
            pp_noise = max(noise_data) - min(noise_data)

            # Calculate signal-to-noise ratio and set threshold:
            snr = pp_signal / pp_noise
            SNR[array_nm].append(  snr)
            thresh = 5

            # We want to save the SAC files in independent directories for each
            # event:

            # SAVE GOOD TRACES:
            if snr > thresh:
                for trace in unfilt_streams[key]:

                    path = main_path + 'SAC/GQ_SAC/' + fband + '/' + ev_time + '/'
                    if not os.path.exists(path): os.makedirs(path)

                    filename = path + ev_time + '.' + network + '.' \
                               + array_nm + '.' + trace.stats.station + '.' \
                               + trace.stats.channel + '.' \
                               + str(trace.stats.sac.mag) + '.SAC'
                    trace.write( filename, format = 'SAC')

        else:

            print('Stream for event ' + key + ' contained 0 traces')
            unusable_streams.append(key)

    # Save the inverse of the sampling rate into a file: we take it from
    # one of the streams in unfilt_streams (it doesn't matter which one).
    delta = unfilt_streams[key][0].stats.delta
    h = open( main_path + array_nm + '_delta.pckl', 'wb')
    pickle.dump( delta, h)
    h.close( )

    print('List of unusable streams: ')
    print(' ')
    print( unusable_streams)
    print(' ')
    print('Everything worked just fine')

    print('It took the script ' + str(datetime.now() - stime) + ' to run so far')
    print(' ======================================================= ')


###############################################################################
####                    END OF THE FUNCTION                                ####
###############################################################################





def QT_sanity_check_EFM (array_nm, gq_sac_path, plots_path, fband, comp = 'all'):

    '''
    Plot a section for each event  with good quality data to check that the
    traces really are good. I only check the filtered vertical components, as
    the radials and transverses should be as noisy as the vertical ones.

    I want the traces to be aligned in these section plots, so I use the
    create_stream_EFM and stream_plotting_trim_stack functions to create a
    stream of filtered, trimmed and aligned traces and then plot it.

    Arguments:
        -array_nm (str): name of the array the stations are part of (for path).
        -gq_sac_path (str): path to the specific directory where our quality
                            controlled sac files are.
        -plots_path (str): path to the directory the plots will be saved into.
        -comp (str): component(s) to plot in the figures either single component ('Z', 'R'
                     or 'T') or 'all' to plot all traces. Default is 'all'.

        Output:
            -Plots of traces recorded for each event, station and channel are saved into a
             directory.
    '''

    #                           GET RAW SAC FILES                             #

    # List of all directories inside the gq_sac_path directory:
    dirs = glob( gq_sac_path + fband + '/*/')
    N = len(dirs)
    print('Number of events with good quality data is ' + str(N))
    print('')

    fbands = {'A':[0.5,1], 'B':[0.75,1.5], 'C':[1,2], 'D':[1.5,3],
              'E':[2,4], 'F':[2.5,5], 'G':[3,6], 'H':[3.5,7]}

    # Choose model to calculate theoretical traveltimes:
    model = TauPyModel( model = 'prem')
    
    for directory in dirs:

        ev_date = directory[-16:-1]
        print('         Event date : ' + ev_date + ', ' + fband + '          ')

        # Plot normalised, non-aligned traces just to double check the quality of the 
        # data and that there are no secondary arrivals in our time window of interest.
        
        # Create figure:
        f, ax = plt.subplots (figsize = (20, 10))
        
        # Load files:
        if comp == 'all': files = glob (directory + '*.SAC')
        if comp != 'all':
            files = glob (directory + '*' + comp + '*.SAC')
            
        chans = []; stream = Stream()
        for i, file in enumerate(files):
            st = read (file)
            st.filter ('bandpass', freqmin = fbands[fband][0], freqmax = fbands[fband][1], 
                       corners = 2, zerophase = True)
            tr = st[0]
            stream.append (tr)
            srate = tr.stats.sampling_rate
            chans.append (tr.stats.station + ', ' + tr.stats.channel)
            
            # Get theoretical arrival times:
            if i == 0:
                arrs = []; arr_times = []
                ev_date = UTCDateTime( ev_date )# Data from 2 minutes BEFORE the event!
                dist_degs = tr.stats.sac.gcarc # distance to event in degrees
                ev_dep = tr.stats.sac.evdp / 1000 # depth in km
                # We are only interested in P wave arrivals:
                arrivals = model.get_travel_times (source_depth_in_km = ev_dep,
                                                   distance_in_degree = dist_degs)
                for j, arrival in enumerate(arrivals):
                    if j < 10:
                        arrs.append (arrival.name)
                        arr_times.append (arrival.time + 120)
                    
            t_P = arr_times[0]
            t_P_ind = ( np.abs( tr.times() - t_P) ).argmin()
            
        # Get maximum amplitude for the normalization of the traces to the vertical
        # component:
        maxamps = []
        for tr in stream:
            if 'Z' in tr.stats.channel:
                maxamps.append (np.max (np.abs (tr.data[int(t_P_ind - 25*srate):int(t_P_ind + 100*srate)])))
        tr_max = np.array(maxamps).max()
        # Plot traces:
        for i, tr in enumerate(stream):
            ax.plot (tr.times(), (tr.data/tr_max) + i*1.8, 'k', linewidth = 0.8)
            
        # Plot theoretical arrivals:
        colors = ['red', 'orange', 'limegreen', 'cornflowerblue', 'darkviolet', 'coral',
                  'gold', 'olive', 'darkcyan', 'fuchsia']
        for i in range(len(arrs)):
            ax.axvline (arr_times[i], linewidth = 1.5, color = colors[i], label = arrs[i])

        ax.set_yticks (list(1.8*np.arange(len(stream))))
        ax.set_yticklabels (chans, fontsize = 14)
        ax.tick_params (axis = 'x', labelsize = 14)
        ax.grid (linestyle = 'dashed')
        ax.legend (loc = 'upper left', fontsize = 14)
        ax.set_xlim ([t_P-50, t_P+250])
        ax.set_ylim ([-tr_max-1.8, tr_max + len(files)*1.8])
        ax.set_xlabel ('Time (s)', fontsize = 16)
        ax.set_title (str(ev_date) + ', ' + array_nm, fontsize = 18)
        
        # Save figure:
        if not os.path.exists (plots_path + fband + '/'): os.makedirs (plots_path + fband + '/')
        fname = plots_path + fband + '/' + array_nm + '_' + directory[-16:-1] + '.png'
        f.savefig ( fname, bbox_inches = 'tight')
        plt.close('all')
    
  
###############################################################################
####                    END OF THE FUNCTION                                ####
###############################################################################






# I want to time how long it took for the script to run:
sttime = datetime.now()

# Define network, array name and data source:
network = ''
array_nm = ''
data_source = ''

# Define which step(s) of the quality control process you want to run:
QT_step1 = False
QT_step2 = False
QT_step3 = True

# Define frequency band names and data components:
fbs = ['A','B','C','D','E','F','G','H']
comps = ['Z','T','R']

# Define needed paths: a) path to the main directory where all our data and
# directories are; b) path to the directories where our raw SAC data live (we
# may have more than one, if we divided the dataset); c) path to the directory
# where we want to store the quality control plots.
main_path = '/path/to/parent/directory/where/our/data/live/' + data_source + '/' \
            + network + '/' + array_nm + '/'
raw_sac_paths = glob(main_path + 'SAC/raw_SAC*/')

# Run QT:
if QT_step1 == True:
    for raw_sac_path in raw_sac_paths:

        print(network)
        print(array_nm)
        print(raw_sac_path)

    for fband in fbs:
        SNR = quality_test_EFM(network, array_nm, data_source, main_path,
                               raw_sac_path, fband)

    print('It took the quality control function ' + str(datetime.now() - sttime) \
	  + ' s to run')


# Run sanity check: this function plots the 1-comp or 3-comp traces for each good quality
# event and saves them to the specified directory:
if QT_step2 == True:
    print('Running sanity check...')
    gq_sac_path = main_path + 'SAC/GQ_SAC/'
    plots_path = main_path + 'SAC/QT_plots/'
    
    for fband in fbs:
        print( 'Frequency band: ' + fband)
        QT_sanity_check_EFM (array_nm, gq_sac_path, plots_path, fband, comp = 'all')
    
    print('It took the whole script ' + str(datetime.now() - sttime) + ' to run')



###############################################################################

#     RUN THIS PART OF THE CODE ONLY AFTER MANUALLY INSPECTING THE PLOTS      #

# It is highly recommended to manually check ALL the plots created by the function
# above to get rid of bad events that may have passed the initial quality control.
# Manually move the plots from the QT_plots directory for each frequency band to 
# an "Unusable events" directory for the same fband within the SAC directory so the 
# code below can remove these events from the dataset.
if QT_step3 == True:
    
    for fband in fbs:
    
        figs = glob (main_path + 'SAC/Unusable_events/' + fband + '/*')
    
        fig_ev_dates = []
        for fig in figs:
            ev_date = fig[-19:-4]
            fig_ev_dates.append(ev_date)
    
        dirs = glob( main_path + '/SAC/GQ_SAC/' + fband + '/*')
        dest = main_path + 'SAC/Unusable_events/' + fband + '/'
    
        for directory in dirs:
            ev_date = directory[-15:]
            if ev_date in fig_ev_dates:
                shutil.move( directory, dest)
    
    
    # Sanity check:
    for fband in fbs:
    
        unusable_figs = glob( main_path + 'SAC/Unusable_events/' + fband + '/*.png')
        unusable_dirs = glob( main_path + 'SAC/Unusable_events/' + fband + '/*/')
    
        dir_ev_dates = []
        for directory in unusable_dirs:
            dir_ev_dates.append( directory[-16:-1] )
    
        for fig in unusable_figs:
            fig_ev_date = fig[-19:-4]
            if fig_ev_date not in dir_ev_dates:
                print('Event on ' +  fig_ev_date + ' for fband ' + fband \
                      + ' should be in Unusable events directory')
    
    
        if len(unusable_figs) != len(unusable_dirs):
            print('Something went wrong for fband ' + fband + '!')
        else:
            print('Number of dirs in Unusable_events for fband ' + fband \
                  + ' is correct! Well done!')
    
      







