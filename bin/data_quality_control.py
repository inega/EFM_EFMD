'''
Created on Dec 10, 2017
@author: Itahisa Gonzalez Alvarez

    This script contains a set of functions required to remove bad quality data
    from the dataset used for the EFM/EFMD, as well as the necessary code to
    perform the quality test. The minimum SNR used is 5.

    Good data for each fband and array are sent to separate directories for each
    event and fband.

'''

import os
import copy
import pickle
import numpy as np
from glob import glob
from datetime import datetime
from shutil import move as move
from vel_models import get_velocity_model
from fk_analysis import get_Pwave_arrivals
from obspy import UTCDateTime, read, Stream
from EFM_EFMD_tools import create_stream_EFM, datetime_string, stream_plotting_trim_stack



def raw_traces_streams_EFM(network, array_nm, data_source, raw_sac_path, fband):

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

    all_files = glob(raw_sac_path + '*.SAC')# List of all files inside the wvf
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
        event_date_string = datetime_string (event_date)

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
            tr.detrend('linear')
            tr.taper(max_percentage = 0.05, type = 'hann')

            # Change distance units to m:
            tr[0].stats.sac.dist = 1000*tr[0].stats.sac.dist

            # Previous filtering, apply to all traces:
            tr.filter('highpass', freq = 0.33333)

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
                    new_data = np.append(tr[0].data, zers)
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
                    zers = np.zeros((diff, 1))
                    new_data = np.append(tr[0].data, zers)
                    tr[0].data = new_data

            unfilt_streams[key].append(tr[0])

            # Filter trace into the desired frequency band:
            tr2 = copy.deepcopy(tr)
            tr2.filter('bandpass', freqmin = fbands[fband][0],
                       freqmax = fbands[fband][1], corners = 2,
                       zerophase = True)
            streams[key].append(tr2[0])

    print('Streams successfully created!')

    return unfilt_streams, streams

###############################################################################
####                    END OF THE FUNCTION                                ####
###############################################################################



def quality_test_EFM(network, array_nm, data_source, main_path, raw_sac_path,
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
    unfilt_streams, filt_streams = raw_traces_streams_EFM(network, array_nm,
                                                          data_source,
                                                          raw_sac_path, fband)
    # n = len(filt_streams)
    unusable_streams = []

    print('Getting SNR and saving good traces into new directories...')

    for key in filt_streams:

        print(key)

        # Some streams only had radial and transverse traces and some had no
        # traces at all (not all stations recorded some events). I can't use
        # those (I need 3 comp data) so remove all streams with only two traces:
        if len(filt_streams[key]) != 0 and len(filt_streams[key]) != 2:

            # Call the get_P_wave arrivals to get the theoretical P wave arrival
            # for each trace.
            ev_time = key[:15]
            t_P,t_P_date = get_Pwave_arrivals(filt_streams[key], ev_time,
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
            t = np.arange(npts) / srate

            # Pre-define signal-to-noise ratio (SNR) as a dictionary:
            SNR = {}; SNR[array_nm] = []

            # Get indices for the p wave arrival, start and end of the time window:
            p_arr_index = (np.abs(t - t_p)).argmin()
            t0_index = (np.abs(t - t_0)).argmin()
            t1_index = (np.abs(t - t_1)).argmin()

            # We take the noise window from the beginning of the trace up to 5
            # seconds before the theoretical P wave arrival (number of
            # samples = time * sampling_rate):
            noise_index = int(p_arr_index - (5 * srate))

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
            SNR[array_nm].append( snr)
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
                    trace.write(filename, format = 'SAC')

        else:

            print('Stream for event ' + key + ' contained 0 traces')
            unusable_streams.append(key)

    # Save the inverse of the sampling rate into a file: we take it from
    # one of the streams in unfilt_streams (it doesn't matter which one).
    delta = unfilt_streams[key][0].stats.delta
    h = open(main_path + array_nm + '_delta.pckl', 'wb')
    pickle.dump(delta, h)
    h.close()

    print('List of unusable streams: ')
    print(' ')
    print(unusable_streams)
    print(' ')
    print('Everything worked just fine')

    print('It took the script ' + str(datetime.now() - stime) + ' to run so far')
    print(' ======================================================= ')


###############################################################################
####                    END OF THE FUNCTION                                ####
###############################################################################



def QT_sanity_check_EFM(network, array_nm, gq_sac_path, plots_path, inv_path,
                        vel_source, vel_model_fname, fband, num_layers):

    '''
    Plot a section for each event  with good quality data to check that the
    traces really are good. I only check the filtered vertical components, as
    the radials and transverses should be as noisy as the vertical ones.

    I want the traces to be aligned in these section plots, so I use the
    create_stream_EFM and stream_plotting_trim_stack functions to create a
    stream of filtered, trimmed and aligned traces and then plot it.

    Arguments:
        -network (str): name of the network the stations are part of (for path).
        -array_nm (str): name of the array the stations are part of (for path).
        -gq_sac_path (str): path to the specific directory where our quality
                            controlled sac files are.
        -plots_path (str): path to the directory the plots will be saved into.
        -inv_path (str): file name and path of the inventory file with info
                        about stations locations.
        -fband (str): frequency band we want to filter the data into.
        -num_layers (int): number of layers in the lithospheric model (not
                           important, required only to get the one-way
                           traveltime through the lithosphere).

        Output:
            -Section plots for each event and frequency band are saved into a
            separate directory.
    '''

    #                           GET RAW SAC FILES                             #

    # List of all directories inside the gq_sac_path directory:
    dirs = glob(gq_sac_path + fband + '/*/')
    N = len(dirs)
    print('Number of events with good quality data is ' + str(N))

    # Get tJ from the get_velocity_model function.
    vel_model = get_velocity_model(array_nm, vel_source, vel_model_fname, num_layers,
                       units = 'm')
    tJ = vel_model['tJ']

    for directory in dirs:

        ev_date = directory[-16:-1]
        print(' ============================================================ ')
        print(' ')
        print('         Event date : ' + ev_date + ', ' + fband + '          ')
        print(' ')

        try:

            # Create stream: this step may fail because some events have less
            # than 5-6 good traces.
            stream = create_stream_EFM(array_nm, ev_date, directory, comp,
                                       fband, tJ, fname = None)

            import matplotlib.pyplot as plt
            for tr in stream:
                plt.plot(tr.data, 'k', linewidth = 0.8)
            ###
            # Plot section:
            full_plots_path = plots_path + array_nm + '/' + fband + '/'
            fname = full_plots_path + array_nm + '_' + ev_date + '_Z_QT_check.png'
            if not os.path.exists(full_plots_path):
                os.makedirs(full_plots_path)
            stream_plotting_trim_stack(array_nm, ev_date, stream, comp, tJ,
                                       filename = fname, show_plots = False)

        except:
            pass

###############################################################################
####                    END OF THE FUNCTION                                ####
###############################################################################



# I want to time how long it took for the script to run:
sttime = datetime.now()

# Define network, array name and data source:
network = ''
array_nm = ''
data_source = ''

# Start of the period of interest:
# starttime = UTCDateTime('2018-01-01T00:00:00')

# Define frequency band names and data components:
fbs = ['A','B','C','D','E','F','G','H']
comps = ['Z','T','R']

# Define needed paths: a) path to the main directory where all our data and
# directories are; b) path to the directories where our raw SAC data live (we
# may have more than one, if we divided the dataset); c) path to the directory
# where we want to store the quality control plots.
main_path = '/path/to/parent/directory/where/our/data/live/'
raw_sac_paths = glob(main_path + 'SAC/raw_SAC*/')
plots_path = main_path + 'SAC/QT_plots/'

# Run QT:
for raw_sac_path in raw_sac_paths:

    print(network)
    print(array_nm)
    print(raw_sac_path)

    for fband in fbs:
        SNR = quality_test_EFM(network, array_nm, data_source, main_path,
                                raw_sac_path, fband)

print('It took the quality control function ' + str(datetime.now() - sttime) \
      + ' s to run')


# Run sanity check:
print('Running sanity check...')
gq_sac_path = main_path + 'SAC/GQ_SAC/'
inv_path = main_path + 'station_inventory_' + network + '_' + array_nm + '.xml'
num_layers = 1 #Not really important, we only need the one-way travel time
               # through the lithosphere, which barely changes

# Define component, velocity model source, model type and units. We need
# these to get the velocity model data.
comp = 'Z'
vel_source = ''
units = 'm'
# Path and file name of the file containing the velocity data: the format of
# this file is in the README file.
vel_model_fname = '/path/to/file/with/velocity/data.csv'
for fband in fbs:
    QT_sanity_check_EFM(network, array_nm, gq_sac_path, plots_path, inv_path,
                        vel_source, vel_model_fname, fband, num_layers)

print('It took the whole script ' + str(datetime.now() - sttime) + ' to run')




###############################################################################

#     RUN THIS PART OF THE CODE ONLY AFTER MANUALLY INSPECTING THE PLOTS      #

# It is highly recommended to manually check ALL the plots created by the functions
# above to get rid of bad events that may have passed the initial quality control.
# E.g. events with a smaller event within the time window of interest, clear
# secondary arrivals (PcP, etc) and events that we won't be able to use for
# whatever reasons.
# Move the section plots to an "Unusable events" directory so the code below can
# remove these events from the dataset.

# arrays = ['']
# ev_types = ['Unusable_events'] # You can define more than one type

# for array_nm in arrays:
#     for fband in fbs:
#         for ev_type in ev_types:

#             # List of figures stored in ev_type directory:
#             figs = glob('/path/to/ev_type/directory/*')

#             fig_ev_dates = []
#             for fig in figs:
#                 # Get event date from figure file name:
#                 ev_date = fig[-30:-15]
#                 # Add it to list of events:
#                 fig_ev_dates.append(ev_date)

#             # Define path to GQ events directory for this array and fband:
#             dirs = glob('/path/to/GQ/events/data/directory/for/this/fband/*')
#             # Define destination where weird or unusable events data will be
#             #saved:
#             dest = '/path/to/unusable/events/data/directory/for/this/fband/'

#             # Move all traces for weird or unusable events to dest:
#             for directory in dirs:
#                 ev_date = directory[-15:]
#                 if ev_date in fig_ev_dates:
#                     move(directory, dest)

# # Sanity check:
# for array_nm in arrays:
#     for fband in fbs:

#         unusable = glob('/path/to/unusable/events/data/directory/for/this/fband/*')
#         u_figs = glob('/path/to/unusable/events/figures/for/this/fband/*')

#         # Get event dates from unusable event data directories:
#         dir_ev_dates = []
#         for directory in unusable:
#             dir_ev_dates.append(directory[-15:])

#         # Compare event dates in GQ data directories for all fbands with dates
#         # on dir_ev_dates:
#         for fig in u_figs:
#             fig_ev_date = fig[-30:-15]
#             if fig_ev_date not in dir_ev_dates:
#                 print('Event on ' +  fig_ev_date + ' for fband ' + fband \
#                       + ' should be in Unusable events directory')

#         if len(u_figs)  != len(unusable):
#             print('Something went wrong for fband ' + fband + '!')
#         else:
#             print('Number of dirs in Unusable_events for fband ' + fband \
#                   + ' is correct! Well done!')









