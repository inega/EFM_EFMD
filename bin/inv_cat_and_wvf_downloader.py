'''
Created on Oct 10 2017

Script to download event data, station inventory and waveforms from IRIS.

@author: Itahisa González Álvarez

'''

## DOWNLOADING DATA FROM IRIS WEB SERVICES ##

import os
import numpy as np
from obspy import UTCDateTime
from obspy.clients.fdsn import Client

#######            DEFINE PARAMETERS               #############

# Define main path to save new files:
main_path = '/path/to/directory/where/we/want/to/store/the/data/'

# Define client:
client = Client('IRIS')

# Array parameters:
network = ''
array_nm = ''
st_names = ['']

#Event parameters for catalogue downloads are INDEPENDENT of the network we
# get seismograms from, but we need to define minimum and maximum magnitudes,
# depth and radius to the array.
minmagnitude = 5.0; maxmagnitude = 7
mindepth = 200
minrad = 30; maxrad = 80 # radius in degrees

# Time period of interest: we will define the start time in string format
# so we can also use it for the inventory file name.
t_str = ''
t1 = UTCDateTime(t_str); t2 = UTCDateTime('')

for st_nm in st_names:

    channel = 'BH*'
    location = '*'

    # Define path to directory where we want to store files:
    path = main_path + network + '/' + array_nm + '/'

    print('Array: ' +  array_nm)
    print()
    print('Files will be saved to directory: ')
    print(path)
    print()
    #print('Subarray: ', subarray[0:-1])

    #######################################################################

    ########      PART 1: DOWNLOAD STATION INVENTORY     ################

    # UNCOMMENT THESE LINES ONLY IF YOU DON'T ALREADY HAVE A STATION INVENTORY!

    # I could download and store the inventory in the HD, but in this case I
    # am only keeping it in memory and using it.

    # In case there is a subarray (optional):
    inv = client.get_stations(network = network, station = st_nm,
                              starttime = t1, endtime = t2, level = 'response')
    print(inv)

    inv_fname = path + 'station_inventory_' + network + '_' + array_nm + '.xml'
    if not os.path.exists(path): os.makedirs(path)
    inv.write(inv_fname, format = 'STATIONXML')

    # ##################                                     ####################

    # #  IN CASE YOU ALREADY HAVE AN INVENTORY FILE:

    # #     READ INVENTORY FILE:

    # inv_fname = path + 'station_inventory_' + network + '_' + array_nm + '.xml'
    # inv = read_inventory(inv_fname)
    # print(inv)

    #Station locations:
    sta_lat = [sta.latitude for net in inv for sta in net]
    sta_lon = [sta.longitude for net in inv for sta in net]
    lat = np.mean(sta_lat); lon = np.mean(sta_lon)

    ###########################################################################



    #########   PART 2: DOWNLOAD EVENT CATALOGUE      #####################

    cat = client.get_events(starttime = t1, endtime = t2, minradius = minrad,
                            maxradius = maxrad, longitude = lon, latitude = lat,
                            minmagnitude = minmagnitude, maxmagnitude = maxmagnitude,
                            mindepth = mindepth)
    print()
    print('Number of deep events in the catalog = ', len(cat))
    cat_fname = path + 'deep_events_cat_' + network + '_' + array_nm + '_' \
                + t_str + '.xml'
    cat.write(cat_fname, format = 'QUAKEML')

    #Events latitude, longitude and time:
    ev_lat = [ev.origins[0].latitude for ev in cat]
    ev_lon = [ev.origins[0].longitude for ev in cat]
    ev_time = [ev.origins[0].time for ev in cat]

    # Get time strings for file and directory names:
    t_str = []
    for v in range(len(ev_time)):
        year = (str(ev_time[v].year))

        if len(str(ev_time[v].month)) < 2: month = '0' + str(ev_time[v].month)
        else: month = (str(ev_time[v].month))

        if len(str(ev_time[v].day)) < 2: day = '0' + str(ev_time[v].day)
        else: day = (str(ev_time[v].day))

        if len(str(ev_time[v].hour)) < 2: hour = '0' + str(ev_time[v].hour)
        else: hour = (str(ev_time[v].hour))

        if len(str(ev_time[v].minute)) < 2: minute = '0' + str(ev_time[v].minute)
        else: minute = (str(ev_time[v].minute))

        if len(str(ev_time[v].second)) < 2: second = '0' + str(ev_time[v].second)
        else: second = (str(ev_time[v].second))

        t_str.append(year + month + day + 'T' + hour + minute + second)

    etime = np.array(ev_time)
    ev_preset = etime - 120; ev_offset = etime + 3480# We want 1h long records
    ev_preset = list(ev_preset); ev_offset = list(ev_offset)

    # Export events info to a file:
    file = open(path + array_nm + '_events_list.txt', 'w')
    for i in range(len(ev_preset)):
        line = [str(ev_preset[i]) + ', ' + str(ev_offset[i]) + '\n']
        file.writelines(line)

    file.close()

    ###########################################################################



    ###   PART 3: DOWNLOAD WAVEFORMS FOR THE EVENTS WITHIN THE CATALOGUE   ###

    # Save the dates there is no data for:
    nodata = []; lengths = []

    # Define a filter band to prevent amplifying noise during the deconvolution
    # The list or tuple defines the four corner frequencies (f1, f2, f3, f4)
    # of a cosine taper which is one between f2 and f3 and tapers to zero for
    # f1 < f < f2 and f3 < f < f4. f3 and f4 are 1 Hz and 0.5 Hz lower than the
    # Nyquist frequency respectively.
    # Get sampling rate from inventory file (CAREFUL! I'm assuming the sampling
    # rate is the same for all stations! Define individual filters for different
    # stations in that case!)
    srate = inv[0][0][0].sample_rate
    nyq_freq = srate / 2
    pre_filt = (0.001, 0.25, nyq_freq - 1, nyq_freq - 0.5)

    # Sanity check!
    print('Sampling rate is ' + str(srate) + ' Hz.')
    print('Pre-filter is = ' + str(pre_filt))
    print()

    for i, ev in enumerate(cat):

        try:
            # The client.get_waveforms requires:
            # ('Network', 'Station', 'Location', 'Channel', t1, t2)
            # We can use '*' to download from all stations and locations or
            # select just one
            st = client.get_waveforms(network = network, station = st_nm,
                                      channel = channel, location = location,
                                      starttime = ev_preset[i],
                                      endtime = ev_offset[i],
                                      attach_response = True)# This step takes
                                                              # a long time!
        except:
            print('No data could be downloaded for event on ' + t_str[i])
        try:
            print('Length of stream for event on ', str(ev_time[i]),
                  ' is ', len(st))
            lengths.append(len(st))
            # Save stream object into a file:
            file_path = path + 'WVF/'
            if not os.path.exists(file_path): os.makedirs(file_path)
            filename = file_path + 'wvf_' + str(t_str[i]) + '_' + network \
                       + '_' + array_nm + '.mseed'
            st.write(filename, format = 'MSEED')
        except:
            print('Instrument response for event on ' + t_str[i] \
                  + 'could not be removed or file could not be saved...')


###############################################################################

print('You\'ve got data!!')