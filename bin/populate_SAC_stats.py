'''
Created on Oct 16, 2017

Script to open mseed waveform files, add the necessary metadata and save
individual traces as SAC files.

Author: Itahisa González Álvarez

'''

import os
import numpy as np
from glob import glob
from obspy.core.util import AttribDict
from obspy import UTCDateTime, read, read_inventory, read_events, Stream

from obspy.clients.iris import Client
distaz_client=Client()
from obspy.clients.fdsn import Client
client=Client()


#############       DEFINE ARRAY PARAMETERS        ###########################

network = '';
array_nm = ''
st_nm = ''
data_source = ''
data_format = ''
location = '*'

# Define path to where files live:
main_path = '/path/to/parent/directory/where/our/downloaded/data/live'

# Starting date and time of period of interest:
t_str = ''
t1 = UTCDateTime(t_str)

#################       READ STATION INVENTORY           ######################

# NOTE: If there are any subarrays and you have NOT merged their respective
# inventories into a total inventory, please do so now. The rest of this script
# will NOT take subarrays or stations outside the inv file into account.
inv = read_inventory(main_path + 'station_inventory_' + network + '_' + array_nm \
                     + '.xml')
#Define network from inventory:
net = inv[0]; stn = []
print(inv)

#Station locations ONLY to get the new catalogs:
#Don't delete this, as lat and lon will be necessary to download the catalog files
#The sta_lat and sta_lon that are calculated inside the loop contain only the value
# for the station that is being considered at the time.
sta_lat = [sta.latitude for net in inv for sta in net]
sta_lon = [sta.longitude for net in inv for sta in net]
sta_elev = [sta.elevation for net in inv for sta in net]
lat = np.mean(sta_lat); lon = np.mean(sta_lon)

###############################################################################





#############          READ EVENT CATALOGUE            #########################
evs = read_events(main_path + 'deep_events_cat_' + network + '_' + array_nm \
                  + '_' + t_str + '.xml')


##########          CHANGE CURRENT WORKING DIRECTORY        ###################

# Change current working directory to specify where do we want to store our sac
# files:

sac_path = main_path + 'SAC/raw_SAC/'
if not os.path.exists(sac_path): os.makedirs(sac_path)
os.chdir(sac_path)

######     READ MINISEED FILES:    ##############

path = main_path + 'WVF/'
files = glob(path + '*.' + data_format)# LIST OF FILES INSIDE THE WVF DIRECTORY
N = len(files)

print('Number of waveform files: ', N)


######   POPULATION OF SAC FILES STATS         ################

# Check how many traces each stream has:
lens = []

# Define a filter band to prevent amplifying noise during the deconvolution
# The list or tuple defines the four corner frequencies (f1, f2, f3, f4) of a
# cosine taper which is one between f2 and f3 and tapers to zero for f1 < f < f2
# and f3 < f < f4. f3 and f4 are 1 Hz and 0.5 Hz lower than the Nyquist
# frequency respectively.
# Get sampling rate from inventory file (CAREFUL! I'm assuming the sampling
# rate is the same for all stations! Define individual filters for different
# stations in that case!)
srate = inv[0][0][0].sample_rate
nyq_freq = srate / 2
pre_filt = (0.001, 0.25, nyq_freq - 1, nyq_freq - 0.5)

# Sanity check!
print()
print('Sampling rate is ' + str(srate) + ' Hz.')
print('Pre-filter is = ' + str(pre_filt))
print()

# Get attributes and write sac files:

for q in range(N):

    # Initialize a new Obspy stream:
    st = Stream()

    #print(q)
    stream = read(files[q])# This contains all the traces in the mseed file

    # Streams may be longer than the network size, as some stations have more
    # than one channel.
    # Also, some streams contain more than one event, I need to
    # take this into account! This means I need to get the event
    # date from the traces, NOT FROM THE MSEED file name!
    # Just in case there are breaks in the traces:
    stream.merge()

    # Some arrays have infrasound stations, as well as other non-seismic
    # instruments. Make sure the stream does NOT contain non-seismic data!
    for tr in stream:
        if tr.stats.channel != 'BDF': st.append(tr)

    # Correct network name (for IMS data only):
    if data_source == 'IMS':
        for trace in st:
            trace.stats.network = 'AU'

    # Get stream length:
    lens.append(len(st))

    try:
        # Remove instrument response (only if we didn't do it before!)
        st.remove_response(pre_filt = pre_filt, output = 'VEL', inventory = inv,
                           zero_mean = True, taper = True)
    
        etimes = str(st[0].stats.starttime + 120)[:-8]
    
        for ev in evs:
            if str(ev.origins[0].time)[:-8] == etimes:
                print(etimes+', event found in the catalogue')
                event_lat = ev.origins[0].latitude
                event_lon = ev.origins[0].longitude
                event_depth = ev.origins[0].depth
                event_magnitude = ev.magnitudes[0].mag
                event_time = ev.origins[0].time
                wvf_starttime = event_time-120;wvf_endtime=event_time+3480
                magnitude_type = 55
                array = array_nm
    
                year = str(event_time.year); month = str(event_time.month);
                day = str(event_time.day); hour = str(event_time.hour)
                minute = str(event_time.minute); second = str(event_time.second)
                if len(month)<2: month ='0'+month
                if len(day)<2: day ='0'+day
                if len(hour)<2: hour ='0'+hour
                if len(minute)<2: minute ='0'+minute
                if len(second)<2: second ='0'+second
                t_str = year + month + day + 'T' + hour + minute + second
                break
            else:
                pass
    
        for trace in st:
    
            if trace.stats.npts > 60000:
    
                # Define necessary attributes for this trace and station:
                net_name = trace.stats.network# Network name
                stn_name = trace.stats.station#Name of the station
                channel = trace.stats.channel
                loc = trace.stats.location# Location of the station within the array
                starttime = trace.stats.starttime# Trace starttime
                endtime = trace.stats.endtime# Trace endtime
                sampling_rate = trace.stats.sampling_rate# Sampling rate
                delta = trace.stats.delta# Time interval between samples
                npts = trace.stats.npts# Number of data in the file
                calib = trace.stats.calib
    
                # Now I need to define some other attributes that may be useful:
    
                for station in net:
                    #print(station.code+' - '+stn_name)
                    if station.code == stn_name:
                        station_lat = station.latitude
                        station_lon = station.longitude
                        station_elev = station.elevation
    
                        try:
                            # Calculate distance, azimuth, backazimuth, etc from
                            # each station to each event
                            distaz = distaz_client.distaz(station_lat, station_lon,
                                                          event_lat, event_lon)
                            dist_km = distaz['distancemeters']/1000
    
                            # Save all the stats I want in my sac file:
                            trace.stats.sac = AttribDict({'knetwk': net_name,
                                                        'kstnm': stn_name,
                                                        'kcmpnm': channel,
                                                        'ko': event_time,
                                                        'delta': delta,
                                                        'user1': calib,
                                                        'npts': npts,
                                                        'stla': station_lat,
                                                        'stlo': station_lon,
                                                        'stel': station_elev,
                                                        'imagtyp': magnitude_type,
                                                        'evla': event_lat,
                                                        'evlo': event_lon,
                                                        'evdp': event_depth,
                                                        'mag': event_magnitude,
                                                        'gcarc': distaz['distance'],
                                                        'dist': dist_km,
                                                        'baz': distaz['backazimuth'],
                                                        'az': distaz['azimuth']})
    
                            trace.write('%s.%s.%s.%s.%s.%s.SAC' %(t_str, net_name,
                                        array_nm, stn_name, channel, event_magnitude),
                                        format='SAC')
                            #print('SAC files successfully saved')
                            #print('--------------------------')
                            break
    
                        except:
                            print('distaz failed, trace not saved')
                else:
                    pass

    except:
        print('Wrong instrument response!')
    
print('Total number of files you should have is ' + str(sum(lens)))









