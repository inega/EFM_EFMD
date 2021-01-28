'''
Created on Oct 18, 2017

Script to open mseed waveform files, add the necessary metadata and save
individual traces as SAC files.

Author: Itahisa González Álvarez

'''

import copy
from glob import glob
from obspy import read, read_inventory, Stream


from obspy.clients.iris import Client
distaz_client=Client()
from obspy.clients.fdsn import Client
client=Client()

print(' ')

#############       DEFINE ARRAY PARAMETERS        ###########################
network = '';
array_nm = ''
location = '*'
data_source = ''

# Define path to where files live:
main_path = '/path/to/parent/directory/where/we/have/our/data/'

#################       READ STATION INVENTORY           ######################

inv = read_inventory(main_path + 'station_inventory_' + network + '_' \
                     + array_nm + '.xml')

#Define network from inventory:
nwk = inv[0]
#print(inv)

# Create a channel list to check whether we need to rotate from ZNE/Z12 to ZRT.
chan_list = []; net = {}
for station in nwk:
    for channel in station:
        chan_list.append(channel.code)

if 'BH1' in chan_list or 'BH2' in chan_list:
    print('Rotating from Z12 to ZRT...')
else:
    print('Rotating from ZNE to ZRT...')

# If BH1, BH2 channels are present, create a dictionary with stations BH1, BH2
# azimuths:
if 'BH1' in chan_list or 'BH2' in chan_list:
    for station in nwk:
        net[station.code] = {'BH1_azi':station.channels[0].azimuth,
                             'BH2_azi':station.channels[1].azimuth}

###############################################################################



##############          READ RAW SAC FILES:            #######################

# NOTE: The creation of station streams takes up too much memory, so I divided
# my SAC files into smaller batches so I can do it.

sac_dirs = glob(main_path + 'SAC/raw_SAC*/')
# Directories we want to ignore:
bad_dirs = [main_path + 'SAC/bad_dir_1/', main_path + 'SAC/bad_dir_2/']

for directory in sac_dirs:

    if directory not in bad_dirs:

        files = glob(directory + '*.SAC')# LIST OF ALL FILES INSIDE THE WVF DIRECTORY
        N = len(files)

        print(' ************************************************************ ')
        print(' ')
        print('          ROTATING TRACES FROM Z12 OR ZNE TO ZRT              ')
        print('Number of raw SAC files in directory ', directory, ' is ', N,'')
        print(' ')
        print(' *************************************************************')
        print(' ')

        #######################################################################

        # Create a stream with the traces we have for each event, channel and
        # station: this is a dictionary whose keys are the date of the event and
        # the code of the station.
        streams = {}
        print('Creating streams for every station and event, this step may take \
              a long time to run...')
        for q in range(N):
            kk = read(files[q])
            # I use the station code from the trace in the file because some
            # stations' codes are three characters long while some are four, so
            # it is more difficult to get them from the file names.
            key = files[q].replace(directory, '')[0:15] + ',' + kk[0].stats.station
            streams[key] = Stream()

        for q in range(N):
            kk = read(files[q])
            key = files[q].replace(directory,'')[0:15] + ',' + kk[0].stats.station
            streams[key].append(kk[0])

        print('Streams for every station and event successfully created!')
        print('Rotating traces...')

        #       ROTATE TRACES FROM BH1-BH2-BHZ TO BHR-BHT-BHZ                 #
        for key in streams:

            # We only can/want to rotate streams with three traces, so we ignore \
            # the rest:
            if len(streams[key]) == 3:

                try:

                    # MAKE EXTRA SURE THAT THE FIRST TRACE HERE IS THE VERTICAL!
                    # BH1 needs to be the second trace and BH2 the third one for
                    # Z12 to ZRT rotation or BHN needs to be the second trace
                    # and BHE the third one for ZNE to ZRT rotation.
                    scheck = streams[key]
                    sancheck = copy.deepcopy(scheck)# Copy of the original stream
                    st = Stream()# This is the stream I will manually rotate

                    for trace in sancheck:
                        if trace.stats.channel == 'BHZ':
                            st.append(trace)
                    for trace in sancheck:
                        if trace.stats.channel == 'BH1' or trace.stats.channel == 'BHN':
                            st.append(trace)
                    for trace in sancheck:
                        if trace.stats.channel=='BH2' or trace.stats.channel == 'BHE':
                            st.append(trace)

                    # Use obspy built-in function to rotate traces:

                    if 'BH1' in chan_list or 'BH2' in chan_list:
                        st2 = copy.deepcopy(st)# This is the stream I rotate from
                                             # Z12 to ZNE with obspy.rotate
                        st2.rotate('->ZNE', back_azimuth = st2[0].stats.sac.baz,
                                   inventory = inv)
                        # Copy of st2 with only horizontal components
                        st3 = Stream(); st3.append(st2[1]); st3.append(st2[2])

                        # This step changes the traces in st2, as I am NOT
                        # making a copy of them for st3! The stream I will want
                        # to keep is st2, NOT st3.
                        st3.rotate('NE->RT', back_azimuth=st2[0].stats.sac.baz,
                                   inventory=inv)

                    else:
                        st2=copy.deepcopy(st)# This is the stream I rotate from
                                             # ZNE to ZRT with obspy.rotate

                        # This step changes the traces in st2, as I am NOT
                        # making a copy of them for st3!
                        st2.rotate('NE->RT', back_azimuth=st2[0].stats.sac.baz,
                                   inventory = inv)

                    #Save traces into new SAC files:
                    for trace in st2:
                        if trace.stats.channel != 'BHZ':
                            filename = directory + key[0:15] + '.' + network \
                                        + '.' + array_nm + '.' \
                                        + trace.stats.station + '.' \
                                        + trace.stats.channel + '.' \
                                        + str(trace.stats.sac.mag) + '.SAC'
                            trace.write(filename, format = 'SAC')

                    print('Traces for event on ' + str(st[0].stats.starttime+120) \
                          + ' , station ' + st[0].stats.station \
                          + ', successfully rotated!')

                except:

                    print('Weird error for event on ' + str(st[0].stats.starttime+120) \
                          + '!!')


###############################################################################







