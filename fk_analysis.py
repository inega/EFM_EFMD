'''
Created on Nov 16, 2017
@author: Itahisa Gonzalez Alvarez

Functions to run an FK analysis on a stream and determine the slowness vector,
backazimuth and time shifts required to align the traces.

'''

import copy
import numpy as np
from obspy import UTCDateTime
import matplotlib.pyplot as plt
from obspy.taup import TauPyModel
from obspy.signal.util import util_geo_km



def fk_slowness_vector(stream, smin, smax, s_int, tmin, tmax, units):

    '''
    Carry out an FK analysis and obtain the slowness vector for a seismic array
    and event.

    Arguments:
         - stream (Obspy stream) = traces recorded by array stations for the
                                   event
         - smin (float) = minimum slowness to consider
         - smax (float) = maximum slowness to consider
         - s_int (float) = slowness interval/step size in our grid
         - tmin (int or float) = start of the time window to use, in seconds
                                 before theoretical P wave arrival
         - tmax (int or float) = end of the time window to use, in seconds
                                 after theoretical P wave arrival
         - units (str) = km or degrees

    Output:
         - dictionary containing the slowness, backazimuth, power matrix,
           slowness matrix, backazimuth matrix, start time, end time, event
           time and units. Slowness  values are given in s/units and backazimuth
           in degrees measured clockwise from north

    '''

    # Make a deep copy of the original stream:
    st = copy.deepcopy( stream)

    # Define some useful quantities:
    num_stations = len( stream)
    delta = stream[0].stats.delta# Time step

    # Get station distances to the array center:
    if units == 'km':
        rel_x, rel_y, ar_z = np.split( get_rel_station_coordinates_km( st),
                                    3, axis = 1)
    if units == 'degrees':
        rel_x, rel_y, ar_z = np.split( get_rel_station_coordinates_degs( st),
                                    3, axis = 1)

    # Obtain theoretical P wave arrival time for each station (in seconds):
    # Getting theoretical P wave arrivals to trim the traces to the time window
    # of interest.
    ev_date = st[0].stats.starttime + 120# Data starts 2 mins before the event
    t_P, t_P_date = get_Pwave_arrivals( st, ev_date, model = 'prem')

    # We need to trim the traces to a short time window around the P wave arrival.
    # Find starttime and endtime: from minimum theoretical P wave arrival
    # date - tmin to maximum theoretical P wave arrival + 10 seconds. It is
    # convenient to use dates to trim the traces, instead of numbers of seconds
    # from the beginning, it's less confusing.
    starttime = min ( t_P_date ) - tmin
    endtime = max( t_P_date ) + tmax

    #Cut traces to the desired time window:
    for trace in st:
        trace.trim( starttime, endtime)

    ###########################################################################

    # Obtain Fourier transform for each trace:

    # Number of samples in each trace:
    npts = st[0].stats.npts

    # Length of real FFT is only half of that of time series data:
    specs = np.zeros( ( num_stations, int( npts / 2) + 1), dtype = complex)

    # Calculate spectra and put them in specs:
    for i, trace in enumerate( st):
        specs[i, :] = np.fft.rfft( trace.data)

    # Calculate frequencies:
    freqs = np.fft.fftfreq( npts, delta)[0:int( npts / 2) + 1]

    ###########################################################################

    # Create slowness grid.
    step = int( ( ( smax - smin) / s_int) + 1)
    slow_gridx = np.linspace( smin, smax, step)
    slow_gridy = copy.deepcopy( slow_gridx)

    ###########################################################################

    # Run FK analysis:

    # Predefine variables:
    fk = np.zeros( ( step, step));power = copy.deepcopy( fk)
    slo_matrix = copy.deepcopy( fk);bazim_matrix = copy.deepcopy( fk)

    for vv in range( int( step)):
        for ww in range( int( step)):

            # Get time shifts:
            dt = slow_gridx[ww] * rel_x + slow_gridy[vv] * rel_y

            # Get "beam", the total energy integral: it's the product of the
            # spectra and the exponential of time shifts by a constant, all
            # divided by the number of stations. The np.sum is the integral.
            beam = np.sum( np.exp( - 1j * 2 * np.pi * np.outer( dt, freqs)) * \
                          specs / num_stations, axis = 0)

            # Get power: it is the square of beam, remember it is a complex
            # number, so we don't want the imaginary part!
            fk = np.vdot( beam, beam).real
            power[vv, ww] = fk # Add to the power matrix

            # Get slowness magnitude:
            slo_mag = np.sqrt( slow_gridx[ww] ** 2 + slow_gridy[vv] ** 2)
            slo_matrix[vv, ww] = slo_mag # Add to slowness magnitude matrix

            # Get backazimuth:
            bazimuth = np.degrees( np.arctan2( slow_gridx[ww], slow_gridy[vv]))
            if bazimuth<0:
                bazimuth += 360
            bazim_matrix[vv, ww] = bazimuth # Add value to the backazimuth matrix

    ###########################################################################

    # Get results:

    #Get index of maximum energy:
    max_pow_index = np.where( power == power.max())#This contains 2 indices!
    max_pow_index_0 = max_pow_index[0][0];max_pow_index_1 = max_pow_index[1][0]

    #Get slowness, bazim and time shifts for that specific set of indices:
    slo_mag_value = slo_matrix[max_pow_index_0, max_pow_index_1]
    bazim_value = bazim_matrix[max_pow_index_0, max_pow_index_1]

    fk_results = {'Slowness': slo_mag_value,
                  'Backazimuth': bazim_value,
                  'Power_matrix': power,
                  'Slowness_matrix': slo_matrix,
                  'Backazimuth_matrix': bazim_matrix,
                  'Starttime': starttime,
                  'Endtime': endtime,
                  'Event_time': ev_date,
                  'Units': units}

    return( fk_results)

###############################################################################
####                       END OF THE FUNCTION                             ####
###############################################################################




def get_timeshifts( stream, smin, smax, s_int, tmin, tmax, units):
    '''
    Function that uses the fk_slowness_vector function to calculate the
    slowness magnitude and backazimuth value for a particular event and array.
    Both functions use get_rel_station_coordinates to get relative distances from
    each station to the center of the array.

    Arguments:
         - stream (Obspy stream) = traces recorded by array stations for the
                                   event
         - smin (float) = minimum slowness to consider
         - smax (float) = maximum slowness to consider
         - s_int (float) = slowness interval/step size in our grid
         - tmin (int or float) = start of the time window to use, in seconds
                                 before theoretical P wave arrival
         - tmax (int or float) = end of the time window to use, in seconds
                                 after theoretical P wave arrival
         - units (str) = km or degrees

    Output:

         - time shifts, in seconds, needed to align the traces within the
           input stream

    '''

    # Run fk analysis:
    fk_results = fk_slowness_vector( stream, smin, smax, s_int, tmin, tmax,
                                    units)

    # We need only slo_mag_value and bazim_value from fk_results.
    slowness_magnitude = fk_results['Slowness']# In s / degree
    backazimuth = fk_results['Backazimuth']

    # Get slowness vector components:
    #N - S direction:
    slowy = np.cos( np.radians( backazimuth)) * slowness_magnitude# In s/units
    #E - W direction:
    slowx = np.sin( np.radians( backazimuth)) * slowness_magnitude# In s/units

    # Get array geometry:
    if units == 'km':
        rel_x, rel_y, ar_z = np.split( get_rel_station_coordinates_km( stream),
                                    3, axis = 1)
    if units == 'degrees':
        rel_x, rel_y, ar_z = np.split( get_rel_station_coordinates_degs( stream),
                                    3, axis = 1)

    # Get time shifts: we want each time shift associated with its station.
    tshifts = []
    for v in range( len( stream)):
        tshifts.append( slowx * rel_x[v] + slowy * rel_y[v])


    return tshifts

###############################################################################
####                       END OF THE FUNCTION                             ####
###############################################################################





def get_array_coordinates( stream):

    '''
    This function uses an inventory file to calculate the relative distance
    from each station of the array to its center, as well as the coordinates
    of the array center. The reason I use the coordinates within the stats of
    the traces in the stream instead of the inventory file is because I need
    these coords to be in the exact same order of the traces in the stream.

    Arguments:
        -stream (Obspy stream) = traces recorded at the array for the event

    Returns:
        -coords, cen_lon, cen_lat, cen_ele (list) = list containing the relative
                                                    distances from each station
                                                    to the array center, as well
                                                    as the center's coordinates

    '''

    # Get number of stations:
    num_stations=len( stream)

    # Get coordinates of stations in array. Looks for SAC headers stla, stlo, stel.
    coords = np.empty( ( num_stations, 3))

    # Get station name, lon, latitude, and elevation
    for i in range( num_stations):
        coords[i, 0]=stream[i].stats.sac.stlo
        coords[i, 1]=stream[i].stats.sac.stla
        coords[i, 2]=stream[i].stats.sac.stel

    # Get coordinates of the center of the array:
    cen_lon=coords[:, 0].mean()
    cen_lat=coords[:, 1].mean()
    cen_ele=coords[:, 2].mean()


    return [coords, cen_lon, cen_lat, cen_ele]

###############################################################################
####                       END OF THE FUNCTION                             ####
###############################################################################




def get_rel_station_coordinates_km( stream):

    '''
    This function calculates the relative distances from each station in the
    array to its center, in km. The reason I use the coordinates within the
    stats of the traces in the stream instead of the inventory file is because
    I need these coords to be in the exact same order of the traces in the
    stream.

    Arguments:
        -stream (Obspy stream) = traces recorded at the array for the event

    Returns:
        -coords (Numpy array) = relative distances from stations to the array
                                center

    '''

    # Get array coordinates:
    coords, cen_lon, cen_lat, cen_ele = get_array_coordinates( stream)

    # Get relative coordinates (distances) of the stations with respect to the
    # array center:

    # Util_geo_km transforms lon, lat to km with reference to origin lon and lat.
    for i in range( len( coords)):
        x, y=util_geo_km( cen_lon, cen_lat, coords[i, 0], coords[i, 1])
        coords[i, 0]=x
        coords[i, 1]=y
        coords[i, 2]-=cen_ele

    return coords

###############################################################################
####                       END OF THE FUNCTION                             ####
###############################################################################





def get_rel_station_coordinates_degs( stream):

    '''
    This function calculates the relative distances from each station in the
    array to its center, in degrees. The reason I use the coordinates within
    the stats of the traces in the stream instead of the inventory file is
    because I need these coords to be in the exact same order of the traces in
    the stream.

    Arguments:
        -stream (Obspy stream) = traces recorded by array stations for the
                                 event
    Returns:
        -coords (Numpy array) = relative distances from stations to the array
                                center

    '''

    # Get array coordinates:
    coords, cen_lon, cen_lat, cen_ele = get_array_coordinates( stream)

    # Get relative coordinates (distances) of the stations with respect to the
    # array center:

    # Get coordinates relative to the center (offsets):
    for i in range( len( coords)):
        coords[i, 0]=( coords[i, 0]-cen_lon)*np.cos( np.radians( cen_lat))
        coords[i, 1]-=cen_lat
        coords[i, 2]-=cen_ele

    return coords

###############################################################################
####                       END OF THE FUNCTION                             ####
###############################################################################





def get_Pwave_arrivals( stream, ev_time, model = 'prem'):

    '''
    Function that calculates the theoretical P wave arrivals for the traces
    present in a stream. Traces need to contain the distance to and depth of
    the event in the metadata for this function to work.

    Arguments:
        -stream (Obspy stream): stream containing all the traces we want to
                                calculate the theoretical P wave arrival time for.
        -ev_time (str): event time in the format 'YYYYMMDDTHHMMSS'.
        -model (str): Earth model to use to calculate the theoretical P wave
                      arrival times. Default is PREM.

    Output:
        -t_P (list): theoretical P wave arrival times (in seconds from the
                    start of the trace, considering that our traces go from 2
                    minutes before the start of the event until 58 minutes later)
                    for the traces in stream, in the same order the traces are
                    listed in the stream.
        -t_P_date (list): theoretical P wave arrival dates and times in
                          UTCDateTime format for the traces in stream, in the
                          same order the traces are listed in the stream.

    '''

    #print('Getting theoretical P wave arrival times. Model used is ' + model + '...')

    # Choose model to calculate theoretical traveltimes:
    model = TauPyModel( model = model)

    # Get theoretical P wave arrival times:
    t_P = [];t_P_date = []

    for trace in stream:
        ev_time = UTCDateTime( ev_time )# Data from 2 minutes BEFORE the event!
        dist_degs = trace.stats.sac.gcarc # distance to event in degrees
        ev_dep = trace.stats.sac.evdp / 1000 # depth in km
        # We are only interested in P wave arrivals:
        arrivals = model.get_travel_times( source_depth_in_km = ev_dep,
                                          distance_in_degree = dist_degs)

        # Save travel times:
        for arrival in arrivals:
            if arrival.name == 'P': P_time = arrival.time

        P_arrs = 120 + P_time # Time from start of the trace
        t_P.append( P_arrs)
        t_P_date.append( ev_time + P_time)

    return t_P, t_P_date

###############################################################################
####                       END OF THE FUNCTION                             ####
###############################################################################





def fk_plot( array_nm, stream, smin, smax, s_int, tmin, tmax, units,
            fname = None, show_plot = False):

    '''
    Function plots the results from the FK_slowness_vector function.

    Arguments:
         - array_nm (str) = name of the array
         - event date (str) = date of the event, in the YYYYMMDDTHHMM format
         - stream (Obspy stream) = traces recorded by array stations for the
                                   event
         - smin (float) = minimum slowness to consider
         - smax (float) = maximum slowness to consider
         - s_int (float) = slowness interval/step size in our grid
         - tmin (int or float) = start of the time window to use, in seconds
                                 before theoretical P wave arrival
         - tmax (int or float) = end of the time window to use, in seconds
                                 after theoretical P wave arrival
         - units (str) = km or degrees

    Output:
         - Plot of the result of the fk analysis

    '''

    fk_results = fk_slowness_vector( stream, smin, smax, s_int, tmin, tmax,
                                    units)

    # Extract results from the dictionary:
    slo_value = fk_results['Slowness']
    bazim_value = fk_results['Backazimuth']
    power_matrix = fk_results['Power_matrix']
    starttime = fk_results['Starttime']
    endtime = fk_results['Endtime']
    ev_date = fk_results['Event_time']

    # Print results to screen:
    print( 'Event: ' + ev_date + ', slowness = ' \
          + str( np.round( fk_results['Slowness'], 4)) + 's/'  + units \
          + ', backazimuth = ' + str( np.round( stream[0].stats.sac.baz, 4) ) \
          + 'degs, fk backazimuth = ' \
          + str( np.round( fk_results['Backazimuth'], 4))  + 'degs')

    # Create slowness grid.
    num_steps = int( ( ( smax - smin) / s_int) + 1 )
    slow_gridx = np.linspace( smin, smax, num_steps)
    slow_gridy = slow_gridx.copy()

    # Create figure:
    f, ax = plt.subplots( figsize = ( 18, 18))
    f.tight_layout( rect = [0.055, 0.035, 0.999, 0.85])
    f.suptitle( 'FK Analysis results: array ' + str( array_nm) + ', event date: ' \
               + str( ev_date), fontsize = 24)

    title = 'Time window from ' + str( round( starttime - ev_date, 2)) \
            + ' to ' +  str( round( endtime - ev_date, 2)) \
            + ' after the event. ' + '\n Slowness =  ' \
            + str( np.round( slo_value, 3)) + ' s /' + units \
            + ', backazimuth =  ' + str( np.round( bazim_value, 4)) + ' deg'
    ax.set_title( title, fontsize = 12)

    # Create title and suptitle:
    if units == 'km':
        ax.set_xlabel( 'slowness east (s/km)', fontsize = 10)
        ax.set_ylabel( 'slowness north (s/km)', fontsize = 12)
    elif units == 'degrees':
        ax.set_xlabel( 'slowness east (s/deg)', fontsize = 10)
        ax.set_ylabel( 'slowness north (s/deg)', fontsize = 12)

    im = ax.contourf( slow_gridx, slow_gridy, power_matrix, 16, cmap = 'afmhot_r')
    ax.grid( linestyle = 'dashed', alpha = 0.4)
    cb = f.colorbar( im, ax=ax)
    cb.set_label( 'Power', fontsize = 12)
    ax.set_xlim( - smax, smax);
    ax.set_ylim( - smax, smax);
    ax.tick_params( axis = 'both', which = 'major', labelsize = 10)

    if fname != None:
        f.savefig( fname, bbox_inches = 'tight')

    if show_plot == False:
        plt.close()

###############################################################################
####                       END OF THE FUNCTION                             ####
###############################################################################





def plot_array( stream, fname = None, show_plot = False):
    '''
    This function uses the stations coordinates stores in the stats of the
    traces and uses them to plot the array configuration and center.

    Arguments:
        -stream (Obspy stream) = traces recorded at the array for an event
        -fname (str) = path and file name to save the figure (optional)
        -show_plot (bool) = True/False whether to show the resulting figure

    Returns:
        -Plot of the station configuration and location of the array center

    '''

    # Get array coordinates:
    coords, cen_lon, cen_lat, cen_ele = get_array_coordinates( stream)

    # Plot station position:
    for i in range( len( coords)):
        plt.plot( coords[i, 0], coords[i, 1], '^', markersize = 20,
                 label = stream[i].stats.station)

    plt.plot( cen_lon, cen_lat, 'k*', markersize=20)
    plt.legend( ncol=2)

    if fname != None:
        plt.savefig( fname, bbox_inches = 'tight')
    if show_plot != True:
        plt.close()

###############################################################################
####                       END OF THE FUNCTION                             ####
###############################################################################




