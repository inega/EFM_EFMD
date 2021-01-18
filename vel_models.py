'''
Created on Feb 03, 2018
@author: Itahisa Gonzalez Alvarez

This script contains functions that calculate or summarise the velocity models
for the arrays used in our EFM/EFMD analysis.

'''

import csv
import numpy as np
import scipy as sp
import scipy.interpolate

def get_velocity_data( array_nm, vel_source, vel_model_fname ):

    '''
    This function contains the velocity data from one or more sources for all
    studies arrays/stations.

    Arguments:
        - array_nm: (str) code name for the array
        - vel_source: (str) name of the velocity data source

    Output:
        - List with crust and lithosphere thickness and two numpy arrays
          containing depths and velocities of the velocity model.

    '''

    with open( vel_model_fname, 'r') as fd:
        reader = csv.reader( fd)
        for i, row in enumerate( reader):
            if i == 0: vel_source = row[0]
            if i == 1: array_nm = row[0]
            if i == 2: crustal_thickness = np.int( row[0])
            if i == 3: lithos_thickness = np.int( row[0])
            if i == 4:
                depths = []
                for val in row:
                    val = float( val)
                    depths.append( val)
                depths = np.array( depths)
            if i == 5:
                vels = []
                for val in row:
                    val = float( val)
                    vels.append( val)
                vels = np.array( vels)

    # Build list:
    vel_data = [crustal_thickness, lithos_thickness, depths, vels]

    return vel_data

###############################################################################
#                      END OF THE FUNCTION                                    #
###############################################################################




def get_velocity_model( array_nm, vel_source, vel_model_fname, num_layers,
                       units = 'm'):

    '''
    Function that loads the velocity data for the array/station and calculates
    the traveltimes through each layer in the model, the cumulative traveltimes
    and the time it takes the seismic waves to reach the free surface.

    Arguments:
        - array_nm : (str) code/name of the array/station
        - vel_source: (str) velocity data source name
        - vel_model_fname: (str) path and file name of the file containing the
                            velocity data. The format needs of this file is
                            explained in the README file
        - num_layers: (int) number of layers in the model
        - units: (str) preferred units for the outputs (either 'm' or 'km')

    Outputs:
        Dictionary containing:
            -num_layers: (int) number of layers in the model
            -L: (np.array) bottom depth of each one of the layers in my model
            -v: (np.array) mean P wave velocity for each layer in the model
            -dtjs: (np.array) vertical traveltimes through each layer (in s)
            -tjs: (np.array) vertical traveltimes from the bottom of the model
                  to the top of each layer (in s)
            -tJ: (float) vertical traveltime from the bottom of the model to
                 the free surface (in s)

    '''

    ###########################################################################
    #                    PART 1: LOAD RAW VELOCITY DATA                       #
    ###########################################################################

    # Vel data is: crust thickness[0], lithos_thickness[1], depths vector[2],
    # velocities vector[3].
    vel_data = get_velocity_data( array_nm, vel_source, vel_model_fname )

    # Adapt units:
    if units == 'km':
        vel_data = vel_data
    if units == 'm':
        for i in range(len(vel_data)):
            vel_data[i] = vel_data[i] * 1000

    # Define layers bottoms:
    if num_layers == 1:
        layers_bottoms = [ vel_data[1]]
    elif num_layers == 2:
        layers_bottoms = [ vel_data[0], vel_data[1]]
    elif num_layers == 3:
        layers_bottoms = [ vel_data[0]/2,  vel_data[0], vel_data[1]]

    # Define depths and vels:
    depths = np.append( 0, vel_data[2])
    vels = np.append( vel_data[3][0], vel_data[3])

    # Interpolate so we have more data points in the depths and velocity arrays.
    new_length = 100
    new_depths = np.linspace (0, depths.max(), new_length)
    new_vels = sp.interpolate.interp1d( depths, vels, kind = 'linear')(new_depths)

    ###########################################################################
    #          PART 2: CREATE VELOCITY MODELS FOR EACH MODEL                  #
    #             TYPE AND EACH METHOD IN THE EFMD                            #
    ###########################################################################

    # 2a. EFM/EFMD VELOCITY MODELS

    # Rename layers_bottoms:
    Ls = layers_bottoms

    # Get layer thicknesses:
    thicks = [ layers_bottoms[0] ]
    for i in range( num_layers - 1 ):
        thicks.append( layers_bottoms[i+1] - layers_bottoms[i])

    # Get indices of new_depths values closest to Ls values:
    inds = [0] # Add a zero so we can effectively slice new_vels
    for l in Ls:
        inds.append ( (np.abs( l-new_depths )).argmin() )

    # Create initial velocity vector:
    vs = []
    for i in range(len(inds)-1):
        vs.append( np.round( np.mean( new_vels[ inds[i]:inds[i+1] ] ), 3 ))

    ###########################################################################
    #         PART 3: GET TRAVELTIMES THROUGH THE MODEL                       #
    ###########################################################################

    # CAREFUL!!! Traveltimes should be given in the same order seismic waves
    # travel through the layers! This means traveltimes through the bottom layer
    # should go first/last and traveltimes through the layer containing the free
    # surface should  be at the center of the numpy array.

    # Define vertical traveltimes through each layer (dts):
    dtjs0 = np.array(thicks) / vs

    # Define the time when the wavefront reaches the free surface:
    tJ = np.sum( dtjs0 )

    # Reverse both Ls and dtjs:
    inv_dtjs = []; inv_Ls = []; inv_thicks = []; inv_vs = []
    for i in np.arange(num_layers-1, -1, -1):
        inv_Ls.append( Ls[i] )
        inv_vs.append( vs[i] )
        inv_thicks.append( thicks[i] )
        inv_dtjs.append( dtjs0[i] )

    # Make L, v, dtjs and tjs symmetric to account for the total reflection at
    # the free surface: we should have an odd number of layers, with layer J
    # (the one containing the free surface) being a duplicate in thickness and
    # traveltime of layer num_layers - 1 in inv_thicks, inv_Ls and dtjs.

    # Duplicate thickness and traveltime for layer containing the free surface:
    inv_dtjs[-1] = inv_dtjs[-1] * 2
    inv_thicks[-1] = inv_thicks[-1] * 2

    thicks = np.append( np.array(inv_thicks), thicks[1:])
    dtjs = np.append( np.array(inv_dtjs), dtjs0[1:])
    v = np.append( np.array(inv_vs), vs[1:])

    # Define vertical traveltimes from the bottom of the model to the top of
    # each layer (ts):
    tjs = np.cumsum( dtjs)

    ###########################################################################
    #             PART 4: CREATE FINAL DICTIONARY                             #
    ###########################################################################

    # Let's create a dictionary summarising the model's characteristics (number
    # of layers, bottom depths, etc)

    # Let's put ALL results into a dictionary:
    vel_model = {'num_layers': num_layers,
                'L': np.array( Ls ),
                'thicks': thicks,
                'v': v,
                'dtjs': dtjs,
                'tjs': tjs,
                'tJ': tJ,
                'layer_bottoms': layers_bottoms
                }

    return vel_model

###############################################################################
#                       END OF THE FUNCTION
###############################################################################

