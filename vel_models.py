'''
Created on Feb 03, 2018
@author: Itahisa Gonzalez Alvarez

This script contains functions that calculate or summarise the velocity models
for the arrays used in our EFM/EFMD analysis.

'''

import numpy as np
import scipy as sp
import scipy.interpolate
import matplotlib.pyplot as plt


def get_velocity_data( array_nm, vel_source ):

    '''
    This function loads velocity data from CRUST1.0 and AuSREM models for ASAR, WRA and PSA.

    Arguments:
        - array_nm: (str) code name for the array
        - vel_source: (str) source of the velocity data

    Output:
        - If vel_source is CRUST1, it returns a list with crust thickness, top depth of the model,
          depths and velocities.
        - If vel_source is AuSREM, it returns a list with crust thickness and two numpy arrays
          containing depths and velocities.

    '''

    ################################################################################################
    #                                     CRUST 1.0 DATA
    ################################################################################################
    #                                           ASAR
    ################################################################################################

    CRUST1_ASAR_top_depth = 0.68
    CRUST1_ASAR_bottom_depths = np.array([0.18, -0.92, -16.98, -32.57, -48.17, 200])
    CRUST1_ASAR_vels = np.array([2.5, 4.50, 6.20, 6.50, 7.10, 8.30 ])

    ################################################################################################
    #                                           WRA
    ################################################################################################

    CRUST1_WRA_top_depth = 0.3
    CRUST1_WRA_bottom_depths = np.array([0.0, -14.41, -30.62, -45.04, 200])
    CRUST1_WRA_vels = np.array([2.5, 6.10, 6.50, 6.90, 8.34 ])

    ################################################################################################
    #                                           PSA
    ################################################################################################

    CRUST1_PSA_top_depth = 0.3
    CRUST1_PSA_bottom_depths = np.array([-9.69, -19.68, -29.98, 200])
    CRUST1_PSA_vels = np.array([6.20, 6.40, 6.80, 8.24 ])



    ################################################################################################
    #                                     AuSREM DATA
    ################################################################################################

    # Create depth vector (the same for all models):
    AuSREM_bottom_depths = np.concatenate((np.arange(-5,-50,-5), np.arange(-50,-225,-25)))

    ################################################################################################
    #                                           ASAR
    ################################################################################################

    AuSREM_ASAR_vels = np.array([6.028, 6.293, 6.425, 6.497, 6.625, 6.813, 7.111, 7.311, 7.437, 7.8850,
                                 8.0165, 8.1195, 8.1870, 8.2725, 8.3395, 8.4020])

    ################################################################################################
    #                                           WRA
    ################################################################################################

    AuSREM_WRA_vels = np.array([6.248, 6.308, 6.219, 6.645, 6.960, 7.291, 7.642, 7.806, 7.985, 8.0505,
                                8.1248, 8.1625, 8.2003, 8.2755, 8.3460, 8.4230])

    ####################################################################################################
    #                                           PSA
    ####################################################################################################

    AuSREM_PSA_vels = np.array([6.089, 6.211, 6.391, 6.451, 6.564, 7.401, 7.958, 8.280, 8.34, 8.1815,
                                8.1970, 8.1955, 8.1930, 8.2395, 8.2915, 8.3640])

    # Create dictionaries with the velocity information for each data source and add crust thickness:
    ASAR_crust_thickness = 46; ASAR_lithos_thickness = 200
    WRA_crust_thickness = 46; WRA_lithos_thickness = 200
    PSA_crust_thickness = 32; PSA_lithos_thickness = 200

    vel_data = {'CRUST1': {
                            'YKA':  [PSA_crust_thickness, PSA_lithos_thickness, CRUST1_PSA_top_depth, CRUST1_PSA_bottom_depths, CRUST1_PSA_vels],
                            'ASAR': [ASAR_crust_thickness, ASAR_lithos_thickness, CRUST1_ASAR_top_depth, CRUST1_ASAR_bottom_depths, CRUST1_ASAR_vels],
                            'WRA':  [WRA_crust_thickness, WRA_lithos_thickness, CRUST1_WRA_top_depth, CRUST1_WRA_bottom_depths, CRUST1_WRA_vels],
                            'PSA':  [PSA_crust_thickness, PSA_lithos_thickness, CRUST1_PSA_top_depth, CRUST1_PSA_bottom_depths, CRUST1_PSA_vels]},
                'AuSREM': {
                            'YKA':  [PSA_crust_thickness, PSA_lithos_thickness, AuSREM_bottom_depths, AuSREM_PSA_vels],
                            'ASAR': [ASAR_crust_thickness, ASAR_lithos_thickness, AuSREM_bottom_depths, AuSREM_ASAR_vels],
                             'WRA': [WRA_crust_thickness, WRA_lithos_thickness, AuSREM_bottom_depths, AuSREM_WRA_vels],
                             'PSA': [PSA_crust_thickness, PSA_lithos_thickness, AuSREM_bottom_depths, AuSREM_PSA_vels]}}

    return vel_data[vel_source][array_nm]

####################################################################################################
#                                        END OF THE FUNCTION
####################################################################################################





def get_velocity_model(array_nm, vel_source, num_layers, layers_bottoms, units = 'm'):

    '''
    Function that uses the lithos velocity models to calculate the vertical traveltimes through each
    layer of my model, the cumulative traveltimes and the time it takes the seismic waves to reach the
    free surface.

    Arguments:
        - array_nm : (str) representative code for the seismic array
        - vel_source: (str) source for the velocity data (either AuSREM or CRUST1)
        - num_layers: (int) number of layers in the model
        - layers_bottoms: (list) bottom depths of each layer in the model, in order from top to bottom
        - units: (str) preferred units for the outputs (either 'm' or 'km')

    Outputs:
        Dictionary containing:
            -num_layers: number of layers in the model
            -L: bottom depth of each one of the layers in my model (in km)
            -v: P wave velocity for each layer in the model (in km/s)
            -dtjs: vertical traveltimes through each layer (in s)
            -tjs: vertical traveltimes from the bottom of the model to the top of each layer (in s)
            -tJ: vertical traveltime from the bottom of the model to the free surface (in s)
            -TFWM_vels: mean velocities from the surface to the bottom of each layer. This is the
                        velocity model we need to use in the TFWM.
            -lithos_model: dictionary containing all layers in the model and the P wave velocity of
                            each one of them as calculated for the EFM or TFWM.
            -plot_vels: Numpy array with velocities for plots.
            -plot_depths: Numpy array with depths for plots.

    '''

    ################################################################################################
    #                                                                                              #
    #                         PART 1: LOAD RAW VELOCITY DATA                                       #
    #                                                                                              #
    ################################################################################################

    # Vel data is: crust thickness[0], lithos_thickness[1], depths vector[2], velocities vector[3].
    vel_data = get_velocity_data( array_nm, vel_source )

    # Adapt units:
    if units == 'km':
        vel_data = vel_data
    if units == 'm':
        for i in range(len(vel_data)):
            vel_data[i] = vel_data[i] * 1000

    # Define depths and vels:
    depths = np.append( 0, -vel_data[2])
    vels = np.append( vel_data[3][0], vel_data[3])

    # Interpolate so we have more data points in the depths and velocity arrays.
    new_length = 100
    new_depths = np.linspace (0, depths.max(), new_length)
    new_vels = sp.interpolate.interp1d( depths, vels, kind = 'linear')(new_depths)


    ################################################################################################
    #                                                                                              #
    #        PART 2: CREATE VELOCITY MODELS FOR EACH MODEL TYPE AND EACH METHOD IN THE EFMD        #
    #                                                                                              #
    ################################################################################################

    #                              2a. EFM/EFMD VELOCITY MODELS                                    #

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

    ################################################################################################
    #                                                                                              #
    #                          PART 3: GET TRAVELTIMES THROUGH THE MODEL                           #
    #                                                                                              #
    ################################################################################################

    # CAREFUL!!! Traveltimes should be given in the same order seismic waves travel through them!
    # This means traveltimes through the bottom layer should go first/last and traveltimes through
    # the layer containing the free surface should  be at the center of the numpy array.

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

    # Make L, v, dtjs and tjs symmetric to account for the total reflection at the free surface: we
    # should have an odd number of layers, with layer J (the one containing the free surface) being
    # a duplicate in thickness and traveltime of layer num_layers - 1 in inv_thicks, inv_Ls and dtjs.

    # Duplicate thickness and traveltime for layer containing the free surface:
    inv_dtjs[-1] = inv_dtjs[-1] * 2
    inv_thicks[-1] = inv_thicks[-1] * 2

    thicks = np.append( np.array(inv_thicks), thicks[1:])
    dtjs = np.append( np.array(inv_dtjs), dtjs0[1:])
    v = np.append( np.array(inv_vs), vs[1:])

    # Define vertical traveltimes from the bottom of the model to the top of each layer (ts):
    tjs = np.cumsum( dtjs)

    #

      ################################################################################################
    #                                                                                              #
    #                              PART 4: TFWM VELOCITY MODELS                                    #
    #                                                                                              #
    ################################################################################################

        #                                2b. TFWM VELOCITY MODELS                                      #

    # The TFWM doesn't allow layered models and it requires constant velocities through the whole
    # scattering layer.
    # To obtain this velocity model I calculate the weighted mean velocity from the surface to the bottom of
    # each layer, using traveltimes through the layers as weights.
    weights = np.cumsum( dtjs0 )

    TFWM_vels = []; TFWM_ttimes = []
    for i in range(num_layers):
        weighted_mean_vel = np.sum( vs[:i+1]*weights[:i+1] ) / np.sum( weights[:i+1] )
        TFWM_vels.append( np.round( weighted_mean_vel, 3 ) )
        TFWM_ttimes.append( weights[i] )

    ################################################################################################
    #                                                                                              #
    #                          PART 4: CREATE ADDITIONAL DICTIONARIES                              #
    #                                                                                              #
    ################################################################################################

    # Let's create a dictionary summarising the model's characteristics (number of layers, bottom
    # depths, )

    lithos_model = {}
    for i in range(num_layers):

        key = 'L' + str(i+1)
        lithos_model[key] = {'Bottom': Ls[i],
                            'EFM_vel': vs[i],
                            'TFWM_vel': TFWM_vels[i]}

    # Let's put ALL results into a dictionary:
    vel_model = {'num_layers': num_layers,
                  'L': np.array( Ls ),
                  'thicks': thicks,
                  'v': v,
                  'dtjs': dtjs,
                  'tjs': tjs,
                  'tJ': tJ,
                  'TFWM_vels': TFWM_vels,
                  'TFWM_ttimes': TFWM_ttimes,
                  'lithos_model': lithos_model,
                  'plot_vels': new_vels,
                  'plot_depths': new_depths}

    return vel_model


####################################################################################################
#                                        END OF THE FUNCTION
####################################################################################################





def get_velocity_model_even(array_nm, vel_source, model_type, units = 'm'):

    '''
    Function that uses the lithos velocity models to calculate the vertical traveltimes through each
    layer of my model, the cumulative traveltimes and the time it takes the seismic waves to reach the
    free surface.

    Arguments:
        - array_nm : (str) representative code for the seismic array
        - vel_source: (str) source for the velocity data (either AuSREM or CRUST1)
        - model_type: (str) I am considering two different types of models.
            * Model type 'I': lithosphere divided into four equally thick layers.
            * Model type 'II': crust divided into two equally thick layers, upper mantle divided as
                                well into two equally thick layers.
        - units: (str) preferred units for the outputs (either 'm' or 'km')

    Outputs:
        Dictionary containing:
            -num_layers: number of layers in the model
            -L: bottom depth of each one of the layers in my model (in km)
            -v: P wave velocity for each layer in the model (in km/s)
            -dtjs: vertical traveltimes through each layer (in s)
            -tjs: vertical traveltimes from the bottom of the model to the top of each layer (in s)
            -tJ: vertical traveltime from the bottom of the model to the free surface (in s)
            -TFWM_vels: mean velocities from the surface to the bottom of each layer. This is the
                        velocity model we need to use in the TFWM.
            -lithos_model: dictionary containing all layers in the model and the P wave velocity of
                            each one of them as calculated for the EFM or TFWM.
            -plot_vels: Numpy array with velocities for plots.
            -plot_depths: Numpy array with depths for plots.

    '''

    ################################################################################################
    #                                                                                              #
    #                         PART 1: LOAD RAW VELOCITY DATA                                       #
    #                                                                                              #
    ################################################################################################

    # Vel data is: crust thickness, depths vector, velocities vector.
    vel_data = get_velocity_data( array_nm, vel_source )

    # Adapt units:
    if units == 'km':
        vel_data = vel_data
    if units == 'm':
        for i in range(len(vel_data)):
            vel_data[i] = vel_data[i] * 1000

    # Define depths and vels:
    depths = np.append( 0, -vel_data[1])
    vels = np.append( vel_data[2][0], vel_data[2])

    # Get crust and lithosphere bottom depths:
    crust_bottom_depth = vel_data[0]
    lithos_bottom_depth = depths.max()
    upper_mantle_thickness = (lithos_bottom_depth - crust_bottom_depth)

    # Interpolate so we have more data points in the depths and velocity arrays.
    new_length = 1000
    new_depths = np.linspace (0, depths.max(), new_length)
    new_vels = sp.interpolate.interp1d( depths, vels, kind = 'linear')(new_depths)


    ################################################################################################
    #                                                                                              #
    #        PART 2: CREATE VELOCITY MODELS FOR EACH MODEL TYPE AND EACH METHOD IN THE EFMD        #
    #                                                                                              #
    ################################################################################################

    #                              2a. EFM/EFMD VELOCITY MODELS                                    #

    # Get layer thicknesses from model type. L should have twice as many values as
    # layers there are in the model, and they should be symmetric:

    if model_type == 'I':
        num_layers = 4
        # This model has four equally thick layers going down to the bottom of the lithosphere.
        L_step = lithos_bottom_depth / 4
        Ls = np.arange( L_step, lithos_bottom_depth + L_step, L_step)

    elif model_type == 'II':
        num_layers = 4
        crust_step = crust_bottom_depth / 2 # Thickness of the crust layers
        um_step = upper_mantle_thickness / 2 # Thickness of the upper mantle layers
        Ls = np.array( [ crust_step, 2*crust_step, crust_bottom_depth + um_step, crust_bottom_depth + um_step*2] )

    # Get indices of new_depths values closest to Ls values:
    inds = [0] # Add a zero so we can effectively slice new_vels
    for l in Ls:
        inds.append ( (np.abs( l-new_depths )).argmin() )

    # Create initial velocity vector:
    vs = []
    for i in range(len(inds)-1):
        vs.append( np.round( np.mean( new_vels[ inds[i]:inds[i+1] ] ), 3 ))


    ################################################################################################
    #                                                                                              #
    #                          PART 3: GET TRAVELTIMES THROUGH THE MODEL                           #
    #                                                                                              #
    ################################################################################################

    # CAREFUL!!! Traveltimes should be given in the same order seismic waves travel through them!
    # This means traveltimes through the bottom layer should go first/last and traveltimes through
    # the layer containing the free surface should  be at the center of the numpy array.

    # Define vertical traveltimes through each layer (dts): we need to define thickness of each layer
    # first.
    l_thicks = [Ls[0]] # Ls[0] is already the thickness of the top layer in the model
    for i in range(num_layers-1):
        l_thicks.append( Ls[i+1] - Ls[i])
    # Get traveltimes through each layer:
    dtjs = np.array(l_thicks) / vs

    # Reverse both Ls and dtjs:
    inv_dtjs = []; inv_Ls = []; inv_thicks = []
    for i in np.arange(num_layers-1, -1, -1):
        inv_Ls.append( Ls[i] )
        inv_thicks.append( l_thicks[i] )
        inv_dtjs.append( dtjs[i] )

    # Make L, v, dtjs and tjs symmetric to account for the total reflection at the free surface:
    rev_L = []; rev_v = []; rev_dtjs = []; rev_thicks = []
    for i in np.arange( num_layers-1, -1, -1):
        rev_L.append( Ls[i])
        rev_thicks.append( l_thicks[i])
        rev_v.append( vs[i])
        rev_dtjs.append( dtjs[i])
    L = np.append( rev_L, Ls)
    thicks = np.append( rev_thicks, l_thicks)
    v = np.append( rev_v, vs)
    dtjs = np.append( rev_dtjs, dtjs )

    # Define vertical traveltimes from the bottom of the model to the top of each layer (ts):
    tjs = np.cumsum( dtjs)

    # Define the time when the wavefront reaches the free surface:
    tJ = tjs[num_layers - 1]

    #

      ################################################################################################
    #                                                                                              #
    #                              PART 4: TFWM VELOCITY MODELS                                    #
    #                                                                                              #
    ################################################################################################

        #                                2b. TFWM VELOCITY MODELS                                      #

    # The TFWM doesn't allow layered models and it requires constant velocities through the whole
    # scattering layer.
    # To obtain this velocity model I calculate the weighted mean velocity from the surface to the bottom of
    # each layer, using traveltimes through the layers as weights.
    weights = np.cumsum( dtjs[num_layers:] )

    TFWM_vels = []; TFWM_ttimes = []
    for i in range(num_layers):
        weighted_mean_vel = np.sum( vs[:i+1]*weights[:i+1] ) / np.sum( weights[:i+1] )
        TFWM_vels.append( np.round( weighted_mean_vel, 3 ) )
        TFWM_ttimes.append( weights[i] )


    ################################################################################################
    #                                                                                              #
    #                          PART 4: CREATE ADDITIONAL DICTIONARIES                              #
    #                                                                                              #
    ################################################################################################

    # Let's create a dictionary summarising the model's characteristics (number of layers, bottom
    # depths, )

    lithos_model = {}
    for i in range(num_layers):

        key = 'L' + str(i+1)
        lithos_model[key] = {'Bottom': Ls[i],
                            'EFM_vel': vs[i],
                            'TFWM_vel': TFWM_vels[i]}

    # Let's put ALL results into a dictionary:
    vel_model = {'num_layers': num_layers,
                  'L': np.array( L ),
                  'thicks': thicks,
                  'v': v,
                  'dtjs': dtjs,
                  'tjs': tjs,
                  'tJ': tJ,
                  'TFWM_vels': TFWM_vels,
                  'TFWM_ttimes': TFWM_ttimes,
                  'lithos_model': lithos_model,
                  'plot_vels': new_vels,
                  'plot_depths': new_depths}

    return vel_model


####################################################################################################
#                                        END OF THE FUNCTION
####################################################################################################




def plot_velocity_models( vel_source ):

    '''
    This function plots the velocity models obtained from get_velocity_model for all arrays.

    Arguments:
        -array_nm: (str) name or code of the seismic array
        -vel_source: (str) source of the velocity data ('AuSREM' or 'CRUS1')
        -units: (str) units for depths and velocities (either 'SI' or 'km')

    Output:
        Single plot with all velocity models for all model types.

    '''

    # Define array names:
    arrays = ['WRA', 'PSA', 'ASAR']

    # The model type does not matter, since the velocity profile is the same for all of them:
    mod_type = 'II'

    #colors = [['orange', 'darkred', 'orangered', 'coral'],
    #          ['dodgerblue', 'navy', 'mediumblue', 'deepskyblue'],
    #          ['mediumaquamarine', 'darkslategrey', 'seagreen', 'forestgreen']]
    colors = {'ASAR': 'orangered',
              'WRA': 'deepskyblue',
              'PSA': 'forestgreen'}

    f, ax = plt.subplots( figsize = (5, 15) )
    for array in arrays:

        # Load velocity data:
        vel_model = get_velocity_data( array, vel_source)
        plot_vels = vel_model[3]
        plot_depths = -vel_model[2]

        ax.plot( plot_vels, plot_depths, linewidth = 3, color = colors[array], label = array)

        ax.set_title( 'AuSREM Velocity Models', fontsize = 32, pad = 40)
        ax.set_xlabel( 'Velocity (km/s)', fontsize = 30)
        ax.set_ylabel( 'Depth (km)', fontsize = 30)
        ax.set_ylim( 210, 0)
        ax.grid()
        ax.legend( loc='best', fontsize = 30 )
        ax.tick_params( axis = 'both', labelsize = 22, width = 2)
        for side in ax.spines.keys():
            ax.spines[side].set_linewidth(2)


    fname = '/nfs/a9/eeinga/Results/AU/Velocity_models/Velocity_models_all_arrays'
    plt.savefig( fname + '.pdf', bbox_inches = 'tight')
    plt.savefig( fname + '.png', bbox_inches = 'tight')
    plt.close()



####################################################################################################
#                                        END OF THE FUNCTION
####################################################################################################


# vel_source = 'AuSREM'
# plot_velocity_models( vel_source )