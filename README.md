This repository contains all the files needed to characterize the small-scale heterogeneity
structure and attenuation of the lithosphere using the EFM/EFMD Bayesian approach. 

The network, array name, velocity source name and paths of the directories where we want
to store our data and/or results need to be added to the scripts. All executable files
are stored in the /bin directory and modules than need to be loaded in the /lib directory.
    
Steps:

0 - Create a python environment with all the necessary dependencies for the analysis. The
    seismodelling_env.yml file can be used for this step.

1 - Download data. The inv_cat_and_wvf_downloader.py script can be used to download event
    and stations metadata from IRIS, as well as waveforms in mseed format. Instrument response
    can be removed from the data during either in this step or the next.
    
2 - Convert waveforms to SAC format and add event information to SAC headers and stats.
    The populate_SAC_stats.py script can be used for this task.

3 - Rotate horizontal N and E component traces to radial and transverse components. The
    script horizontal_components_rotation.py contains the necessary code to do this.

4 - Remove events with low signal-to-noise ratio. Only traces with SNR > 5 will be used
    in the analysis. The data_quality_control.py script contains functions that carry
    out the data quality control for the whole dataset.
    
5 - EFM/EFMD analysis. The EFM_EFMD_inversion.py script carries out all the necessary 
    steps in the analysis. The script also allows us to choose which steps of the 
    analysis we want to carry out, since some of them only need to be done once, while 
    others will likely be repeated a number of times. 
    First, we need to calculate the normalized coda envelopes for all frequency bands. 
    Then, we run the EFM analysis and obtain the intrinsic, scattering and diffusion 
    quality factors and their frequency dependency. Next, the EFMD analysis needs to be 
    run multiple times, in order to guarantee the convergence of the algorithm. Fine 
    tuning of the step sizes defined in the EFMD_Bayesian_modelling is critical to 
    ensure correct acceptance rates. Finally, the different chains obtained from the 
    EFMD are combined into a single set of results. Careful! The files we wish to 
    combine need to be moved into a separate subdirectory before combining them to 
    avoid mistakes. The EFM_EFMD inversion will plot the combined results and save 
    the figures into this new subdirectory.
    
Please do not hesitate to contact me for any questions or issues running these codes.
    
NOTE: For synthetic tests, we need to create the synthetic envelopes for a given
      combination of scattering parameters first, and save them in the same 
      directory and format than our normalised coda envelopes from the EFM. The 
      EFM_EFMD_Bayesian_modelling function can be used for this calculation.
      The algorithm will assume that, in the case of synthetic tests, the input
      parameter values will also be stored in the same dictionary as the synthetic
      envelopes, so they can be compared with the ones obtained from the inversion.

#######################################################################################

Velocity data:
      The velocity model for each station/array needs to be saved to a csv file.
      The format of this file is as follow:
      Line 1 = name of the velocity data source (CRUST1,etc)
      Line 2 = array/station code
      Line 3 = crustal thickness in km (int)
      Line 4 = lithospheric thickness in km (int)
      Line 5 = depth values
      Line 6 = velocity values
      
      Example:
	  CRUST1
	  PSA
	  32
	  200
	  9.69,19.68,29.98,200
	  6.20,6.40,6.80,8.24

#######################################################################################

Extra/auxiliary functions are stored in these files in the lib directory:

    EFM_EFMD_tools.py - functions to create streams of traces and do basic processing/plotting
			and other auxiliary functions
    
    trace_alignment.py - functions to align traces within a stream and stack them

    fk_analysis.py - functions to carry out an FK analysis, get array geometry and/or get 
		     theoretical P wave arrivals
   
    V_models - functions to load velocity data and create the EFM/EFMD velocity model

    EFM - core functions of the EFM
      
    F_EFM - auxiliary functions for the EFM

    EFMD_Bayesian_modelling - core functions of the EFMD

    F_EFMD - auxiliary functions for the EFMD

      


[![DOI](https://zenodo.org/badge/330757440.svg)](https://zenodo.org/badge/latestdoi/330757440)


      
