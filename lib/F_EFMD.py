'''

First created on Jan 17, 2019
Author: Itahisa Gonzalez Alvarez

    This script contains secondary functions required to run parts 1 and 2
    of the EFMD Bayesian analysis.

'''

import copy
import string
import pickle
import numpy as np
import matplotlib as mpl
from numba import jit, njit
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FormatStrFormatter
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


#@jit #(nopython = True)
def syn_envelope_preprocessing (model, s_env, delta, tjs, cf, N, v, tJ, vJ,
                                inv_Qi_val, maxamp_ind):

    '''
    This function implements the first three steps in the EFMD envelope modelling
    technique described in Korn (1997). For a given frequency band and model
    consisting on values of correlation length and RMS velocity fluctuations for
    each layer, it calculates the corresponding scattering Q values
    and the direct wave energy decay over time.

    Arguments:
        -model: (np.array) correlation length and RMS velocity fluctuations for
                each layer in the model
        -s_env: (np.array) stacked envelope for this frequency band
        -delta: (float) inverse of the sampling rate for this array
        -tjs: (np.array) cumulative traveltimes through the layers of the model
        -cf: (float) central frequency
        -N: (int) total number of layers in the model
        -v: (np.array) mean P wave velocity in each layer of the model
        -tJ: (float) time it takes P waves to reach the free surface of the model
        -vJ: (float) velocity of the layer containing the free surface
        -inv_Qi_val: (float) inverse intrinsic quality factor for this fband
        -maxamp_ind: (int) Index of the maximum amplitude of the data normalised
                     coda envelope

    Output:
        -mod_inv_Qs: (np.array) Inverse scattering Q values for each layer in
                     the model
        -t: (np.array) Time vector
        -tJ_ind: (int) Index corresponding to the time sample representing the
                 direct wave arrival to the free surface
        -Ed: (np.array) Direct wave energy decay over time
        -EdJ: (float) Direct wave energy measured at the free surface

    '''

    ###########################################################################

    # SYNTHETIC ENVELOPE CALCULATION STEP 1:
    # Calculate Qs-1 for each layer in  model for each fband: we use Fang &
    # Muller's equation for an exponential ACF. THESE ARE INVERSE Qs, CAREFUL!
    c1 = 28.73; c2 = 16.77; c3 = 2.40

    factor = np.divide(np.multiply(model[:,0], 2*np.pi*cf), v)
    Qs_1_num =  np.multiply(np.power(model[:,1], 2),
                            np.multiply(c1, np.power(factor, 3)))
    Qs_1_den = 1 + np.multiply(c2, np.power(factor, 2)) \
               + np.multiply(c3, np.power(factor, 4))
    mod_inv_Qs = np.divide(Qs_1_num, Qs_1_den)

    ###########################################################################

    # SYNTHETIC ENVELOPE CALCULATION STEP 2:
    # Get the energy of the direct wave at the free surface (EdJ) from the
    # squared envelope measured at my stations (eq. 12 from Korn, (1997):

    # Define time:
    t = np.arange(0, len(s_env) * delta, delta)# All the squared normalised
                                                # envelopes should have the same
                                                # number of samples, so it does
                                                # not matter which one I use to
                                                # define time.
    Nt = len(t)

    # Get the indices for each tj (I'll need these later):.
    tjs_inds = []
    for tj in tjs:
        tjs_inds.append(np.abs(tj - t).argmin())
    tjs_inds = np.array(tjs_inds)# tjs_inds[3] should be equal to Nt/2
    # Get index for tJ (time to reach the free surface):
    tJ_ind = (np.abs(t - tJ)).argmin()

    # Square the envelopes for each fband:
    sq_env = np.power(s_env, 2)

    # Calculate the integral over the time window of the direct wave arrival:
    # Define x (time):
    x = t[:maxamp_ind]; dx = delta
    # Define y (coda data):
    y = sq_env[:maxamp_ind]
    # Integrate:
    Id = np.multiply(2, np.trapz(y, x, dx = dx))

    # Multiply the integral by vJ to get the spectral energy density at the
    # free surface: this is only the central sample in my Ed for each fband!
    EdJ = np.multiply(vJ, Id)

    ###########################################################################

    # SYNTHETIC ENVELOPE CALCULATION STEP 3:
    # Calculate the time decay of the energy of the direct wave for each layer
    # of our model. Each Ed will be a vector with the same length of t.

    Ed = np.zeros(Nt)
    wQ = np.zeros(Nt)

    # Add the EdJ as the central sample of the Ed vector:
    Ed[tJ_ind] = EdJ

    # Calculate the product of Qs-1 and omega, and Qi-1 and omega:
    wQ_s = 2 * np.pi * cf * np.array(mod_inv_Qs)

    # The wQs vectors should have a constant value of wQ within each layer:
    for i in range(Nt):
        for j in range(N):
            if j == 0 and i <= tjs_inds[j]:
                wQ[i] = wQ_s[j]
            if j != 0 and tjs_inds[j-1] < i <= tjs_inds[j]:
                wQ[i] = wQ_s[j]

    # Get the product of w and inverse Qi values:
    wQi = 2 * np.pi * cf * inv_Qi_val

    # Calculate the Eds for the first half of the time vector and all fbands:
    # Define a vector that goes backwards from J:
    hs = np.arange(tJ_ind, 0, -1)

    for h in hs:
        if h == hs[0] and Ed[h]  !=  EdJ:
            print('Something is wrong!')
        else:
            Ed[h-1] = Ed[h] / (np.exp(-1 * (wQi + wQ[h]) * (t[h]-t[h-1])))

    # Calculate Eds for the second half of the time vector and all fbands:
    hs = np.arange(tJ_ind + 1, Nt, 1)

    for h in hs:
        if h == hs[0] and Ed[h-1]  !=  EdJ:
            print('Something is wrong!')
        else:
            Ed[h] =  Ed[h-1] * (np.exp(-1 * (wQi + wQ[h]) * (t[h]-t[h-1])))

    return mod_inv_Qs, t, tJ_ind, Ed, EdJ

###############################################################################
####                    END OF THE FUNCTION                                ####
###############################################################################



def get_synthetic_envelope(model, s_env, delta, tjs, dtjs, cf, N, v, tJ, vJ,
                           inv_Qi_val, maxamp_ind):

    '''
    This function calculates the synthetic envelope according to the envelope
    modelling technique described in Korn (1997). It uses the
    syn_envelope_preprocessing function that carries out the first three steps
    in the calculation and returns the inverse scattering quality factor (Qs^-1)
    values for each layer in the model, inverse intrinsic quality factor (Qi^-1)
    for the corresponding frequency band, time vector, index of the direct wave
    arrival to the free surface, direct wave energy decay over time, direct wave
    energy at the free surface value and end of the direct wave time window, all
    of them necessary for the final steps in the process. The main reason to
    split the calculation into two functions was to speed up the algorithm.
    Finally, this function solves the system of differential equations defined
    in EFMD_coda_energy_ODEsys and computes the synthetic envelope from the
    results.
    NOTE: The synthetic envelope does NOT reproduce the direct wave arrival,
          only the coda decay from tJ seconds after the direct wave arrival.

    Arguments:
        -model: (np.array) correlation length and RMS velocity fluctuations for
               each layer in the model
        -s_env: (np.array) stacked envelope for this frequency band
        -delta: (float) inverse of the sampling rate for this array
        -tjs: (np.array) cumulative traveltimes through the layers of the model
        -dtjs: (np.array) travel times for each layer in the model
        -cf: (float) central frequency
        -N: (int) total number of layers in the model
        -v: (np.array) mean P wave velocity in each layer of the model
        -tJ: (float) time it takes P waves to reach the free surface of the model
        -vJ: (float) velocity of the layer containing the free surface
        -inv_Qi_val: (float) inverse intrinsic quality factor for this fband
        -maxamp_ind: (int) index of the maximum amplitude of the data normalised
                     coda envelope

    Output:
        -syn_env_results: (dict) Dictionary containing the synthetic envelope
                          ('syn_env'), time vector ('t'), index at which tJ
                          (P wave reaches the free surface of the model) happens
                          ('tJ_ind') and total coda energy within each layer
                          ('Ecjs').

    '''

    # Import results from
    mod_inv_Qs, t, tJ_ind, Ed, EdJ = syn_envelope_preprocessing (model, s_env,
                                                                 delta, tjs, cf,
                                                                 N, v, tJ, vJ,
                                                                 inv_Qi_val,
                                                                 maxamp_ind)
    Nt = len(t)

    # Create t copy so I can get the right Ed:
    t_copy = copy.deepcopy(t)

    # SYNTHETIC ENVELOPE CALCULATION STEP 4:
    # Solve system of ODEs to get spectral energy density for each layer

    # Establish initial values: this is the energy at time 0 for each layer. At
    # t = 0, only the bottom layer has any energy, and this is the first value
    # in the corresponding Ed.
    Ecj0 = np.zeros(N)
    Ecj0[0] = Ed[0]
    Ec0 = Ed[0]

    # For each model (combination of Qss), and each fc, calculate Eds for each
    # layer SEPARATELY. Don't forget that Qs are INVERSE values.

    # Solve system of differential equations: this returns a matrix with N
    # columns and Nt rows. Columns correspond to Ec for layers 1 to N for this
    # specific model.

    # Use odeint to solve system of differential equations:
    Ecj = odeint(EFMD_coda_energy_ODEsys, Ecj0, t,
                 args = (mod_inv_Qs, inv_Qi_val, Ed, Ec0, dtjs, tjs, cf, t_copy))

    # Let's rearrange this results so it is easier to read in the future:
    Ecjs = []
    for n in range(N):
        Ecjs.append(Ecj[:,n])

        # Sanity check:
        if len(Ecjs[n])  !=  Nt:
            print('ERROR: Ecj and t not the same length')

    ###########################################################################

    # CALCULATE SYNTHETIC ENVELOPE:
    # Generate the synthetic envelope that I will compare with the normalised
    # coda envelopes obtained from data(see third term in eq. 12 from Hock et
    # al. (2000)).
    EcJ = Ecjs[int(N/2)]

    syn_env = np.sqrt(np.divide(np.multiply(2, EcJ), np.multiply(tJ, EdJ)))

    # The synthetic envelope needs to be shifted in time, as this method assumes
    # the energy is measured at the bottom of the layer, instead of at the top.
    # The time shift is, therefore, the travel time through the layer at the
    # top of the model (the one containing the free surface).

    # Number of samples to shift:
    sam_shift = int(np.round((1 / delta) * (dtjs[int(N/2)] / 2)))

    # Since we want to shift the synthetic envelope forward in time, I will
    # add as many zeros at the beginning as samplas are in sam_shift and
    # remove the same number of samples from the end of the synthetic envelope.
    shift_zeros = np.zeros(sam_shift)
    new_syn_env = np.append(shift_zeros, syn_env[:-sam_shift])
    syn_env = new_syn_env.copy()

    syn_env_results = {'syn_env': syn_env,
                       'time': t,
                       'tJ_ind': tJ_ind,
                       'Ecjs': Ecjs}

    return syn_env_results

###############################################################################
####                    END OF THE FUNCTION                                ####
###############################################################################



@jit(nopython = True)
def EFMD_coda_energy_ODEsys(Ec, t, Qsmod_inp, inv_Qi_val, Ed_inp, Ec0, dtjs_inp,
                            tjs_inp, cf, t_copy):
    '''
    Function to solve the system of coupled differential equations from the EFMD
    (Korn (1997)).

    Arguments:
        -Ec: (np.array) initial guess for coda energy for each layer. These are
             the functions we are solving for
        -t: (np.array) time vector
        -inv_Qsmod: (list) values of Qs^-1 for each layer in the model calculated
                    from parameter values
        -inv_Qi_val: (float) value of Qi^-1 obtained from the EFM for this fband
        -Ed: (np.array) direct wave energy for all time data points
        -Ec0: (float) Coda energy at time 0
        -dtjs: (np.array) travel times through each layer in the model
        -tjs: (np.array) cumulative traveltimes from the bottom of the model to
              the top of each layer
        -cf: (float) central frequency
        -t_copy: (np.array) copy of the time vector

    Returns:
        -Derivative of Ec with respect to time, to be integrated by odeint
    '''

    # Get number of layers:
    N = len(tjs_inp)

    # Declare some boundary conditions: it doesn't matter which dt9 value we
    # use here, since Ec9 = 0
    Ec9 = 0; dt9 = 1/4; t9 = tjs_inp[-1] + dt9
    Ec0 = 0; dt0 = 1/4; t0 = 0;

    # Add Ec0 and Ec9 to functions we are solving for:
    Ec = np.concatenate((np.array([Ec0]), Ec, np.array([Ec9])));

    # Add dt0 and dt9 to travel times through each layer:
    dts = np.concatenate((np.array([dt0]), dtjs_inp, np.array([dt9])))

    # Define cumulative travel times:
    ts = np.concatenate((np.array([t0]), tjs_inp, np.array([t9])));

    # Get the correct Ed sample I need:
    Ed = np.interp(t, t_copy, Ed_inp);

    dEcdt = []
    for j in range(1, N+1):
        dEcdt.append(
                (((1/(4 * dts[j-1])) * Ec[j-1] * step_fun(t, ts[j-1]))
                + ((1/(4 * dts[j+1])) * Ec[j+1] * step_fun(t, ts[j]))
                - ((1/(4 * dts[j])) * Ec[j] * step_fun(t, ts[j]))
                - ((1/(4 * dts[j])) * Ec[j] * step_fun(t, ts[j-1]))
                - ((2 * np.pi * cf * inv_Qi_val) * Ec[j] * step_fun(t, ts[j-1]))
                + ((2 * np.pi * cf * Qsmod_inp[j-1]) * Ed * step_fun(t, ts[j-1]) \
                   * inv_step_fun(t, ts[j]))))

    return dEcdt

###############################################################################
####                    END OF THE FUNCTION                                ####
###############################################################################



@njit
def EFMD_model_update(model, num_layers, a_step_size, E_step_size, min_a, max_a,
                      min_E, max_E) :

    '''
    This function takes the current model of the EFMD and randomly updates one
    of the parameters.

    Arguments:
        - model: (np.array) Last accepted model in the EFMD.
        - num_layers: (int) Number of layers of the model.
        - a_step_size: (float) Step size for the correlation length.
        - E_step_size: (float) Step size for the RMS velocity fluctuations.
        - min_a: (float) Minimum accepted value for the correlation length
        - min_E: (float) Minimum accepted value for the velocity fluctuations
        - thicks: (np.array) Thickness of the model layers
        - max_E: (float) Maximum accepted value for the velocity fluctuations
        - lambda_max: (np.array) Maximum wavelength for each layer in the model

    Returns:
        - new_model: (np.array) Updated model
        - j: (int) Index pointing to which parameter was updated
             (0 = correlation length, 1 = velocity fluctuations)

    '''

    # Define the total number of layers in model:
    N = num_layers * 2 - 1

    # Let's randomly select one of the layers and one of the two parameters to
    # update.
    # I will update the current parameter values by either adding or subtracting
    # the step size. The decision to add or subtract is made randomly.

    # Randomly choose a layer. The layer containing thefree surface should always
    # have the same structural parameters as the layer immediately below.
    i = np.random.randint(0, num_layers, 1)[0]

    # Randomly choose one of the two parameters (a=0, E=1)
    j = np.random.randint(0, 2, 1)[0]

    updated_model = model.copy()

    # Define min, max and step size for each parameter:
    min_a_vals = min_a[:num_layers]; max_a_vals = max_a[:num_layers]
    if j == 0:
        step_size = a_step_size
        param_min = min_a_vals[i]
        param_max = max_a_vals[i]
    elif j == 1:
        step_size = E_step_size
        param_min = min_E
        param_max = max_E


    # Update chosen parameter:
    updated_model[i][j] = model[i][j] \
                         + np.random.uniform(-step_size, step_size, 1)[0]

    # If the parameters in the updated_model are below or above the limits set
    # by min_a/E, max_E and thick, the parameters should bounce back within the
    # limits.

    # Check whether there are any parameters out of bounds: I will assume the
    # parameters that have not just been updated are ok.
    num_anom_params = get_number_of_anom_params(i, j, updated_model, param_min,
                                                param_max)

    if num_anom_params != 0:
        while num_anom_params != 0:
            # Make parameters bounce back:
            new_model = EFMD_model_update_bounce_back (i, j, updated_model,
                                                       param_min, param_max)
            # Check for anomalous values of the parameters again:
            num_anom_params = get_number_of_anom_params(i, j, new_model,
                                                        param_min, param_max)
    else:
        new_model = updated_model.copy()

    # The model should be symmetric with respect to the free surface.
    if i != (num_layers - 1):
        for k in range(num_layers):
            new_model[N - 1 - k][0] = new_model[k][0]
            new_model[N - 1 - k][1] = new_model[k][1]

    # Sanity checks!
    for r in range(num_layers):
        if (new_model[-(r+1)][0] != new_model[r][0]) or \
            new_model[-(r+1)][0] != new_model[r][0]:
            print('Symmetry broken!')
            print(new_model)

    anom_params2 = 0
    for h in range(len(new_model)):
        if new_model[h][0] < min_a[h] or new_model[h][0] > max_a[h]:
            anom_params2 += 1
        if new_model[h][1] < min_E or new_model[h][1] > max_E:
            anom_params2 += 1
    if anom_params2 != 0:
        print('THE MODEL UPDATE FAILED!')
        print(i, j, new_model)

    return new_model, j

###############################################################################
####                    END OF THE FUNCTION                                ####
###############################################################################



@njit
def EFMD_model_update_bounce_back (i, j, model, param_min, param_max):

    '''
    Function that makes EFMD model parameters bounce back in case they take
    values outside the boundaries in the model update process.

    Arguments:
        i: (int) Index pointing to number of the layer whose parameter is
           being updated
        j: (int) Integer pointing to parameter number that is being updated
        model: (np.array) Model containing structural parameters for all layers
        param_min: (float) Minimum accepted value of the parameter being updated
        param_max: (float) Maximum accepted value of the parameter being updated
        wlength: (float) 5*maximum wavelength considered in the modelling

    Returns:
        updated_model: (np.array) Values of the parameters in the model, now
                        within the accepted boundaries

    '''

    new_model = model.copy()

    if new_model[i][j] < param_min:
        delta = param_min - new_model[i][j]
        new_model[i][j] = param_min + delta

    if new_model[i][j] > param_max:
        delta = new_model[i][j] - param_max
        new_model[i][j] = param_max - delta

    return new_model

###############################################################################
####                    END OF THE FUNCTION                                ####
###############################################################################



@njit
def get_number_of_anom_params (i, j, model, param_min, param_max):

    '''
    This function takes any model used in the EFMD and checks whether there are
    any anomalous parameters (params with values outside accepted boundaries)
    and, in case there are any, it returns their number.
    It will only check the parameters that have just been updated, assuming
    the rest are correct.

    Arguments:
        model: (np.array) Values of the parameters in each layer of the model.
        param_min: (float) Minimum accepted value for the correlation length.
        param_max: (float) Maximum accepted value for the correlation length is
                    the thickness of the model layer whose parameter is being
                    updated.
        wlength: (float) Maximum wavelength considered in the modelling. If
                 smaller than the current layer thickness, this will be the
                 maximum accepted value for the correlation length.

    Returns:
        num_anom_params: (int) Number of parameters in the model with values
                         outside allowed boundaries.

    '''

    num_anom_params = 0

    if model[i][j] < param_min or model[i][j] > param_max:
        num_anom_params += 1

    return num_anom_params

###############################################################################
####                    END OF THE FUNCTION                                ####
###############################################################################



@njit
def get_loglik (data, syn_data, inv_Qd):

    '''
    This functions calculates the loglikelihood for a given model by comparing
    the normalised coda envelope and the synthetic normalised envelope, both
    obtained as described in Korn (1997).

    Arguments:
        -data: (np.array) normalised coda envelope for this frequency band
        -syn_data: (np.array) synthetic normalised coda envelope for this
                   frequency band and model
        -inv_Qd: (np.array) inverse of the variance-covariance matrix from
                 the data

    Returns:
        -loglik: (float) loglikelihood for this model

    '''

    # Obtain loglikelihood:
    loglik = (np.dot(np.dot((data - syn_data).transpose(), inv_Qd),
                      (data - syn_data))) / 2

    return loglik

###############################################################################
####                    END OF THE FUNCTION                                ####
###############################################################################



@njit
def update_hists_2D_syn_envs(hists_2D_syn_envs, syn_env, hist_range):

    '''
    Function that updates the hist_2D_syn_envs matrix by checking which bin
    each sample in syn_env belongs to and adding 1 to the corresponding matrix
    element.

    Arguments:
        - hists_2D_syn_envs: (np.array) Its dimensions are
                             len(syn_env) x len(hist_range)
        - syn_env: (np.array)
            Synthetic envelope being added to the 2D histogram
        - hist_range: (np.array) Bins or intervals to which data samples in
                      syn_env can belong to.

    Returns:
        - new_hists_2D_syn_envs: (np.array) Updated hists_2D_syn_envs matrix

    '''

    nrows = len(hist_range)
    nsamps = len(syn_env)
    new_hists_2D_syn_envs = hists_2D_syn_envs.copy()

    for w in range(nsamps):
        for i in range(nrows - 1):
            if syn_env[w] >= hist_range[i] and syn_env[w] < hist_range[i+1]:
                new_hists_2D_syn_envs[i][w] += 1

    return new_hists_2D_syn_envs

###############################################################################
####                    END OF THE FUNCTION                                ####
###############################################################################



@njit
def step_fun(x, a):

    '''
    Heaviside or step function that takes an array x and compares each element
    with a. It returns 0 if x is lower than a, 0.5 if x=a and 1 if it is larger
    than a.

    '''

    if x == a:  return 0.5
    elif x < a: return 0
    elif x > a: return 1

###############################################################################
####                    END OF THE FUNCTION                                ####
###############################################################################



@njit
def inv_step_fun(x, a):

    '''
    Heaviside or step function that takes an array x and compares each element
    with a. It returns 1 if x is lower than a, 0.5 if x=a and 0 if it is larger
    than a.

    '''

    if x == a:  return 0.5
    elif x < a: return 1
    elif x > a: return 0

###############################################################################
####                    END OF THE FUNCTION                                ####
###############################################################################



#@jit
def EFMD_modelling_summary(k, N_iters, num_layers, units, kept_models,
                           kept_logliks, kept_syn_envs, times, rejections,
                           good_accepts, rand_accepts, a_accepts, a_updates,
                           E_accepts, E_updates):

    '''
    This function summarises the EFMD results up to some percentage of the total
    number of iterations in the Monte Carlo Markov Chain (MCMC).

    Arguments:
        -k: (int) current number of iteration
        -N_iters: (int) total number of iterations to be done in the MCMC
        -num_layers: (int) number of layers in the model
        -units: (str) units of distance used in the modelling (either 'm' or 'km')
        -kept_models: (list) accepted models from the MCMC
        -kept_logliks: (list) loglikelihoods of the accepted models from the MCMC
        -kept_syn_envs: (list) synthetic envelopes for all frequency bands for
                        each accepted model from the MCMC
        -times: (list) times per iteration in the MCMC
        -rejections: (int) total number of rejected models in the MCMC
        -good_accepts: (int) total number of models accepted with lower
                       loglikelihood than the previous one
        -rand_accepts: (int) total number of models randomly accepted
        -a_accepts: (int) total number of accepted a updates
        -a_updates: (int) total number of times a was updated
        -E_accepts: (int) total number of accepted E updates
        -E_updates: (int) total number of times E was updated

    Returns:
        -It prints a summary of current results to screen
    '''

    print(' ')
    print('Summarising results so far... ')
    print(' ')

    # Print summary of the results so far:
    print(str(int(100*k/N_iters)) + '% of iterations done!')
    print('Average time per iteration was = ' + str(np.mean(times)))
    print('Number of kept models is ' + str(len(kept_models)))
    print('Number of good models accepted is ' + str(good_accepts))
    print('Number of random models accepted is ' + str(rand_accepts))
    print('Minimum loglikelihood found so far is ' + str(min(kept_logliks)))

    # Print ACCEPTANCE rate:
    accepts_perc = np.round(((N_iters - rejections) / N_iters) * 100, 4)
    if accepts_perc > 70:
        print('Acceptance rate is ' + str(accepts_perc) + '%, too high!')
    elif accepts_perc < 30 :
        print('Acceptance rate is ' + str(accepts_perc) + '%, too low!')
    else: print('Acceptance rate is ' + str(accepts_perc) + '%')

    a_accepts_perc = np.round((a_accepts / a_updates) * 100, 3)
    E_accepts_perc = np.round((E_accepts / E_updates) * 100, 3)
    print('Correlation length acceptance rate is ' + str(a_accepts_perc) + '%')
    print('Velocity fluctuations acceptance rate is ' + str(E_accepts_perc) + '%')

    # Save structural parameters for each layer from kept models (we are only
    # interested in the first half of our model). I also create E_perc to save
    # E values in percentage, so they are easier to interpret in histograms and
    # scatter plots.
    a = {}; E_perc = {}; layer_labels = []

    for i in range(num_layers):
        layer_labels.append('L' + str(i+1))
    for mod in kept_models:
        for key in layer_labels:
            a[key] = []
            E_perc[key] = []
    for mod in kept_models:
        for i, key in enumerate(layer_labels):
            a[key].append(mod[num_layers - 1 + i][0])
            E_perc[key].append(mod[num_layers - 1 + i][1]*100)

    print(' ')

    # Print mean values of the parameters for each layer:
    for key in a:
        print('Mean correlation length for ' + key + ' is ' \
              + str(np.round(np.mean(np.array(a[key])), 4)) + units \
              + ' and mean velocity fluctuations are ' \
              + str(np.round(np.mean(np.array(E_perc[key])), 4)) + '%')

    print(' ')
    print('        --------------------------------                 ')
    print(' ')

###############################################################################
####                    END OF THE FUNCTION                                ####
###############################################################################




def EFMD_preprocess_data(array_nm, fbands, s_nces_fname, s_envs_fname,
                         vel_model, delta, Q_i, resample = True,
                         syn_test = False):

    '''
    This function loads and preprocess the data necessary for the EFMD Bayesian
    inversion. Preprocessing includes extracting required data from files, get
    velocity model information, get indices pointing to start/end of the time
    windows of interest, etc.

    Arguments:
        - array_nm: (str) Name of the seismic array.
        - fbands: (dict) Frequencies to be used in the analysis. Keys are names
                  assigned to each fband and each element is a list with freqmin
                  and freqmax for each frequency band (in Hz) of interest.
                  (Example: fband = {'A': [0.5,1], 'B': [1,2]}).
        - s_nces_fname: (str) Path and file name of dataset consisting on
                        normalised coda envelopes and their standard deviation.
                        For synthetic tests, the input model is also included.
        - s_envs_fname: (str) Path and file name of dataset consisting on
                        NON-normalised envelopes and their standard deviation.
        - vel_model: (dict) Characteristics of the velocity model to be used in
                     the EFMD analysis. For each layer, it contains thickness
                     and mean P wave velocity. Traveltimes through each layer,
                     cumulative traveltimes from the bottom of the model to the
                     top of each layer, number of layers in the model, velocities
                     for the TFWM and EFM are also included. Dictionary created
                     by the get_velocity_model function from vel_models.
        - delta: (float) Inverse of the sampling rate for the data.
        - Q_i: (dict) Dictionary with intrinsic quality factor Q_i values for
                all fbands. Keys should be the same from fbands dictionary.
        - resample: (bool) Indication of whether to resample the data to a new
                    sampling rate or not (default is True).
        - syn_test: (bool) Indication of whether to show or not the plots at
                    the end of the analysis (default is False).

    Returns:
        - num_layers: (int) Number of layers in the model (without mirroring
                      on the other side of the free surface).
        - N: (int) Absolute number of layers in the model.
        - L: (np.array) Thickness of each layer in the model (without mirroring
             on the other side of the free surface).
        - tjs: (np.array) Cumulative traveltime through each layer of the model.
        - dtjs: (np.array) Traveltimes through each layer of the model.
        - v: (np.array) Velocity of each layer of the model.
        - tJ: (float) Traveltime through the stack of layers, from the model
              bottom to the free surface.
        - vJ: (float) Velocity of the layer containing the free surface.
        - thicks: (np.array) Thickness of all layers in the model.
        - lambda_min: (np.array) Minimum wavelength in each layer of the model.
        - lambda_max: (np.array) Maximum wavelength in each layer of the model.
        - input_model: (np.array) Value of the structural parameters for each
                       layer in the model.
        - s_nces: (dict) Normalised coda envelopes for each frequency band.
        - data_envs: (dict) Normalised coda envelopes trimmed to the coda time
                     window.
        - s_envs: (dict) Non-normalised envelopes for each frequency band.
        - maxamp_inds: (dict) Index of the maximum amplitude of the squared
                       non-normalised envelope for each fband.
        - DW_end_inds: (dict) Index of the end of the direct wave time window
                        for each frequency band.
        - cfs: (dict) Central frequency for each frequency band.
        - inv_Qi_vals: (dict) Inverse of the intrinsic quality factor for each
                        frequency band.
        - inv_Qds: (dict) Inverse of the variance-covariance matrix of the data
                    for each fband.
        - delta: (float) Inverse of the sampling rate of the data.
        - resampled_s_nces_dataset: (dict) Resampled dataset consisting on
                                    normalised coda envelopes and their standard
                                    deviation.
        - resampled_s_envs_dataset: (dict) Resampled dataset consisting on
                                    NON-normalised envelopes and their standard
                                    deviation.

    '''

    #                ****   LOAD DATA   ****                                  #

    # Get stacked, corrected and normalised coda envelopes for all events and
    # fbands: these are my DATA.
    fopen = open(s_nces_fname,'rb')
    s_nces_dataset = pickle.load(fopen)
    fopen.close()

    # Load non-normalized stacked envelopes for all arrays, events and fband.
    # These are the ones I will use to do the modelling and compute the synthetic
    # envelopes. This is a dictionary with a single envelope for each array and
    # fband.
    fopen = open(s_envs_fname,'rb')
    s_envs_dataset = pickle.load(fopen)
    fopen.close()

    ###########################################################################

    if resample == True:

        # Resampling our data makes the EFMD  significantly faster!

        # Define new sampling rate:
        old_delta = delta
        new_delta = 1/15

        resampled_s_nces_dataset = data_resampler (array_nm, s_nces_dataset,
                                                   fbands, old_delta, new_delta)

        s_nces = {}; s_nces_std = {}
        for fband in fbands:
            s_nces[fband] = resampled_s_nces_dataset[array_nm][fband]['s_nces']
            s_nces_std[fband] = resampled_s_nces_dataset[array_nm][fband]['s_nces_std']

        resampled_s_envs_dataset = data_resampler(array_nm, s_envs_dataset,
                                                  fbands, old_delta, new_delta)

        s_envs = {}; s_envs_std = {}
        for fband in fbands:
            s_envs[fband] = resampled_s_envs_dataset[array_nm][fband]['s_envs']
            s_envs_std[fband] = resampled_s_envs_dataset[array_nm][fband]['s_envs_std']

        # Redefine delta and datasets:
        delta = new_delta
        s_nces_dataset = copy.deepcopy(resampled_s_nces_dataset)
        s_envs_dataset = copy.deepcopy(resampled_s_envs_dataset)

    else:

        s_nces = {}; s_nces_std = {}
        for fband in fbands:
            s_nces[fband] = s_nces_dataset[array_nm][fband]['s_nces']
            s_nces_std[fband] = s_nces_dataset[array_nm][fband]['s_nces_std']

        s_envs = {}; s_envs_std = {}
        for fband in fbands:
            s_envs[fband] = s_envs_dataset[array_nm][fband]['s_envs']
            s_envs_std[fband] = s_envs_dataset[array_nm][fband]['s_envs_std']

    ###########################################################################

    # Extract velocity model characteristics from vel_model:
    num_layers = vel_model['num_layers']
    v = vel_model['v']
    L = np.array(vel_model['L'])
    dtjs = vel_model['dtjs']
    tjs = vel_model['tjs']
    tJ = vel_model['tJ']
    thicks = vel_model['thicks']

    # Half the central value in thicks! It's twice as much as the actual
    # thickness of the layer. We need to do this here because we only use the
    # thicknesses to define the upper limit for correlation lengths.
    thicks[num_layers - 1] = thicks[num_layers - 1]/2

    # Calculate the minimum and maximum wavelengths within each layer: they will
    # help define the threshold for the correlation lengths.
    fmin = 0.5; fmax = 7;# Hz
    lambda_min = v / fmax # Same units as v
    lambda_max = v / fmin # Same units as v

    # Define velocity of the J-th layer (the one containing the free surface):
    vJ = v[num_layers - 1] #in m/s !!!

    # Set total number of layers in the model:
    N = num_layers*2 - 1

    if syn_test == True:
        input_model = s_nces_dataset[array_nm]['input_model']

    #                 ****   PREPROCESS DATA   ****                           #

    # Compute variance-covariance matrix for all frequency bands. There's no
    # need to calculate them every time (millions of times!) in the for loop,
    # they will always be the same!

    # I will need the index of the maximum amplitude and end of the direct wave
    # time window for each fband.
    maxamp_inds = {}; DW_end_inds = {}

    # I will also need the data normalised coda envelope and variance-covariance
    # matrix for each frequency band:
    data_envs = {}; inv_Qds = {}

    # I will also need the inverse of the intrinsic quality factor Q_i for each
    # frequency band:
    inv_Qi_vals = {}

    # Finally, define the central frequency for each frequency band:
    cfs = {}

    for fband in fbands:

        # Load stacked but not-normalised envelopes:
        s_env = s_envs[fband]

        # Square the envelope for each fband:
        sq_env = np.power(s_env, 2)

        # Get the index of the maximum amplitude, the direct wave goes from 5
        # seconds before maximum amplitude to tJ seconds after maximum amplitude.
        maxamp_ind = (np.abs(sq_env - max(sq_env))).argmin()#ok
        DW_end_ind = np.int(np.round(maxamp_ind + (1 / delta) * tJ))
        maxamp_inds[fband] = maxamp_ind
        DW_end_inds[fband] = DW_end_ind

        # Get data, syn_data and data_vars to be used for the calculation of the
        # likelihood (we only use the coda for this, since the synthetic
        # envelope does NOT reproduce the direct wave arrival):
        data = s_nces[fband][DW_end_ind:-150]
        data_vars = s_nces_std[fband][DW_end_ind:-150]**2
        data_envs[fband] = data

        # Create the variance-covariance matrix: it should be a diagonal matrix
        # where the diag. elements are the variances for each data sample.
        Qd = np.zeros((len(data), len(data)))
        Qd[np.arange(len(data)), np.arange(len(data))] = data_vars
        inv_Qd = np.linalg.inv(Qd)
        inv_Qds[fband] = inv_Qd

        # Calculate inverse intrinsic quality factors:
        inv_Qi_vals[fband] = 1 / Q_i[fband]

        # Calculate central frequency:
        cfs[fband] = (fbands[fband][0] + fbands[fband][1]) / 2


    if syn_test == True:
        return num_layers, N, L, tjs, dtjs, v, tJ, vJ, thicks, lambda_min, \
               lambda_max, input_model, s_nces, data_envs, s_envs, maxamp_inds, \
               DW_end_inds, cfs, inv_Qi_vals, inv_Qds, delta, s_nces_dataset, \
               s_envs_dataset
    else:
        return num_layers, N, L, tjs, dtjs, v, tJ, vJ, thicks, lambda_min, \
               lambda_max, s_nces, data_envs, s_envs, maxamp_inds, DW_end_inds, \
               cfs, inv_Qi_vals, inv_Qds, delta, s_nces_dataset, s_envs_dataset

###############################################################################
####                    END OF THE FUNCTION                                ####
###############################################################################




def data_resampler (array_nm, dataset, fbands, old_delta, new_delta):

    '''
    Function to resample (downsample) the initial data envelopes to speed up
    the forward calculation of the synthetic envelopes.

    Arguments:
        - array_nm: (str) Name of the seismic array.
        - dataset: (dict) Dictionary with a structure like
                   array_nm / fband / s_nces-s_nces_std, in which the normalised
                   coda envelopes and their standard deviations are numpy arrays.
        - fbands: (dict) Frequencies to be used in the analysis. Keys are names
                  assigned to each fband and each element is a list with freqmin
                  and freqmax for each frequency band (in Hz) of interest.
                  (Example: fband = {'A': [0.5,1], 'B': [1,2]}).
        - delta: (float) Inverse of the sampling rate of our data.
        - resampling_factor: (int or float) Factor by which to downsample the
                             original data.

    Returns:
        - resampled_dataset: (dict) Dictionary with the same structure of dataset
                             but containing the new resampled data and their
                             standard deviations.

    '''

    # Define time vector: it doesn't matter which frequency band I use here,
    # they should all have the same number of data samples.
    resampled_dataset = copy.deepcopy(dataset)
    resampling_factor = new_delta / old_delta

    for fband in fbands:
        for key in dataset[array_nm][fband]:

            t = np.arange(0, len(dataset[array_nm][fband][key])) * old_delta
            new_t = np.arange(0, len(dataset[array_nm][fband][key]) / \
                              resampling_factor) * new_delta

            data = dataset[array_nm][fband][key]

            new_data = np.interp(new_t, t, data)
            resampled_dataset[array_nm][fband][key] = new_data

    return resampled_dataset

###############################################################################
####                    END OF THE FUNCTION                                ####
###############################################################################




def EFMD_get_RM_MLM (array_nm, fbands, data_envs, s_envs, delta, vel_model,
                     kept_models, kept_logliks, a, E, maxamp_inds, DW_end_inds,
                     cfs, inv_Qi_vals, inv_Qds, parameter, comb_results = False):

    '''
    This function takes the results from the EFMD Bayesian inversion and obtains
    the Minimum Loglikelihood Model (MLM) and the Representative Model (RM). The
    MLM corresponds to the tested model with highest likelihood (or lowest
    loglikelihood), while the RM is an representative model calculated from the
    ensemble of accepted models.

    Arguments:
        - array_nm: (str) Name of the seismic array.
        - fbands: (dict) Frequencies to be used in the analysis. Keys are names
                  assigned to each fband and each element is a list with freqmin
                  and freqmax for each frequency band (in Hz) of interest.
                  (Example: fband = {'A': [0.5,1], 'B': [1,2]}).
        - data_envs: (dict) Data envelopes restricted to the coda time window.
                     Keys should be the same from fbands dictionary.
        - s_envs: (dict) Stacked non-normalised envelopes for all frequency
                  bands, as well as their respective standard deviations. First
                  level key is the array name, second level keys are fbands.
        - delta: (float) Inverse of the sampling rate of our data.
        - vel_model: (dict) Characteristics of the velocity model to be used in
                     the EFMD analysis. For each layer, it contains thickness
                     and mean P wave velocity. Traveltimes through each layer,
                     cumulative traveltimes from the bottom of the model to the
                     top of each layer, number of layers in the model, velocities
                     for the TFWM and EFM are also included. Dictionary created
                     by the get_velocity_model function from vel_models.
        - kept_models: (list) Ensemble of accepted models obtained from the MCMC
                       from the Bayesian EFMD inversion.
        - kept_logliks: (list) Loglikelihoods of all accepted models from the
                        MCMC from the Bayesian EFMD inversion.
        - a: (dict) Values of the correlation length for each layer in the
             physical model, obtained from the entire ensemble of accepted
             models. Keys are 'LX', X being the layer number.
        - E: (dict) Values of the RMS velocity fluctuations for each layer in
             the physical model, obtained from the entire ensemble of accepted
             models. Keys are 'LX', X being the layer number.
        - maxamp_inds: (dict) Index pointing at the sample corresponding to the
                        maximum amplitude in the envelopes contained in s_envs.
                        Keys are the same from fbands dictionary.
        - DW_end_inds: (dict) Index pointing at the sample corresponding to the
                        end of the direct wave time window. Keys are the same
                        from fbands dictionary.
        - cfs: (dict) Central frequency of each frequency band. Keys are the
                same from fbands dictionary.
        - inv_Qi_vals: (dict) Dictionary with inverses of the intrinsic quality
                              factor Q_i values for all fbands. Keys should be
                              the same from fbands dictionary.
        - inv_Qds: (dict) Inverse of the variance-covariance matrix for each
                   frequency band. Keys should be the same from fbands dictionary.
        - parameter: (str) Statistical parameter to use to calculate the model
                     which is representative of the ensemble obtained from the
                     EFMD. (Either Mean, Median or Mode)
        - comb_results: (bool) Indication of whether EFMD_results come from a
                        single MCMC or from a combination of multiple chains.
                        Default is False.

    Returns:
        - RM: (np.array) Values of the structural parameters for each layer of
              the computational model for the Representative Model (representative
              of the ensemble obtained from the EFMD).
        - RM_loglik: (float) Loglikelihood corresponding to the RM.
        - RM_syn_envs: (dict) Synthetic envelopes for the RM and all fbands.
        - RM_percs: (np.array) 5-95 percentiles for the parameters in each layer.
        - MLM: (np.array) Values of the structural parameters for each layer of
                the computational model for the Maximum Loglikelihood Model.
        - MLM_loglik: (float) Loglikelihood of the MLM.
        - MLM_syn_envs: (dict) Synthetic envelopes corresponding to the MLM.

    '''

    # Extract velocity model characteristics from vel_model:
    num_layers = vel_model['num_layers']
    v = vel_model['v']
    dtjs = vel_model['dtjs']
    tjs = vel_model['tjs']
    tJ = vel_model['tJ']
    vJ = v[num_layers - 1] #in m/s !!!

    # Define number of accepted models:
    num_models = len(kept_models)

    # Define total number of layers in our models:
    N = num_layers*2 - 1

    # Find Representative Model (RM) (Mean/Mode/Median of a, E_perc) and minimum
    # loglikelihood model (MLM) (model with minimum loglikelihood).
    min_loglik = min(kept_logliks)
    for i in range(num_models):
        if kept_logliks[i] == min_loglik:
            MLM = kept_models[i]

    rep_E = {}; rep_a = {}; rep_a_percs = {}; rep_E_percs = {}

    if parameter == 'Mean':
        for key in E:
            rep_a[key] = np.round(np.mean(a[key]), 0)
            rep_E[key] = np.round(np.mean(E[key]), 4)

    elif parameter == 'Mode':
        for key in E:

            rep_a[key] = calculate_mode(a[key])
            rep_E[key] = calculate_mode(E[key])

    elif parameter == 'Median':
        for key in E:
            rep_a[key] = np.round(np.median(a[key]), 0)
            rep_E[key] = np.round(np.median(E[key]), 4)

    for key in E:
        rep_a_percs[key] = np.percentile(a[key], [5, 95])
        rep_E_percs[key] = np.percentile(E[key], [5, 95])

    # Representative Model: be careful with the order of the structural
    # parameters!!! L3 should go first and L1 should go last! (for the first
    # half of the model).
    RM = []; RM_percs = []
    for i in np.arange(num_layers-1, -1, -1):
        key = 'L' + str(i+1)
        RM.append([rep_a[key], rep_E[key]])
        RM_percs.append([rep_a_percs[key], rep_E_percs[key]])
    for i in np.arange(num_layers-2, -1, -1):
        RM.append(RM[:-1][i])
        RM_percs.append(RM_percs[:-1][i])
    RM = np.array(RM); RM_percs = np.array(RM_percs)
    print('Representative Model (RM) obtained using the ' + parameter \
          + ' of a and E for each layer')

    # Iterate over all frequency bands:
    RM_logliks = []; RM_syn_envs = {}; MLM_logliks = []; MLM_syn_envs = {}
    for fband in fbands:

        # Get indices for the maximum amplitude of the data normalised coda
        # envelope for this frequency band and the end of the direct wave time
        # window:
        maxamp_ind = maxamp_inds[fband]
        DW_end_ind = DW_end_inds[fband]
        data = data_envs[fband]

        # Get central frequency and intrinsic quality factor for this frequency
        # band:
        cf = cfs[fband]
        inv_Qi_val = inv_Qi_vals[fband]

        # Get stacked not-normalised coda envelope for this frequency band:
        s_env = s_envs[fband]

        # Create the variance-covariance matrix: it should be a diagonal matrix
        # where the diag. elements are the variances for each data sample.
        inv_Qd = inv_Qds[fband]

        #######################################################################

        #        RM SYN ENVELOPES AND LOGLIKS:

        # Calculate synthetic envelope:
        RM_syn_env_results = get_synthetic_envelope(RM, s_env, delta, tjs, dtjs,
                                                    cf, N, v, tJ, vJ, inv_Qi_val,
                                                    maxamp_ind)
        RM_syn_env = RM_syn_env_results['syn_env']
        RM_syn_envs[fband] = RM_syn_env

        # Get data, syn_data and data_vars to be used for the calculation of the
        # likelihood (we only use the coda for this, since the synthetic envelope
        # does NOT reproduce the direct wave arrival):
        RM_syn_data = RM_syn_env[DW_end_ind:-150]

        # Calculate loglikelihood for this frequency band:
        RM_loglik = get_loglik (data, RM_syn_data, inv_Qd)

        # Add loglikelihood to list to average over all fbands:
        RM_logliks.append(RM_loglik)

        #######################################################################

        #        MLM SYN ENVELOPES AND LOGLIKS:

        # Calculate synthetic envelope:
        MLM_syn_env_results = get_synthetic_envelope(MLM, s_env, delta, tjs,
                                                     dtjs, cf, N, v, tJ, vJ,
                                                     inv_Qi_val, maxamp_ind)
        MLM_syn_env = MLM_syn_env_results['syn_env']
        MLM_syn_envs[fband] = MLM_syn_env

        # Get data, syn_data and data_vars to be used for the calculation of the
        # likelihood (we only use the coda for this, since the synthetic envelope
        # does NOT reproduce the direct wave arrival):
        MLM_syn_data = MLM_syn_env[DW_end_ind:-150]

        # Calculate loglikelihood for this frequency band:
        MLM_loglik = get_loglik (data, MLM_syn_data, inv_Qd)

        # Add loglikelihood to list to average over all fbands:
        MLM_logliks.append(MLM_loglik)

    # Calculate the mean loglikelihood for the RM:
    RM_loglik = np.mean(np.array(RM_logliks))

    # Calculate the mean loglikelihood for the MLM:
    MLM_loglik = np.mean(np.array(MLM_logliks))

    return RM, RM_loglik, RM_syn_envs, RM_percs, MLM, MLM_loglik, MLM_syn_envs


###############################################################################
####                    END OF THE FUNCTION                                ####
###############################################################################




@njit
def calculate_mode(x):

    '''
    This function uses numpy.histogram to calculate the mode of an array of
    numbers. The number of bins in the histogram will be the square root of the
    number of elements in x.

    Arguments:
        -x: (np.array) Collection of numbers we want to calculate the mode of.

    Returns:
        -mode_val: (float) Mode of the set of numbers.

    '''

    # Calculate the histogram of x:
    #nbins = int(np.sqrt(len(x)))# This is WAY TOO MANY
    nbins = 1000
    hist = np.histogram(x, bins = nbins)

    # Get the length/size and the edges of the bins:
    bins_lengths = hist[0]
    bins_edges = hist[1]

    # Find the index of the tallest bin and its center, this will be the mean:
    bins_max = (np.abs(bins_lengths - bins_lengths.max())).argmin()
    mode_val = (bins_edges[bins_max] + bins_edges[bins_max+1])/2

    return mode_val


###############################################################################
####                    END OF THE FUNCTION                                ####
###############################################################################



def EFMD_plot_results(array_nm, fbands, units, vel_model, dataset,
                      DW_end_inds, delta, EFMD_results, figs_fname,
                      scat_layer_label, parameter, syn_test = False,
                      comb_results = False, showplots = False):

    '''
    This function plots the results from the EFMD.

    Arguments:
        - array_nm: (str) Name of the seismic array.
        - fbands: (dict) Frequencies to be used in the analysis. Keys are names
                  assigned to each fband and each element is a list with freqmin
                  and freqmax for each frequency band (in Hz) of interest.
                  (Example: fband = {'A': [0.5,1], 'B': [1,2]}).
        - units: (str) Units for distances and velocities (either 'm' or 'km').
        - vel_model: (dict) Characteristics of the velocity model to be used in
                     the EFMD analysis. For each layer, it contains thickness
                     and mean P wave velocity. Traveltimes through each layer,
                     cumulative traveltimes from the bottom of the model to the
                     top of each layer, number of layers in the model, velocities
                     for the TFWM and EFM are also included. Dictionary created
                     by the get_velocity_model function from vel_models.
        - dataset: (dict) Dataset to be used in the plots. If real data is being
                    used, the dictionary should contain resampled, stacked and
                    normalised coda envelopes for all frequency bands, as well
                    as the standard deviation of these data, in the same format
                    used for the inversion. For synthetic tests, file should
                    contain the input model, synthetic envelopes corresponding
                    to input model and standard deviation of the data.
        - DW_end_inds: (dict) Indices of time sample at which the direct wave
                       time window ends.
        - delta: (float) Inverse of the sampling rate of our data.
        - EFMD_results: (dict) Set of results obtained from the Bayesian EFMD
                        inversion. It includes all accepted models, their
                        likelihoods, histogram matrices representing the
                        synthetic envelopes of all accepted models at all fbands,
                        the time it took each iteration to run, the number of
                        unexpected errors, the percentage of iterations completed
                        at the time of saving the file, a and E values for each
                        layer both in units/decimal form and km/percentaje, total
                        number of times a and E were accepted, total number of
                        times a and E were updated, total acceptance rate, total
                        number of iterations, structural parameters of the best
                        fitting and minimum loglikelihood models, their synthetic
                        envelopes at all frequencies and their loglikelihoods.
        - fig_name: (str) Path and common part of figures' file name.
        - scat_layer_label: (str) Needed for synthetic tests only. Either ''
                            (for 'all' scattering layers) or 'LX' (for 'single'
                            scattering layer, where X is the number of the layer
                            containing the scattering).
        -syn_test: (bool) Indication of whether real data is being inverted or
                    a synthetic test is being conducted.
        - parameter: (str) Statistical parameter to use to calculate the model
                     which is representative of the ensemble obtained from the
                     EFMD. (Either Mean, Median or Mode)
        - comb_results: (bool) Indication of whether EFMD_results come from a
                        single MCMC or from a combination of multiple chains.
                        Default is False.
        - showplots: (bool, optional) Indication of whether to show or not the
                     plots at the end of the analysis. Default is False.

    Output:

        * Histograms and values of the likelihoods for kept models.
        * Histograms of the values of the structural parameters for all layers
          in the model.
        * 2D Histograms of the values of the structural parameters for all layers
          in the model.
        * Data normalised coda envelopes vs. synthetic envelopes for all accepted
          models.

        Figures are saved both in .png and .pdf formats.

    '''

    # Extract needed results from EFMD_results:
    num_models = len(EFMD_results['kept_models'])
    kept_logliks = np.array(EFMD_results['kept_logliks'])
    if comb_results == True: logliks_all = EFMD_results['logliks_all']
    a = EFMD_results['a']
    a_km = EFMD_results['a_km']
    E_perc = EFMD_results['E_perc']
    a_accepts_rate = EFMD_results['a_accepts_rate']
    E_accepts_rate = EFMD_results['E_accepts_rate']
    accepts_rate = EFMD_results['total_accepts_rate']
    hists_2D_syn_envs = EFMD_results['hists_2D_syn_envs']
    hist_range = EFMD_results['hists_2D_range']
    N_iters = EFMD_results['N_iters']
    RM = EFMD_results['RM_structural_params']
    RM_syn_envs = EFMD_results['RM_syn_envs']
    MLM = EFMD_results['MLM_structural_params']
    MLM_syn_envs = EFMD_results['MLM_syn_envs']

    ###########################################################################

    # Get stacked, corrected and normalised coda envelopes for all events and
    # fbands: these are my DATA.
    if syn_test == True:
        input_model = dataset[array_nm]['input_model']

    s_nces = {}; s_nces_std = {}
    for fband in fbands:
        s_nces[fband] = dataset[array_nm][fband]['s_nces']
        s_nces_std[fband] = dataset[array_nm][fband]['s_nces_std']

    # Get layer bottoms from velocity model:
    L = np.array(vel_model['L'])
    num_layers = vel_model['num_layers']
    N = num_layers*2 - 1

    # Set number of bins for histograms:
    num_bins = 500

    # Create layer labels for plots:
    if syn_test == True:
        if units == 'm':
            layers_plot_labels = str(num_layers) + ' layers model, ' \
                                + scat_layer_label + ', layer bottoms at ' \
                                + str(np.round((L/1000), 2)) + ' km'
        else:
            layers_plot_labels = str(num_layers) + ' layers model, ' \
                                + scat_layer_label + ', layer bottoms at ' \
                                + str(np.round(L, 2)) + ' ' + units
    else:
        if units == 'm':
            layers_plot_labels = str(num_layers) + ' layers model, layer bottoms at ' \
                                + str(np.round((L/1000), 2)) + ' km'
        else:
            layers_plot_labels = str(num_layers) + ' layers model, layer bottoms at ' \
                                + str(np.round(L, 2)) + ' ' + units

    layer_labels = []; plot_layer_labels = []
    for i in range(num_layers):
        layer_labels.append('L' + str(i+1))
        plot_layer_labels.append('Layer ' + str(i+1))

    # Input model string:
    if syn_test == True:
        input_model_str = ''
        for i in range(num_layers - 1, N, 1):
            input_model_str += '[' + str(input_model[i][0]/1000) + ', ' \
            + str(100*input_model[i][1]) + ']'

    # RM string:
    RM_str = ''
    for i in range(num_layers - 1, N, 1):
        RM_str += '[' + str(np.round(RM[i][0]/1000, 3)) \
        + ', ' + str(np.round(100 * RM[i][1], 3)) + ']'

    # MLM model string:
    MLM_str = ''
    for i in range(num_layers - 1, N, 1):
        MLM_str += '[' + str(np.round(MLM[i][0]/1000, 3)) \
        + ', ' + str(np.round(100*MLM[i][1], 3)) + ']'

    # Get representative value of the parameters in each layer:
    rep_a = []; rep_E = []
    for i in range(num_layers - 1, N):
        rep_a.append(RM[i][0])
        rep_E.append(RM[i][1])

    ###########################################################################

    # Plot histograms of the likelihoods for kept models:

    f, (ax1, ax2) = plt.subplots(2, 1, figsize = (20,10))
    if syn_test == True:
        f.suptitle(array_nm + ' syn test, N_iters = ' + str(N_iters) + ', ' \
                   + str(np.round(accepts_rate, 4)) + ' % global A.R. ' \
                   + layers_plot_labels + ', max loglik = ' \
                   + str(np.round(np.max(-np.array(kept_logliks)) , 4)) \
                   + '\n a A.R. = ' + str(a_accepts_rate) \
                   + '%, $\epsilon$ A.R. = ' + str(E_accepts_rate) \
                   + '%.', fontsize = 20)
    else:
        f.suptitle(array_nm + ', N_iters = ' + str(N_iters) + ', ' \
                   + str(np.round(accepts_rate, 4)) + ' % global A.R. ' \
                   + layers_plot_labels + ', max loglik = ' \
                   + str(np.round(np.max(-np.array(kept_logliks)) , 4)) \
                   + '\n a A.R. = ' + str(a_accepts_rate) \
                   + '%, $\epsilon$ A.R. = ' + str(E_accepts_rate) \
                   + '%.', fontsize = 20)
    ax1.plot(np.arange(num_models), -np.array(kept_logliks),
             color = 'lightseagreen', linewidth = 2)
    ax1.set_xlabel('Number of accepted models', fontsize = 20)
    ax1.set_ylabel('Posterior probability \n exponent', fontsize = 20)
    #ax1.set_ylim([-30, 1])
    ax1.tick_params(axis = 'both', labelsize = 18)
    ax1.grid()
    ax2.hist(kept_logliks, color = 'lightseagreen', bins = num_bins)
    ax2.set_xlabel('Loglikelihoods', fontsize = 20)
    ax2.set_ylabel('Frequency (counts)', fontsize = 20)
    ax2.tick_params(axis = 'both', labelsize = 18)
    ax2.grid()

    figname = figs_fname + '_kept_model_likelihoods_histograms.'
    plt.savefig(figname + 'png', dpi = 300, bbox_inches = 'tight')
    plt.savefig(figname + 'pdf', bbox_inches = 'tight')
    if showplots == False: plt.close()

    ###########################################################################

    # Plot histograms of the structural parameters for each layer:

    if num_layers == 1:

        key = layer_labels[0]
        lab = plot_layer_labels[0]
        f, axes = plt.subplots(num_layers, 2, figsize = (20, 10))
        if units == 'km':
            axes[0].hist(a[key], bins = num_bins, color = 'darkorange',
                         label = lab)
        else: axes[0].hist(a_km[key], bins = num_bins, color = 'darkorange')
        axes[1].hist(E_perc[key], bins = num_bins, color = 'lightseagreen')

        if syn_test == True:
            axes[0].axvline(input_model[num_layers-1][0]/1000, color = 'k',
                            linestyle = 'dashed', linewidth = 2,
                            label = 'Input model value')
            axes[1].axvline(input_model[num_layers-1][1]*100, color = 'k',
                            linestyle = 'dashed', linewidth = 2,
                            label = 'Input model value')
        axes[0].axvline(RM[num_layers-1][0]/1000, color = 'r', linestyle = 'dashdot',
                        linewidth = 2, label = parameter)
        axes[1].axvline(RM[num_layers-1][1]*100, color = 'r', linestyle = 'dashdot',
                        linewidth = 2, label = parameter)
        axes[0].axvline(MLM[num_layers-1][0]/1000, color = 'b', linestyle = 'dashdot',
                        linewidth = 2, label = 'MLM')
        axes[1].axvline(MLM[num_layers-1][1]*100, color = 'b', linestyle = 'dashdot',
                        linewidth = 2, label = 'MLM')

        axes[1].yaxis.tick_right()
        axes[0].grid()
        axes[1].grid()

        axes[0].set_xlabel('Correlation length (km)', fontsize = 20)
        axes[1].set_xlabel('RMS Velocity Fluctuations (%)', fontsize = 20)
        axes[0].tick_params(axis = 'both', labelsize = 18)
        axes[1].tick_params(axis = 'both', labelsize = 18)
        axes[0].legend(loc = 'upper right', title = lab, title_fontsize = 20,
                       fontsize = 18)
        axes[1].legend(loc = 'upper right', title = lab, title_fontsize = 20,
                       fontsize = 18)

    else:

        f, axes = plt.subplots(num_layers, 2, sharex = 'none', sharey = 'none',
                               figsize = (20, 10))

        for i, key in enumerate(layer_labels):

            lab = plot_layer_labels[i]

            if units == 'km':
                axes[i,0].hist(a[key], bins = num_bins, color = 'darkorange',
                               label = lab)
            else: axes[i,0].hist(a_km[key], bins = num_bins, color = 'darkorange')
            axes[i, 1].hist(E_perc[key], bins = num_bins, color = 'lightseagreen')
            if syn_test == True:
                axes[i, 0].axvline(input_model[num_layers-1+i][0]/1000,
                                   color = 'k', linestyle = 'dashed',
                                   linewidth = 2, label = 'Input model value')
                axes[i, 1].axvline(input_model[num_layers-1+i][1]*100,
                                   color = 'k', linestyle = 'dashed', linewidth = 2,
                                   label = 'Input model value')
            axes[i, 0].axvline(RM[num_layers-1+i][0]/1000, color = 'r',
                               linestyle = 'dashed', linewidth = 2,
                               label = parameter)
            axes[i, 1].axvline(RM[num_layers-1+i][1]*100, color = 'r',
                               linestyle = 'dashed', linewidth = 2,
                               label = parameter)
            axes[i, 0].axvline(MLM[num_layers-1+i][0]/1000, color = 'b',
                               linestyle = 'dashdot', linewidth = 2,
                               label = 'MLM')
            axes[i, 1].axvline(MLM[num_layers-1+i][1]*100, color = 'b',
                               linestyle = 'dashdot', linewidth = 2,
                               label = 'MLM')
            axes[i, 1].yaxis.tick_right()
            axes[i, 0].grid()
            axes[i, 1].grid()

            axes[num_layers-1, 0].set_xlabel('Correlation length (km)',
                                             fontsize = 20)
            axes[num_layers-1, 1].set_xlabel('RMS Velocity Fluctuations (%)',
                                             fontsize = 20)
            axes[i,0].tick_params(axis = 'both', labelsize = 18)
            axes[i,1].tick_params(axis = 'both', labelsize = 18)
            axes[i,0].legend(loc = 'best', framealpha = 0.7, title = lab,
                             title_fontsize = 20, fontsize = 18)
            axes[i,1].legend(loc = 'best', framealpha = 0.7, title = lab,
                             title_fontsize = 20, fontsize = 18)

    if syn_test == True:
        f.suptitle(array_nm + ' syn test, N_iters = ' + str(N_iters) + ', ' \
                   + str(np.round(accepts_rate, 4)) + '% A.R. ' \
                   + layers_plot_labels + '\n a A.R. = ' + str(a_accepts_rate) \
                   + '%, $\epsilon$ A.R. = ' + str(E_accepts_rate) \
                   + '%. Input model = ' + input_model_str + ', RM = ' + RM_str \
                   + ', MLM = ' + MLM_str, fontsize = 20)
    else:
        f.suptitle(array_nm + ', N_iters = ' + str(N_iters) + ', ' \
                   + str(np.round(accepts_rate, 4)) + '% A.R. ' \
                   + layers_plot_labels + '\n a A.R. = ' + str(a_accepts_rate) \
                   + '%, $\epsilon$ A.R. = ' + str(E_accepts_rate) \
                   + '%, RM = ' + RM_str + ', MLM = ' + MLM_str, fontsize = 20)

    figname = figs_fname + '_structural_params_histograms.'
    plt.savefig(figname + 'png', dpi = 300, bbox_inches = 'tight')
    plt.savefig(figname + 'pdf', bbox_inches = 'tight')
    if showplots == False: plt.close()

    ###########################################################################

    # Plot 2D histograms of the structural parameters for each layer:

    ncol = num_layers; nrow = 1
    ls = layer_labels
    f, axes = plt.subplots(nrow, ncol, figsize = (20,10))

    for j in range(ncol):
        key = ls[j]
        lab = plot_layer_labels[j]

        if num_layers == 1:
            ax = axes
            ax.set_ylabel('RMS Velocity Fluctuations (%)', fontsize = 24)
        else:
            ax = axes[j]
            axes[0].set_ylabel('RMS Velocity Fluctuations (%)', fontsize = 24)

        if units == 'km': im = ax.hist2d(a[key], E_perc[key], bins = num_bins,
                                         norm = mpl.colors.LogNorm(), cmap = 'Blues')
        else: im = ax.hist2d(a_km[key], E_perc[key], bins = num_bins,
                             norm = mpl.colors.LogNorm(), cmap = 'afmhot_r')

        ax.set_title(lab, fontsize = 24)
        ax.set_xlabel('Correlation length (km)', fontsize = 24)
        ax.tick_params(axis = 'both', labelsize = 24)
        ax.grid()
        if syn_test == True:
            ax.scatter(input_model [num_layers - 1 + j][0] / 1000,
                       input_model [num_layers - 1 + j][1] * 100, s = 500,
                       marker = '+', c = 'b', linewidth = 4,
                       label = 'Input model value')
        ax.scatter(RM [num_layers - 1 + j][0] / 1000,
                   RM [num_layers - 1 + j][1] * 100, s = 500, marker = '+',
                   c = 'k', linewidth = 4, label = parameter)
        ax.scatter(MLM [num_layers - 1 + j][0] / 1000,
                   MLM [num_layers - 1 + j][1] * 100, s = 500, marker = '+',
                   c = 'deepskyblue', linewidth = 4, label = 'MLM')
        ax.legend(loc = 'best', fontsize = 24)

    cbar_ax = f.add_axes([0.95, 0.134, 0.021, 0.687])
    cbar = f.colorbar(im[3], cbar_ax)
    cbar.set_label(label = 'Frequency (counts)', rotation = -90, labelpad = 18,
                   fontsize = 24)

    if syn_test == True:
        f.suptitle(array_nm + ' syn test, N_iters = ' + str(N_iters) + ', ' \
                   + str(np.round(accepts_rate, 4)) + '% A.R. ' \
                   + layers_plot_labels + '\n a A.R. = ' + str(a_accepts_rate) \
                   + '%, $\epsilon$ A.R. = ' + str(E_accepts_rate) \
                   + '%. \n Input model = ' + input_model_str + ', RM = ' \
                   + RM_str + ', MLM = ' + MLM_str, fontsize = 20)
    else:
        f.suptitle(array_nm + ', N_iters = ' + str(N_iters) + ', ' \
                   + str(np.round(accepts_rate, 4)) + '% A.R. ' \
                   + layers_plot_labels + '\n a A.R. = ' + str(a_accepts_rate) \
                   + '%, $\epsilon$ A.R. = ' + str(E_accepts_rate) \
                   + '%, RM = ' + RM_str + ', MLM = ' + MLM_str, fontsize = 20)

    figname = figs_fname + '_structural_params_2D_histograms.'
    plt.savefig(figname + 'png', dpi = 300, bbox_inches = 'tight')
    plt.savefig(figname + 'pdf', bbox_inches = 'tight')
    if showplots == False: plt.close()

    ###########################################################################

    # Define time vector (it doesn't matter which frequency we use, the data
    # envelopes should all be equally long):
    t = np.arange(0, len(s_nces['H'])) * delta

    # Plot 2D histogram of all synthetic envelopes and input model synthetic
    # envelopes together:
    if len(fbands) == 8:
        nrow = 2; ncol = 4
        fbs = [['A', 'B', 'C', 'D'],['E', 'F', 'G', 'H']]
    elif len(fbands) == 5:
        ncol = 3; nrow = 2
        fbs = [['D', 'E', 'F'], ['G', 'H']]
    f, axes = plt.subplots(nrow, ncol, figsize = (20,10))

    for i, ax_row in enumerate(axes):
        for j, ax_col in enumerate(ax_row):
            if len(fbands) == 5 and (i, j) != (1, 2):
                ax = axes[i][j]
                fband = fbs[i][j]

            elif len(fbands) == 8:
                ax = axes[i][j]
                fband = fbs[i][j]

            ax.axvspan(t[DW_end_inds[fband]], t[-150], color = 'seagreen',
                       alpha = 0.2, label = 'Coda time window')
            im = ax.contourf(t, hist_range, hists_2D_syn_envs[fband],
                             norm = mpl.colors.LogNorm(), cmap = 'afmhot_r')
            ax.plot(t, s_nces[fband], 'k', linewidth = 2, label =  'Input model')
            ax.plot(t, RM_syn_envs[fband], color = 'blue', linestyle = 'dashed',
                    linewidth = 2, label =  'RM')
            ax.plot(t, MLM_syn_envs[fband], color = 'c', linestyle = 'dashdot',
                    linewidth = 2, label =  'MLM')
            if i  ==  1: ax.set_xlabel('Time (s)', fontsize = 24)
            if j  ==  0: ax.set_ylabel('Amplitude', fontsize = 24)

            # Legend:
            legend_title = str(fbands[fband][0]) + ' to ' + str(fbands[fband][1]) \
                           + ' Hz'
            if len(fbands) == 8:
                ax.legend(loc = 'best', title = legend_title, title_fontsize = 16,
                          ncol = 2, fancybox = True, framealpha = 0.55,
                          fontsize = 12)
                ax.tick_params(axis = 'both', labelsize = 24)
                ax.grid()
            elif len(fbands) == 5 and (i,j) != (1,2):
                ax.legend(loc = 'best', title = legend_title, title_fontsize = 16,
                          fancybox = True, ncol = 2, columnspacing = 0.25,
                          framealpha = 0.55, fontsize = 12)
                ax.tick_params(axis = 'both', labelsize = 24)
                ax.grid()

            # Axes:
            if syn_test == True:
                if num_layers == 1:
                    ax.set_ylim(0, 0.8)
                elif num_layers == 2:
                    if 'L2' in scat_layer_label: ax.set_ylim(0, 3.1)
                    elif 'L1' in scat_layer_label: ax.set_ylim(0, 0.8)
                    else: ax.set_ylim(0, 0.5)
                elif num_layers == 3:
                    ax.set_ylim(0, 0.3)
            else:
                if num_layers != 1:
                    ax.set_ylim(0, 0.3)
                else:
                    ax.set_ylim(0, 0.5)
            ax.set_xlim(20, t[-1])

    if len(fbands) == 8:
        cbar_ax = f.add_axes([0.95, 0.115, 0.015, 0.74])
        cbar = f.colorbar(im, cax = cbar_ax)
    elif len(fbands) == 5:
        cbar = f.colorbar(im, cax = axes[-1][-1])
    cbar.set_label(label = 'Frequency (counts)', rotation = -90, labelpad = 18,
                   fontsize = 24)

    if syn_test == True:
        f.suptitle(array_nm + ' syn test, N_iters = ' + str(N_iters) + ', ' \
                   + str(np.round(accepts_rate, 4)) + '% A.R. ' \
                   + layers_plot_labels + '\n a A.R. = ' + str(a_accepts_rate) \
                   + '%, $\epsilon$ A.R. = ' + str(E_accepts_rate) \
                   + '%. \n Input model = ' + input_model_str + ', RM = ' \
                   + RM_str, fontsize = 20)
    else:
        f.suptitle(array_nm + ', N_iters = ' + str(N_iters) + ', ' \
                   + str(np.round(accepts_rate, 4)) + '% A.R. ' \
                   + layers_plot_labels + '\n a A.R. = ' + str(a_accepts_rate) \
                   + '%, $\epsilon$ A.R. = ' + str(E_accepts_rate) \
                   + '%, RM = ' + RM_str + ', MLM = ' + MLM_str, fontsize = 20)

    figname = figs_fname + '_DATA_vs_RM_MLM_all_tested_models_envelopes.'
    plt.savefig(figname + 'png', dpi = 300, bbox_inches = 'tight')
    plt.savefig(figname + 'pdf', bbox_inches = 'tight')
    if showplots == False: plt.close()

###############################################################################
####                    END OF THE FUNCTION                                ####
###############################################################################



def EFMD_plot_results_summary ( array_nm, fbands, units, vel_model,
                               EFMD_results, figs_fname, scattering_layer,
                               parameter, syn_test = False, comb_results = False,
                               showplots = False):

    '''
    This function plots a summary of the results from the EFMD in a single figure.

    Arguments:
        - array_nm: (str)
            Name of the seismic array.
        - fbands: (dict)
            Frequencies to be used in the analysis. Keys are names assigned to
            each fband and each element is a list with freqmin and freqmax for
            each frequency band (in Hz) of interest.
            (Example: fband = {'A': [0.5,1], 'B': [1,2]}).
        - units: (str)
            Units for distances and velocities (either 'm' or 'km').
        - vel_model: (dict)
            Characteristics of the velocity model to be used in the EFMD analysis.
            For each layer, it contains thickness and mean P wave velocity.
            Traveltimes through each layer, cumulative traveltimes from the bottom
            of the model to the top of each layer, number of layers in the model,
            velocities for the TFWM and EFM are also included. Dictionary created
            by the get_velocity_model function from vel_models.
        - EFMD_results: (dict)
            Set of results obtained from the Bayesian EFMD inversion. It includes
            all accepted models, their likelihoods, histogram matrices representing
            the synthetic envelopes of all accepted models at all fbands, the time
            it took each iteration to run, the number of unexpected errors, the
            percentage of iterations completed at the time of saving the file,
            a and E values for each layer both in units/decimal form and
            km/percentaje, total number of times a and E were accepted, total
            number of times a and E were updated, total acceptance rate, total
            number of iterations, structural parameters of the best fitting and
            minimum loglikelihood models, their synthetic envelopes at all
            frequencies and their loglikelihoods.
        -fig_name: (str)
            Path and common part of figures' file name.
        - scattering_layer: (str)
            Needed for synthetic tests only. Either '' (for 'all' scattering
            layers or real data inversions), '1' or '2'.
        - parameter: (str)
            Statistical parameter to use to calculate the model which is
            representative of the ensemble obtained from the EFMD. (Either Mean,
            Median or Mode)
        -syn_test: (bool)
            Indication of whether real data is being inverted or a synthetic
            test is being conducted.
        - comb_results: (bool)
            Indication of whether EFMD_results come from a single MCMC or from
            a combination of multiple chains. Default is False.
        - showplots: (bool, optional)
            Indication of whether to show or not the plots at the end of the
            analysis. Default is False.

    Output:

        * Single figure containing the histograms and values of the likelihoods
          for all combined chains, the histograms of the values of the structural
          parameters for all layers in the model, 2D Histograms of the values of
          the structural parameters for all layers in the model, the data
          normalised coda envelopes vs. synthetic envelopes for all accepted
          models and fbands.

        Figures will be saved both in .png and .pdf formats.

    '''

    # Extract needed results from EFMD_results:
    kept_logliks = np.array( EFMD_results['kept_logliks'] )
    if comb_results == True:
        logliks_all = EFMD_results['logliks_all']
        num_chains = len( logliks_all)
    else:
        num_chains = 1
        logliks_all = kept_logliks
    a = EFMD_results['a']
    a_km = EFMD_results['a_km']
    E_perc = EFMD_results['E_perc']
    hists_2D_syn_envs = EFMD_results['hists_2D_syn_envs']
    hist_range = EFMD_results['hists_2D_range']
    RM_percs = EFMD_results['RM_structural_params_percs']
    dataset = EFMD_results['resampled_s_nces_dataset']
    delta = EFMD_results['resampled_delta']
    DW_end_inds = EFMD_results['DW_end_inds']

    ###########################################################################

    # Get stacked, corrected and normalised coda envelopes for all events and
    # fbands: these are my DATA.
    if syn_test == True:
        input_model = dataset[array_nm]['input_model']

    s_nces = {}; s_nces_std = {}
    for fband in fbands:
        s_nces[fband] = dataset[array_nm][fband]['s_nces']
        s_nces_std[fband] = dataset[array_nm][fband]['s_nces_std']

    # Get layer bottoms from velocity model:
    L = np.array( vel_model['L'] )
    num_layers = vel_model['num_layers']

    # Create layer labels for plots:
    if scattering_layer == 'all': scat_layer_label = ''
    else: scat_layer_label = 'L' + str(scattering_layer) + '_scattering'

    layers_plot_labels = str( num_layers) + ' layers model, '

    if syn_test == True:
        if scat_layer_label != '':
            layers_plot_labels = layers_plot_labels + scat_layer_label + ', '

    layers_plot_labels = layers_plot_labels + 'layer bottoms at '

    for i, val in enumerate(L):
        if units == 'm': rval = np.round( val/1000, 2)
        else: rval = val

        if len(L) == 1:
            layers_plot_labels = layers_plot_labels + str( rval ) + ' km'
        elif len(L)>=2 and val != L[-1] and val != L[-2]:
            layers_plot_labels = layers_plot_labels + str( rval ) + ', '
        elif len(L)>=2 and val == L[-2]:
            layers_plot_labels = layers_plot_labels + str( rval )
        else:
            layers_plot_labels = layers_plot_labels + ' and ' +str( rval ) + ' km'

    layer_labels = []; plot_layer_labels = []
    for i in range(num_layers):
        layer_labels.append( 'L' + str(i+1) )
        plot_layer_labels.append( 'Layer ' + str(i+1) )

    ###########################################################################

    # Some specs for plots:

    # Data line color:
    if syn_test == True: lcolor = 'dodgerblue'
    else: lcolor = 'dodgerblue' #'dimgray'

    # Text sizes:
    if num_layers == 1:
        major_text_size = 18
        minor_text_size = 16
        legend_text_size = 14
    elif num_layers == 2:
        major_text_size = 16
        minor_text_size = 14
        legend_text_size = 12
    else:
        major_text_size = 14
        minor_text_size = 12
        legend_text_size = 12

    # Spacing between subplots:
    verspace = 0.15
    if num_layers != 1: horspace = 0.38
    else: horspace = verspace

    # Figure dimensions:
    if num_layers == 1:# and syn_test == True:
        dimx = 15; dimy = 15
    else:
        dimx = 15; dimy = 10

    # Create text labels for the subplots:
    tlabels = list( string.ascii_lowercase)

    # Create empty variable to host axes numbers:
    axnum = 0

    # Create figure:
    f = plt.figure( constrained_layout = False, figsize = ( dimx, dimy))

    if len(fbands) == 8: gs = GridSpec ( len(fbands), 4*(num_layers) +4,
                                        figure = f, wspace = horspace,
                                        hspace = verspace)
    elif len(fbands) == 5: gs = GridSpec ( len(fbands)*2, 4*(num_layers) +4,
                                          figure = f, wspace = horspace,
                                          hspace = verspace)

    ###########################################################################

    # Plot loglikelihoods for all combined chains:
    chains_axes = []
    #ylabels = ['Posterior', 'probability', 'exponent']
    ax00 = f.add_subplot( gs[:num_chains, :-4])
    ax00.tick_params(labelcolor='w', axis = 'both', which = 'both',
                     direction = 'in', length = 0, color = 'white', top=False,
                     bottom=False, left=False, right=False)

    if len(fbands) == 8 and syn_test == False and num_layers != 1: pad_val = 60
    elif syn_test == False and num_layers == 1: pad_val = 80
    else: pad_val = 25

    if num_chains > 3: ax00_label = 'Posterior probability exponent'
    else: ax00_label = 'Posterior probability\n exponent'

    ax00.set_ylabel( ax00_label, labelpad = pad_val, fontsize = major_text_size)
    ax00.spines['top'].set_color('none')
    ax00.spines['bottom'].set_color('none')
    ax00.spines['left'].set_color('none')
    ax00.spines['right'].set_color('none')

    if num_layers != 1:
        num_ticks = 6
    else:
        num_ticks = 4

    for w in range( num_chains):

        if w == 0:
            ax = f.add_subplot( gs[w,:-4])
            chains_axes.append( ax)
        else:
            ax = f.add_subplot( gs[w,:-4], sharex = chains_axes[0])
            chains_axes.append( ax)

            # Add axis to axnum:
            axnum += 1

        ax.plot( -logliks_all[w], color = 'maroon', linewidth = 2)

        ax.grid()
        ax.tick_params( axis = 'both', which = 'major', bottom = False,
                       right = False, labelsize = minor_text_size)
        ax.xaxis.tick_top()
        ax.xaxis.set_major_locator(plt.MaxNLocator( num_ticks))

        if w != 0:
            plt.setp( ax.get_xticklabels(), visible = False)
            if syn_test == False and num_layers == 1 and array_nm == 'PSA':
                ax.yaxis.set_major_formatter( FormatStrFormatter('%.2e'))
        else:
            ax.set_title( 'Number of accepted models', fontsize = major_text_size)
            ax.xaxis.set_major_formatter( FormatStrFormatter('%.0f'))
            if syn_test == False and num_layers == 1 and array_nm == 'PSA':
                ax.yaxis.set_major_formatter( FormatStrFormatter('%.2e'))
        if syn_test == False and num_layers == 1 and array_nm == 'ASAR':
            ax.set_ylim([-11050, -10450])

        # Place text label inside the subplot:
        if num_layers == 1: xy = (0.010, 0.087)
        elif num_layers == 2:
            if len(fbands) == 8: xy = (0.007, 0.131)
            else: xy = (0.007, 0.155)
        elif num_layers == 3: xy = (0.007, 0.122)
        ax.annotate( tlabels[axnum], xy = xy, xycoords = 'axes fraction',
                    bbox = dict( facecolor = 'white', alpha = 0.7),
                    fontsize = major_text_size, color = 'k')

    ###########################################################################

    # 1D and 2D histograms for the structural parameters need to be done in a
    # specific order for each layer in the model. 2D should go first, then 1D
    # for a, then 1D for E.

    # Define axes limits for those cases in which we need to zoom in:
    if syn_test == True and scat_layer_label != '':
        scat_layer = scat_layer_label[:2]
    if scat_layer_label == '':
        scat_layer = scat_layer_label

    for k in range( num_layers):

        key = layer_labels[k]
        lab = plot_layer_labels[k]
        plot_factor = k*4

        if num_layers == 1: m = 0
        elif num_layers == 2: m = k+1
        elif num_layers == 3: m = k+2

        # Define number of bins for histograms (it should be the same for all
        # of them):
        num_bins = 500

        #######################################################################

        # Plot 1D histogram for the RMS velocity fluctuations:
        if len(fbands) == 8:
            ax0 = f.add_subplot( gs[ num_chains:-1, 0 + plot_factor])
        elif len(fbands) == 5:
            ax0 = f.add_subplot( gs[ num_chains:-2, 0 + plot_factor])

        # Plot 9-95 percentiles:
        ax0.axhspan( RM_percs[m][1, 0]*100, RM_percs[m][1, 1]*100,
                    color = 'rosybrown', alpha = 0.2)

        # Plot histogram:
        ax0.hist( E_perc[key], bins = num_bins, color = 'darkred',
                 orientation = 'horizontal')

        # Draw horizontal line with the input and representative models values
        # of the parameter:
        if syn_test == True:
             ax0.axhline( input_model[m][1]*100, color = lcolor, linewidth = 3,
                         linestyle = (0, (0.1, 2)), dash_capstyle = 'round',
                         label = 'Input value', alpha = 0.8)

        # Define axes:
        ax0.invert_xaxis()
        ax0.set_xticklabels([])
        ax0.tick_params( axis='x', which='both', bottom=False, top=False,
                        labelbottom=False)
        if k == 0: ax0.set_ylabel( 'RMS Velocity Fluctuations (%)',
                                  fontsize = major_text_size )

        if (syn_test == False and num_layers == 1):
            if array_nm == 'PSA':
                ax0.set_yscale('log')
                ax0.set_ylim([ 0.0045, 0.0089])
                ax0.set_yticks( np.array([0.005, 0.006, 0.007, 0.008]))
                ax0.yaxis.set_major_formatter( FormatStrFormatter( '%.3f') )
            else:
                ax0.set_yscale('log')
                ax0.set_ylim([ 0.0045, 1.05])
                ax0.set_yticks( np.array([ 0.005, 0.01, 0.05, 0.1, 0.5]))
                ax0.yaxis.set_major_formatter( FormatStrFormatter( '%.2f') )

        elif ( syn_test == False and num_layers == 2 and len(fbands) == 8):
            ax0.set_yscale('log')
            ax0.yaxis.set_major_formatter( FormatStrFormatter( '%.2f') )
            ax0.set_yticks( np.array([0.01, 0.05, 0.1, 0.5, 1.0]))
            # ax0.xaxis.set_major_locator(plt.MaxNLocator(4))

        elif syn_test == True and num_layers == 1:
            ax0.yaxis.set_major_formatter( FormatStrFormatter('%.3f') )
            # ax0.set_ylim([ 4.995, 5.04])

        elif ( syn_test == False and num_layers == 2 and len(fbands) == 5 \
              and array_nm == 'PSA') or ( syn_test == True and num_layers == 3) or \
            ( syn_test == True and num_layers == 2):
            ax0.yaxis.set_major_formatter( FormatStrFormatter('%.1f') )

        elif ( syn_test == True and num_layers == 2 \
              and 'L1' in scat_layer_label and k == 0):
            ax0.set_ylim([6.68, 7.12])
            ax0.axes.get_yaxis().set_ticks( [6.7, 6.8, 6.9, 7.0, 7.1])

        elif ( syn_test == True and num_layers == 2 \
              and 'L2' in scat_layer_label and k == 1):
            ax0.yaxis.set_major_formatter( FormatStrFormatter( '%.2f') )
            # ax0.xaxis.set_major_locator(plt.MaxNLocator(5))
            ax0.set_ylim([6.978, 7.012])
            ax0.axes.get_yaxis().set_ticks( [6.980, 6.990, 7.000, 7.010])

        elif ( syn_test == True and num_layers == 2 and scat_layer_label == '' \
              and k == 1):
            ax0.set_ylim([3.78, 4.22])
            ax0.axes.get_yaxis().set_ticks( [3.8, 3.9, 4.0, 4.1, 4.2])

        else: ax0.yaxis.set_major_formatter( FormatStrFormatter('%.0f') )

        ax0.grid()
        ax0.tick_params( axis = 'y', which = 'both', left = True, right = True,
                        labelsize = minor_text_size, pad = 0.5 )

        # Add axis to axnum:
        axnum += 1

        # Place text label inside the subplot:
        if num_layers == 1: xy = (0.049, 0.018)
        elif num_layers == 2:
            if scat_layer == 'L1' and k == 0:
                xy = (0.07, 0.938)
            elif scat_layer == 'L2' or syn_test == False:
                xy = (0.07, 0.938)
            else:
                xy = (0.070, 0.028)
        elif num_layers == 3:
            if k != 2: xy = (0.121, 0.026)
            else: xy = (0.121, 0.946)

        ax0.annotate( tlabels[axnum], xy = xy,
                     xycoords = 'axes fraction',
                     bbox = dict( facecolor = 'white', alpha = 0.7),
                     fontsize = major_text_size, color = 'k')

        # Get ax0 box limits (left, bottom, width, height):
        ax0_l, ax0_b, ax0_w, ax0_h = ax0.get_position().bounds

        #######################################################################

        # Plot 2D histogram of the structural parameters:
        if len(fbands) == 8:
            ax1 = f.add_subplot( gs[ num_chains:-1,
                                (1 + plot_factor):(4 + plot_factor)],
                                sharey = ax0)# 2D histogram
        elif len(fbands) == 5:
            ax1 = f.add_subplot( gs[ num_chains:-2,
                                (1 + plot_factor):(4 + plot_factor)],
                                sharey = ax0)# 2D histogram

        # Plot 9-95 percentiles:
        ax1.axvspan( RM_percs[m][0, 0]/1000, RM_percs[m][0, 1]/1000,
                    color = 'rosybrown', alpha = 0.2)
        ax1.axhspan( RM_percs[m][1, 0]*100, RM_percs[m][1, 1]*100,
                    color = 'rosybrown', alpha = 0.2)

        if syn_test == True:
            ax1.axvline( input_model[m][0]/1000, color = lcolor, linewidth = 3,
                        linestyle = (0, (0.1, 2)), dash_capstyle = 'round',
                        label = 'Input value', alpha = 0.8)
            ax1.axhline( input_model[m][1]*100, color = lcolor, linewidth = 3,
                        linestyle = (0, (0.1, 2)), dash_capstyle = 'round',
                        alpha = 0.8)

        # Legend:
        if (syn_test == False and num_layers == 2 and len(fbands) == 8) or \
            (syn_test == False and num_layers == 1 and array_nm == 'ASAR') or \
            (syn_test == True and num_layers == 3 and k != 2):
            leg_loc = 'center right'
        elif (syn_test == True and 'L1' in scat_layer_label and k == 1) or \
            (syn_test == True and num_layers == 3 and k == 2) or \
            (syn_test == False and num_layers == 1 and array_nm == 'PSA'):
            leg_loc = 'center left'
        elif (syn_test == True and num_layers == 2 and scat_layer_label == ''):
              leg_loc = 'lower left'
        elif (syn_test == True and 'L1' in scat_layer_label and k == 0) or \
            (syn_test == True and 'L2' in scat_layer_label):
            leg_loc = 'upper right'
        else:
            leg_loc = 'lower right'

        ax1.legend( loc = leg_loc, title = lab, title_fontsize = minor_text_size,
                       fontsize = legend_text_size)

        # Plot histogram:
        if units == 'km': im = ax1.hist2d( a[key], E_perc[key], bins = num_bins,
                                          norm = mpl.colors.LogNorm(),
                                          cmap = 'afmhot_r' )
        else: im = ax1.hist2d( a_km[key], E_perc[key], bins = num_bins,
                              cmap = 'afmhot_r', norm = mpl.colors.LogNorm())

        # Set x axis scale to log:
        if ( syn_test == False and num_layers == 2) or \
            ( syn_test == False and num_layers == 3) or \
            ( syn_test == True and num_layers == 3) or \
            ( syn_test == True and 'L1' in scat_layer_label and k == 1 ) or \
            ( syn_test == True and 'L2' in scat_layer_label and k == 0 ):
            ax1.set_xscale('log')
            ax1.set_xticks( np.array([ 1, 5, 10, 20]))
        else:
            ax1.xaxis.set_major_locator(plt.MaxNLocator(4))

        # Set y axis scale to log:
        if (syn_test == False and num_layers == 1):
            # ax1.set_yscale('log')
            if array_nm == 'PSA':
                ax1.set_xlim([ 10, 32])
                ax1.set_ylim([ 0.0045, 0.0089])
                ax1.set_yticks( np.array([0.005, 0.006, 0.007, 0.008]))
            else:
                ax1.set_ylim([ 0.0045, 1.05])
                ax1.set_yticks( np.array([ 0.005, 0.01, 0.05, 0.1, 0.5]))


        elif (syn_test == False and num_layers == 2 and len(fbands) == 8):
            ax1.set_yscale('log')
            ax1.set_yticks( np.array([0.01, 0.05, 0.1, 0.5, 1.0]))
            ax1.yaxis.set_major_formatter( FormatStrFormatter( '%.2f') )

        elif ( syn_test == True and num_layers == 2 and scat_layer_label == '' \
              and k == 1):
            ax1.set_ylim([3.78, 4.22])
            # ax1.yaxis.set_major_locator(plt.MaxNLocator(4))
            ax1.axes.get_yaxis().set_ticks( [3.8, 3.9, 4.0, 4.1, 4.2])

        elif ( syn_test == True and num_layers == 2 and 'L1' in scat_layer_label \
              and k == 0):
            ax1.set_ylim([6.68, 7.12])
            ax1.axes.get_yaxis().set_ticks( [6.7, 6.8, 6.9, 7.0, 7.1])

        elif ( syn_test == True and num_layers == 2 and 'L2' in scat_layer_label \
              and k == 1):
            ax1.yaxis.set_major_formatter( FormatStrFormatter( '%.2f') )
            # ax0.xaxis.set_major_locator(plt.MaxNLocator(5))
            ax1.set_ylim([6.978, 7.012])
            ax1.axes.get_yaxis().set_ticks( [6.980, 6.990, 7.000, 7.010])
        elif (syn_test == True and num_layers == 1):
            # ax1.set_ylim([ 4.96, 5.04])
            # ax1.set_xlim([ 4.7, 5.2])
            # ax1.yaxis.set_major_formatter( FormatStrFormatter( '%.2f') )
            ax1.yaxis.set_major_locator(plt.MaxNLocator(4))

        # Remove axes ticks labels:
        ax1.tick_params( axis = 'both', which = 'both', left = True, right = False,
                        top = False, bottom = True, labelbottom = False,
                        labelleft = False)
        ax1.grid( axis = 'both', which = 'major')

        # Define location of the colorbar inside the plot:
        if (syn_test == True and 'L2' in scat_layer_label) \
        or (syn_test == True and 'L1' in scat_layer_label and k == 0)\
        or (syn_test == True and num_layers == 3 and k != 2):
            cbar_loc = 'lower center'
        else:
            cbar_loc = 'upper center'

        # Create colorbar inside the plot:
        cbaxes = inset_axes(ax1, width = "90%", height = "5%", loc = cbar_loc)
        plt.colorbar(im[3], ax = ax1, cax=cbaxes, ticks = [10**1, 10**3, 10**5, 10**7],
                     orientation='horizontal')
        cbaxes.set_xlabel( xlabel = 'Frequency (counts)', fontsize = major_text_size )
        if cbar_loc == 'lower center':
            cbaxes.xaxis.set_ticks_position('top')
            cbaxes.xaxis.set_label_position('top')
        cbaxes.tick_params( axis = 'both', labelsize = minor_text_size )

        # Add axis to axnum:
        axnum += 1

        # Place text label inside the subplot:
        if num_layers == 1: xy = (0.014, 0.018)
        elif num_layers == 2:
            if scat_layer == 'L1':
                if k == 0: xy = (0.027, 0.939)
                if k == 1: xy = (0.025, 0.028)
            elif scat_layer == 'L2':
                xy = (0.023, 0.938)
            elif syn_test == False:
                xy = (0.021, 0.028)
            else:
                xy = (0.928, 0.029)
        elif num_layers == 3:
            if k != 2: xy = (0.029, 0.946)
            else: xy = (0.029, 0.026)

        ax1.annotate( tlabels[axnum], xy = xy,
                    xycoords = 'axes fraction',
                    bbox = dict( facecolor = 'white', alpha = 0.7),
                    fontsize = major_text_size, color = 'k')

        # I need to move the 2D histograms and corr. length histograms for
        # the 2 layer models:
        # [left, bottom, width, height]
        ax1_l, ax1_b, ax1_w, ax1_h = ax1.get_position().bounds

        if num_layers == 2:
            if k == 0: ax1.set_position([ ax1_l - 0.008, ax1_b, ax1_w, ax1_h])
            elif k == 1:
                ax0.set_position([ ax1_l - 0.008 - ax0_w, ax0_b, ax0_w, ax0_h])
        if num_layers == 3:
            if k == 0: ax1.set_position([ ax1_l - 0.007, ax1_b, ax1_w, ax1_h])
            elif k == 1:
                ax0.set_position([ ax1_l - 0.0075 - ax0_w, ax0_b, ax0_w, ax0_h])
                ax1.set_position([ ax1_l - 0.0035, ax1_b, ax1_w, ax1_h])
            elif k == 2:
                ax0.set_position([ ax1_l - 0.004 - ax0_w, ax0_b, ax0_w, ax0_h])

        #######################################################################

        # Plot 1D histogram for the correlation length:

        if len(fbands) == 8:
            ax2 = f.add_subplot( gs[-1, (1 + plot_factor):(4 + plot_factor)],
                                sharex = ax1)
        elif len(fbands) == 5:
            ax2 = f.add_subplot( gs[-2:, (1 + plot_factor):(4 + plot_factor)],
                                sharex = ax1)

        # 5-95 percentiles:
        ax2.axvspan( RM_percs[m][0, 0]/1000, RM_percs[m][0, 1]/1000,
                    color = 'rosybrown', alpha = 0.2)

        if units == 'km': ax2.hist( a[key], bins = num_bins, color = 'maroon')
        else: ax2.hist( a_km[key], bins = num_bins, color = 'maroon')

        # Plot vertical lines for the input model and representative model values
        # of the parameter:
        if syn_test == True:
            ax2.axvline( input_model[m][0]/1000, color = lcolor, linewidth = 3,
                        linestyle = (0, (0.1, 2)), dash_capstyle = 'round',
                        label = 'Input value', alpha = 0.8)

        # Define axes:
        ax2.invert_yaxis()
        ax2.set_yticklabels([])
        # Set x axis scale to log:
        if ( syn_test == False  and num_layers == 2) or \
            ( syn_test == False and num_layers == 3) or \
            ( syn_test == True and num_layers == 3) or \
            ( syn_test == True and 'L1' in scat_layer_label and k == 1 ) or \
            ( syn_test == True and 'L2' in scat_layer_label and k == 0 ):
            ax2.set_xscale('log')
            ax2.set_xticks( np.array([ 1, 5, 10, 20]))
            ax2.xaxis.set_major_formatter( FormatStrFormatter('%.0f') )
        elif syn_test == True and scat_layer == 'L2' and k == 1:
            ax2.set_xlim ( 0.990, 1.010)
            ax2.xaxis.set_major_locator(plt.MaxNLocator(3))
            ax2.xaxis.set_major_formatter( FormatStrFormatter('%.2f') )
        # elif (syn_test == True and num_layers == 1):
            # ax2.set_xlim([ 4.7, 5.2])
        else:
            ax2.xaxis.set_major_locator(plt.MaxNLocator(4))
        ax2.set_xlabel( 'Correlation length (km)', fontsize = major_text_size)
        ax2.tick_params( axis='y', which='both', left=False, right=False)
        ax2.tick_params( axis = 'x', which='both', bottom = True, top = True,
                        labelbottom = True, labelsize = minor_text_size )

        if syn_test == False and num_layers == 1 and array_nm == 'PSA':
            ax1.set_xlim([ 10, 32])

        ax2.grid( which = 'major', axis = 'x')

        # Add axis to axnum:
        axnum += 1

        # Place text label inside the subplot:
        if num_layers == 1: xy = (0.014, 0.083)
        elif num_layers == 2:
            if scat_layer == '' and syn_test == True: 
                if k == 0: xy = (0.953, 0.125)
                elif k == 1: xy = (0.955, 0.125)
            elif syn_test == False and len(fbands) == 8: xy = (0.021, 0.125)
            elif syn_test == False and len(fbands) == 5: xy = (0.021, 0.070)
            else: 
                if k == 0: xy = (0.026, 0.125)
                elif k == 1: xy = (0.024, 0.125)
        elif num_layers == 3: xy = (0.029, 0.115)

        ax2.annotate( tlabels[axnum], xy = xy,
                     xycoords = 'axes fraction',
                     bbox = dict( facecolor = 'white', alpha = 0.7),
                     fontsize = major_text_size, color = 'k')

        # I need to move the 2D histograms and corr. length histograms for
        # the 2 layer models:
        ax2_l, ax2_b, ax2_w, ax2_h = ax2.get_position().bounds

        if num_layers == 2:
            if k == 0:
                ax2.set_position([ ax2_l - 0.008, ax2_b, ax2_w, ax2_h])
        if num_layers == 3:
            if k == 0: ax2.set_position([ ax2_l - 0.007, ax2_b, ax2_w, ax2_h])
            elif k == 1: ax2.set_position([ ax2_l - 0.0035, ax2_b, ax2_w, ax2_h])

    ###########################################################################

    # Plot 2D histograms of the synthetic envelopes, (synthetic or real )data
    # envelopes and RM envelopes:

    # Define time vector (it doesn't matter which frequency we use, the data
    # envelopes should all be equally long):
    t = np.arange(0, len( s_nces['H'] )) * delta

    for i, fband in enumerate( fbands):

        if len(fbands) == 8: ax = f.add_subplot( gs[ i, -4:])
        elif len(fbands) == 5: ax = f.add_subplot( gs[ 2*i:2*i+2, -4:])

        ax.axvspan( t[DW_end_inds[fband]], t[-150], color = 'rosybrown',
                   alpha = 0.3)
        im = ax.contourf( t, hist_range, hists_2D_syn_envs[fband],
                         norm = mpl.colors.LogNorm(), cmap = 'afmhot_r')
        if i == 0:
            if syn_test == True:
                ax.plot( t, s_nces[fband], color = lcolor, linewidth = 3,
                        linestyle = (0, (0.1, 2)), dash_capstyle = 'round',
                        alpha = 0.9, label =  'Input data')
            else:
                ax.plot( t, s_nces[fband], color = lcolor, linewidth = 1.5,
                        alpha = 0.8, label =  'Data')

        else:
            if syn_test == True:
                ax.plot( t, s_nces[fband], color = lcolor, linewidth = 3,
                        alpha = 0.9, linestyle = (0, (0.1, 2)),
                        dash_capstyle = 'round',)
            else:
                ax.plot( t, s_nces[fband], color = lcolor, linewidth = 1.5,
                        alpha = 0.8 )

       # Axes:
        if fband != 'H':
            ax.set_xticklabels([])
        else:
            ax.set_xlabel( 'Time (s)', fontsize = major_text_size )
        ax.set_yticklabels([])

        if syn_test == True:
            if num_layers == 1:
                ax.set_ylim(0, 0.8)
            elif num_layers == 2:
                if 'L1' in scat_layer_label: ax.set_ylim(0, 0.9)
                elif 'L2' in scat_layer_label: ax.set_ylim(0, 4.0)
                else: ax.set_ylim(0, 0.5)

            elif num_layers == 3:
                ax.set_ylim( 0, 0.4)
        else:
            if num_layers == 2 or ( num_layers != 1 and array_nm != 'WRA'):
                ax.set_ylim(0, 0.3)
            elif num_layers == 1:
                ax.set_ylim(0, 0.5)
            elif num_layers == 3 and array_nm == 'WRA':
                ax.set_ylim( 0, 0.5)
        ax.set_xlim( 20, t[-1])

        ax.tick_params( axis = 'both', labelsize = minor_text_size )
        ax.tick_params( axis='y', which='both', left=False, right=False,
                       labelbottom=False)

        # Grid and legend:
        ax.grid()
        leg_title = str(fbands[fband][0]) + ' to ' + str(fbands[fband][1]) + ' Hz'
        ax.legend( loc = 'upper left', fancybox = True, ncol = 2, framealpha = 0.55,
                  title = leg_title, columnspacing = 1, title_fontsize = minor_text_size,
                  handlelength = 1, labelspacing = 0.05, fontsize = legend_text_size)

        # Colorbars:
        cbaxes = inset_axes(ax, width = "2.5%", height = "70%", loc = 'center right')
        plt.colorbar(im, ax = ax, cax = cbaxes,
                     ticks = [10**1, 10**4, 10**7], orientation = 'vertical')
        cbaxes.yaxis.set_ticks_position('left')
        cbaxes.tick_params( axis = 'both', labelsize = minor_text_size )

        # Place text label inside the subplot:
        if num_layers == 1: xy = (0.010, 0.083)
        elif num_layers == 2 and len(fbands) == 8: xy = (0.016, 0.125)
        elif num_layers == 2 and len(fbands) == 5: xy = (0.016, 0.070)
        elif num_layers == 3: xy = xy = (0.021, 0.115)

        ax.annotate( tlabels[w + (num_layers*3) + i + 1], xy = xy,
                     xycoords = 'axes fraction',
                     bbox = dict( facecolor = 'white', alpha = 0.7),
                     fontsize = major_text_size, color = 'k')

    figname = figs_fname + '_FULL_RESULTS_SUMMARY.'
    plt.savefig(figname + 'png', dpi = 300, bbox_inches = 'tight')
    # plt.savefig(figname + 'pdf', bbox_inches = 'tight')
    if showplots == False: plt.close()

###############################################################################
####                    END OF THE FUNCTION                                ####
###############################################################################




def EFMD_plot_results_summary_simple (array_nm, fbands, units,
                                      vel_model, EFMD_results, figs_fname,
                                      scattering_layer, parameter, syn_test = False,
                                      comb_results = False, showplots = False):
    '''
    This function plots a simpler summary of the results of the EFMD. The
    resulting figure is exactly the same as the one from the
    EFMD_plot_results_summary function, except it doesn't include the logliks
    of the models kept in each chain.

    Arguments:
        - array_nm: (str)
            Name of the seismic array.
        - fbands: (dict)
            Frequencies to be used in the analysis. Keys are names assigned to
            each fband and each element is a list with freqmin and freqmax for
            each frequency band (in Hz) of interest.
            (Example: fband = {'A': [0.5,1], 'B': [1,2]}).
        - units: (str)
            Units for distances and velocities (either 'm' or 'km').
        - vel_model: (dict)
            Characteristics of the velocity model to be used in the EFMD analysis.
            For each layer, it contains thickness and mean P wave velocity.
            Traveltimes through each layer, cumulative traveltimes from the bottom
            of the model to the top of each layer, number of layers in the model,
            velocities for the TFWM and EFM are also included. Dictionary created
            by the get_velocity_model function from vel_models.
        - EFMD_results: (dict)
            Set of results obtained from the Bayesian EFMD inversion. It includes
            all accepted models, their likelihoods, histogram matrices representing
            the synthetic envelopes of all accepted models at all fbands, the time
            it took each iteration to run, the number of unexpected errors, the
            percentage of iterations completed at the time of saving the file,
            a and E values for each layer both in units/decimal form and
            km/percentaje, total number of times a and E were accepted, total
            number of times a and E were updated, total acceptance rate, total
            number of iterations, structural parameters of the best fitting and
            minimum loglikelihood models, their synthetic envelopes at all
            frequencies and their loglikelihoods.
        - fig_name: (str)
            Path and common part of figures' file name.
        - scattering_layer: (str)
            Needed for synthetic tests only. Either '' (for 'all' scattering
            layers or real data inversions), '1' or '2'.
        - parameter: (str)
            Statistical parameter to use to calculate the model which is
            representative of the ensemble obtained from the EFMD. (Either Mean,
            Median or Mode)
        - syn_test: (bool)
            Indication of whether real data is being inverted or a synthetic
            test is being conducted.
        - comb_results: (bool)
            Indication of whether EFMD_results come from a single MCMC or from
            a combination of multiple chains. Default is False.
        - showplots: (bool, optional)
            Indication of whether to show or not the plots at the end of the
            analysis. Default is False.

    Output:

        * Single figure containing the histograms of the values of the structural
          parameters for all layers in the model, 2D Histograms of the values of
          the structural parameters for all layers in the model, the data
          normalised coda envelopes vs. synthetic envelopes for all accepted
          models and fbands.

        Figures will be saved both in .png and .pdf formats.

    '''

    # Extract needed results from EFMD_results:
    kept_logliks = np.array(EFMD_results['kept_logliks'])
    if comb_results == True:
        logliks_all = EFMD_results['logliks_all']
        num_chains = len(logliks_all)
    else:
        num_chains = 1
        logliks_all = kept_logliks
    a = EFMD_results['a']
    a_km = EFMD_results['a_km']
    E_perc = EFMD_results['E_perc']
    hists_2D_syn_envs = EFMD_results['hists_2D_syn_envs']
    hist_range = EFMD_results['hists_2D_range']
    RM_percs = EFMD_results['RM_structural_params_percs']
    dataset = EFMD_results['resampled_s_nces_dataset']
    delta = EFMD_results['resampled_delta']
    DW_end_inds = EFMD_results['DW_end_inds']

    ###########################################################################

    # Get stacked, corrected and normalised coda envelopes for all events and
    # fbands: these are my DATA.
    if syn_test == True:
        input_model = dataset[array_nm]['input_model']

    s_nces = {}; s_nces_std = {}
    for fband in fbands:
        s_nces[fband] = dataset[array_nm][fband]['s_nces']
        s_nces_std[fband] = dataset[array_nm][fband]['s_nces_std']

    # Get layer bottoms from velocity model:
    L = np.array(vel_model['L'])
    num_layers = vel_model['num_layers']

    # Create layer labels for plots:
    if scattering_layer == 'all': scat_layer_label = ''
    else:
        scat_layer_label = scat_layer_label = 'L' + str(scattering_layer) \
                           + '_scattering'

    layers_plot_labels = str(num_layers) + ' layers model, '

    if syn_test == True:
        if scat_layer_label != '':
            layers_plot_labels = layers_plot_labels + scat_layer_label + ', '

    layers_plot_labels = layers_plot_labels + 'layer bottoms at '

    for i, val in enumerate(L):
        if units == 'm': rval = np.round(val/1000, 2)
        else: rval = val

        if len(L) == 1:
            layers_plot_labels = layers_plot_labels + str(rval) + ' km'
        elif len(L)>=2 and val != L[-1] and val != L[-2]:
            layers_plot_labels = layers_plot_labels + str(rval) + ', '
        elif len(L)>=2 and val == L[-2]:
            layers_plot_labels = layers_plot_labels + str(rval)
        else:
            layers_plot_labels = layers_plot_labels + ' and ' +str(rval) + ' km'

    layer_labels = []; plot_layer_labels = []
    for i in range(num_layers):
        layer_labels.append('L' + str(i+1))
        plot_layer_labels.append('Layer ' + str(i+1))

    ###########################################################################

    # Some specs for plots:

    # Data line color:
    if syn_test == True: lcolor = 'dodgerblue'
    else: lcolor = 'dodgerblue' #'dimgray'

    # Text sizes:
    if num_layers != 3:
        major_text_size = 22
        minor_text_size = 20
        legend_text_size = 16
    else:
        major_text_size = 18
        minor_text_size = 16
        legend_text_size = 14

    # Spacing between subplots:
    verspace = 0.15
    if num_layers != 1: horspace = 0.3
    else: horspace = verspace

    # Create figure:

    f = plt.figure(constrained_layout = False, figsize = (35, 15))

    if len(fbands) == 8:
        gs = GridSpec (len(fbands), 4*(num_layers) +4, figure = f,
                       wspace = horspace, hspace = verspace)
    elif len(fbands) == 5:
        gs = GridSpec (len(fbands)*2, 4*(num_layers) +4, figure = f,
                       wspace = horspace, hspace = verspace)

    ###########################################################################

    # 1D and 2D histograms for the structural parameters need to be done in a
    # specific order for each layer in the model. 1D for a should go first, then
    # 2D, then 1D for E.

    # Define axes limits for those cases in which we need to zoom in:
    if syn_test == True and scat_layer_label != '':
        scat_layer = scat_layer_label[:2]
    if scat_layer_label == '':
        scat_layer = scat_layer_label

    for k in range(num_layers):

        key = layer_labels[k]
        lab = plot_layer_labels[k]
        plot_factor = k*4

        if num_layers == 1: m = 0
        elif num_layers == 2: m = k+1
        elif num_layers == 3: m = k+2

        #######################################################################

        # Define number of bins for histograms (it should be the same for all of
        # them):
        num_bins = 500

        # Plot 2D histogram of the structural parameters:
        if len(fbands) == 8:
            ax1 = f.add_subplot(gs[:-1, (1 + plot_factor):(4 + plot_factor)])
        elif len(fbands) == 5:
            ax1 = f.add_subplot(gs[:-2, (1 + plot_factor):(4 + plot_factor)])

        # Plot 9-95 percentiles:
        ax1.axvspan(RM_percs[m][0, 0]/1000, RM_percs[m][0, 1]/1000,
                    color = 'rosybrown', alpha = 0.2)
        ax1.axhspan(RM_percs[m][1, 0]*100, RM_percs[m][1, 1]*100,
                    color = 'rosybrown', alpha = 0.2)

        if syn_test == True:
            ax1.axvline(input_model[m][0]/1000, color = lcolor, linewidth = 5,
                        linestyle = (0, (0.1, 3)), dash_capstyle = 'round',
                        label = 'Input value', alpha = 0.8)
            ax1.axhline(input_model[m][1]*100, color = lcolor, linewidth = 5,
                        linestyle = (0, (0.1, 3)), dash_capstyle = 'round',
                        alpha = 0.8)

        # Legend:
        if (syn_test == False and num_layers == 2 and len(fbands) == 8) or \
            (syn_test == False and num_layers == 1 and array_nm == 'ASAR'):
            leg_loc = 'center right'
        elif (syn_test == True and 'L1' in scat_layer_label and k == 1) or \
            (syn_test == True and num_layers == 3 and k == 2) or \
            (syn_test == False and num_layers == 1 and array_nm == 'PSA'):
            leg_loc = 'center left'
        elif (syn_test == True and num_layers == 2 and scat_layer_label == ''):
              leg_loc = 'lower left'
        elif (syn_test == True and 'L1' in scat_layer_label and k == 0) or \
            (syn_test == True and 'L2' in scat_layer_label) or \
            (syn_test == True and num_layers == 3 and k != 2):
            leg_loc = 'upper right'
        else:
            leg_loc = 'lower right'

        ax1.legend(loc = leg_loc, title = lab, title_fontsize = minor_text_size,
                       fontsize = legend_text_size)

        # Plot histogram:
        if units == 'km':
            im = ax1.hist2d(a[key], E_perc[key], bins = num_bins,
                            norm = mpl.colors.LogNorm(), cmap = 'afmhot_r')
        else:
            im = ax1.hist2d(a_km[key], E_perc[key], bins = num_bins,
                            cmap = 'afmhot_r', norm = mpl.colors.LogNorm())

        # Set x axis scale to log:
        if (syn_test == False and num_layers == 2) or \
            (syn_test == False and num_layers == 3) or \
            (syn_test == True and num_layers == 3) or \
            (syn_test == True and 'L1' in scat_layer_label and k == 1) or \
            (syn_test == True and 'L2' in scat_layer_label and k == 0):
            ax1.set_xscale('log')
            ax1.set_xticks(np.array([1, 5, 10, 20]))
        else:
            ax1.xaxis.set_major_locator(plt.MaxNLocator(4))

        # Set y axis scale to log:
        if (syn_test == False and num_layers == 1):
            ax1.set_yscale('log')
            if array_nm == 'PSA':
                ax1.set_yticks(np.array([0.001, 0.005, 0.01]))
                ax1.set_ylim([0.0045, 0.01])
                ax1.set_xlim([10, 32])
            else:
                ax1.set_yticks(np.array([0.01, 0.05, 0.1, 0.5, 1]))
                ax1.set_ylim([0.0045, 1.01])

        elif (syn_test == False and num_layers == 2 and len(fbands) == 8):
            ax1.set_yscale('log')
            ax1.set_yticks(np.array([0.01, 0.1, 1]))

        # Remove axes ticks labels:
        ax1.tick_params(axis = 'both', which = 'both', left = True,
                        right = False, top = False, bottom = True,
                        labelbottom = False, labelleft = False)
        ax1.grid(axis = 'both', which = 'major')

        # Define location of the colorbar inside the plot:
        if (syn_test == True and 'L2' in scat_layer_label) \
        or (syn_test == True and 'L1' in scat_layer_label and k == 0)\
        or (syn_test == True and num_layers == 3 and k != 2):
            cbar_loc = 'lower center'
        else:
            cbar_loc = 'upper center'

        # Create colorbar inside the plot:
        cbaxes = inset_axes(ax1, width = "90%", height = "5%", loc = cbar_loc)
        plt.colorbar(im[3], ax = ax1, cax=cbaxes, ticks = [10**1, 10**3, 10**5, 10**7],
                     orientation='horizontal')
        cbaxes.set_xlabel(xlabel = 'Frequency (counts)', fontsize = major_text_size)
        if cbar_loc == 'lower center':
            cbaxes.xaxis.set_ticks_position('top')
            cbaxes.xaxis.set_label_position('top')
        cbaxes.tick_params(axis = 'both', labelsize = minor_text_size)

        #######################################################################
        # Plot 1D histogram for the correlation length:

        if len(fbands) == 8:
            ax2 = f.add_subplot(gs[-1, (1 + plot_factor):(4 + plot_factor)],
                                sharex = ax1)
        elif len(fbands) == 5:
            ax2 = f.add_subplot(gs[-2:, (1 + plot_factor):(4 + plot_factor)],
                                sharex = ax1)

        # 5-95 percentiles:
        ax2.axvspan(RM_percs[m][0, 0]/1000, RM_percs[m][0, 1]/1000,
                    color = 'rosybrown', alpha = 0.2)

        if units == 'km': ax2.hist(a[key], bins = num_bins, color = 'maroon')
        else: ax2.hist(a_km[key], bins = num_bins, color = 'maroon')

        # Plot vertical lines for the input model and representative model values
        # of the parameter:
        if syn_test == True:
            ax2.axvline(input_model[m][0]/1000, color = lcolor, linewidth = 5,
                        linestyle = (0, (0.1, 3)), dash_capstyle = 'round',
                        label = 'Input value', alpha = 0.8)

        # Define axes:
        ax2.invert_yaxis()
        ax2.set_yticklabels([])
        # Set x axis scale to log:
        if (syn_test == False  and num_layers == 2) or \
            (syn_test == False and num_layers == 3) or \
            (syn_test == True and num_layers == 3) or \
            (syn_test == True and 'L1' in scat_layer_label and k == 1) or \
            (syn_test == True and 'L2' in scat_layer_label and k == 0):
            ax2.set_xscale('log')
            ax2.set_xticks(np.array([1, 5, 10, 20]))
            ax2.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
        else:
            ax2.xaxis.set_major_locator(plt.MaxNLocator(4))
        ax2.set_xlabel('Correlation length (km)', fontsize = major_text_size)
        ax2.tick_params(axis='y', which='both', left=False, right=False)
        ax2.tick_params(axis = 'x', which='both', bottom = True, top = True,
                        labelbottom = True, labelsize = minor_text_size)
        # ax2.tick_params(axis = 'x', which='minor', bottom = True, top = True,
                        # labelbottom = False)

        if syn_test == False and num_layers == 1 and array_nm == 'PSA':
            ax1.set_xlim([10, 32])

        ax2.grid(which = 'major', axis = 'x')

        #######################################################################


        # Plot 1D histogram for the RMS velocity fluctuations:
        if len(fbands) == 8:
            ax2 = f.add_subplot(gs[:-1, 0 + plot_factor], sharey = ax1)
        elif len(fbands) == 5:
            ax2 = f.add_subplot(gs[:-2, 0 + plot_factor], sharey = ax1)

        # Plot 9-95 percentiles:
        ax2.axhspan(RM_percs[m][1, 0]*100, RM_percs[m][1, 1]*100,
                    color = 'rosybrown', alpha = 0.2)

        # Plot histogram:
        ax2.hist(E_perc[key], bins = num_bins, color = 'darkred',
                 orientation = 'horizontal')

        # Draw horizontal line with the input and representative models values
        # of the parameter:
        if syn_test == True:
             ax2.axhline(input_model[m][1]*100, color = lcolor, linewidth = 5,
                         linestyle = (0, (0.1, 3)), dash_capstyle = 'round',
                         label = 'Input value', alpha = 0.8)

        # Define axes:
        ax2.invert_xaxis()
        ax2.set_xticklabels([])
        ax2.tick_params(axis='x', which='both', bottom=False, top=False,
                        labelbottom=False)
        if k == 0: ax2.set_ylabel('RMS Velocity Fluctuations (%)',
                                  fontsize = major_text_size)

        if (syn_test == False and num_layers == 1):
            if array_nm == 'PSA':
                ax2.set_yscale('log')
                ax2.set_yticks(np.array([0.001, 0.005, 0.01]))
                ax2.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
                ax2.yaxis.set_minor_formatter(FormatStrFormatter('%.3f'))
                ax2.xaxis.set_major_locator(plt.MaxNLocator(3))
                ax2.xaxis.set_minor_locator(plt.MaxNLocator(1))
                ax2.set_ylim([0.0045, 0.01])
            else:
                ax2.set_yscale('log')
                ax2.set_yticks(np.array([0.01, 0.05, 0.1, 0.5, 1]))
                ax2.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
                ax2.xaxis.set_major_locator(plt.MaxNLocator(5))
                ax2.set_ylim([0.0045, 1.05])

        elif (syn_test == False and num_layers == 2 and len(fbands) == 8):
            ax2.set_yscale('log')
            ax2.set_yticks(np.array([0.01, 0.1, 1]))
            ax2.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            ax2.xaxis.set_major_locator(plt.MaxNLocator(4))

        elif syn_test == True and num_layers == 1:
            ax2.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))

        elif (syn_test == False and num_layers == 2 and len(fbands) == 5 and \
              array_nm == 'PSA') or (syn_test == True and num_layers == 3) or \
            (syn_test == True and num_layers == 2 and 'L' not in scat_layer_label) or \
            (syn_test == True and num_layers == 2 and 'L1' in scat_layer_label and k == 1) or \
            (syn_test == True and num_layers == 2 and 'L2' in scat_layer_label and k == 0):
            ax2.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

        elif (syn_test == True and num_layers == 2 and 'L1' in scat_layer_label and k == 0) or \
            (syn_test == True and num_layers == 2 and 'L2' in scat_layer_label and k == 1):
            ax2.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

        else: ax2.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))

        ax2.grid()
        ax2.tick_params(axis = 'y', which = 'both', left = True, right = True,
                        labelsize = minor_text_size)


    ###########################################################################

    # Plot 2D histograms of the synthetic envelopes, (synthetic or real)data
    # envelopes and RM envelopes:

    # Define time vector (it doesn't matter which frequency we use, the data
    # envelopes should all be equally long):
    t = np.arange(0, len(s_nces['H'])) * delta

    for i, fband in enumerate(fbands):

        if len(fbands) == 8: ax = f.add_subplot(gs[i, -4:])
        elif len(fbands) == 5: ax = f.add_subplot(gs[2*i:2*i+2, -4:])

        ax.axvspan(t[DW_end_inds[fband]], t[-150], color = 'rosybrown',
                   alpha = 0.3)
        im = ax.contourf(t, hist_range, hists_2D_syn_envs[fband],
                         norm = mpl.colors.LogNorm(), cmap = 'afmhot_r')
        if i == 0:
            if syn_test == True:
                ax.plot(t, s_nces[fband], color = lcolor, linewidth = 5,
                        linestyle = (0, (0.1, 3)), dash_capstyle = 'round',
                        alpha = 0.9, label =  'Input data')
            else:
                ax.plot(t, s_nces[fband], color = lcolor, linewidth = 3,
                        alpha = 0.8, label =  'Data')
        else:
            if syn_test == True:
                ax.plot(t, s_nces[fband], color = lcolor, linewidth = 5,
                        alpha = 0.9, linestyle = (0, (0.1, 3)),
                        dash_capstyle = 'round',)
            else:
                ax.plot(t, s_nces[fband], color = lcolor, linewidth = 3,
                        alpha = 0.8)

       # Axes:
        if fband != 'H':
            ax.set_xticklabels([])
        else:
            ax.set_xlabel('Time (s)', fontsize = major_text_size)
        ax.set_yticklabels([])

        if syn_test == True:
            if num_layers == 1:
                ax.set_ylim(0, 0.8)
            elif num_layers == 2:
                if 'L1' in scat_layer_label: ax.set_ylim(0, 0.9)
                elif 'L2' in scat_layer_label: ax.set_ylim(0, 4.0)
                else: ax.set_ylim(0, 0.5)

            elif num_layers == 3:
                ax.set_ylim(0, 0.4)
        else:
            if num_layers == 2 or (num_layers != 1 and array_nm != 'WRA'):
                ax.set_ylim(0, 0.3)
            elif num_layers == 1:
                ax.set_ylim(0, 0.5)
            elif num_layers == 3 and array_nm == 'WRA':
                ax.set_ylim(0, 0.5)
        ax.set_xlim(20, t[-1])

        ax.tick_params(axis = 'both', labelsize = minor_text_size)
        ax.tick_params(axis='y', which='both', left=False, right=False,
                       labelbottom=False)

        # Grid and legend:
        ax.grid()
        leg_title = str(fbands[fband][0]) + ' to ' + str(fbands[fband][1]) + ' Hz'
        ax.legend(loc = 'upper left', fancybox = True, ncol = 2,
                  framealpha = 0.55, title = leg_title, columnspacing = 1,
                  title_fontsize = minor_text_size, handlelength = 1,
                  labelspacing = 0.05, fontsize = legend_text_size)

        # Colorbars:
        cbaxes = inset_axes(ax, width = "2.5%", height = "70%",
                            loc = 'center right')
        plt.colorbar(im, ax = ax, cax = cbaxes,
                     ticks = [10**1, 10**4, 10**7], orientation = 'vertical')
        cbaxes.yaxis.set_ticks_position('left')
        cbaxes.tick_params(axis = 'both', labelsize = minor_text_size-2)

    figname = figs_fname + '_FULL_RESULTS_SUMMARY_simple.'
    plt.savefig(figname + 'png', dpi = 300, bbox_inches = 'tight')
    # plt.savefig(figname + 'pdf', bbox_inches = 'tight')
    if showplots == False: plt.close()

###############################################################################
####                    END OF THE FUNCTION                                ####
###############################################################################



def EFMD_plot_x_histograms (array_nm, s_nces_fname, EFMD_results, fbands,
                            figs_fname, syn_test = False, comb_results = False,
                            showplots = False):

    '''
    This function plots the cross 2D histograms of the scattering parameters in
    all layers of the models.

    Arguments:
        - array_nm: (str)
            Name of the seismic array.
        - s_nces_fname: (str) Path and file name of dataset consisting on
                        normalised coda envelopes and their standard deviation.
                        For synthetic tests, the input model is also included.
        - EFMD_results: (dict)
            Set of results obtained from the Bayesian EFMD inversion. It includes
            all accepted models, their likelihoods, histogram matrices
            representing the synthetic envelopes of all accepted models at all
            fbands, the time it took each iteration to run, the number of
            unexpected errors, the percentage of iterations completed at the
            time of saving the file, a and E values for each layer both in
            units/decimal form and km/percentaje, total number of times a and
            E were accepted, total number of times a and E were updated, total
            acceptance rate, total number of iterations, structural parameters
            of the best fitting and minimum loglikelihood models, their
            synthetic envelopes at all frequencies and their loglikelihoods.
        - fbands: (dict)
            Frequencies to be used in the analysis. Keys are names assigned to
            each fband and each element is a list with freqmin and freqmax for
            each frequency band (in Hz) of interest.
            (Example: fband = {'A': [0.5,1], 'B': [1,2]}).
        -figs_fname: (str)
            Path and file name for figures.
        -syn_test: (bool)
            Indication of whether real data is being inverted or a synthetic
            test is being conducted.
        - comb_results: (bool)
            Indication of whether EFMD_results come from a single MCMC or from
            a combination of multiple chains. Default is False.
        - showplots: (bool) (optional)
            Indication of whether to show or not the plots at the end of the
            analysis. Default is False.

    Output:

        * Single figure with the cross histograms for all parameters and layers.

        Figures will be saved both in .png and .pdf formats.

    '''

    # Extract needed results from EFMD_results:
    a_km = EFMD_results['a_km']
    E_perc = EFMD_results['E_perc']
    num_layers = len(a_km)
    if syn_test == True:
        fopen = open(s_nces_fname, 'rb')
        syn_dataset = pickle.load(fopen)
        fopen.close()

        IM = syn_dataset[array_nm]['input_model'][num_layers-1:]

        # Let's work on the input model so the parameters are easier to plot:
        input_a = {}; input_E = {}
        for w in range(num_layers):
            key = 'L' + str(w+1)
            input_a [key] = IM [w][0]/1000
            input_E [key] = IM [w][1]*100

    ###########################################################################

    # Create figure:

    # title_size = 22
    if num_layers != 3:
        major_text_size = 14
        minor_text_size = 12
    else:
        major_text_size = 14
        minor_text_size = 12


    f = plt.figure(constrained_layout = False, figsize = (10, 12))

    gs = GridSpec (2*num_layers, 2*num_layers, figure = f, wspace = 0.05,
                   hspace = 0.05)

    num_bins = 350

    ###########################################################################

    # Plot histograms:

    # We need the number of rows/columns in the plot:
    val = 2*num_layers - 1

    for j in range(2*num_layers):
        for i in range(2*num_layers):

            # CREATE PLOTS:

            if j > i:
                # We don't need these plots, they're redundant.
                pass
            else:

                ax = f.add_subplot(gs[i, j])

                # #############################################################

                # Details:

                if num_layers == 2:
                    ax.xaxis.set_major_locator(plt.MaxNLocator(4))
                    ax.yaxis.set_major_locator(plt.MaxNLocator(4))
                    ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
                    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

                if num_layers == 3:
                    if i%2 == 0 and i != j:
                        ax.set_yscale('log')
                        ax.set_yticks(np.array([1, 5, 10, 20]))
                        ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))

                    if j%2 == 0:
                        ax.set_xscale('log')
                        ax.set_xticks(np.array([1, 5, 10, 20]))
                        ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))

                if syn_test == False and num_layers == 2 and len(fbands) == 8:
                    if j%2==0:
                        ax.set_xscale('log')
                        ax.set_xticks(np.array([1, 5, 10, 20]))
                        ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
                    else:
                        ax.set_xscale('log')
                        ax.set_xticks(np.array([0.01, 0.05, 0.1]))
                        ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
                        # ax.set_xlim([0.0045399, 0.11])

                    if i != j:
                        if i %2==0 and i != 0:
                            ax.set_yscale('log')
                            ax.set_yticks(np.array([1, 5, 10, 20]))
                            ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
                        else:
                            ax.set_yscale('log')
                            ax.set_yticks(np.array([0.01, 0.05, 0.1]))
                            ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
                            # ax.set_ylim([0.0045399, 0.11])

                if syn_test == True and \
                    np.array_equal(IM, np.array([[6e3, 1e-2],[1e3, 7e-2]])):
                    ax.xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
                    if i!=j:
                        ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))

                # Set tick labels and ticks right:

                if i == j:
                    ax.axes.get_yaxis().set_visible(False)
                    if i != val:
                        ax.set_xticklabels([])

                if i < val and j > 0:
                    ax.set_xticklabels([])
                    ax.set_yticklabels([])

                if i != j and i < val and j == 0: ax.set_xticklabels([])

                if i != j and i == val and j > 0: ax.set_yticklabels([])

                if i != j:
                    ax.tick_params(axis = 'both', which = 'both', direction = 'in',
                                   top = True, bottom = True, left = True,
                                   right = True)
                else:
                    ax.tick_params(axis = 'both', which = 'both', direction = 'in',
                                   top = False, bottom = True, left = True,
                                   right = True)

                ###############################################################

                # Get axis labels:

                if num_layers == 2:
                    axis_labels = ['$a_{L1}$(km)', '$\epsilon_{L1}$(%)',
                                    '$a_{L2}$(km)', '$\epsilon_{L2}$(%)']
                elif num_layers == 3:
                    axis_labels = ['$a_{L1}$(km)', '$\epsilon_{L1}$(%)',
                                    '$a_{L2}$(km)', '$\epsilon_{L2}$(%)',
                                    '$a_{L3}$(km)', '$\epsilon_{L3}$(%)']
                if j == 0:
                    ax.set_ylabel(axis_labels[i], fontsize = major_text_size)
                if i == 2*num_layers -1:
                    ax.set_xlabel(axis_labels[j], fontsize = major_text_size)

                ###############################################################

                # SET UP PLOT:

                # Specify what to plot in x and y axes in each subplot:
                if i <= 1: ikey = 'L1'
                elif 1 < i <= 3: ikey = 'L2'
                elif 3 < i <= 5: ikey = 'L3'

                if j <= 1: jkey = 'L1'
                elif 1 < j <= 3: jkey = 'L2'
                elif 3 < j <= 5: jkey = 'L3'

                # If i or j are even numbers, we plot a_km. If they're odd, we
                # plot E_perc.
                if i%2 == 0:
                    y = a_km[ikey]

                else:
                    y = E_perc[ikey]

                if j%2 == 0:
                    x = a_km[jkey];

                else:
                    x = E_perc[jkey];

                ###############################################################

                # Plot histograms on them:
                if i == j:
                    ax.hist(x, bins = num_bins, color = 'maroon')
                else:
                    ax.hist2d(x, y, bins = num_bins, cmap = 'afmhot_r',
                              norm = mpl.colors.LogNorm(vmin = 1, vmax = 1100))
                ax.set_xlim([x.min(), x.max()])
                ax.grid()

                ###############################################################

                # Add label to each subplot:
                sp_label = (str(i+1) + '-' + str(j+1))
                xy = (0.7, 0.87)
                ax.annotate(sp_label, xy = xy,
                xycoords = 'axes fraction',
                bbox = dict(facecolor = 'white', alpha = 0.7),
                fontsize = minor_text_size, color = 'k')

    figname = figs_fname + '_parameters_cross_histograms.'
    plt.savefig(figname + 'png', dpi = 300, bbox_inches = 'tight')
    # plt.savefig(figname + 'pdf', bbox_inches = 'tight')
    if showplots == False: plt.close()

###############################################################################
####                    END OF THE FUNCTION                                ####
###############################################################################
