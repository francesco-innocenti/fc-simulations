from utils import setup_matlab, plot_network, simulate_data, plot_timeseries


eng = setup_matlab()

# test data generation parameters
n_trials = float(5)
n_obs = float(500)
sfreq = float(200)

# VAR model parameters
adj_net = eng.tnet5()
moact = float(6)
rho = 0.95
wvar = 0.5
rmi = 0.5

# VAR model order estimation parameters
moregmode = 'LWR'
mosel = 'LRT'
momax = float(2) * moact

# VAR model parameter estimation
regmode = 'LWR'

# MVGC (time domain) statistical inference
alpha = 0.05
stest = 'F'
mhtc = 'FDRD'

# MVGC (frequency domain)
fres = []

# set RNG
seed = 123
eng.rng_seed(seed)

# plot mode (for matlab functions)
plotm = float(0)

# generate random VAR test data
#data = simulate_data(n_trials, n_obs, adj_net, moact, rho, wvar, rmi)

# generate random VAR coefficients for test network
AA = eng.var_rand(adj_net, moact, rho, wvar)
n_vars = eng.size(AA, 1)

# generate random residuals covariance (in fact correlation) matrix
VV = eng.corr_rand(n_vars, rmi)

# report information on the generated VAR model
info = eng.var_info(AA, VV)

# generate multi-trial VAR time series data with normally distributed
# residuals for generated VAR coefficients and residuals covariance matrix
data = eng.var_to_tsdata(AA, VV, n_obs, n_trials)

# remove temporal mean and normalise by temporal variance - not strictly
# necessary, but may help numerical stability if data has very large or very
# small values
data = eng.demean(data, True)

# calculate VAR model order estimation criteria up to specified maximum
moaic, mobic, mohqc, molrt = eng.tsdata_to_varmo(data, momax, moregmode, [],
                                                 [], nargout=4)

# select & report VAR model order
morder = eng.moselect(eng.sprintf("VAR model order selection (max = " +
                                  str(momax) + ")"), mosel, 'ACT', moact,
                      'AIC', moaic, 'BIC', mobic, 'HQC', mohqc, 'LRT', molrt)
assert morder > 0, "Selected zero model order! GCs will all be zero!"
if morder >= momax:
    print("WARNING: selected maximum model order (may have been set too low)")

# estimate VAR model of selected order from data
A, V = eng.tsdata_to_var(data, morder, regmode, nargout=2)

# report information on the estimated VAR
info = eng.var_info(A, V)

# calculate spectral pairwise-conditional causalities resolution from VAR model
# parameters. If not specified, we set the frequency resolution to something
# sensible. Warn if resolution is very large, as this may cause problems
if eng.isempty(fres):
    fres = eng.var2fres(A,V)
    max_fres = 2**14
    if fres > max_fres:
        print(f"WARNING: estimated frequency resolution {fres} exceeds maximum; "
              f"setting to {max_fres}")
        fres = max_fres
    else:
        print(f"Using frequency resolution {fres}")

fabserr = eng.var_check_fres(A, V, fres)
print(f"Absolute integration error = {fabserr}")

f = eng.var_to_spwcgc(A, V, fres)
assert ~eng.isbad(f, False), "Spectral GC estimation failed"

# for comparison, we also calculate the actual pairwise-conditional spectral
# causalities
ff = eng.var_to_spwcgc(AA, VV, fres)
assert ~eng.isbad(ff, False), "Spectral GC calculation failed"

# get frequency vector corresponding to specified sampling rate
freqs = eng.sfreqs(fres, sfreq)

# plot spectral causal graphs
if eng.isnumeric(plotm):
    plotm = plotm + 1

eng.plot_sgc((ff, f), freqs,
             'Spectral Granger causalities (blue = actual, red = estimated)',
             [], plotm, nargout=0)
