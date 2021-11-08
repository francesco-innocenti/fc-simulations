import matlab.engine

eng = matlab.engine.start_matlab()

# add MVGC2 path
eng.addpath('/Users/FrancescoInnocenti/Documents/MATLAB/MVGC2/demo')
eng.addpath('/Users/FrancescoInnocenti/Documents/MATLAB/MVGC2/utils')

# test data generation
n_trials = float(4)
n_obs = float(500)
fs = float(200)

# actual (ground truth) VAR model generation parameters
tnet = eng.tnet5()
moact = float(6)
rho = 0.95
wvar = 0.5
rmi = 0.5

# generate random VAR test data

# set RNG
seed = float(123)
eng.rng(seed)

# generate random VAR coefficients for test network
AA = eng.var_rand(tnet, moact, rho, wvar)