import matlab.engine


def setup_matlab():
    eng = matlab.engine.start_matlab()

    # add MVGC2 paths
    eng.addpath("/Users/FrancescoInnocenti/Documents/MATLAB/MVGC2/core")
    eng.addpath("/Users/FrancescoInnocenti/Documents/MATLAB/MVGC2/utils")
    eng.addpath("/Users/FrancescoInnocenti/Documents/MATLAB/MVGC2/demo")
