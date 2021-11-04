import matlab.engine

eng = matlab.engine.start_matlab()

# add MVGC2 path
eng.addpath('/Users/FrancescoInnocenti/Documents/MATLAB/MVGC2/demo')

data = eng.tnet5()
print(data)
