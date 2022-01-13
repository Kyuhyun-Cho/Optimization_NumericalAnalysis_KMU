import os
import Optimization.Optimization as Optimization
from Optimization.SGD import *
from Tester.Function import *
from Tester.UnitTest import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

init = [-2, 2]

test_Qualitative(SGD(), Rosenbrock_function(), init)
test_Quantitative(SGD(), Rosenbrock_function(), init)