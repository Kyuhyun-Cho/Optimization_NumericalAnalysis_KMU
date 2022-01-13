import os
from Optimization.Adam import *
from Tester.Function import *
from Tester.UnitTest import *

init = [-2, 2]

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

test_Qualitative(Adam(), Rosenbrock_function(), init)
test_Quantitative(Adam(), Rosenbrock_function(), init)


