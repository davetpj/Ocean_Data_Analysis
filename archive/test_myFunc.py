import numpy as np
from myFunc import *

alist = [80, 70, 90, np.NaN, 70]

weight = [15, 15, 40, 15, 15]

my_point = weight_average(alist, weight)

print(my_point)
