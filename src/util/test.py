import numpy
from math import *
from entities.fraction import Fraction

f1 = Fraction(1, 1)
f2 = Fraction(2, 2)

f3 = f1+f2
f3.compute_extended_euclidean(1, 2)
print(f3.num)
print(f3.denom)
