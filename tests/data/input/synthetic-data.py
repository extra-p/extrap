#!/usr/bin/env python

"""
This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)

Copyright (c) 2020,
Technische Universitaet Darmstadt, Germany

This software may be modified and distributed under the terms of
a BSD-style license. See the LICENSE file in the base
directory for details.
"""
from math import pow, log
import random

"""
This program can be used to generate synthetic data inputs to test the multiparameter model creator of ExtraP.
"""

JITTER = 0.02  # 2% noise


def log2(x):
    return log(x, 2)


class Function:
    c_0 = 0.0
    c_1 = 0.0
    c_2 = 0.0

    @staticmethod
    def getRandomCoefficient():
        rand = random.random()  # in [0.0, 1.0)
        c = (rand * 5) - 2  # in [-2.0, 3.0)
        return 10 ** c

    def eval(self, x, y, z):
        return self.c_0 + self.eval1(x, y, z) + self.eval2(x, y, z)

    def eval_with_noise(self, x, y, z):
        value = self.eval(x, y, z)
        rand = random.random()  # in [0.0, 1.0)
        rand = rand * 2.0 - 1.0  # scale to [-1.0, 1.0)
        return value * (1 + rand * JITTER)

    def eval1(self, x, y, z):
        return self.c_1 * (x ** 2 * log2(y) ** 1)

    def eval2(self, x, y, z):
        return self.c_2 * (x ** 1 * y ** 2 * z ** 1)

    @staticmethod
    def getFunc(nparams):
        f = Function()
        f.c_0 = 1.0
        f.c_1 = 0.2
        if nparams < 3:
            f.c_2 = 0
        else:
            f.c_2 = 0.1

        return f


nparams = 3
reps = 5

x_range = [20, 30, 40, 50, 60]
y_range = [2]
z_range = [1]

if nparams > 1:
    y_range = [1, 2, 3, 4, 5]
if nparams > 2:
    z_range = [100, 200, 300, 400, 500]

data_file = "input_data_" + str(nparams) + "p.txt"
outfile = open(data_file, 'w')

outfile.write("PARAMETER x\n")
if nparams > 1:
    outfile.write("PARAMETER y\n")
if nparams > 2:
    outfile.write("PARAMETER z\n")
outfile.write("\n")

for z in z_range:
    for x in x_range:
        outfile.write("POINTS")
        for y in y_range:
            if nparams == 1:
                outfile.write(" ( {} )".format(x))
            if nparams == 2:
                outfile.write(" ( {} {} )".format(x, y))
            if nparams == 3:
                outfile.write(" ( {} {} {} )".format(x, y, z))
        outfile.write("\n")
    outfile.write("\n")

outfile.write("REGION reg\n")
outfile.write("METRIC metr\n")

f = Function.getFunc(nparams)

for z in z_range:
    for x in x_range:
        for y in y_range:
            outfile.write("DATA")
            for i in range(1, reps + 1):
                outfile.write(" {}".format(f.eval_with_noise(x, y, z)))
            outfile.write("\n")

outfile.close()
