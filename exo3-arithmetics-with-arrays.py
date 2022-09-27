# -*- coding: utf-8 -*-

# Python packages
import numpy
import os


def run_exercise():
    """Arithmetic with numpy arrays.
    """
    nnodes = 10
    spacedim = 3
    xyz = numpy.empty((nnodes, spacedim), dtype=numpy.float64)
    for i in range(0, nnodes):
       xyz[i, 0] = i
       xyz[i, 1] = i ** 2
       xyz[i, 2] = i ** 3
    print(xyz)

    return


if __name__ == '__main__':

    run_exercise()
    print('End.')
