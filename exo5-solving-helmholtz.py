# -*- coding: utf-8 -*-
"""
..warning:: The explanations of the functions in this file and the details of
the programming methodology have been given during the lectures.
"""


# Python packages

import matplotlib.pyplot
import matplotlib.pylab
from mpl_toolkits.mplot3d import Axes3D
import numpy
import os
import scipy.io
import scipy.linalg
import scipy.sparse
import scipy.sparse.linalg
import sys

from scipy.stats import linregress
import random


# MRG packages
import zsolutions4students as solutions


def geometrical_loc(nx, ny, xmin=0.0, xmax=1.0, ymin=0.0, ymax=1.0):
    spacedim = 3
    nnodes = (nx + 1) * (ny + 1)
    node_coords = numpy.empty((nnodes, spacedim), dtype=numpy.float64)
    nodes_per_elem = 3
    nelems = nx * ny * 2-8
    p_elem2nodes = numpy.empty((nelems + 1,), dtype=numpy.int64)
    p_elem2nodes[0] = 0
    for i in range(0, nelems):
        p_elem2nodes[i + 1] = p_elem2nodes[i] + nodes_per_elem
    elem2nodes = numpy.empty((nelems * nodes_per_elem,), dtype=numpy.int64)

    # elements
    node_to_dodge = [(nx//2, ny//2), (nx//2-1, ny//2-1),
                     (nx//2-1, ny//2), (nx//2, ny//2-1)]
    k = 0
    for j in range(0, ny):
        for i in range(0, nx):
            if (i, j) not in node_to_dodge:
                elem2nodes[k + 0] = j * (nx + 1) + i
                elem2nodes[k + 1] = j * (nx + 1) + i + 1
                elem2nodes[k + 2] = (j + 1) * (nx + 1) + i + 1
                k += nodes_per_elem
                elem2nodes[k + 0] = j * (nx + 1) + i
                elem2nodes[k + 1] = (j + 1) * (nx + 1) + i + 1
                elem2nodes[k + 2] = (j + 1) * (nx + 1) + i
                k += nodes_per_elem
    # elem_type = numpy.empty((nelems,), dtype=numpy.int64)
    # elem_type[:] = VTK_TRIANGLE

    # coordinates of (nx+1)*(ny+1) nodes of cartesian grid
    k = 0
    for j in range(0, ny + 1):
        yy = ymin + (j * (ymax - ymin) / ny)
        for i in range(0, nx + 1):
            xx = xmin + (i * (xmax - xmin) / nx)
            node_coords[k, :] = xx, yy, 0.0
            k += 1

    return node_coords, p_elem2nodes, elem2nodes


def shuffle(node_coords, p_elem2nodes, elem2nodes, xmin=0.0, xmax=1.0, ymin=0.0, ymax=1.0, nelemsx=10, nelemsy=10):
    n = len(node_coords)
    for i in range(n):
        node = node_coords[i]
        x, y, z = node[0], node[1], node[2]
        ratiox, ratioy = (xmax-xmin)/nelemsx, (ymax-ymin)/nelemsy
        c1, c2 = random.choice([ratiox/3, 2*ratiox/3]
                               ), random.choice([ratioy/3, 2*ratioy/3])
        node_coords[i, :] = numpy.array([x+c1, y+c2, z])
    return node_coords, p_elem2nodes, elem2nodes


def pagged_mat(mat):
    n = mat.shape[0]
    new_mat = numpy.zeros((n+2, n+2), int)
    new_mat[1:n+1, 1:n+1] = mat
    return new_mat


def mat_res_helmholtz():
    nx, ny = 20, 20
    M = numpy.ones(18)
    M = pagged_mat(M)
    for i in range(1, 9):
        for j in range(9, 11):
            M[i, j] = 0
    for i in range(11, 20):
        for j in range(9, 11):
            M[i, j] = 0

    return M


def detect_boundary_mat(mat):
    node_to_dodge = []
    ny, nx = mat.shape[0], mat.shape[1]
    for i in range(ny):
        for j in range(nx):
            ii = ny-1-i
            jj = j
            if mat[ii, jj] == 1:
                if (jj-1 >= 0 and ii+1 < ny and mat[ii+1, jj-1] == 0) or (0 <= jj-1 and mat[ii, jj-1] == 0) or (0 <= jj-1 and 0 <= ii-1 and mat[ii-1, jj-1] == 0) or (0 <= ii-1 and mat[ii-1, jj] == 0) or (0 <= ii-1 and jj+1 < nx and mat[ii-1, jj+1] == 0) or (jj+1 < nx and mat[ii, jj+1] == 0) or (ii+1 < ny and jj+1 < nx and mat[ii+1, jj+1] == 0) or (ii+1 < ny and mat[ii+1, jj] == 0):
                    node_to_dodge.append(j*(nx+1)+i)
    return node_to_dodge


def build_matrix(mat, xmin, xmax, ymin, ymax):
    spacedim = 3
    nx, ny = mat.shape[1], mat.shape[0]
    nnodes = (nx + 1) * (ny + 1)
    node_coords = numpy.empty((nnodes, spacedim), dtype=numpy.float64)
    nodes_per_elem = 4
    nelems = mat.sum()
    p_elem2nodes = numpy.empty((nelems + 1,), dtype=numpy.int64)
    p_elem2nodes[0] = 0
    for i in range(0, nelems):
        p_elem2nodes[i + 1] = p_elem2nodes[i] + nodes_per_elem
    elem2nodes = numpy.empty((nelems * nodes_per_elem,), dtype=numpy.int64)

    # elements
    k = 0
    for j in range(0, ny):
        for i in range(0, nx):
            if mat[i][j] == 1:
                elem2nodes[k + 0] = j * (nx + 1) + i
                elem2nodes[k + 1] = j * (nx + 1) + i + 1
                elem2nodes[k + 2] = (j + 1) * (nx + 1) + i + 1
                elem2nodes[k + 3] = (j + 1) * (nx + 1) + i
                k += nodes_per_elem

    # elem_type = numpy.empty((nelems,), dtype=numpy.int64)
    # elem_type[:] = VTK_TRIANGLE

    # coordinates of (nx+1)*(ny+1) nodes of cartesian grid
    r = 0
    for j in range(0, ny+1):
        yy = ymin + (j * (ymax - ymin) / ny)
        for i in range(0, nx+1):
            xx = xmin + (i * (xmax - xmin) / nx)
            node_coords[r, :] = xx, yy, 0.0
            r += 1

    # local to global numbering
    # node_l2g = numpy.arange(0, nnodes, 1, dtype=numpy.int64)

    return node_coords, p_elem2nodes, elem2nodes


def fractalize_mat(mat):
    # always matrix = mat_test
    mat[0][2] = 1
    mat[1][3] = 0

    mat[3][0] = 1
    mat[2][1] = 0

    mat[5][3] = 1
    mat[4][2] = 0

    mat[2][5] = 1
    mat[3][4] = 0

    return mat


mat_test = numpy.array([[0, 0, 0, 0, 0, 0], [0, 1, 1, 1, 1, 0], [0, 1, 1, 1, 1, 0], [
                       0, 1, 1, 1, 1, 0], [0, 1, 1, 1, 1, 0], [0, 0, 0, 0, 0, 0]])
fractalized_mat_sample_global = fractalize_mat(mat_test)


def quadruple_mat(mat):
    n = mat.shape[0]
    new_mat = numpy.zeros((4*n, 4*n), int)
    for i in range(0, n):
        for j in range(0, n):
            if mat[i][j] == 1:
                for x in range(4):
                    for y in range(4):
                        new_mat[4*i+x][4*j+y] = 1
    return new_mat


def duplicate_mat(mat):
    n = mat.shape[0]
    new_mat = numpy.zeros((2*n, 2*n), int)
    for i in range(0, n):
        for j in range(0, n):
            if mat[i][j] == 1:
                for x in range(2):
                    for y in range(2):
                        new_mat[2*i+x][2*j+y] = 1
    return new_mat


def pagging(mat):
    n = mat.shape[0]
    new_mat = numpy.zeros((n+2, n+2), int)
    for i in range(1, n+1):
        new_mat[i][1:n+1] = mat[i-1][0:n]
    return new_mat


# def fractalize_mat_order_rec(order):
#     if order == 1:
#         return fractalized_mat_sample_global

#     fractalized_mat_sample = fractalize_mat_order_rec(order-1)
#     n = fractalized_mat_sample.shape[0]
#     fractalized_mat_sample_pag = pagging(fractalized_mat_sample)
#     isolated_ones = []
#     for i in range(n+2):
#         for j in range(n+2):
#             if 0 <= i-1 < n+2 and 0 <= i+1 < n+2 and fractalized_mat_sample_pag[i-1][j] == 0 and fractalized_mat_sample_pag[i+1][j] == 0 and fractalized_mat_sample_pag[i][j] == 1:
#                 isolated_ones.append((i, j))
#             if 0 <= j-1 < n+2 and 0 <= j+1 < n+2 and fractalized_mat_sample_pag[i][j-1] == 0 and fractalized_mat_sample_pag[i][j+1] == 0 and fractalized_mat_sample_pag[i][j] == 1:
#                 isolated_ones.append((i, j))
#     new_mat = fractalized_mat_sample
#     new_mat = pagging(new_mat)
#     new_mat = quadruple_mat(new_mat)

#     for t in isolated_ones:
#         new_mat[4*t[0]-1:4*t[0]+5, 4*t[1]-1:4*t[1]+5] = new_mat[4*t[0] -
#                                                                 1:4*t[0]+5, 4*t[1]-1:4*t[1]+5] + fractalized_mat_sample_global

#     p = new_mat.shape[0]
#     for i in range(p):
#         for j in range(p):
#             if new_mat[i][j] >= 2:
#                 new_mat[i][j] = 1
#     return new_mat
# ------------------------------------------------------
def equipe_mat(mat):
    # always matrix = fractalize(mat_test)
    M = mat.copy()
    M[1, 3] = -1
    M[2, 1] = -1
    M[3, 4] = -1
    M[4, 2] = -1

    return M


def fractalize_mat_order_rec(order):
    if order == 1:
        return fractalized_mat_sample_global

    fractalized_mat_sample = fractalize_mat_order_rec(order-1)
    n = fractalized_mat_sample.shape[0]
    fractalized_mat_sample_pag = pagging(fractalized_mat_sample)
    isolated_ones = []
    for i in range(n+2):
        for j in range(n+2):
            # if 0 <= i-1 < n+2 and 0 <= i+1 < n+2 and fractalized_mat_sample_pag[i-1][j] == 0 and fractalized_mat_sample_pag[i+1][j] == 0 and fractalized_mat_sample_pag[i][j] == 1:
            #     isolated_ones.append((i, j))
            # if 0 <= j-1 < n+2 and 0 <= j+1 < n+2 and fractalized_mat_sample_pag[i][j-1] == 0 and fractalized_mat_sample_pag[i][j+1] == 0 and fractalized_mat_sample_pag[i][j] == 1:
            #     isolated_ones.append((i, j))
            if fractalized_mat_sample_pag[i, j] == 1:
                isolated_ones.append((i, j))
    new_mat = fractalized_mat_sample
    new_mat = pagging(new_mat)
    new_mat = quadruple_mat(new_mat)

    mat_to_add = equipe_mat(fractalized_mat_sample_global)

    for t in isolated_ones:
        new_mat[4*t[0]-1:4*t[0]+5, 4*t[1]-1:4*t[1]+5] = new_mat[4*t[0] -
                                                                1:4*t[0]+5, 4*t[1]-1:4*t[1]+5] + mat_to_add

    p = new_mat.shape[0]
    for i in range(p):
        for j in range(p):
            if new_mat[i][j] >= 2:
                new_mat[i][j] = 1
    return new_mat

# ..todo: Uncomment for displaying limited digits
# numpy.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})


def split_quadrangle_into_triangle(node_coords, p_elem2nodes, elem2nodes):
    new_node_coords = node_coords
    number_elem = len(p_elem2nodes)-1
    spacedim = node_coords.shape[1]
    new_p_elem2nodes = numpy.zeros(2*number_elem+1, dtype=int)
    for i in range(0, 2*number_elem+1):
        new_p_elem2nodes[i] = 3*i
    new_elem2nodes = numpy.zeros(6*number_elem, dtype=int)
    for i in range(number_elem):
        x, y = p_elem2nodes[i], p_elem2nodes[i+1]
        u1, u2, u3 = new_p_elem2nodes[2 *
                                      i], new_p_elem2nodes[2*i+1], new_p_elem2nodes[2*i+2]
        necessary_nodes = elem2nodes[x:y]
        new_elem2nodes[u1:u2] = numpy.array(
            [necessary_nodes[0], necessary_nodes[1], necessary_nodes[2]])
        new_elem2nodes[u2:u3] = numpy.array(
            [necessary_nodes[0], necessary_nodes[2], necessary_nodes[3]])
    return new_node_coords, new_p_elem2nodes, new_elem2nodes


def run_exercise_solution_helmholtz_dddd():

    # -- set equation parameters
    wavenumber = numpy.pi
    # -- set geometry parameters
    xmin, xmax, ymin, ymax = 0.0, 1.0, 0.0, 1.0
    nelemsx, nelemsy = 20, 20

    # -- generate mesh
    nnodes = (nelemsx + 1) * (nelemsy + 1)
    nelems = nelemsx * nelemsy * 2
    node_coords, p_elem2nodes, elem2nodes, node_l2g = solutions._set_square_trimesh(
        xmin, xmax, ymin, ymax, nelemsx, nelemsy)
    # .. todo:: Modify the line below to define a different geometry.
    # p_elem2nodes, elem2nodes, node_coords, node_l2g = ...
    nnodes = node_coords.shape[0]
    nelems = len(p_elem2nodes)-1

    # -- plot mesh
    fig = matplotlib.pyplot.figure(1)
    ax = matplotlib.pyplot.subplot(1, 1, 1)
    ax.set_aspect('equal')
    ax.axis('off')
    solutions._plot_mesh(p_elem2nodes, elem2nodes, node_coords, color='orange')
    matplotlib.pyplot.show()

    # -- set boundary geometry
    # boundary composed of nodes
    # .. todo:: Modify the lines below to select the ids of the nodes on the boundary of the different geometry.
    nodes_on_north = solutions._set_square_nodes_boundary_north(node_coords)
    nodes_on_south = solutions._set_square_nodes_boundary_south(node_coords)
    nodes_on_east = solutions._set_square_nodes_boundary_east(node_coords)
    nodes_on_west = solutions._set_square_nodes_boundary_west(node_coords)
    nodes_on_boundary = numpy.unique(numpy.concatenate(
        (nodes_on_north, nodes_on_south, nodes_on_east, nodes_on_west)), )
    # ..warning: the ids of the nodes on the boundary should be 'global' number.
    # nodes_on_boundary = ...

    # ..warning: for teaching purpose only
    # -- set exact solution
    solexact = numpy.zeros((nnodes, 1), dtype=numpy.complex128)
    laplacian_of_solexact = numpy.zeros((nnodes, 1), dtype=numpy.complex128)
    for i in range(nnodes):
        x, y, z = node_coords[i, 0], node_coords[i, 1], node_coords[i, 2]
        # set: u(x,y) = e^{ikx}
        solexact[i] = numpy.exp(complex(0., 1.)*wavenumber*x)
        laplacian_of_solexact[i] = complex(
            0., 1.)*wavenumber*complex(0., 1.)*wavenumber * solexact[i]
    # ..warning: end

    # -- set dirichlet boundary conditions
    values_at_nodes_on_boundary = numpy.zeros(
        (nnodes, 1), dtype=numpy.complex128)
    values_at_nodes_on_boundary[nodes_on_boundary] = solexact[nodes_on_boundary]

    # -- set finite element matrices and right hand side
    f_unassembled = numpy.zeros((nnodes, 1), dtype=numpy.complex128)

    # ..warning: for teaching purpose only
    for i in range(nnodes):
        # evaluate: (-\Delta - k^2) u(x,y) = ...
        f_unassembled[i] = - laplacian_of_solexact[i] - \
            (wavenumber ** 2) * solexact[i]
    # ..warning: end

    coef_k = numpy.ones((nelems, 1), dtype=numpy.complex128)
    coef_m = numpy.ones((nelems, 1), dtype=numpy.complex128)
    K, M, F = solutions._set_fem_assembly(
        p_elem2nodes, elem2nodes, node_coords, f_unassembled, coef_k, coef_m)
    A = K - wavenumber**2 * M
    B = F

    # -- apply Dirichlet boundary conditions
    A, B = solutions._set_dirichlet_condition(
        nodes_on_boundary, values_at_nodes_on_boundary, A, B)

    # -- solve linear system
    sol = scipy.linalg.solve(A, B)

    # -- plot finite element solution
    solreal = sol.reshape((sol.shape[0], ))
    # _ = solutions._plot_contourf(nelems, p_elem2nodes, elem2nodes, node_coords, numpy.real(solreal))
    # _ = solutions._plot_contourf(nelems, p_elem2nodes, elem2nodes, node_coords, numpy.imag(solreal))
    #
    # ..warning: for teaching purpose only
    # -- plot exact solution
    solexactreal = solexact.reshape((solexact.shape[0], ))
    _ = solutions._plot_contourf(
        nelems, p_elem2nodes, elem2nodes, node_coords, numpy.real(solexactreal))
    _ = solutions._plot_contourf(
        nelems, p_elem2nodes, elem2nodes, node_coords, numpy.imag(solexactreal))
    # # ..warning: end

    # ..warning: for teaching purpose only
    # -- plot exact solution - approximate solution
    solerr = solreal - solexactreal
    norm_err = numpy.linalg.norm(solerr)
    _ = solutions._plot_contourf(
        nelems, p_elem2nodes, elem2nodes, node_coords, numpy.real(solerr))
    _ = solutions._plot_contourf(
        nelems, p_elem2nodes, elem2nodes, node_coords, numpy.imag(solerr))
    # # ..warning: end

    return


def geometrical_loc_sol(mat, r):
    # so that we can have more details on the the grid (un maillage plus fin)
    mat = duplicate_mat(mat)
    # -- set geometry parameters
    xmin, xmax, ymin, ymax = 0.0, 1.0, 0.0, 1.0
    nelemsx, nelemsy = mat.shape[1], mat.shape[0]
    # -- set equation parameters
    # wavenumber = numpy.pi*(nelemsx/5)
    # wavenumber = numpy.pi
    h = (ymax-ymin)/nelemsy
    wavenumber = numpy.pi/(r*h)
    # wavenumber = 1/(2*((ymax-ymin)/nelemsy)) # according to F-Simon presentation (slide13)
    # wavenumber = 1/(2*((ymax-ymin)/nelemsy))
    # -- generate mesh
    nnodes = (nelemsx + 1) * (nelemsy + 1)
    nelems = nelemsx * nelemsy * 2
    node_coords, p_elem2nodes, elem2nodes, node_l2g = solutions._set_square_trimesh(
        xmin, xmax, ymin, ymax, nelemsx, nelemsy)
    # .. todo:: Modify the line below to define a different geometry.
    # p_elem2nodes, elem2nodes, node_coords, node_l2g = ...
    nnodes = node_coords.shape[0]
    nelems = len(p_elem2nodes)-1

    # -- plot mesh
    new_node_coords, new_p_elem2nodes, new_elem2nodes = build_matrix(
        mat, xmin, xmax, ymin, ymax)
    fig = matplotlib.pyplot.figure(1)
    ax = matplotlib.pyplot.subplot(1, 1, 1)
    ax.set_aspect('equal')
    ax.axis('off')
    solutions._plot_mesh(new_p_elem2nodes, new_elem2nodes,
                         new_node_coords, color='orange')
    matplotlib.pyplot.show()

    # -- set boundary geometry
    # boundary composed of nodes
    # .. todo:: Modify the lines below to select the ids of the nodes on the boundary of the different geometry.
    # nodes_on_north = solutions._set_square_nodes_boundary_north(node_coords)
    # nodes_on_south = solutions._set_square_nodes_boundary_south(node_coords)
    # nodes_on_east = solutions._set_square_nodes_boundary_east(node_coords)
    # nodes_on_west = solutions._set_square_nodes_boundary_west(node_coords)
    node_to_dodge = detect_boundary_mat(mat)
    # in this case we should add some extra nodes
    nodes_on_boundary = numpy.unique(numpy.array(node_to_dodge), )
    # ..warning: the ids of the nodes on the boundary should be 'global' number.
    # nodes_on_boundary = ...

    # ..warning: for teaching purpose only
    # -- set exact solution
    solexact = numpy.zeros((nnodes, 1), dtype=numpy.complex128)
    laplacian_of_solexact = numpy.zeros((nnodes, 1), dtype=numpy.complex128)
    for i in range(nnodes):
        x, y, z = node_coords[i, 0], node_coords[i, 1], node_coords[i, 2]
        # set: u(x,y) = e^{ikx}
        solexact[i] = numpy.exp(complex(0., 1.)*wavenumber*x)
        laplacian_of_solexact[i] = complex(
            0., 1.)*wavenumber*complex(0., 1.)*wavenumber * solexact[i]
    # ..warning: end

    # -- set dirichlet boundary conditions
    values_at_nodes_on_boundary = numpy.zeros(
        (nnodes, 1), dtype=numpy.complex128)
    values_at_nodes_on_boundary[nodes_on_boundary] = solexact[nodes_on_boundary]

    # -- set finite element matrices and right hand side
    f_unassembled = numpy.zeros((nnodes, 1), dtype=numpy.complex128)

    # ..warning: for teaching purpose only
    for i in range(nnodes):
        # evaluate: (-\Delta - k^2) u(x,y) = ...
        f_unassembled[i] = - laplacian_of_solexact[i] - \
            (wavenumber ** 2) * solexact[i]
    # ..warning: end

    coef_k = numpy.ones((nelems, 1), dtype=numpy.complex128)
    coef_m = numpy.ones((nelems, 1), dtype=numpy.complex128)
    K, M, F = solutions._set_fem_assembly(
        p_elem2nodes, elem2nodes, node_coords, f_unassembled, coef_k, coef_m)
    A = K - wavenumber**2 * M
    B = F

    # -- apply Dirichlet boundary conditions
    A, B = solutions._set_dirichlet_condition(
        nodes_on_boundary, values_at_nodes_on_boundary, A, B)

    # -- solve linear system
    sol = scipy.linalg.solve(A, B)

    # -- plot finite element solution
    solreal = sol.reshape((sol.shape[0], ))
    _ = solutions._plot_contourf(
        nelems, p_elem2nodes, elem2nodes, node_coords, numpy.real(solreal))
    _ = solutions._plot_contourf(
        nelems, p_elem2nodes, elem2nodes, node_coords, numpy.imag(solreal))
    #
    # ..warning: for teaching purpose only
    # -- plot exact solution
    solexactreal = solexact.reshape((solexact.shape[0], ))
    # _ = solutions._plot_contourf(nelems, p_elem2nodes, elem2nodes, node_coords, numpy.real(solexactreal))
    # _ = solutions._plot_contourf(nelems, p_elem2nodes, elem2nodes, node_coords, numpy.imag(solexactreal))
    # # ..warning: end

    # ..warning: for teaching purpose only
    # -- plot exact solution - approximate solution
    solerr = solreal - solexactreal
    norm_err = numpy.linalg.norm(solerr)
    _ = solutions._plot_contourf(
        nelems, p_elem2nodes, elem2nodes, node_coords, numpy.real(solerr))
    _ = solutions._plot_contourf(
        nelems, p_elem2nodes, elem2nodes, node_coords, numpy.imag(solerr))
    # # ..warning: end
    print(nelemsx)
    return

####################################################################################################


def geometrical_loc_sol_bis(mat, r):

    # -- set geometry parameters
    xmin, xmax, ymin, ymax = 0.0, 1.0, 0.0, 1.0
    nelemsx, nelemsy = mat.shape[1], mat.shape[0]
    # -- set equation parameters
    # wavenumber = numpy.pi*(nelemsx/5)
    # wavenumber = numpy.pi
    h = (ymax-ymin)/nelemsy
    wavenumber = numpy.pi/(r*h)
    # wavenumber = 1/(2*((ymax-ymin)/nelemsy)) # according to F-Simon presentation (slide13)
    # wavenumber = 1/(2*((ymax-ymin)/nelemsy))
    # -- generate mesh
    nnodes = (nelemsx + 1) * (nelemsy + 1)
    nelems = nelemsx * nelemsy * 2
    node_coords, p_elem2nodes, elem2nodes, node_l2g = solutions._set_square_trimesh(
        xmin, xmax, ymin, ymax, nelemsx, nelemsy)
    # .. todo:: Modify the line below to define a different geometry.
    # p_elem2nodes, elem2nodes, node_coords, node_l2g = ...

    # -- plot mesh
    new_node_coords, new_p_elem2nodes, new_elem2nodes = build_matrix(
        mat, xmin, xmax, ymin, ymax)
    new_node_coords, new_p_elem2nodes, new_elem2nodes = split_quadrangle_into_triangle(
        new_node_coords, new_p_elem2nodes, new_elem2nodes)
    fig = matplotlib.pyplot.figure(1)
    ax = matplotlib.pyplot.subplot(1, 1, 1)
    ax.set_aspect('equal')
    ax.axis('off')
    solutions._plot_mesh(new_p_elem2nodes, new_elem2nodes,
                         new_node_coords, color='orange')
    matplotlib.pyplot.show()

    nnodes = node_coords.shape[0]
    nelems = len(new_p_elem2nodes)-1

    # -- set boundary geometry
    # boundary composed of nodes
    # .. todo:: Modify the lines below to select the ids of the nodes on the boundary of the different geometry.
    # nodes_on_north = solutions._set_square_nodes_boundary_north(node_coords)
    # nodes_on_south = solutions._set_square_nodes_boundary_south(node_coords)
    # nodes_on_east = solutions._set_square_nodes_boundary_east(node_coords)
    # nodes_on_west = solutions._set_square_nodes_boundary_west(node_coords)
    node_to_dodge = detect_boundary_mat(mat)
    # in this case we should add some extra nodes
    nodes_on_boundary = numpy.unique(numpy.array(node_to_dodge), )
    # ..warning: the ids of the nodes on the boundary should be 'global' number.
    # nodes_on_boundary = ...

    # ..warning: for teaching purpose only
    # -- set exact solution
    solexact = numpy.zeros((nnodes, 1), dtype=numpy.complex128)
    laplacian_of_solexact = numpy.zeros((nnodes, 1), dtype=numpy.complex128)
    for i in range(nnodes):
        x, y, z = node_coords[i, 0], node_coords[i, 1], node_coords[i, 2]
        # set: u(x,y) = e^{ikx}
        solexact[i] = numpy.exp(complex(0., 1.)*wavenumber*x)
        laplacian_of_solexact[i] = complex(
            0., 1.)*wavenumber*complex(0., 1.)*wavenumber * solexact[i]
    # ..warning: end

    # -- set dirichlet boundary conditions
    values_at_nodes_on_boundary = numpy.zeros(
        (nnodes, 1), dtype=numpy.complex128)
    values_at_nodes_on_boundary[nodes_on_boundary] = solexact[nodes_on_boundary]

    # -- set finite element matrices and right hand side
    f_unassembled = numpy.zeros((nnodes, 1), dtype=numpy.complex128)

    # ..warning: for teaching purpose only
    for i in range(nnodes):
        # evaluate: (-\Delta - k^2) u(x,y) = ...
        f_unassembled[i] = - laplacian_of_solexact[i] - \
            (wavenumber ** 2) * solexact[i]
    # ..warning: end
    # print(node_to_dodge)
    coef_k = numpy.ones((nelems, 1), dtype=numpy.complex128)
    coef_m = numpy.ones((nelems, 1), dtype=numpy.complex128)
    K, M, F = solutions._set_fem_assembly(
        p_elem2nodes, elem2nodes, node_coords, f_unassembled, coef_k, coef_m)
    A = K - wavenumber**2 * M
    B = F
    print(elem2nodes[p_elem2nodes[0]:p_elem2nodes[1]])
    # -- apply Dirichlet boundary conditions
    A, B = solutions._set_dirichlet_condition(
        nodes_on_boundary, values_at_nodes_on_boundary, A, B)

    # -- solve linear system
    sol = scipy.linalg.solve(A, B)

    # -- plot finite element solution
    solreal = sol.reshape((sol.shape[0], ))
    # _ = solutions._plot_contourf(
    #     nelems, p_elem2nodes, elem2nodes, node_coords, numpy.real(solreal))
    # _ = solutions._plot_contourf(
    #     nelems, p_elem2nodes, elem2nodes, node_coords, numpy.imag(solreal))
    #
    # ..warning: for teaching purpose only
    # -- plot exact solution
    solexactreal = solexact.reshape((solexact.shape[0], ))
    # _ = solutions._plot_contourf(nelems, p_elem2nodes, elem2nodes, node_coords, numpy.real(solexactreal))
    # _ = solutions._plot_contourf(nelems, p_elem2nodes, elem2nodes, node_coords, numpy.imag(solexactreal))
    # # ..warning: end

    # ..warning: for teaching purpose only
    # -- plot exact solution - approximate solution
    solerr = solreal - solexactreal
    norm_err = numpy.linalg.norm(solerr)
    # _ = solutions._plot_contourf(
    #     nelems, p_elem2nodes, elem2nodes, node_coords, numpy.real(solerr))
    # _ = solutions._plot_contourf(
    #     nelems, p_elem2nodes, elem2nodes, node_coords, numpy.imag(solerr))
    # # ..warning: end

    return
####################################################################################################


def find_alpha():
    error_values = []
    nelem_values = [5, 10, 15, 20, 25, 30]
    h_values = [1/x for x in nelem_values]
    nelem_values_log = [numpy.log(x) for x in nelem_values]
    h_values_log = [numpy.log(x) for x in h_values]
    for nelem in nelem_values:
        wavenumber = numpy.pi
        # -- set geometry parameters
        xmin, xmax, ymin, ymax = 0.0, 1.0, 0.0, 1.0
        nelemsx, nelemsy = nelem, nelem

        # -- generate mesh
        nnodes = (nelemsx + 1) * (nelemsy + 1)
        nelems = nelemsx * nelemsy * 2
        node_coords, p_elem2nodes, elem2nodes, node_l2g = solutions._set_square_trimesh(
            xmin, xmax, ymin, ymax, nelemsx, nelemsy)
        # .. todo:: Modify the line below to define a different geometry.
        # p_elem2nodes, elem2nodes, node_coords, node_l2g = ...
        nnodes = node_coords.shape[0]
        nelems = len(p_elem2nodes)-1

        # -- set boundary geometry
        # boundary composed of nodes
        # .. todo:: Modify the lines below to select the ids of the nodes on the boundary of the different geometry.
        nodes_on_north = solutions._set_square_nodes_boundary_north(
            node_coords)
        nodes_on_south = solutions._set_square_nodes_boundary_south(
            node_coords)
        nodes_on_east = solutions._set_square_nodes_boundary_east(node_coords)
        nodes_on_west = solutions._set_square_nodes_boundary_west(node_coords)
        nodes_on_boundary = numpy.unique(numpy.concatenate(
            (nodes_on_north, nodes_on_south, nodes_on_east, nodes_on_west)), )
        # ..warning: the ids of the nodes on the boundary should be 'global' number.
        # nodes_on_boundary = ...

        # ..warning: for teaching purpose only
        # -- set exact solution
        solexact = numpy.zeros((nnodes, 1), dtype=numpy.complex128)
        laplacian_of_solexact = numpy.zeros(
            (nnodes, 1), dtype=numpy.complex128)
        for i in range(nnodes):
            x, y, z = node_coords[i, 0], node_coords[i, 1], node_coords[i, 2]
            # set: u(x,y) = e^{ikx}
            solexact[i] = numpy.exp(complex(0., 1.)*wavenumber*x)
            laplacian_of_solexact[i] = complex(
                0., 1.)*wavenumber*complex(0., 1.)*wavenumber * solexact[i]
        # ..warning: end

        # -- set dirichlet boundary conditions
        values_at_nodes_on_boundary = numpy.zeros(
            (nnodes, 1), dtype=numpy.complex128)
        values_at_nodes_on_boundary[nodes_on_boundary] = solexact[nodes_on_boundary]

        # -- set finite element matrices and right hand side
        f_unassembled = numpy.zeros((nnodes, 1), dtype=numpy.complex128)

        # ..warning: for teaching purpose only
        for i in range(nnodes):
            # evaluate: (-\Delta - k^2) u(x,y) = ...
            f_unassembled[i] = - laplacian_of_solexact[i] - \
                (wavenumber ** 2) * solexact[i]
        # ..warning: end

        coef_k = numpy.ones((nelems, 1), dtype=numpy.complex128)
        coef_m = numpy.ones((nelems, 1), dtype=numpy.complex128)
        K, M, F = solutions._set_fem_assembly(
            p_elem2nodes, elem2nodes, node_coords, f_unassembled, coef_k, coef_m)
        A = K - wavenumber**2 * M
        B = F

        # -- apply Dirichlet boundary conditions
        A, B = solutions._set_dirichlet_condition(
            nodes_on_boundary, values_at_nodes_on_boundary, A, B)

        # -- solve linear system
        sol = scipy.linalg.solve(A, B)

        # -- plot finite element solution
        solreal = sol.reshape((sol.shape[0], ))

        solexactreal = solexact.reshape((solexact.shape[0], ))

        solerr = solreal - solexactreal
        norm_err = numpy.linalg.norm(solerr)
        error_values.append(norm_err)
    error_values_log = [numpy.log(x) for x in error_values]
    slope, intercept, r_value, p_value, std_err = linregress(
        h_values_log, error_values_log)
    matplotlib.pyplot.plot(h_values_log, error_values_log)
    matplotlib.pyplot.xlabel("nelem values (log scale)")
    matplotlib.pyplot.ylabel("error values (log scale)")
    matplotlib.pyplot.title("slope : {}".format(slope))
    matplotlib.pyplot.show()
    return


def find_beta():
    error_values = []
    k_values = [x*numpy.pi for x in [0.5, 1, 2, 3, 4, 5]]
    k_values_log = [numpy.log(x) for x in k_values]
    for k in k_values:
        wavenumber = k
        # -- set geometry parameters
        xmin, xmax, ymin, ymax = 0.0, 1.0, 0.0, 1.0
        nelemsx, nelemsy = 10, 10

        # -- generate mesh
        nnodes = (nelemsx + 1) * (nelemsy + 1)
        nelems = nelemsx * nelemsy * 2
        node_coords, p_elem2nodes, elem2nodes, node_l2g = solutions._set_square_trimesh(
            xmin, xmax, ymin, ymax, nelemsx, nelemsy)
        # .. todo:: Modify the line below to define a different geometry.
        # p_elem2nodes, elem2nodes, node_coords, node_l2g = ...
        nnodes = node_coords.shape[0]
        nelems = len(p_elem2nodes)-1

        # -- set boundary geometry
        # boundary composed of nodes
        # .. todo:: Modify the lines below to select the ids of the nodes on the boundary of the different geometry.
        nodes_on_north = solutions._set_square_nodes_boundary_north(
            node_coords)
        nodes_on_south = solutions._set_square_nodes_boundary_south(
            node_coords)
        nodes_on_east = solutions._set_square_nodes_boundary_east(node_coords)
        nodes_on_west = solutions._set_square_nodes_boundary_west(node_coords)
        nodes_on_boundary = numpy.unique(numpy.concatenate(
            (nodes_on_north, nodes_on_south, nodes_on_east, nodes_on_west)), )
        # ..warning: the ids of the nodes on the boundary should be 'global' number.
        # nodes_on_boundary = ...

        # ..warning: for teaching purpose only
        # -- set exact solution
        solexact = numpy.zeros((nnodes, 1), dtype=numpy.complex128)
        laplacian_of_solexact = numpy.zeros(
            (nnodes, 1), dtype=numpy.complex128)
        for i in range(nnodes):
            x, y, z = node_coords[i, 0], node_coords[i, 1], node_coords[i, 2]
            # set: u(x,y) = e^{ikx}
            solexact[i] = numpy.exp(complex(0., 1.)*wavenumber*x)
            laplacian_of_solexact[i] = complex(
                0., 1.)*wavenumber*complex(0., 1.)*wavenumber * solexact[i]
        # ..warning: end

        # -- set dirichlet boundary conditions
        values_at_nodes_on_boundary = numpy.zeros(
            (nnodes, 1), dtype=numpy.complex128)
        values_at_nodes_on_boundary[nodes_on_boundary] = solexact[nodes_on_boundary]

        # -- set finite element matrices and right hand side
        f_unassembled = numpy.zeros((nnodes, 1), dtype=numpy.complex128)

        # ..warning: for teaching purpose only
        for i in range(nnodes):
            # evaluate: (-\Delta - k^2) u(x,y) = ...
            f_unassembled[i] = - laplacian_of_solexact[i] - \
                (wavenumber ** 2) * solexact[i]
        # ..warning: end

        coef_k = numpy.ones((nelems, 1), dtype=numpy.complex128)
        coef_m = numpy.ones((nelems, 1), dtype=numpy.complex128)
        K, M, F = solutions._set_fem_assembly(
            p_elem2nodes, elem2nodes, node_coords, f_unassembled, coef_k, coef_m)
        A = K - wavenumber**2 * M
        B = F

        # -- apply Dirichlet boundary conditions
        A, B = solutions._set_dirichlet_condition(
            nodes_on_boundary, values_at_nodes_on_boundary, A, B)

        # -- solve linear system
        sol = scipy.linalg.solve(A, B)

        # -- plot finite element solution
        solreal = sol.reshape((sol.shape[0], ))

        solexactreal = solexact.reshape((solexact.shape[0], ))

        solerr = solreal - solexactreal
        norm_err = numpy.linalg.norm(solerr)
        error_values.append(norm_err)
    error_values_log = [numpy.log(x) for x in error_values]
    slope, intercept, r_value, p_value, std_err = linregress(
        k_values_log, error_values_log)
    matplotlib.pyplot.plot(k_values_log, error_values_log)
    matplotlib.pyplot.xlabel("k values (log scale)")
    matplotlib.pyplot.ylabel("h values (log scale)")
    matplotlib.pyplot.title("slope : {}".format(slope))
    matplotlib.pyplot.show()
    return


def compute_distance(node1, node2):
    return numpy.sqrt((node1[0]-node2[0])**2 + (node1[1]-node2[1])**2 + (node1[2]-node2[2])**2)


def triangle_area(a, b, c):
    semi_perim = (a+b+c)/2
    return numpy.square(semi_perim*(semi_perim-a)*(semi_perim-b)*(semi_perim-c))


def radius_inscribed(a, b, c):
    s = (a+b+c)/2
    rho = numpy.sqrt((s-a)*(s-b)*(s-c)/s)
    return rho


def radius_circumscribed(a, b, c):
    area = triangle_area(a, b, c)
    return (a*b*c)/(4*area)


def compute_h_for_triangle(a, b, c, choice):
    # choice = int(input("0 for hmax,, 1 for haverage, 2 rayon du cercle inscrit, 3 rayon interieur/exterieur"))
    if choice == 0:
        return max(a, b, c)
    if choice == 1:
        return (a+b+c)/3
    if choice == 2:
        return (6*radius_inscribed(a, b, c))/numpy.square(3)
    if choice == 3:
        return (radius_circumscribed(a, b, c) / radius_inscribed(a, b, c))


def compute_h_for_grid(node_coords, p_elem2nodes, elem2nodes, choice):
    number_elem = len(p_elem2nodes)-1
    spacedim = node_coords.shape[1]
    elem_quality = numpy.empty(number_elem, dtype=numpy.float64)
    h_values = []
    for i in range(number_elem):
        x, y = p_elem2nodes[i], p_elem2nodes[i+1]
        necessary_nodes = elem2nodes[x:y]
        a, b, c = compute_distance(node_coords[necessary_nodes[0]], node_coords[necessary_nodes[1]]), compute_distance(
            node_coords[necessary_nodes[1]], node_coords[necessary_nodes[2]]), compute_distance(node_coords[necessary_nodes[2]], node_coords[necessary_nodes[0]])
        h_values.append(compute_h_for_triangle(a, b, c, choice))
    return (min(h_values), max(h_values), sum(h_values)/len(h_values))


def solve_given_grid(node_coords, p_elem2nodes, elem2nodes):
    nnodes = node_coords.shape[0]
    nelems = len(p_elem2nodes)-1

    wavenumber = numpy.pi
    # -- set boundary geometry
    # boundary composed of nodes
    # .. todo:: Modify the lines below to select the ids of the nodes on the boundary of the different geometry.
    nodes_on_north = solutions._set_square_nodes_boundary_north(node_coords)
    nodes_on_south = solutions._set_square_nodes_boundary_south(node_coords)
    nodes_on_east = solutions._set_square_nodes_boundary_east(node_coords)
    nodes_on_west = solutions._set_square_nodes_boundary_west(node_coords)
    nodes_on_boundary = numpy.unique(numpy.concatenate(
        (nodes_on_north, nodes_on_south, nodes_on_east, nodes_on_west)), )
    # ..warning: the ids of the nodes on the boundary should be 'global' number.
    # nodes_on_boundary = ...

    # ..warning: for teaching purpose only
    # -- set exact solution
    solexact = numpy.zeros((nnodes, 1), dtype=numpy.complex128)
    laplacian_of_solexact = numpy.zeros((nnodes, 1), dtype=numpy.complex128)
    for i in range(nnodes):
        x, y, z = node_coords[i, 0], node_coords[i, 1], node_coords[i, 2]
        # set: u(x,y) = e^{ikx}
        solexact[i] = numpy.exp(complex(0., 1.)*wavenumber*x)
        laplacian_of_solexact[i] = complex(
            0., 1.)*wavenumber*complex(0., 1.)*wavenumber * solexact[i]
    # ..warning: end

    # -- set dirichlet boundary conditions
    values_at_nodes_on_boundary = numpy.zeros(
        (nnodes, 1), dtype=numpy.complex128)
    values_at_nodes_on_boundary[nodes_on_boundary] = solexact[nodes_on_boundary]

    # -- set finite element matrices and right hand side
    f_unassembled = numpy.zeros((nnodes, 1), dtype=numpy.complex128)

    # ..warning: for teaching purpose only
    for i in range(nnodes):
        # evaluate: (-\Delta - k^2) u(x,y) = ...
        f_unassembled[i] = - laplacian_of_solexact[i] - \
            (wavenumber ** 2) * solexact[i]
    # ..warning: end

    coef_k = numpy.ones((nelems, 1), dtype=numpy.complex128)
    coef_m = numpy.ones((nelems, 1), dtype=numpy.complex128)
    K, M, F = solutions._set_fem_assembly(
        p_elem2nodes, elem2nodes, node_coords, f_unassembled, coef_k, coef_m)
    A = K - wavenumber**2 * M
    B = F

    # -- apply Dirichlet boundary conditions
    A, B = solutions._set_dirichlet_condition(
        nodes_on_boundary, values_at_nodes_on_boundary, A, B)

    # -- solve linear system
    sol = scipy.linalg.solve(A, B)

    # -- plot finite element solution
    solreal = sol.reshape((sol.shape[0], ))

    solexactreal = solexact.reshape((solexact.shape[0], ))

    solerr = solreal - solexactreal
    norm_err = numpy.linalg.norm(solerr)
    return numpy.log(norm_err)


def find_alpha_2(choice):
    error_values = []
    h_values, h_values1, h_values2 = [], [], []
    wavenumber = numpy.pi
    # -- set geometry parameters
    xmin, xmax, ymin, ymax = 0.0, 1.0, 0.0, 1.0
    nelemsx, nelemsy = 10, 10

    # -- generate mesh
    nnodes = (nelemsx + 1) * (nelemsy + 1)
    nelems = nelemsx * nelemsy * 2
    node_coords, p_elem2nodes, elem2nodes, node_l2g = solutions._set_square_trimesh(
        xmin, xmax, ymin, ymax, nelemsx, nelemsy)
    # -- generate other meshes by shuffling the original one
    node_coords1, p_elem2nodes1, elem2nodes1, node_l2g1 = solutions._set_square_trimesh(
        xmin, xmax, ymin, ymax, 15, 15)
    node_coords2, p_elem2nodes2, elem2nodes2, node_l2g2 = solutions._set_square_trimesh(
        xmin, xmax, ymin, ymax, 20, 20)
    # --

    h_maillage_grid = [compute_h_for_grid(node_coords, p_elem2nodes, elem2nodes, choice), compute_h_for_grid(
        node_coords1, p_elem2nodes1, elem2nodes1, choice), compute_h_for_grid(node_coords2, p_elem2nodes2, elem2nodes2, choice)]
    h_values = [h_maillage[0] for h_maillage in h_maillage_grid]
    h_values1 = [h_maillage[1] for h_maillage in h_maillage_grid]
    h_values2 = [h_maillage[2] for h_maillage in h_maillage_grid]
    h_values_log = [numpy.log(x) for x in h_values]
    h_values_log1 = [numpy.log(x) for x in h_values1]
    h_values_log2 = [numpy.log(x) for x in h_values2]

    error_values = [solve_given_grid(node_coords, p_elem2nodes, elem2nodes), solve_given_grid(
        node_coords1, p_elem2nodes1, elem2nodes1), solve_given_grid(node_coords2, p_elem2nodes2, elem2nodes2)]

    # slope, intercept, r_value, p_value, std_err = linregress(h_values_log, error_values_log)
    matplotlib.pyplot.plot(h_values_log, error_values)
    matplotlib.pyplot.plot(h_values_log1, error_values)
    matplotlib.pyplot.plot(h_values_log2, error_values)
    matplotlib.pyplot.xlabel("h values (log scale)")
    matplotlib.pyplot.ylabel("error values (log scale)")
    # matplotlib.pyplot.title("slope : {}".format(slope))
    matplotlib.pyplot.show()
    return


def find_alpha_3(choice):
    error_values = []
    nelem_values = [5, 10, 15, 20, 25, 30]
    h_values1, h_values2, h_values3 = [], [], []
    nelem_values_log = [numpy.log(x) for x in nelem_values]
    # h_values_log = [numpy.log(x) for x in h_values]
    for nelem in nelem_values:

        wavenumber = numpy.pi
        # -- set geometry parameters
        xmin, xmax, ymin, ymax = 0.0, 1.0, 0.0, 1.0
        nelemsx, nelemsy = nelem, nelem

        # -- generate mesh
        nnodes = (nelemsx + 1) * (nelemsy + 1)
        nelems = nelemsx * nelemsy * 2
        node_coords, p_elem2nodes, elem2nodes, node_l2g = solutions._set_square_trimesh(
            xmin, xmax, ymin, ymax, nelemsx, nelemsy)
        # .. todo:: Modify the line below to define a different geometry.
        # p_elem2nodes, elem2nodes, node_coords, node_l2g = ...
        nnodes = node_coords.shape[0]
        nelems = len(p_elem2nodes)-1

        h_values1.append(compute_h_for_grid(
            node_coords, p_elem2nodes, elem2nodes, choice)[0])
        h_values2.append(compute_h_for_grid(
            node_coords, p_elem2nodes, elem2nodes, choice)[1])
        h_values3.append(compute_h_for_grid(
            node_coords, p_elem2nodes, elem2nodes, choice)[2])
        # -- set boundary geometry
        # boundary composed of nodes
        # .. todo:: Modify the lines below to select the ids of the nodes on the boundary of the different geometry.
        nodes_on_north = solutions._set_square_nodes_boundary_north(
            node_coords)
        nodes_on_south = solutions._set_square_nodes_boundary_south(
            node_coords)
        nodes_on_east = solutions._set_square_nodes_boundary_east(node_coords)
        nodes_on_west = solutions._set_square_nodes_boundary_west(node_coords)
        nodes_on_boundary = numpy.unique(numpy.concatenate(
            (nodes_on_north, nodes_on_south, nodes_on_east, nodes_on_west)), )
        # ..warning: the ids of the nodes on the boundary should be 'global' number.
        # nodes_on_boundary = ...

        # ..warning: for teaching purpose only
        # -- set exact solution
        solexact = numpy.zeros((nnodes, 1), dtype=numpy.complex128)
        laplacian_of_solexact = numpy.zeros(
            (nnodes, 1), dtype=numpy.complex128)
        for i in range(nnodes):
            x, y, z = node_coords[i, 0], node_coords[i, 1], node_coords[i, 2]
            # set: u(x,y) = e^{ikx}
            solexact[i] = numpy.exp(complex(0., 1.)*wavenumber*x)
            laplacian_of_solexact[i] = complex(
                0., 1.)*wavenumber*complex(0., 1.)*wavenumber * solexact[i]
        # ..warning: end

        # -- set dirichlet boundary conditions
        values_at_nodes_on_boundary = numpy.zeros(
            (nnodes, 1), dtype=numpy.complex128)
        values_at_nodes_on_boundary[nodes_on_boundary] = solexact[nodes_on_boundary]

        # -- set finite element matrices and right hand side
        f_unassembled = numpy.zeros((nnodes, 1), dtype=numpy.complex128)

        # ..warning: for teaching purpose only
        for i in range(nnodes):
            # evaluate: (-\Delta - k^2) u(x,y) = ...
            f_unassembled[i] = - laplacian_of_solexact[i] - \
                (wavenumber ** 2) * solexact[i]
        # ..warning: end

        coef_k = numpy.ones((nelems, 1), dtype=numpy.complex128)
        coef_m = numpy.ones((nelems, 1), dtype=numpy.complex128)
        K, M, F = solutions._set_fem_assembly(
            p_elem2nodes, elem2nodes, node_coords, f_unassembled, coef_k, coef_m)
        A = K - wavenumber**2 * M
        B = F

        # -- apply Dirichlet boundary conditions
        A, B = solutions._set_dirichlet_condition(
            nodes_on_boundary, values_at_nodes_on_boundary, A, B)

        # -- solve linear system
        sol = scipy.linalg.solve(A, B)

        # -- plot finite element solution
        solreal = sol.reshape((sol.shape[0], ))

        solexactreal = solexact.reshape((solexact.shape[0], ))

        solerr = solreal - solexactreal
        norm_err = numpy.linalg.norm(solerr)
        error_values.append(norm_err)
    error_values_log = [numpy.log(x) for x in error_values]
    h_values1_log = [numpy.log(x) for x in h_values1]
    h_values2_log = [numpy.log(x) for x in h_values2]
    h_values3_log = [numpy.log(x) for x in h_values3]

    slope1, intercept1, r_value1, p_value1, std_err1 = linregress(
        h_values1_log, error_values_log)
    slope2, intercept2, r_value2, p_value2, std_err2 = linregress(
        h_values2_log, error_values_log)
    slope3, intercept3, r_value3, p_value3, std_err3 = linregress(
        h_values2_log, error_values_log)
    matplotlib.pyplot.plot(h_values1_log, error_values_log,
                           label="{}".format(slope1))
    matplotlib.pyplot.plot(h_values2_log, error_values_log,
                           label="{}".format(slope2))
    matplotlib.pyplot.plot(h_values3_log, error_values_log,
                           label="{}".format(slope3))
    matplotlib.pyplot.xlabel("nelem values (log scale)")
    matplotlib.pyplot.ylabel("error values (log scale)")
    # matplotlib.pyplot.title("slope : {}".format(slope))
    matplotlib.pyplot.legend()
    matplotlib.pyplot.show()
    return
    # print(h_values1_log, h_values2_log, h_values3_log)


def eig_for_given_solution(mat, r):
    xmin, xmax, ymin, ymax = 0.0, 1.0, 0.0, 1.0
    nelemsx, nelemsy = mat.shape[1], mat.shape[0]
    nnodes = (nelemsx + 1) * (nelemsy + 1)
    nelems = nelemsx * nelemsy * 2
    node_coords, p_elem2nodes, elem2nodes, node_l2g = solutions._set_square_trimesh(
        xmin, xmax, ymin, ymax, nelemsx, nelemsy)

    nnodes = node_coords.shape[0]
    nelems = len(p_elem2nodes)-1

    # -- plot mesh
    new_node_coords, new_p_elem2nodes, new_elem2nodes = build_matrix(
        mat, xmin, xmax, ymin, ymax)

    nnodes = node_coords.shape[0]
    nelems = len(p_elem2nodes)-1
    h = (ymax-ymin)/nelemsy
    wavenumber = numpy.pi/(r*h)
    # -- set boundary geometry
    # boundary composed of nodes
    node_to_dodge = detect_boundary_mat(mat)
    # in this case we should add some extra nodes
    nodes_on_boundary = numpy.unique(numpy.array(node_to_dodge), )

    # ..warning: for teaching purpose only
    # -- set exact solution
    solexact = numpy.zeros((nnodes, 1), dtype=numpy.complex128)
    laplacian_of_solexact = numpy.zeros((nnodes, 1), dtype=numpy.complex128)
    for i in range(nnodes):
        x, y, z = node_coords[i, 0], node_coords[i, 1], node_coords[i, 2]
        # set: u(x,y) = e^{ikx}
        solexact[i] = numpy.exp(complex(0., 1.)*wavenumber*x)
        laplacian_of_solexact[i] = complex(
            0., 1.)*wavenumber*complex(0., 1.)*wavenumber * solexact[i]
    # ..warning: end

    # -- set dirichlet boundary conditions
    values_at_nodes_on_boundary = numpy.zeros(
        (nnodes, 1), dtype=numpy.complex128)
    values_at_nodes_on_boundary[nodes_on_boundary] = solexact[nodes_on_boundary]

    # -- set finite element matrices and right hand side
    f_unassembled = numpy.zeros((nnodes, 1), dtype=numpy.complex128)

    # ..warning: for teaching purpose only
    for i in range(nnodes):
        # evaluate: (-\Delta - k^2) u(x,y) = ...
        f_unassembled[i] = - laplacian_of_solexact[i] - \
            (wavenumber ** 2) * solexact[i]
    # ..warning: end

    coef_k = numpy.ones((nelems, 1), dtype=numpy.complex128)
    coef_m = numpy.ones((nelems, 1), dtype=numpy.complex128)
    K, M, F = solutions._set_fem_assembly(
        p_elem2nodes, elem2nodes, node_coords, f_unassembled, coef_k, coef_m)
    A = K - wavenumber**2 * M
    B = F

    # -- apply Dirichlet boundary conditions
    A, B = solutions._set_dirichlet_condition(
        nodes_on_boundary, values_at_nodes_on_boundary, A, B)

    # -- solve linear system
    sol = scipy.linalg.solve(A, B)

    # -- plot finite element solution
    solreal = sol.reshape((sol.shape[0], ))

    solexactreal = solexact.reshape((solexact.shape[0], ))

    solerr = solreal - solexactreal
    norm_err = numpy.linalg.norm(solerr)
    spectrum = numpy.linalg.eigvals(A)
    spectrum = numpy.unique(spectrum)
    im_spectrum = numpy.real(sol)/numpy.pi
    re_spectrum = numpy.imag(sol)/numpy.pi
    im_spectrum = numpy.sort(im_spectrum, axis=None)
    re_spectrum = numpy.sort(re_spectrum, axis=None)
    matplotlib.pyplot.plot(re_spectrum, im_spectrum)
    matplotlib.pyplot.show()
    # print(sol.shape)
    return


if __name__ == '__main__':

    # run_exercise_solution_helmholtz_dddd()
    geometrical_loc_sol(fractalize_mat_order_rec(2), 2)
    # geometrical_loc_sol(mat_res_helmholtz(), 5)
    # eig_for_given_solution(fractalize_mat_order_rec(1), 5)
    # find_alpha()
    # find_beta()
    # find_alpha_2()
    # find_alpha_3(0)
    print('End.')
