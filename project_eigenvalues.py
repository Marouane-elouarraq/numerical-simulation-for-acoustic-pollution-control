# Python packages
from platform import node
from re import I
import matplotlib.pyplot
import matplotlib.pylab
from mpl_toolkits.mplot3d import Axes3D
import numpy
import random
import os
from pyparsing import java_style_comment
import scipy.io
import scipy.linalg
import scipy.sparse
import scipy.sparse.linalg
import sys
from math import floor
from scipy.stats import linregress
import random

# MRG packages
import zsolutions4students as solutions
# ---------------------------------------------------------------------------------------
# Question 1:


def build_matrix(mat, xmin, xmax, ymin, ymax):
    '''function that build a mesh given the input matrix (that's to say: transforming a matrix into a mesh).
    the input matrix contains zeros and ones, the presence of 1 at position (i,j) means the presence 
    of an element in the output grid at the position (i,j)
    '''
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

    #
    k = 0
    for j in range(0, ny):
        for i in range(0, nx):
            if mat[i][j] == 1:
                elem2nodes[k + 0] = j * (nx + 1) + i
                elem2nodes[k + 1] = j * (nx + 1) + i + 1
                elem2nodes[k + 2] = (j + 1) * (nx + 1) + i + 1
                elem2nodes[k + 3] = (j + 1) * (nx + 1) + i
                k += nodes_per_elem
    #
    r = 0
    for j in range(0, ny+1):
        yy = ymin + (j * (ymax - ymin) / ny)
        for i in range(0, nx+1):
            xx = xmin + (i * (xmax - xmin) / nx)
            node_coords[r, :] = xx, yy, 0.0
            r += 1

    return node_coords, p_elem2nodes, elem2nodes


def fractalize_mat(mat):
    '''this function fractalise mat_test (the matrix given below) to the first order
    this matrix will be useful later'''
    mat[0][2] = 1
    mat[1][3] = 0

    mat[3][0] = 1
    mat[2][1] = 0

    mat[5][3] = 1
    mat[4][2] = 0

    mat[2][5] = 1
    mat[3][4] = 0

    return mat


def equipe_mat(mat):
    # always matrix = fractalize(mat_test)
    M = mat.copy()
    M[1, 3] = -1
    M[2, 1] = -1
    M[3, 4] = -1
    M[4, 2] = -1

    return M


mat_test = numpy.array([[0, 0, 0, 0, 0, 0], [0, 1, 1, 1, 1, 0], [0, 1, 1, 1, 1, 0], [
                       0, 1, 1, 1, 1, 0], [0, 1, 1, 1, 1, 0], [0, 0, 0, 0, 0, 0]])
fractalized_mat_sample_global = fractalize_mat(mat_test)


def quadruple_mat(mat):
    '''this function transform an nxn matrix into a 4nx4n matrix such that every element in the initial
    matrix will give birth of 4x4 matrix containing only this element'''
    n = mat.shape[0]
    new_mat = numpy.zeros((4*n, 4*n), int)
    for i in range(0, n):
        for j in range(0, n):
            if mat[i][j] == 1:
                for x in range(4):
                    for y in range(4):
                        new_mat[4*i+x][4*j+y] = 1
    return new_mat


def padding(mat):
    '''this function consist of adding zeros around the input matrix.'''
    n = mat.shape[0]
    new_mat = numpy.zeros((n+2, n+2), int)
    for i in range(1, n+1):
        new_mat[i][1:n+1] = mat[i-1][0:n]
    return new_mat


def padded_mat(mat):
    n = mat.shape[0]
    new_mat = numpy.zeros((n+2, n+2), int)
    new_mat[1:n+1, 1:n+1] = mat
    return new_mat


def fractalize_mat_order_rec(order):
    if order == 1:
        return fractalized_mat_sample_global

    fractalized_mat_sample = fractalize_mat_order_rec(order-1)
    n = fractalized_mat_sample.shape[0]
    fractalized_mat_sample_pag = padding(fractalized_mat_sample)
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
    new_mat = padding(new_mat)
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


def mat_res_helmholtz():
    nx, ny = 20, 20
    M = numpy.ones(18)
    M = padded_mat(M)
    for i in range(1, 9):
        for j in range(9, 11):
            M[i, j] = 0
    for i in range(11, 20):
        for j in range(9, 11):
            M[i, j] = 0

    return M


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


def draw_fractal(mat, xmin, xmax, ymin, ymax, color='blue'):
    node_coords, p_elem2nodes, elem2nodes = build_matrix(
        mat, xmin, xmax, ymin, ymax)
    node_coords, p_elem2nodes, elem2nodes = split_quadrangle_into_triangle(
        node_coords, p_elem2nodes, elem2nodes)
    fig = matplotlib.pyplot.figure(1)
    ax = matplotlib.pyplot.subplot(1, 1, 1)
    nnodes = numpy.shape(node_coords)[0]
    nelems = numpy.shape(p_elem2nodes)[0]
    for elem in range(0, nelems-1):
        xyz = node_coords[elem2nodes[p_elem2nodes[elem]
            :p_elem2nodes[elem+1]], :]
        if xyz.shape[0] == 3:
            matplotlib.pyplot.plot((xyz[0, 0], xyz[1, 0], xyz[2, 0], xyz[0, 0]),
                                   (xyz[0, 1], xyz[1, 1], xyz[2, 1], xyz[0, 1]), color=color)
        else:
            matplotlib.pyplot.plot((xyz[0, 0], xyz[1, 0], xyz[2, 0], xyz[3, 0], xyz[0, 0]),
                                   (xyz[0, 1], xyz[1, 1], xyz[2, 1], xyz[3, 1], xyz[0, 1]), color=color)

    matplotlib.pyplot.axis('off')
    matplotlib.pyplot.show()
    return

# to generate a fractal geometry run: draw_fractal(fractalize_mat_order_rec(2), 0.0, 1.0, 0.0, 1.0)
# to generate a geometry similar to the helmholtz resonator run: draw_fractal(mat_res_helmholtz(), 0.0, 1.0, 0.0, 1.0)

# ---------------------------------------------------------------------------------------
# Question 2:


def detect_boundary_mat(mat):
    '''we need first of all provide a function that returns nodes on boundary so that we can set initial
    conditions on those nodes later'''
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


def duplicate_mat(mat):
    '''for some precision purpose, I needed to write this function, It mainly duplicates the grid's size'''
    n = mat.shape[0]
    new_mat = numpy.zeros((2*n, 2*n), int)
    for i in range(0, n):
        for j in range(0, n):
            if mat[i][j] == 1:
                for x in range(2):
                    for y in range(2):
                        new_mat[2*i+x][2*j+y] = 1
    return new_mat


def geometrical_loc_sol(mat, r):
    '''This function solves the problem's equation for the geometry given by the matrix mat'''
    # so that we can have more details on the the grid (un maillage plus fin)
    mat = duplicate_mat(mat)
    # -- set geometry parameters
    xmin, xmax, ymin, ymax = 0.0, 1.0, 0.0, 1.0
    nelemsx, nelemsy = mat.shape[1], mat.shape[0]
    # -- set equation parameters
    # wavenumber = numpy.pi*(nelemsx/5)
    # wavenumber = numpy.pi
    h = (ymax-ymin)/nelemsy
    wavenumber = numpy.pi/(r*h)  # according to F-Simon presentation (slide13)
    # wavenumber = 1/(2*((ymax-ymin)/nelemsy))
    # wavenumber = 1/(2*((ymax-ymin)/nelemsy))
    # -- generate mesh
    nnodes = (nelemsx + 1) * (nelemsy + 1)
    nelems = nelemsx * nelemsy * 2
    node_coords, p_elem2nodes, elem2nodes, node_l2g = solutions._set_square_trimesh(
        xmin, xmax, ymin, ymax, nelemsx, nelemsy)

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
    node_to_dodge = detect_boundary_mat(mat)
    nodes_on_boundary = numpy.unique(numpy.array(node_to_dodge), )

    # For the time being we will set the term f the same way it was introduced in the lecture, but later we
    # can easily choose what we want for the noise function
    # -- set exact solution
    solexact = numpy.zeros((nnodes, 1), dtype=numpy.complex128)
    laplacian_of_solexact = numpy.zeros((nnodes, 1), dtype=numpy.complex128)
    for i in range(nnodes):
        x, y, z = node_coords[i, 0], node_coords[i, 1], node_coords[i, 2]
        # set: u(x,y) = e^{ikx}
        solexact[i] = numpy.exp(complex(0., 1.)*wavenumber*x)
        laplacian_of_solexact[i] = complex(
            0., 1.)*wavenumber*complex(0., 1.)*wavenumber * solexact[i]

    # -- set dirichlet boundary conditions
    values_at_nodes_on_boundary = numpy.zeros(
        (nnodes, 1), dtype=numpy.complex128)
    values_at_nodes_on_boundary[nodes_on_boundary] = solexact[nodes_on_boundary]

    # -- set finite element matrices and right hand side
    f_unassembled = numpy.zeros((nnodes, 1), dtype=numpy.complex128)

    for i in range(nnodes):
        # evaluate: (-\Delta - k^2) u(x,y) = ...
        f_unassembled[i] = - laplacian_of_solexact[i] - \
            (wavenumber ** 2) * solexact[i]

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

    return

# To visualise the sol (real and imaginary part) run : geometrical_loc_sol(fractalize_mat_order_rec(2), 2) (for a fractal)
# and geometrical_loc_sol(mat_res_helmholtz(), 5) (for a helmholtz resonator), by changing the argument r you can modify the frequency

# ---------------------------------------------------------------------------------------
# Question 3:


def eigenmode(n=1, m=1):
    '''the eigenvalues are given by this formula: pi²(m²+n²) (since a=b=1), and the eigenfunctions are
    given as below'''
    x = numpy.linspace(0, 1, 200)
    y = numpy.linspace(0, 1, 200)
    X, Y = numpy.meshgrid(x, y)
    Z = numpy.sin(m*numpy.pi*X)*numpy.sin(n*numpy.pi*Y)
    matplotlib.pyplot.pcolormesh(X, Y, Z)
    matplotlib.pyplot.show()
    return
# ---------------------------------------------------------------------------------------
# Question 4:


if __name__ == '__main__':
    # draw_fractal(fractalize_mat_order_rec(2), 0.0, 1.0, 0.0, 1.0)
    # geometrical_loc_sol(fractalize_mat_order_rec(2), 2)
    # geometrical_loc_sol(mat_res_helmholtz(), 5)
    # eigenmode(5, 2)
    print('End!')
