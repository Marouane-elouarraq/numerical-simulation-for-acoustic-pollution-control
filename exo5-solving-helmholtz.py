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

def shuffle(node_coords, p_elem2nodes, elem2nodes, xmin=0.0, xmax=1.0, ymin=0.0, ymax=1.0, nelemsx=10, nelemsy=10):
    n = len(node_coords)
    for i in range(n):
        node = node_coords[i]
        x, y, z = node[0], node[1], node[2]
        ratiox, ratioy = (xmax-xmin)/nelemsx, (ymax-ymin)/nelemsy
        c1, c2 = random.choice([ratiox/3, 2*ratiox/3]), random.choice([ratioy/3, 2*ratioy/3])
        node_coords[i, :] = numpy.array([x+c1, y+c2, z])
    return node_coords, p_elem2nodes, elem2nodes


# ..todo: Uncomment for displaying limited digits
# numpy.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})


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
    nodes_on_boundary = numpy.unique(numpy.concatenate((nodes_on_north, nodes_on_south, nodes_on_east, nodes_on_west)), )
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
    node_coords, p_elem2nodes, elem2nodes, node_l2g = solutions._set_square_trimesh(xmin, xmax, ymin, ymax, nelemsx, nelemsy)
    # -- generate other meshes by shuffling the original one
    node_coords1, p_elem2nodes1, elem2nodes1, node_l2g1 = solutions._set_square_trimesh(xmin, xmax, ymin, ymax, 15, 15)
    node_coords2, p_elem2nodes2, elem2nodes2, node_l2g2 = solutions._set_square_trimesh(xmin, xmax, ymin, ymax, 20, 20)
    # --
    
    h_maillage_grid = [compute_h_for_grid(node_coords, p_elem2nodes, elem2nodes, choice), compute_h_for_grid(node_coords1, p_elem2nodes1, elem2nodes1, choice), compute_h_for_grid(node_coords2, p_elem2nodes2, elem2nodes2, choice)]
    h_values = [h_maillage[0] for h_maillage in h_maillage_grid]
    h_values1 = [h_maillage[1] for h_maillage in h_maillage_grid]
    h_values2 = [h_maillage[2] for h_maillage in h_maillage_grid]
    h_values_log = [numpy.log(x) for x in h_values]
    h_values_log1 = [numpy.log(x) for x in h_values1]
    h_values_log2 = [numpy.log(x) for x in h_values2]
    
    error_values = [solve_given_grid(node_coords, p_elem2nodes, elem2nodes), solve_given_grid(node_coords1, p_elem2nodes1, elem2nodes1), solve_given_grid(node_coords2, p_elem2nodes2, elem2nodes2)]

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

        h_values1.append(compute_h_for_grid(node_coords, p_elem2nodes, elem2nodes, choice)[0])
        h_values2.append(compute_h_for_grid(node_coords, p_elem2nodes, elem2nodes, choice+1)[1])
        h_values3.append(compute_h_for_grid(node_coords, p_elem2nodes, elem2nodes, choice+2)[2])
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


    # slope, intercept, r_value, p_value, std_err = linregress(h_values_log, error_values_log)
    matplotlib.pyplot.plot(h_values1_log, error_values_log)
    matplotlib.pyplot.plot(h_values2_log, error_values_log)
    matplotlib.pyplot.plot(h_values3_log, error_values_log)
    matplotlib.pyplot.xlabel("nelem values (log scale)")
    matplotlib.pyplot.ylabel("error values (log scale)")
    # matplotlib.pyplot.title("slope : {}".format(slope))
    matplotlib.pyplot.show()
    return
    # print(h_values1_log, h_values2_log, h_values3_log)



if __name__ == '__main__':

    # run_exercise_solution_helmholtz_dddd()
    # find_alpha()
    # find_beta()
    # find_alpha_2()
    find_alpha_3(0)
    print('End.')
