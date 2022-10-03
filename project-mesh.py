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

# MRG packages
import solutions

# Question 1 :


def compute_distance(node1, node2):
    return numpy.sqrt((node1[0]-node2[0])**2 + (node1[1]-node2[1])**2 + (node1[2]-node2[2])**2)


def radius_inscribed(a, b, c):
    s = (a+b+c)/2
    rho = numpy.sqrt((s-a)*(s-b)*(s-c)/s)
    return rho


def compute_aspect_ratio_of_element(node_coords, p_elem2nodes, elem2nodes):
    number_elem = len(p_elem2nodes)-1
    spacedim = node_coords.shape[1]
    elem_quality = numpy.empty(number_elem, dtype=numpy.float64)
    for i in range(number_elem):
        x, y = p_elem2nodes[i], p_elem2nodes[i+1]
        necessary_nodes = elem2nodes[x:y]
        number_of_vertices = len(necessary_nodes)
        if number_of_vertices == 4:
            vector0, vector1, vector2, vector3 = node_coords[necessary_nodes[0]]-node_coords[necessary_nodes[1]], node_coords[necessary_nodes[1]] - \
                node_coords[necessary_nodes[2]], node_coords[necessary_nodes[2]] - \
                node_coords[necessary_nodes[3]], node_coords[necessary_nodes[3]
                                                             ]-node_coords[necessary_nodes[0]]
            q = 1 - sum([numpy.abs(numpy.dot(vector0, vector1)), numpy.abs(numpy.dot(vector1, vector2)),
                        numpy.abs(numpy.dot(vector2, vector3)), numpy.abs(numpy.dot(vector3, vector0))])/4
            elem_quality[i] = q
        if number_of_vertices == 3:
            a, b, c = compute_distance(node_coords[necessary_nodes[0]], node_coords[necessary_nodes[1]]), compute_distance(
                node_coords[necessary_nodes[0]], node_coords[necessary_nodes[2]]), compute_distance(node_coords[necessary_nodes[2]], node_coords[necessary_nodes[1]])
            h, rho = max(a, b, c), radius_inscribed(a, b, c)
            q = (numpy.sqrt(3)*rho)/(6*h)
            elem_quality[i] = q

    return elem_quality

# Question 2 :


def compute_edge_length_factor_of_element(node_coords, p_elem2nodes, elem2nodes):
    number_elem = len(p_elem2nodes)-1
    spacedim = node_coords.shape[1]
    elem_quality = numpy.empty(number_elem, dtype=numpy.float64)
    for i in range(number_elem):
        x, y = p_elem2nodes[i], p_elem2nodes[i+1]
        necessary_nodes = elem2nodes[x:y]
        number_of_vertices = len(necessary_nodes)
        if number_of_vertices == 4:
            lenghts = [compute_distance(node_coords[necessary_nodes[0]], node_coords[necessary_nodes[1]]), compute_distance(node_coords[necessary_nodes[1]], node_coords[necessary_nodes[2]]), compute_distance(
                node_coords[necessary_nodes[2]], node_coords[necessary_nodes[3]]), compute_distance(node_coords[necessary_nodes[3]], node_coords[necessary_nodes[0]])]
            m = min(lenghts)
            elem_quality[i] = 4*m/sum(lenghts)
        if number_of_vertices == 3:
            lenghts = [compute_distance(node_coords[necessary_nodes[0]], node_coords[necessary_nodes[1]]), compute_distance(
                node_coords[necessary_nodes[1]], node_coords[necessary_nodes[2]]), compute_distance(node_coords[necessary_nodes[2]], node_coords[necessary_nodes[0]])]
            m = min(lenghts)
            elem_quality[i] = 3*m/sum(lenghts)
    return elem_quality

# Question 3 :

# I didn't understand very what are we supposed to do to answer this question

# Question 4 :


def add_node_to_mesh(node_coords, p_elem2nodes, elem2nodes, nodeid_coords):
    node_coords.append(nodeid_coords)
    return node_coords, p_elem2nodes, elem2nodes,


def add_elem_to_mesh(node_coords, p_elem2nodes, elem2nodes, elemid2nodes):
    elem2nodes.append(elemid2nodes)
    return node_coords, p_elem2nodes, elem2nodes,


def remove_node_to_mesh(node_coords, p_elem2nodes, elem2nodes, nodeid):
    spacedim = node_coords.shape[1]
    nelems = node_coords.shape[0]
    triggered_index = []
    rings = len(p_elem2nodes)
    for i in range(rings-1):
        x, y = p_elem2nodes[i], p_elem2nodes[i+1]
        area = elem2nodes[x:y]
        if nodeid in area:
            triggered_index.append((x, y))
    plage_to_remove = []
    for t in triggered_index:
        plage_to_remove += [i for i in range(t[0], t[1])]
    plage_to_remove = list(set(plage_to_remove))
    new_elem2nodes = numpy.delete(elem2nodes, plage_to_remove)
    for i in range(len(new_elem2nodes)):
        if new_elem2nodes[i] > nodeid:
            new_elem2nodes[i] = new_elem2nodes[i]-1

    new_node_coords = numpy.delete(node_coords, nodeid, axis=0)

    new_p_elem2nodes = p_elem2nodes[:len(p_elem2nodes)-len(triggered_index)]
    return new_node_coords, new_p_elem2nodes, new_elem2nodes


def remove_elem_to_mesh(node_coords, p_elem2nodes, elem2nodes, elemid):
    curr_node_coords, curr_p_elem2nodes, curr_elem2nodes = node_coords, p_elem2nodes, elem2nodes
    x, y = p_elem2nodes[elemid], p_elem2nodes[elemid+1]
    nodes_to_remove = elem2nodes[x:y]
    nodes_to_remove.sort()
    k = 0
    for nodeid in nodes_to_remove:
        curr_node_coords, curr_p_elem2nodes, curr_elem2nodes = remove_node_to_mesh(
            curr_node_coords, curr_p_elem2nodes, curr_elem2nodes, nodeid-k)
        k += 1
    return curr_node_coords, curr_p_elem2nodes, curr_elem2nodes

# Question 5 :


def compute_barycenter_of_element(node_coords, p_elem2nodes, elem2nodes):
    spacedim = node_coords.shape[1]
    nelems = p_elem2nodes.shape[0] - 1
    elem_coords = numpy.zeros((nelems, spacedim), dtype=numpy.float64)
    for i in range(0, nelems):
        nodes = elem2nodes[p_elem2nodes[i]:p_elem2nodes[i+1]]
        elem_coords[i, :] = numpy.average(node_coords[nodes, :], axis=0)
    return elem_coords

# Question 6 :


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


def pagging(mat):
    n = mat.shape[0]
    new_mat = numpy.zeros((n+2, n+2), int)
    for i in range(1, n+1):
        new_mat[i][1:n+1] = mat[i-1][0:n]
    return new_mat


def fractalize_mat_order_rec(order):
    if order == 1:
        return fractalized_mat_sample_global

    fractalized_mat_sample = fractalize_mat_order_rec(order-1)
    n = fractalized_mat_sample.shape[0]
    fractalized_mat_sample_pag = pagging(fractalized_mat_sample)
    isolated_ones = []
    for i in range(n+2):
        for j in range(n+2):
            if 0 <= i-1 < n+2 and 0 <= i+1 < n+2 and fractalized_mat_sample_pag[i-1][j] == 0 and fractalized_mat_sample_pag[i+1][j] == 0 and fractalized_mat_sample_pag[i][j] == 1:
                isolated_ones.append((i, j))
            if 0 <= j-1 < n+2 and 0 <= j+1 < n+2 and fractalized_mat_sample_pag[i][j-1] == 0 and fractalized_mat_sample_pag[i][j+1] == 0 and fractalized_mat_sample_pag[i][j] == 1:
                isolated_ones.append((i, j))
    new_mat = fractalized_mat_sample
    new_mat = pagging(new_mat)
    new_mat = quadruple_mat(new_mat)

    for t in isolated_ones:
        new_mat[4*t[0]-1:4*t[0]+5, 4*t[1]-1:4*t[1]+5] = new_mat[4*t[0] -
                                                                1:4*t[0]+5, 4*t[1]-1:4*t[1]+5] + fractalized_mat_sample_global

    p = new_mat.shape[0]
    for i in range(p):
        for j in range(p):
            if new_mat[i][j] >= 2:
                new_mat[i][j] = 1
    return new_mat


def draw_fractal(mat, xmin, xmax, ymin, ymax, color='blue'):
    node_coords, p_elem2nodes, elem2nodes = build_matrix(
        mat, xmin, xmax, ymin, ymax)

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

    matplotlib.pyplot.show()
    return

# Question 7 :


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


if __name__ == '__main__':
    # compute_aspect_ratio_of_element(node_coords, p_elem2nodes, elem2nodes)
    # compute_edge_length_factor_of_element(node_coords, p_elem2nodes, elem2nodes)
    # add_node_to_mesh(node_coords, p_elem2nodes, elem2nodes, nodeid_coords)
    # add_elem_to_mesh(node_coords, p_elem2nodes, elem2nodes, elemid2nodes)
    # remove_node_to_mesh(node_coords, p_elem2nodes, elem2nodes, nodeid)
    # remove_elem_to_mesh(node_coords, p_elem2nodes, elem2nodes, elemid)
    # compute_barycenter_of_element(node_coords, p_elem2nodes, elem2nodes)
    draw_fractal(fractalize_mat_order_rec(2), 0.0, 1.0, 0.0, 1.0)
    # split_quadrangle_into_triangle(node_coords, p_elem2nodes, elem2nodes)

    print("End.")
