# -*- coding: utf-8 -*-
"""
.. warning:: The explanations of the functions in this file and the details of
the programming methodology have been given during the lectures.
"""


# Python packages
import matplotlib.pyplot
import matplotlib.pylab
from mpl_toolkits.mplot3d import Axes3D
import numpy
import random
import os
import scipy.io
import scipy.linalg
import scipy.sparse
import scipy.sparse.linalg
import sys
from math import floor


# MRG packages
import solutions


def add_node_to_mesh(node_coords, p_elem2nodes, elem2nodes, nodeid_coords):
    # .. todo:: Modify the lines below to add one node to the mesh
    node_coords.append(nodeid_coords)
    return node_coords, p_elem2nodes, elem2nodes,


def add_elem_to_mesh(node_coords, p_elem2nodes, elem2nodes, elemid2nodes):
    # .. todo:: Modify the lines below to add one element to the mesh
    elem2nodes.append(elemid2nodes)
    return node_coords, p_elem2nodes, elem2nodes,


def remove_node_to_mesh(node_coords, p_elem2nodes, elem2nodes, nodeid):
    # .. todo:: Modify the lines below to remove one node to the mesh
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


def compute_barycenter_of_element(node_coords, p_elem2nodes, elem2nodes):
    # ..todo: Modify the lines below to compute the barycenter of one element
    spacedim = node_coords.shape[1]
    nelems = p_elem2nodes.shape[0] - 1
    elem_coords = numpy.zeros((nelems, spacedim), dtype=numpy.float64)
    for i in range(0, nelems):
        nodes = elem2nodes[p_elem2nodes[i]:p_elem2nodes[i+1]]
        elem_coords[i, :] = numpy.average(node_coords[nodes, :], axis=0)
    return elem_coords


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


def rect_area(a, b, c):
    semi_perim = (a+b+c)/2
    return numpy.square(semi_perim*(semi_perim-a)*(semi_perim-b)*(semi_perim-c))


def compute_pointedness_of_element(node_coords, p_elem2nodes, elem2nodes):
    number_elem = len(p_elem2nodes)-1
    spacedim = node_coords.shape[1]
    elem_quality = numpy.zeros(number_elem, dtype=numpy.float64)
    for i in range(number_elem):
        x, y = p_elem2nodes[i], p_elem2nodes[i+1]
        necessary_nodes = elem2nodes[x:y]

        barycenter = numpy.zeros(spacedim, dtype=numpy.float64)
        for j in range(spacedim):
            barycenter[j] = (node_coords[necessary_nodes[0]][j] + node_coords[necessary_nodes[1]]
                             [j] + node_coords[necessary_nodes[2]][j] + node_coords[necessary_nodes[3]][j])/4
        ##
        A1 = rect_area(compute_distance(node_coords[necessary_nodes[0]], node_coords[necessary_nodes[1]]), compute_distance(
            node_coords[necessary_nodes[0]], barycenter), compute_distance(node_coords[necessary_nodes[1]], barycenter))
        ##
        A2 = rect_area(compute_distance(node_coords[necessary_nodes[1]], node_coords[necessary_nodes[2]]), compute_distance(
            node_coords[necessary_nodes[1]], barycenter), compute_distance(node_coords[necessary_nodes[2]], barycenter))
        ##
        A3 = rect_area(compute_distance(node_coords[necessary_nodes[2]], node_coords[necessary_nodes[3]]), compute_distance(
            node_coords[necessary_nodes[2]], barycenter), compute_distance(node_coords[necessary_nodes[3]], barycenter))
        ##
        A4 = rect_area(compute_distance(node_coords[necessary_nodes[3]], node_coords[necessary_nodes[0]]), compute_distance(
            node_coords[necessary_nodes[3]], barycenter), compute_distance(node_coords[necessary_nodes[0]], barycenter))
        ##
        A = A1 + A2 + A3 + A4
        ##
        elem_quality[i] = 4*(min(A1, A2, A3, A4)/A)

    return elem_quality


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


# def hard_shuffle(node_coords, p_elem2nodes, elem2nodes, xmin=0.0, xmax=1.0, ymin=0.0, ymax=1.0, nelemsx=10, nelemsy=10):
#     n = len(node_coords)
#     for i in range(n):
#         node = node_coords[i]
#         x, y, z = node[0], node[1], node[2]
#         ratiox, ratioy = (xmax-xmin)/nelemsx, (ymax-ymin)/nelemsy
#         c1, c2 = random.gauss(0, 0.05), random.gauss(0, 0.05)
#         node_coords[i, :] = numpy.array([x+c1, y+c2, z])
#     return node_coords, p_elem2nodes, elem2nodes


def shuffled_quadrangle_grid(xmin=0.0, xmax=1.0, ymin=0.0, ymax=1.0, nelemsx=10, nelemsy=10):
    node_coords, node_l2g, p_elem2nodes, elem2nodes = solutions._set_quadmesh(
        xmin, xmax, ymin, ymax, nelemsx, nelemsy)
    n = len(node_coords)
    choice = int(input(
        "print 0 for the aspect ratio, 1 for the edge length factor, 2 for the pointedness"))
    if choice == 0:
        new_node_coords, new_p_elem2nodes, new_elem2nodes = shuffle(
            node_coords, p_elem2nodes, elem2nodes)
        criterion_table = compute_aspect_ratio_of_element(
            new_node_coords, new_p_elem2nodes, new_elem2nodes)
    if choice == 1:
        new_node_coords, new_p_elem2nodes, new_elem2nodes = shuffle(
            node_coords, p_elem2nodes, elem2nodes)
        criterion_table = compute_edge_length_factor_of_element(
            new_node_coords, new_p_elem2nodes, new_elem2nodes)
    if choice == 2:
        new_node_coords, new_p_elem2nodes, new_elem2nodes = shuffle(
            node_coords, p_elem2nodes, elem2nodes)
        criterion_table = compute_edge_length_factor_of_element(
            new_node_coords, new_p_elem2nodes, new_elem2nodes)

    if choice == 1 or choice == 2:
        x_axis = [y/50 for y in range(0, 51)]
    else:
        x_axis = [0.9+0.1*y/50 for y in range(0, 51)]
    y_axis = []
    clusters_dict = {i: [] for i in range(50)}
    n = len(criterion_table)
    for i in range(50):
        s = 0
        for x in range(n):
            if x_axis[i] <= criterion_table[x] <= x_axis[i+1]:
                s += 1
                clusters_dict[i].append(x)
        y_axis.append(s)

    matplotlib.pyplot.hist(criterion_table, bins=x_axis)
    matplotlib.pyplot.title('quality criterion histogram')
    matplotlib.pyplot.xlabel("categories considering the computed criterion")
    matplotlib.pyplot.ylabel("number of elements")
    matplotlib.pyplot.show()


############# shows aberrant elements #########################
    d = {}
    for i in clusters_dict:
        if clusters_dict[i] != []:
            d[i] = clusters_dict[i]
    least_cluster = d[min(d, key=lambda x:len(d[x]))]
    fig = matplotlib.pyplot.figure(1)
    ax = matplotlib.pyplot.subplot(1, 1, 1)
    ax.set_aspect('equal')
    ax.axis('off')
    solutions._plot_mesh(new_p_elem2nodes, new_elem2nodes,
                         new_node_coords, color='yellow')
    for elemid in least_cluster:
        solutions._plot_elem(new_p_elem2nodes, new_elem2nodes,
                             new_node_coords, elemid, color='red')
    matplotlib.pyplot.title('display aberrant elements in the shuffled grid')
    matplotlib.pyplot.show()
    return


def shuffled_triangle_grid(xmin=0.0, xmax=1.0, ymin=0.0, ymax=1.0, nelemsx=10, nelemsy=10):
    node_coords, node_l2g, p_elem2nodes, elem2nodes = solutions._set_trimesh(
        xmin, xmax, ymin, ymax, nelemsx, nelemsy)
    n = len(node_coords)

    choice = int(
        input("print 0 for the aspect ratio criterion, 1 for the edge length factor"))
    if choice == 0:
        normalizing_factor = numpy.mean(compute_aspect_ratio_of_element(
            node_coords, p_elem2nodes, elem2nodes))
        new_node_coords, new_p_elem2nodes, new_elem2nodes = shuffle(
            node_coords, p_elem2nodes, elem2nodes)
        pointedness_table = compute_aspect_ratio_of_element(
            new_node_coords, new_p_elem2nodes, new_elem2nodes)/normalizing_factor
    if choice == 1:
        normalizing_factor = numpy.mean(compute_edge_length_factor_of_element(
            node_coords, p_elem2nodes, elem2nodes))
        new_node_coords, new_p_elem2nodes, new_elem2nodes = shuffle(
            node_coords, p_elem2nodes, elem2nodes)
        pointedness_table = compute_edge_length_factor_of_element(
            new_node_coords, new_p_elem2nodes, new_elem2nodes)/normalizing_factor

    pointedness_table_min, pointedness_table_max = min(
        pointedness_table), max(pointedness_table)
    r = floor(pointedness_table_max-pointedness_table_min)+1

    x_axis = [r*y/50 for y in range(0, 51)]
    y_axis = []
    clusters_dict = {i: [] for i in range(50)}
    n = len(pointedness_table)
    for i in range(50):
        s = 0
        for x in range(n):
            if x_axis[i] <= pointedness_table[x] <= x_axis[i+1]:
                s += 1
                clusters_dict[i].append(x)
        y_axis.append(s)

    matplotlib.pyplot.hist(pointedness_table, bins=x_axis)
    matplotlib.pyplot.title('quality criterion histogram')
    matplotlib.pyplot.xlabel("categories considering the computed criterion")
    matplotlib.pyplot.ylabel("number of elements")
    matplotlib.pyplot.show()


############# shows aberrant elements #########################
    d = {}
    for i in clusters_dict:
        if clusters_dict[i] != []:
            d[i] = clusters_dict[i]
    least_cluster = d[min(d, key=lambda x:len(d[x]))]
    fig = matplotlib.pyplot.figure(1)
    ax = matplotlib.pyplot.subplot(1, 1, 1)
    ax.set_aspect('equal')
    ax.axis('off')

    solutions._plot_mesh(new_p_elem2nodes, new_elem2nodes,
                         new_node_coords, color='yellow')
    for elemid in least_cluster:
        solutions._plot_elem(new_p_elem2nodes, new_elem2nodes,
                             new_node_coords, elemid, color='red')
    matplotlib.pyplot.title('display aberrant elements in the shuffled grid')
    matplotlib.pyplot.show()
    return


def criteria_functions_testing():
    choice = int(
        input("type 3 for a grid with triangles, 4 for a grid with quadrangles"))
    if choice == 3:
        shuffled_triangle_grid(xmin=0.0, xmax=1.0, ymin=0.0,
                               ymax=1.0, nelemsx=10, nelemsy=10)
    else:
        shuffled_quadrangle_grid(
            xmin=0.0, xmax=1.0, ymin=0.0, ymax=1.0, nelemsx=10, nelemsy=10)


def node_to_remove_for_fractal(previous_n, n):
    # if you add n elements to the grid bellow this program will generate the list of nodes to remove so that you can draw a fractal
    # the gap between n and previous_n should be 10
    # l = [72, 64, 62, 55, 42, 41, 39, 31, 29, 22, 11, 17, 15, 11, 9, 7, 5, 2]
    l = [2, 5, 7, 9, 11, 12, 34, 56, 24, 31, 33, 44, 45, 60, 67, 70]
    t = l[:]
    for i in range(len(t)):
        k = t[i]//(previous_n+1)
        t[i] = t[i] + 10*k
    tt = [x+10 for x in t]
    for x in tt:
        if x not in t:
            t.append(x)
    t.sort()
    t.reverse()
    l.sort()
    l.reverse()
    if n == 10:
        return l
    return t


def run_exercise_a(n):
    """Generate grid with quadrangles.
    """
    # -- generate grid with quadrangles
    xmin, xmax, ymin, ymax = 0.0, 2.0, 0.0, 2.0
    nelemsx, nelemsy = n+1, n+1
    nelems = nelemsx * nelemsy
    nnodes = (nelemsx + 1) * (nelemsy + 1)
    # .. todo:: Modify the line below to call to generate a grid with quadrangles
    # p_elem2nodes, elem2nodes, node_coords, node_l2g = my_set_quadmesh(...)
    # .. note:: If you do not succeed, uncomment the following line to access the solution
    node_coords, node_l2g, p_elem2nodes, elem2nodes = solutions._set_quadmesh(
        xmin, xmax, ymin, ymax, nelemsx, nelemsy)

    # test functions
    # node_coords, p_elem2nodes, elem2nodes = remove_node_to_mesh(node_coords, p_elem2nodes, elem2nodes, 90)
    # node_coords, p_elem2nodes, elem2nodes = remove_node_to_mesh(node_coords, p_elem2nodes, elem2nodes, 36)
    # node_coords, p_elem2nodes, elem2nodes = remove_node_to_mesh(node_coords, p_elem2nodes, elem2nodes, 18)
    # node_coords, p_elem2nodes, elem2nodes = remove_node_to_mesh(node_coords, p_elem2nodes, elem2nodes, 2)
    # l = [107, 100, 99, 97, 90, 63, 62, 53, 46, 45, 43, 36, 18, 17, 15, 12, 11, 9, 7, 5, 2]
    # ll = [107, 99, 97, 90, 63, 62, 53, 45, 43, 36, 18, 17, 15, 11, 9, 7, 5, 2]
    # l = [167, 160, 159, 157, 150, 149, 147, 140, 103, 102, 93, 92, 83, 76, 75, 73, 66, 65, 63, 56, 38, 28, 27, 25, 22, 21, 19, 17, 15, 12, 11, 9, 7, 5, 2]
    l = node_to_remove_for_fractal(n, n)
    for nodeid in l:
        node_coords, p_elem2nodes, elem2nodes = remove_node_to_mesh(
            node_coords, p_elem2nodes, elem2nodes, nodeid)
    # node_coords, p_elem2nodes, elem2nodes = remove_node_to_mesh(node_coords, p_elem2nodes, elem2nodes, 16)

    fig = matplotlib.pyplot.figure(1)
    ax = matplotlib.pyplot.subplot(1, 1, 1)
    ax.set_aspect('equal')
    ax.axis('off')
    solutions._plot_mesh(p_elem2nodes, elem2nodes, node_coords, color='yellow')
    node = 80
    # solutions._plot_node(p_elem2nodes, elem2nodes,node_coords, node, color='red', marker='o')
    elem = 70
    # solutions._plot_elem(p_elem2nodes, elem2nodes,node_coords, elem, color='orange')
    matplotlib.pyplot.show()

    return


def run_exercise_b():
    """Generate grid with triangles.
    """
    # -- generate grid with triangles
    xmin, xmax, ymin, ymax = 0.0, 1.0, 0.0, 1.0
    nelemsx, nelemsy = 10, 10
    nelems = nelemsx * nelemsy * 2
    nnodes = (nelemsx + 1) * (nelemsy + 1)
    node_coords, node_l2g, p_elem2nodes, elem2nodes = solutions._set_trimesh(
        xmin, xmax, ymin, ymax, nelemsx, nelemsy)
    nodes_on_boundary = solutions._set_trimesh_boundary(nelemsx, nelemsy)

    # test functions
    # node_coords, p_elem2nodes, elem2nodes = remove_elem_to_mesh(node_coords, p_elem2nodes, elem2nodes, 25)

    # -- plot mesh
    fig = matplotlib.pyplot.figure(1)
    ax = matplotlib.pyplot.subplot(1, 1, 1)
    ax.set_aspect('equal')
    ax.axis('off')
    solutions._plot_mesh(p_elem2nodes, elem2nodes, node_coords, color='yellow')
    node = 80
    solutions._plot_node(p_elem2nodes, elem2nodes,
                         node_coords, node, color='red', marker='o')
    elem = 70
    solutions._plot_elem(p_elem2nodes, elem2nodes,
                         node_coords, elem, color='orange')
    matplotlib.pyplot.show()

    return


def run_exercise_c():
    pass


def run_exercise_d():
    pass


def helmholtz_resonator(xmin=0.0, xmax=2.0, ymin=0.0, ymax=1.0):
    nelemsx, nelemsy = 20, 10
    nelems = nelemsx * nelemsy
    nnodes = (nelemsx + 1) * (nelemsy + 1)

    node_coords, node_l2g, p_elem2nodes, elem2nodes = solutions._set_quadmesh(
        xmin, xmax, ymin, ymax, nelemsx, nelemsy)

    # shape the resonator
    # node_coords, p_elem2nodes, elem2nodes = remove_elem_to_mesh(node_coords, p_elem2nodes, elem2nodes, 30)
    node_coords, p_elem2nodes, elem2nodes = remove_elem_to_mesh(
        node_coords, p_elem2nodes, elem2nodes, 30)
    node_coords, p_elem2nodes, elem2nodes = remove_elem_to_mesh(
        node_coords, p_elem2nodes, elem2nodes, 161)
    # -- plot mesh
    fig = matplotlib.pyplot.figure(1)
    ax = matplotlib.pyplot.subplot(1, 1, 1)
    ax.set_aspect('equal')
    ax.axis('off')
    solutions._plot_mesh(p_elem2nodes, elem2nodes, node_coords, color='yellow')
    matplotlib.pyplot.show()
    return


def tight_helmholtz_resonator(xmin=0.0, xmax=2.0, ymin=0.0, ymax=1.0):
    nelemsx, nelemsy = 20, 10
    nelems = nelemsx * nelemsy
    nnodes = (nelemsx + 1) * (nelemsy + 1)

    node_coords, node_l2g, p_elem2nodes, elem2nodes = solutions._set_quadmesh(
        xmin, xmax, ymin, ymax, nelemsx, nelemsy)

    # shape the resonator
    # node_coords, p_elem2nodes, elem2nodes = remove_elem_to_mesh(node_coords, p_elem2nodes, elem2nodes, 30)
    node_coords, p_elem2nodes, elem2nodes = remove_elem_to_mesh(
        node_coords, p_elem2nodes, elem2nodes, 50)
    node_coords, p_elem2nodes, elem2nodes = remove_elem_to_mesh(
        node_coords, p_elem2nodes, elem2nodes, 10)
    node_coords, p_elem2nodes, elem2nodes = remove_elem_to_mesh(
        node_coords, p_elem2nodes, elem2nodes, 138)
    node_coords, p_elem2nodes, elem2nodes = remove_elem_to_mesh(
        node_coords, p_elem2nodes, elem2nodes, 169)
    # -- plot mesh
    fig = matplotlib.pyplot.figure(1)
    ax = matplotlib.pyplot.subplot(1, 1, 1)
    ax.set_aspect('equal')
    ax.axis('off')
    solutions._plot_mesh(p_elem2nodes, elem2nodes, node_coords, color='yellow')
    matplotlib.pyplot.show()
    return


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
        u1, u2, u3 = new_p_elem2nodes[2*i], new_p_elem2nodes[2*i+1], new_p_elem2nodes[2*i+2]
        necessary_nodes = elem2nodes[x:y]
        new_elem2nodes[u1:u2] = numpy.array([necessary_nodes[0], necessary_nodes[1], necessary_nodes[2]])
        new_elem2nodes[u2:u3] = numpy.array([necessary_nodes[0], necessary_nodes[2], necessary_nodes[3]])
    return new_node_coords, new_p_elem2nodes, new_elem2nodes


def test_split():
    xmin, xmax, ymin, ymax = 0.0, 1.0, 0.0, 1.0
    nelemsx, nelemsy = 10, 10
    nelems = nelemsx * nelemsy
    nnodes = (nelemsx + 1) * (nelemsy + 1)
    # .. todo:: Modify the line below to call to generate a grid with quadrangles
    # p_elem2nodes, elem2nodes, node_coords, node_l2g = my_set_quadmesh(...)
    # .. note:: If you do not succeed, uncomment the following line to access the solution
    node_coords, node_l2g, p_elem2nodes, elem2nodes = solutions._set_quadmesh(
        xmin, xmax, ymin, ymax, nelemsx, nelemsy)
    new_node_coords, new_p_elem2nodes, new_elem2nodes = split_quadrangle_into_triangle(
        node_coords, p_elem2nodes, elem2nodes)

    fig = matplotlib.pyplot.figure(1)
    ax = matplotlib.pyplot.subplot(1, 1, 1)
    ax.set_aspect('equal')
    ax.axis('off')
    solutions._plot_mesh(new_p_elem2nodes, new_elem2nodes,
                         new_node_coords, color='yellow')
    matplotlib.pyplot.show()
    return
    # print(new_node_coords, node_coords)


def _set_fractal(xmin, xmax, ymin, ymax, nx, ny):
    
    spacedim = 3
    nnodes = (nx + 1) * (ny + 1)
    node_coords = numpy.empty((nnodes, spacedim), dtype=numpy.float64)
    nodes_per_elem = 4
    nelems = nx * ny
    p_elem2nodes = numpy.empty((nelems + 1,), dtype=numpy.int64)
    p_elem2nodes[0] = 0
    for i in range(0, nelems):
        p_elem2nodes[i + 1] = p_elem2nodes[i] + nodes_per_elem
    elem2nodes = numpy.empty((nelems * nodes_per_elem,), dtype=numpy.int64)

    # elements
    k = 0
    for j in range(0, ny):
        for i in range(0, nx):
            elem2nodes[k + 0] = j * (nx + 1) + i
            elem2nodes[k + 1] = j * (nx + 1) + i + 1
            elem2nodes[k + 2] = (j + 1) * (nx + 1) + i + 1
            elem2nodes[k + 3] = (j + 1) * (nx + 1) + i
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
# little edit, just checking
    node_coords = numpy.delete(node_coords, [i for i in range(0, nx+1) if i!=2 and i!=1], axis = 0)
    elem2nodes = elem2nodes[:nelems*nodes_per_elem-nx*nodes_per_elem-1]
    p_elem2nodes = p_elem2nodes[:nelems+1-nx]
    # node_coords[0] = [0.1, 0.0,  0.0 ]
    # print(node_coords)

    # local to global numbering
    node_l2g = numpy.arange(0, nnodes, 1, dtype=numpy.int64)

    return node_coords, node_l2g, p_elem2nodes, elem2nodes

def _plot_fractal(color='blue'):
    node_coords, node_l2g, p_elem2nodes, elem2nodes = _set_fractal(0.0, 1.0, 0.0, 1.0, 10, 10)


    fig = matplotlib.pyplot.figure(1)
    ax = matplotlib.pyplot.subplot(1, 1, 1)
    nnodes = numpy.shape(node_coords)[0]
    nelems = numpy.shape(p_elem2nodes)[0]
    for elem in range(0, nelems-1):
        xyz = node_coords[ elem2nodes[p_elem2nodes[elem]:p_elem2nodes[elem+1]], :]
        if xyz.shape[0] == 3:
            matplotlib.pyplot.plot((xyz[0, 0], xyz[1, 0], xyz[2, 0], xyz[0, 0]),
                                   (xyz[0, 1], xyz[1, 1], xyz[2, 1], xyz[0, 1]), color=color)
        else:
            matplotlib.pyplot.plot((xyz[0, 0], xyz[1, 0], xyz[2, 0], xyz[3, 0], xyz[0, 0]),
                                   (xyz[0, 1], xyz[1, 1], xyz[2, 1], xyz[3, 1], xyz[0, 1]), color=color)

    matplotlib.pyplot.show()
    return

if __name__ == '__main__':

    # run_exercise_a(10)
    # run_exercise_b()
    # run_exercise_c()
    # run_exercise_d()
    # helmholtz_resonator()
    # criteria_functions_testing()
    # shuffled_quadrangle_grid()
    # test_split()
    _plot_fractal()
    print('End.')
