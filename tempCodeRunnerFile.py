fig = matplotlib.pyplot.figure(1)
    ax = matplotlib.pyplot.subplot(1, 1, 1)
    ax.set_aspect('equal')
    ax.axis('off')
    solutions._plot_mesh(new_p_elem2nodes, new_elem2nodes, new_node_coords, color='yellow')
    matplotlib.pyplot.show()
    return