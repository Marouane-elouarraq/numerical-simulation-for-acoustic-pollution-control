import numpy as np
# points = np.array([[0, 0], [0, 1.1], [1, 0], [1, 1], [0.5, 1.5], [0.8, 0.8]])
points = np.array([[0, 0], [2, 0], [0.3, 1], [2, 1], [0, 2], [0.3, 2], [-2, -1], [4, -1], [1, 5]])
from scipy.spatial import Delaunay
#tri = Delaunay(points)
tri = Delaunay(points, furthest_site=False, incremental=True, qhull_options=None)

import matplotlib.pyplot as plt
plt.triplot(points[:,0], points[:,1], tri.simplices)
plt.plot(points[:,0], points[:,1], 'o')
plt.show()