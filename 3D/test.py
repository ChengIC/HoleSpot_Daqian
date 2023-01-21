import matplotlib.pyplot as plt

import pyvista as pv
from pyvista import examples

# Load an interesting example of geometry
mesh = examples.load_random_hills()

# Establish geometry within a pv.Plotter()
p = pv.Plotter()
zval = p.get_image_depth()