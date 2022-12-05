import numpy as np
import os
import matplotlib.pyplot as plt

from microwaveopt.momentum.design import Design
from microwaveopt.momentum.em_setup import Sampling
from microwaveopt.em_lib import geometry as geom


ex_path = __file__
ex_dir = os.path.abspath(os.path.join(ex_path, os.pardir))
init_proj = os.path.join(ex_dir, "ADS_Original_Files")
new_proj = os.path.join(ex_dir, "ADS_Parametric_Layout")


def blackbox1_4p(x=None, debug=False, afs=False):
    if x is None:
        x = [35, 30.5, 1.5, 4.3]

    delta_w = x[0] - 35
    delta_l = x[1] - 30.5
    hox_new = x[2]
    eox_new = x[3]

    device = Design(init_proj, new_proj)
    # device.ads_path = "/usr/local/ADS2015_01"
    device.load_original()

    if afs:
        device.Sampling = Sampling(mode='adaptive', lower=2, higher=4, max_samples=50)
    else:
        device.Sampling = Sampling(mode='linear', lower=2, higher=4, step=0.04)


    # Modifying layout
    polygons = device.Layout.shapely()
    # Changing W
    polygons[2] = geom.set_dim(polygons[2], delta_w, 0)
    polygons[1] = geom.set_dim(polygons[1], delta_w/2, fix=1)
    polygons[0] = geom.set_dim(polygons[0], delta_w/2, fix=0)
    polygons[2] = geom.set_dim(polygons[2], delta_l, 1, fix=0)

    device.Layout.overwrite_geometry(polygons, mask='P1')

    # Update port
    device.Ports.coordinates[0][0] = (polygons[3].exterior.coords[0][0] + polygons[3].exterior.coords[1][0]) * 1e-3 / 2
    device.Ports.coordinates[0][1] = (polygons[3].exterior.coords[0][1]) * 1e-3

    # Modifying Substrate
    device.Substrate.stack[2].height = hox_new * 1e-3
    device.Substrate.materials[0].permittivity = eox_new

    if debug:
        device.layout_plot(label=True, coords=True)

    # Save and simulate
    device.write_new()
    device.simulate_new(quiet=not debug)
    # device.load_new()

    # Get frequency response
    f, s11 = device.get_s11()

    if debug:
        transfer_mag = np.log10(np.abs(s11))
        plt.plot(f, transfer_mag)
        plt.show()

    return f, s11