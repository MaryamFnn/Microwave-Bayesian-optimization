import numpy as np
import os
import matplotlib.pyplot as plt

from microwaveopt.momentum.design import Design
from microwaveopt.momentum.em_setup import Sampling
from microwaveopt.em_lib import geometry as geom
from microwaveopt.em_lib import filtering
from shapely import affinity

ex_path = __file__
ex_dir = os.path.abspath(os.path.join(ex_path, os.pardir))
init_proj = os.path.join(ex_dir, "ADS_Original_Files")
new_proj = os.path.join(ex_dir, "ADS_Parametric_Layout")

# INITIAL DESIGN PARAMETERS:
w1_0 = 0.1185
w2_0 = 0.1185
l2_0 = 2.1946
l1_0 = l2_0 - 2 * w2_0
l3_0 = 0.2
s_0 = 0.1219
hox_0 = 0.127
eox_0 = 9.9

bounds_1 = np.array([[2, 2.5], [0.1, 0.2], [9.5, 11], [0.10, 0.15]])

def blackbox1_4p(x=None, debug=False, afs=False):
    if x is None:
        x = [l2_0, s_0, hox_0, eox_0]

    delta_l2 = x[0] - l2_0
    delta_s = x[1] - s_0
    hox_new = x[2]
    eox_new = x[3]

    device = Design(init_proj, new_proj)
    # device.ads_path = "/usr/local/ADS2015_01"
    device.load_original()

    fmin = 7  # Ghz
    fmax = 21  # Ghz
    if afs:
        device.Sampling = Sampling(mode='adaptive', lower=fmin, higher=fmax, max_samples=50)
    else:
        device.Sampling = Sampling(mode='linear', lower=fmin, higher=fmax, step=0.2)

    # if debug:
    #     device.layout_plot(label=True, coords=True)

    # Modifying layout
    polygons = device.Layout.shapely()
    # changing L2
    polygons[4] = geom.set_dim(polygons[4], delta_l2, 0)
    polygons[0] = geom.set_dim(polygons[0], delta_l2, 0)
    polygons[2] = geom.set_dim(polygons[2], delta_l2, 0)
    polygons[1] = affinity.translate(polygons[1], xoff=delta_l2 / 2)
    polygons[3] = affinity.translate(polygons[3], xoff=-delta_l2 / 2)
    #
    # # changing S
    polygons[1] = geom.set_dim(polygons[1], delta_s / 2, 1, fix=0)
    polygons[3] = geom.set_dim(polygons[3], delta_s / 2, 1, fix=1)
    polygons[2] = affinity.translate(polygons[2], yoff=delta_s / 2)
    polygons[0] = affinity.translate(polygons[0], yoff=-delta_s / 2)

    device.Layout.overwrite_geometry(polygons, mask='P1')

    # Update port
    device.Ports.coordinates[0][0] = (polygons[4].exterior.coords[0][0]) * 1e-3
    device.Ports.coordinates[0][1] = (polygons[4].exterior.coords[0][1] + polygons[4].exterior.coords[3][1]) * 1e-3 / 2
    device.Ports.coordinates[1][0] = (polygons[4].exterior.coords[1][0]) * 1e-3
    device.Ports.coordinates[1][1] = (polygons[4].exterior.coords[1][1] + polygons[4].exterior.coords[2][1]) * 1e-3 / 2

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
    f, s21 = device.get_s21()

    if debug:
        transfer_mag = 20*np.log10(np.abs(s21))
        plt.plot(f, transfer_mag)
        plt.show()

    s21_db = 20*np.log10(np.abs(s21))
    bw = filtering.bandwidth(f, s21_db, method=1)
    f0 = filtering.central_frequency(f, s21_db, method=1)

    return f, s21, f0, bw


########################################################################################################################
if __name__ == '__main__':
    x_test = [l2_0, s_0, hox_0, eox_0]
    f_samp, s11, _, _ = blackbox1_4p(x_test, debug=True, afs=False)
