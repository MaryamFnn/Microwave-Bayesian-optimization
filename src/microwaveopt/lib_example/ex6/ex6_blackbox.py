import numpy as np
import matplotlib.pyplot as plt

from shapely import affinity
from microwaveopt.momentum.design import Design
from microwaveopt.momentum.em_setup import Sampling
from microwaveopt.em_lib import filtering
from microwaveopt.em_lib import geometry as geom
import os

ex_path = __file__
ex_dir = os.path.abspath(os.path.join(ex_path, os.pardir))
init_proj = os.path.join(ex_dir, "ADS_Original_Files")
new_proj = os.path.join(ex_dir, "ADS_Parametric_Layout")


def blackbox1(x=None, debug=False, afs=False):
    # default values
    l1_0 = 6.977
    l2_0 = 3.02
    g_0 = 0.06
    w_0 = 1
    h_0 = 0.254
    e_0 = 9.9

    if x is None:
        x = [l1_0, l2_0, g_0, w_0, h_0, e_0]

    delta_l1 = x[0] - l1_0
    delta_l2 = x[1] - l2_0
    delta_g = x[2] - g_0
    delta_w = x[3] - w_0
    hox_new = x[4]
    eox_new = x[5]

    device = Design(init_proj, new_proj)
    # device.ads_path = "/usr/local/ADS2015_01"
    device.load_original()

    fmin = 3  # GHz
    fmax = 7
    if afs:
        device.Sampling = Sampling(mode='adaptive', lower=fmin, higher=fmax, max_samples=50)
    else:
        device.Sampling = Sampling(mode='linear', lower=fmin, higher=fmax, step=0.08)

    # Modifying layout
    polygons = device.Layout.shapely()
    # changing l2-l1
    polygons[2] = affinity.translate(polygons[2], xoff=delta_l1)
    polygons[3] = affinity.translate(polygons[3], xoff=delta_l1)

    polygons[0] = geom.set_dim(polygons[0], delta_l1 + delta_l2, dim=0, fix=0)
    polygons[2] = geom.set_dim(polygons[2], delta_l1 + delta_l2, dim=0, fix=0)
    polygons[3] = affinity.translate(polygons[3], xoff=delta_l1 + delta_l2)

    # changing g
    polygons[2] = affinity.translate(polygons[2], yoff=-delta_g)
    polygons[3] = affinity.translate(polygons[3], yoff=-delta_g)

    # changing w
    polygons[0] = geom.set_dim(polygons[0], delta_w, dim=1, fix=0)
    polygons[1] = geom.set_dim(polygons[1], delta_w, dim=0, fix=0)
    polygons[1] = affinity.translate(polygons[1], yoff=delta_w)

    polygons[2] = geom.set_dim(polygons[2], delta_w, dim=1, fix=1)
    polygons[3] = geom.set_dim(polygons[3], delta_w, dim=0, fix=1)
    polygons[3] = affinity.translate(polygons[3], yoff=-delta_w)

    device.Layout.overwrite_geometry(polygons, mask='P1')

    # Update port
    device.Ports.coordinates[0][0] = (polygons[1].exterior.coords[2][0] + polygons[1].exterior.coords[3][0]) * 1e-3 / 2
    device.Ports.coordinates[0][1] = (polygons[1].exterior.coords[2][1]) * 1e-3
    device.Ports.coordinates[1][0] = (polygons[3].exterior.coords[0][0] + polygons[3].exterior.coords[1][0]) * 1e-3 / 2
    device.Ports.coordinates[1][1] = (polygons[3].exterior.coords[0][1]) * 1e-3

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
    f, s21 = device.get_s21()

    if debug:
        transfer_mag = np.log10(np.abs(s21))
        plt.plot(f, transfer_mag)
        plt.show()

    return f, s11, s21


def blackbox_constr(x, *args, **kwargs):
    # # default values
    # x[0] = 6.977
    # x[1] = 3.02
    # x[2] = 0.06
    # x[3] = 1
    # x[4] = 0.254
    # x[5] = 9.9

    f, s11, s21 = blackbox1(x, *args, **kwargs)

    p_loss = np.sqrt(1 - np.abs(s11) ** 2 - np.abs(s21) ** 2)
    p_loss_max = np.max(p_loss)  # <10%

    s21_db = 20 * np.log10(np.abs(s21))
    f = np.asarray(f) / 1e9
    f_0 = filtering.central_frequency(f, s21_db, cut_off_value=-3)
    bw = filtering.bandwidth(f, s21_db, cut_off_value=-3)

    obj1 = np.abs(f_0 - 5)
    obj2 = bw
    constr = p_loss_max

    print([obj1, obj2, constr])

    return obj1, obj2, constr


#######################################################
if __name__ == '__main__':
    # x = [7.77710014, 2.13942827, 0.06, 1, 0.2, 11.]
    # y = blackbox_constr(x=None, debug=True, afs=False)


    def obj(x):
        x_1 = [x[0], x[1], 0.06, 1, x[2], x[3]]
        y = blackbox_constr(x_1, debug=True, afs=False)[-1]
        print(x)
        print(y)
        return y




    from scipy.optimize import minimize

    x0 = np.array([6.977, 3.02, 0.254, 9.9])
    res = minimize(obj, x0, method='Nelder-Mead', tol=1e-6)

    print(res.x)
    print(res)
