"""ORIGINAL EXAMPLE:
l1 = 4.95
l2 = 2.765
l3 = 4.95
w1 = 0.10523

"""
import numpy as np
import matplotlib.pyplot as plt

from shapely.geometry import Polygon
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


def set_params(old_geometry, params):
    # assert len(old_geometry) == len(params), "ERROR! different number of polygon and specified parameters"
    # for i in params:
    #     assert len(i) == 2, f"ERROR! Number of specified dimensions is different from 2 for Polygon {i}"

    new = [Polygon(p.shapely.exterior.coords) for p in old_geometry]

    # changing l2-l1
    new[2] = affinity.translate(new[2], xoff=params[0])
    new[3] = affinity.translate(new[3], xoff=params[0])

    new[0] = geom.set_dim(new[0], params[0] + params[1], dim=0, fix=0)
    new[2] = geom.set_dim(new[2], params[0] + params[1], dim=0, fix=0)
    new[3] = affinity.translate(new[3], xoff=params[0] + params[1])


    # changing g
    new[2] = affinity.translate(new[2], yoff=-params[2])
    new[3] = affinity.translate(new[3], yoff=-params[2])

    # changing w
    new[0] = geom.set_dim(new[0], params[3], dim=1, fix=0)
    new[1] = geom.set_dim(new[1], params[3], dim=0, fix=0)
    new[1] = affinity.translate(new[1], yoff=params[3])

    new[2] = geom.set_dim(new[2], params[3], dim=1, fix=1)
    new[3] = geom.set_dim(new[3], params[3], dim=0, fix=1)
    new[3] = affinity.translate(new[3], yoff=-params[3])

    return new


def parametrization_1(d_l1, dl_2, d_g, d_w, eox, hox, debug=True, sampling_mode='adaptive'):
    d_l1 = round(d_l1, 7)
    d_l2 = round(dl_2, 7)
    d_g = round(d_g, 7)
    d_w = round(d_w, 7)
    eox = round(eox, 7)
    hox = round(hox, 7)

    device = Design(init_proj, new_proj)
    # device.ads_path = "C:\\Program Files\\Keysight\\ADS2019\n"
    device.ads_path = "/usr/local/ADS2015_01"

    # load initial design from init_proj folder
    device.load_original()

    # modify frequency sampling in the design class
    device.Sampling = Sampling(mode=sampling_mode, lower=3, higher=7, max_samples=50, step=0.08)

    # if debug:
    #     device.layout_plot(label=True, coords=True)

    # Modifying Layout
    old_geom = device.Layout.polygons
    # set variation amount for each parametrized polygon
    delta = [d_l1, d_l2, d_g, d_w]
    new_geom = set_params(old_geom, delta)
    device.Layout.overwrite_geometry(new_geom, mask='P1')

    # Modifying Ports
    device.Ports.coordinates[0][0] = (new_geom[1].exterior.coords[2][0] + new_geom[1].exterior.coords[3][0]) * 1e-3 / 2
    device.Ports.coordinates[0][1] = (new_geom[1].exterior.coords[2][1]) * 1e-3
    device.Ports.coordinates[1][0] = (new_geom[3].exterior.coords[0][0] + new_geom[3].exterior.coords[1][0]) * 1e-3 / 2
    device.Ports.coordinates[1][1] = (new_geom[3].exterior.coords[0][1]) * 1e-3
    print(device.Ports.coordinates)

    if debug:
        device.layout_plot(label=True, coords=True)

    # Modifying Substrate
    device.Substrate.stack[2].height = hox * 1e-3
    device.Substrate.materials[0].permittivity = eox

    # Save and simulate
    device.write_new()
    device.simulate_new(quiet=not debug)

    # Get frequency response
    f, smatrix = device.get_smatrix()
    smatrix_db = 20 * np.log10(np.abs(smatrix))
    s21 = smatrix[0, 1]
    s11 = smatrix[0, 0]
    p_loss = np.sqrt(1-np.abs(s11)**2 - np.abs(s21)**2)
    p_loss_max = np.max(p_loss)
    if debug:
        plt.plot(f[1:], smatrix_db[0, 1][1:], )
        plt.plot(f, -3*np.ones(len(f)))
        plt.plot(f, -20 * np.ones(len(f)))
        plt.show()

    f = np.asarray(f)
    s21 = np.asarray(s21)

    return f, s21


# def objective1(response, interval, value):  # , bound='high'):
#
#
#     return


if __name__ == '__main__':

    f_s1 = 4e9
    f_p1 = 4.75e9
    f_p2 = 5.25e9
    f_s2 = 6e9

    # original parameters
    l1_0 = 6.977
    l2_0 = 3.02
    g_0 = 0.06
    w_0 = 1
    e_0 = 9.9
    h_0 = 0.254

    l1 = 7.5
    l2 = 2.5
    g = 0.1
    w = 1
    e = 10.32208869
    h = 0.24141866


    d_l1 = l1 - l1_0        # displacement
    d_l2 = l2 - l2_0        # coupling length
    d_g = g - g_0
    d_w = w - w_0

    footprint = (l1 + 10) * (4 * 2 + l1)
    print(footprint)
    freq, s_db = parametrization_1(d_l1, d_l2, d_g, d_w, e, h, debug=True, sampling_mode='linear')

    linear_interp = True
    if linear_interp:
        freq_new = np.linspace(3e9, 7e9, 100)
        s_db_new = np.interp(freq_new, freq, s_db)
        plt.plot(freq_new, s_db_new)
        plt.show()



    seg1 = s_db[np.where(freq < f_s1)]
    seg2 = s_db[np.where((f_p1 < freq) & (freq < f_p2))]
    seg3 = s_db[np.where((f_p1 < freq) & (freq < f_p2))]
    seg4 = s_db[np.where(freq > f_s2)]
    term1 = np.amax(seg1)
    term2 = np.amin(seg2)
    term3 = np.amin(seg3)
    term4 = np.amax(seg4)

    total_d = term1 - term2 - term3 + term4
    print(total_d)


